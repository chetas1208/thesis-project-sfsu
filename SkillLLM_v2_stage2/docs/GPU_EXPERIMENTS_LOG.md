# GPU experiments log (SkillLLM_v2 Mistral-Small-3.1-24B QLoRA)

Hard rules: GPU1 only (`CUDA_VISIBLE_DEVICES=1`), every GPU command through
`guardrails run --ram 16 gpu1 -- bash <abs script>`, one GPU job at a time, **total cap 8 GPU-hours** across all experiments
below. GPU-h = wall-clock seconds of the GPU job (loop trainer bills train steps + generation, clamping a single event at 120 s so
guardrails pauses are not billed; eval/inference scripts below are billed as wall seconds of generation, load time shown separately).
Prior art for reference (not counted against this cap): v2 loop 3.17 GPU-h, v2 held-out test eval ~1.20 GPU-h.

Evidence rules: success criteria are fixed **before** the run (below, written 2026-10-22 before any GPU work); tuning only on the
fixed 400-doc dev subset; held-out test (974 gold-bearing SkillSpan test docs) run ONCE per final adapter; paired bootstrap /
permutation via `skillllm_v2.eval.significance` (Holm); negative results reported plainly.

## Pre-declared plan and success criteria (written before any GPU job)

### Experiment 1 - short prompt (cost cut)
- Prompt `short` (single candidate, not tuned): system `Extract skill spans.`; user
  `Output {"skills":[{"text":"<exact span>","label":"HARD_SKILL|SOFT_SKILL"}]}\n\nText:\n{text}`. Output schema, label set and target JSON are
  unchanged; the rules/negative-instructions/"SKILL" label prose is removed. Selected with `prompt.version: short` (config) /
  `prompt_version="short"` (`MistralExtractor`, `build_mistral_sft_data`); default stays `original`.
- Measured by `loop_trainer --plan --count-tokens` (CPU, exact tokenizer): original 382 tokens/example (prompt 332 = 300 fixed + 32 text,
  target 50; fixed prompt = 78.5% of tokens), short 124 tokens/example (fixed 41; 33%). Expected 3.08x fewer tokens per training example.
- Run: `configs/skillllm_v2_loop_qlora_v3_shortprompt.yaml` = v2 hyperparameters (r=16, alpha 32, dev 400 docs, patience 3, max_rounds 6,
  640 examples/round, same seed so identical 400-doc dev subset) -> `artifacts/skillllm_v2/qlora_loop_v3_shortprompt`. Only other change:
  `eval.batch_size` 8 -> 16 (dev/probe generation throughput only; greedy). Budget cap for this loop 3.2 GPU-h.
- Held-out test: dev-selected best adapter, ONCE, same scoring path as v1/v2 (`scripts/eval_qlora_loop_test.py`, one doc at a time,
  greedy, max_new_tokens 512) but with the short prompt -> `artifacts/skillllm_v2/qlora_loop_v3_eval`.
- Success (all three required):
  1. Non-inferiority on held-out exact-span F1 vs v2 (0.6614): point estimate v3 >= 0.6514 ("within 0.01"); the 95% paired-bootstrap CI of
     (v3 - v2) is reported. NOTE: with 974 docs the CI half-width is ~0.018 (MDE ~0.03), so a *formal* non-inferiority proof at margin 0.01
     (CI lower bound > -0.01) is only possible if v3 is better than v2 by ~0.009; I report both readings and do not call equivalence shown
     unless the formal one passes.
  2. Train GPU-h per round: >= 2x lower. Primary = pure training-step time per round (`train_seconds`; v2 round 0 = 0.197 GPU-h, from
     logs). Secondary (reported, not the criterion) = total round GPU-h including the dev eval + mining generation.
  3. Generation latency per test doc >= 1.5x lower than v2 (v2: 4303 s / 974 docs = 4.42 s/doc, `qlora_loop_v2_eval/timing_segments.jsonl`).
  - Prior (stated before running): 1 and 2 plausible; 3 doubtful, because generation is decode-bound (the ~50-token JSON answer is
    unchanged), the prompt only affects the prefill.

### Experiment 2 - ensemble distillation / self-training
- Unlabeled text: `SkillLLM_v2/Extract_jobDesc/data/job_descriptions.jsonl` (1,887 postings; local/private, text never copied to git/reports,
  counts only). Step 0 (CPU): remove every posting that overlaps SkillSpan train/dev/test text: a SkillSpan sentence with >= 8 normalised
  (lower-case alphanumeric) tokens counts as overlapping if >= 50% of its 8-gram shingles occur in the posting (100% = normalised
  substring). Report removed count. Also drop individual chunks equal (normalised) to any SkillSpan dev/test sentence.
- Sample up to 600 of the surviving postings (seed 20261022); split into line/sentence chunks (>= 4 words, <= 300 chars) and sample <= 4
  chunks per posting (<= 2,400 chunks).
- Pseudo-labelling: anchor + v1 + v2 adapters (all three were trained with the ORIGINAL prompt, so the original prompt is used regardless of
  Exp 1's outcome; the short prompt could only apply to the v3 adapter), greedy, batch generation. Keep spans predicted by >= 2 of 3
  (exact start/end/label), the same rule as the ensemble. Chunk inclusion: chunks with >= 1 voted span are positives; chunks where no adapter
  predicted anything are confident negatives; chunks whose only spans are 1-vote are excluded as uncertain.
- Training: `configs/skillllm_v2_loop_qlora_v4_distill.yaml`: warm start from v2 best (r=16), same loop trainer, 640 examples/round with
  `extra_frac: 0.5` (1:1 gold : pseudo per round; gold half still uses hard-example mining on gold train probes only; pseudo docs
  drawn without replacement), same dev subset/selection metric, patience 3, max 5 rounds (budget). Output
  `artifacts/skillllm_v2/qlora_loop_v4_distill`. Held-out test ONCE on the dev-selected best round.
- Success (either): (A) v4 > anchor (0.6701) on exact-span F1 with the paired CI (95%) excluding 0; or (B) v4 within 0.01 of the
  3-adapter ensemble (0.6899): point estimate >= 0.6799, CI reported (single adapter = 1x inference cost). Also reported: precision/recall
  vs v2 and the ensemble, and whether recall dropped (pseudo-label noise).
- Holm correction: over the exact-span comparisons of all new adapters vs their references (list in the significance report).

### Experiment 3 - log-probability confidence
- Option `record_logprobs=True` in `MistralExtractor` stores `metadata.logprob = {sum, mean, min, n_tokens}` per span (log-probs of the
  generated tokens overlapping the span's JSON string; None if the span could not be aligned to the raw output). Enabled in the Exp 1 and
  Exp 2 held-out generation runs; no extra generation.
- Analysis (CPU): per predicted span of the new systems (v3, v4): correct = exact (start, end, label) match with gold. Confidence candidates:
  constant 0.86; exp(mean logprob) (primary); exp(sum logprob) (sequence probability); min logprob; vote count (number of anchor/v1/v2
  predicting the same span, 0-3). Metrics: ECE (10 equal-width bins), Brier, AURC, AUROC with document-cluster bootstrap CIs, using
  `skillllm_v2.evals.calibration_evals` primitives (read-only import). Raw exp(.) values are compared for ECE/Brier; ranking metrics
  (AUROC, AURC) for all; recalibrated variants use 5-fold document-grouped cross-fitting (no fit on the scored fold).
- Primary question: is AUROC of mean-logprob > 0.5 (the constant's AUROC) with a CI excluding 0.5, and how does it compare to vote count.
  Descriptive; no multiple-comparison claim.

## Step log (append after every step)

- 2026-10-22 08:54 UTC  Step 1a START: Exp 1 loop v3 (short prompt), GPU1 via guardrails (GREEN), launcher
  `scripts/run_loop_v3_shortprompt.sh`, log `logs/qlora_loop_v3_launch.log`. Code/tests done before launch: prompt_version option
  (MistralExtractor / build_mistral_sft_data / loop_trainer / mistral_lora_trainer), `extra_frac` pseudo-row mixing, logprob capture,
  `--plan --count-tokens`; new tests `tests/skillllm_v2/test_prompt_version_and_logprobs.py` (28 loop/prompt tests pass). Full suite baseline before
  my edits: 405 passed, 4 failed (all in `tests/evals/`, other agents' files; not touched).
  CPU dry-run: 124 vs 382 tokens/example (3.08x fewer; see plan above).
- 2026-10-22 08:56 UTC  Exp 2 step 0 (CPU, no GPU-h): `scripts/prepare_pseudo_corpus.py` -> counts in
  `reports/skillllm_v2/qlora_loop_v4_corpus_counts.json`: 1,887 postings, **55 removed** for SkillSpan overlap (rule above), 1,832
  surviving, 600 sampled, 2,400 chunks selected (<=4 per posting; 34 chunks equal to a SkillSpan dev/test sentence had been dropped).
  Pre-declared addition (before seeing any pseudo-label): the pseudo pool keeps ALL positive chunks and subsamples confident-negative
  chunks so that they are <= 25% of the pool (mirrors the loop's `empty_frac_end`; SkillSpan train itself is 67% empty docs, which the
  loop deliberately down-weights).
- 2026-10-22 (CPU prep while v3 trains, no GPU-h): written and syntax/unit checked: `scripts/pseudo_label_generate.py` (+`run_pseudo_label_generate.sh`),
  `scripts/build_pseudo_sft.py`, `configs/skillllm_v2_loop_qlora_v4_distill.yaml` (+`run_loop_v4_distill.sh`; pre-declared: warm start v2 best,
  lr 7e-5 = v2's last-round LR, extra_frac 0.5, max 5 rounds, baseline_eval true, cap 2.4 GPU-h), `scripts/eval_adapter_test.py` (+v3/v4
  wrappers; prompt_version + logprob options), `scripts/significance_v3_v4.py` (smoke-tested: reproduces v2 numbers), `scripts/logprob_confidence_eval.py`
  (smoke-tested on v2 predictions).
