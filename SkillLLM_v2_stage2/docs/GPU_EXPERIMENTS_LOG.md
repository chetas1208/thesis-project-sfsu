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
- 2026-10-22 08:54-10:15 UTC  **Step 1a DONE: Exp 1 loop v3 (short prompt)**: 5 rounds, stop reason `plateau` (patience 3), best = round 1.
  Dev exact-span F1 (400 docs, same subset as v2): 0.6789, **0.6904**, 0.6838, 0.6785, 0.6855 (v2: 0.6584, 0.6652, 0.6913, 0.7029, 0.6772, 0.7005).
  **GPU-h: 1.333** (loop budget accounting `loop_summary.json`; 4,797 s). Train tokens 390,578 (78k/round vs v2 ~237k/round = 3.0x fewer).
  Per round (mean of 5): training-step time 338 s (0.094 GPU-h) vs v2 ~710 s (0.197 GPU-h, v2 round 0 from its log; v2 did not store
  train_seconds) = **2.1x lower**; dev eval (400 docs, batch 16) ~382 s, mining probe ~239 s; total round ~961 s (0.267 GPU-h) vs v2
  ~1,906 s (0.53 GPU-h) = 2.0x lower (part of that from the larger generation batch 16 vs 8, i.e. not purely the prompt). Peak alloc 18.1 GB.
  Cumulative GPU-h used so far: **1.33 / 8**.
- 2026-10-22 10:15-10:55 UTC  Step 1b: logprob-path GPU smoke (`scripts/smoke_logprob_eval.sh`, v2 best, 8 test docs, one-at-a-time): all 8 docs gave
  predictions identical to the stored v2 test predictions, all 16 spans got an aligned `metadata.logprob`. Cost ~61 s (28 s load + 33 s gen) = **0.017 GPU-h**
  (guardrails was RED for ~40 min first; waited, no bypass). Observation: v2's span log-probs are close to 0 (many at -0.00x): the
  model is very peaked, expect weak but non-zero ranking information. Cumulative GPU-h: **1.35 / 8**.
  Next: held-out test of v3 best (round 1), ONCE.
- 2026-10-22 11:00-12:00 UTC  **Step 1c DONE: Exp 1 held-out test of v3 best (round 1), ONCE** (`scripts/run_qlora_loop_v3_test_eval.sh`, short prompt, logprobs on):
  974 docs, 3,920 s generation + 19 s load = **1.10 GPU-h** (wall; guardrails was RED part of the time, so this may include paused wall time; the
  per-doc rows store `gen_seconds`). Exact-span F1 **0.6439** (P 0.6751 / R 0.6155) vs v2 0.6614, anchor 0.6701.
  v3 - v2 = -0.0176, 95% CI [-0.0356, +0.0008] (p boot 0.061, Holm-family 0.184 on exact span); v3 - anchor = -0.0262 [-0.0462, -0.0057]
  (p 0.0126, Holm-family 0.025). Token F1 v3-v2 = -0.0256 [-0.043, -0.008] (p 0.005, Holm-5 0.023): the short-prompt adapter is measurably worse on token F1.
  Criteria: (1) point within 0.01 of v2: **NOT MET** (-0.0176); formal NI: not met. (2) train-step GPU-h/round 2.1x lower: **MET**
  (338 s vs ~710 s; total round incl. eval/mining 1.98x lower, marginally under 2x). (3) generation latency 4.02 s/doc vs 4.42 s/doc = 1.10x:
  **NOT MET** (decode-bound, as predicted). **Experiment 1 verdict: FAILED** (cost win in training, quality loss, no latency win).
  Reports: `reports/skillllm_v2/qlora_loop_v3_significance.{json,md}`, `qlora_loop_v3_test_metrics.json`.
  Exp 3 on v3 (CPU): `reports/skillllm_v2/logprob_confidence_v3.{json,md}`.
  Cumulative GPU-h: 1.35 + 1.10 = **2.45 / 8**.
- 2026-10-22 12:00 UTC  Step 2a START: Exp 2 pseudo-label generation (anchor, v1, v2; `scripts/run_pseudo_label_generate.sh`, log `logs/pseudo_label_generate.log`).
- 2026-10-22 12:00-13:50 UTC  **Step 2a DONE: pseudo-label generation**: 2,400 chunks x 3 adapters = 7,200 generations (batch 24, greedy, original prompt),
  0 parse failures, 5,664 s = **1.57 GPU-h** (0.71-0.87 s/doc; guardrails RED for parts, wall time). Cumulative GPU-h: 2.45 + 1.58 (incl. load) = **4.03 / 8**.
  Vote result (`reports/skillllm_v2/qlora_loop_v4_pseudo_counts.json`, counts only): distinct predicted spans by votes 1/2/3 = 932 / 653 / 924; kept (>=2 votes) 1,564
  spans (892 HARD, 672 SOFT; 922 unanimous, 642 two-vote); 664 positive chunks, 86 chunks excluded as uncertain (only 1-vote spans), 1,650 confident-negative
  chunks of which 221 used (<=25% share); pseudo pool = 885 rows (75% positive). **367 of 664 positive chunks (55%) also contain dropped 1-vote spans**
  -> their targets omit spans some adapter found: a built-in recall-noise source for the distilled model (to be examined in the v4 result).
  Step 2b START: v4 distillation loop (`scripts/run_loop_v4_distill.sh`, log `logs/qlora_loop_v4_launch.log`, cap 2.4 GPU-h). Worst-case projection
  4.03 + 2.4 + ~1.2 (test eval) = 7.6 <= 8.
- 2026-10-22 14:20 UTC  Step 2b actually started (guardrails queue wait ~2.5 h: RED, and another process of someone else was using GPU1; no bypass). v4 loop warm start from v2 best loaded (92.41M trainable params).
- 2026-10-22 14:20-15:38 UTC  Step 2b (v4 loop) partial: baseline dev (v2 best, same eval path) **0.7061**; round 0 **0.7031**; round 1 **0.6994**; round 2 trained and dev-evaluated but then
  **crashed with CUDA OOM in the mining probe** (a foreign process, PID 2196145, held ~3.4 GB on GPU1 while my job used 19.4 GB; not mine, not touched).
  `loop_state.json` had committed 2 rounds, so round 2 is redone on resume (its ~0.4 GPU-h is wasted). GPU-h so far: loop 0.928 (committed) + ~0.4 (lost round 2) = ~1.33 for v4;
  cumulative **~5.4 / 8**. Remaining ~2.6: held-out test of v4 needs ~1.15, so v4 loop budget is cut: resume with `budget.max_gpu_hours=2.0` (tracker already at 0.928, i.e. <=~1.07 more:
  rounds 2 and 3) and `eval.batch_size=8` (OOM headroom; dev-eval numbers are insensitive to this). Projection: ~5.4 + 1.07 + 1.15 = 7.6 <= 8.
  Note for the evidence record: the dev curve is flat/slightly below the warm-start baseline after 2 rounds, i.e. no dev signal of benefit from the pseudo-labels so far.
- 2026-10-22 15:40-16:43 UTC  **Step 2b DONE: v4 loop** (resume after the OOM): round 2 redone: dev **0.7052**; round 3: **0.7034**; stopped by `budget_gpu_hours` (cap 2.0).
  Dev curve (400 docs): baseline v2-best 0.7061 | r0 0.7031 | r1 0.6994 | **r2 0.7052 (best)** | r3 0.7034 -> no round beat the warm start; differences are within the dev SE (~0.02).
  Loop tracker GPU-h 1.975 (4 rounds + baseline); plus ~0.4 for the crashed round-2 attempt = ~2.4 GPU-h for v4 training. Cumulative GPU-h: 4.03 + 2.4 = **~6.4 / 8**.
  Pre-declared selection: best v4 round by dev = round 2 (`artifacts/skillllm_v2/qlora_loop_v4_distill/best`). Step 2c START: held-out test of v4 best, ONCE (est. 1.15 GPU-h -> ~7.55 total).
- 2026-10-22 16:45-17:58 UTC  **Step 2c DONE: v4 held-out test (round 2), ONCE** (`scripts/run_qlora_loop_v4_test_eval.sh`, original prompt, logprobs on; guardrails RED part of the time): 974 docs, 4,465 s gen
  + 19 s load = **1.25 GPU-h** (4.58 s/doc mean, wall). Exact-span F1 **0.6846** (P 0.7059 / R 0.6645). v4-anchor +0.0145 [-0.0044, +0.0330] p 0.138 (Holm-3 0.276);
  v4-v2 +0.0231 [+0.0054, +0.0403] p 0.0088 (Holm-3 0.026); v4-ensemble -0.0053 [-0.0220, +0.0108] p 0.52. Criterion A not met; criterion B met (point). Details: `reports/skillllm_v2/qlora_loop_v4_{summary.md,significance.*}`.
  Exp 3 on v4 (CPU): `reports/skillllm_v2/logprob_confidence_v4.{md,json}`; combined summary `reports/skillllm_v2/logprob_confidence_summary.md`. Full test suite after all edits: 513 passed, 6 skipped (the 4 earlier `tests/evals` failures are gone, fixed elsewhere).
  **Total GPU-h = 1.333 (v3 loop) + 0.017 (logprob smoke) + 1.10 (v3 test) + 1.58 (pseudo-label gen) + 1.975 (v4 loop tracker) + <=0.4 (lost OOM round) + 1.25 (v4 test) = ~7.65 / 8.**
- 2026-10-23 ~10:00 UTC  Lead asked for state verification after a session stop: all GPU jobs had finished (no process running, nothing missing); no further GPU work was run. `guardrails status` was RED (watcher running).

## Final summary

| Experiment | Pre-declared success criteria | Outcome | GPU-h | Verdict |
|---|---|---|---|---|
| 1. Short prompt (v3) | (1) held-out exact-F1 within 0.01 of v2 (0.6614), CI reported; (2) >= 2x lower train GPU-h/round; (3) >= 1.5x lower generation latency/doc | tokens/example 382 -> 124 (3.08x fewer; fixed prompt 300 -> 41). (1) 0.6439, v3-v2 -0.0176 [-0.0356, +0.0008], v3-anchor -0.0262 [-0.0462, -0.0057]: NOT met. (2) train step 338 s vs ~710 s = 2.1x: met (total round 1.98x). (3) 4.02 vs 4.42 s/doc = 1.10x: NOT met | 1.33 loop + 1.10 test + 0.02 smoke = 2.45 | **FAILED** (cheaper training, worse accuracy, no latency gain) |
| 2. Distillation (v4) | (A) v4 > anchor 0.6701 with paired CI excluding 0, or (B) within 0.01 of ensemble 0.6899 at 1x cost | test 0.6846 (P 0.706 / R 0.664). A: +0.0145 [-0.0044, +0.0330] p 0.138 NOT met. B: -0.0053 vs ensemble [-0.022, +0.011] met (point). vs v2 +0.0231 [+0.0054, +0.0403]. Dev curve never beat the warm start (0.7061 vs best 0.7052); no gold-only control | 1.58 gen + ~2.4 loop (incl. lost OOM round) + 1.25 test = ~5.2 | **PARTIAL: B met, A not; benefit not attributable to pseudo-labels** |
| 3. Logprob confidence | primary: AUROC of exp(mean lp) > 0.5 with CI excluding 0.5; compare with constant 0.86 and vote count | v4 AUROC 0.701 [0.676, 0.725], v3 0.714; AURC 0.168/0.183 vs 0.294/0.325 constant; ~= vote count (AUROC diff +0.006 / -0.010, ns), better AURC; combined best (0.734 / 0.767); raw exp(lp) over-confident (ECE 0.26 > constant 0.15-0.18) | 0 (rode on the held-out runs) | **MET as a ranking signal; not usable as a raw probability without recalibration** |
| Total | cap 8 GPU-h | ~7.65 | ~7.65 | within cap |
