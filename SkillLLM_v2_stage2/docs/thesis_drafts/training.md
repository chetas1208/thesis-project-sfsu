# Training: data, the 80-epoch anchor, the QLoRA loop, the ensemble, and what the evidence supports

Draft for the thesis committee. Status as of 2026-10-22. Scope: the supervised fine-tuning (SFT) of the skill-span extractor (Mistral-Small-3.1-24B-Instruct-2503 plus LoRA adapters) and the three-adapter ensemble built from it. Every number below was read from a file in this repository (path given as `reports/...`, `artifacts/...`, `configs/...`, `docs/...`, `logs/...`) or is marked **derived** (arithmetic on read numbers, shown) or **pending** (from an experiment that has not finished). Nothing here was re-trained or re-run for this draft; token counts in Table T1/T2 were recomputed on CPU from the SkillSpan files with the tokenizer saved next to the anchor adapter.

Companion drafts: `docs/thesis_drafts/pretraining.md` (domain-adaptive pretraining feasibility), `evals.md`, `inference.md`; plan and claims table: `docs/BEST_PLAN.md`. Master audit with the full list of inconsistencies: `docs/INFERENCE_AND_EVALS.md`.

---

## 1. Data

### 1.1 Source and splits

All extraction training and evaluation uses SkillSpan (Zhang, Jensen, Sonniks and Plank, NAACL 2022; bibliography key `zhang2022skillspan` in `docs/thesis/references.bib`), local copy `data/source/skillspan_hf/{train,dev,test}.json` (one JSON row per sentence-like unit: `tokens`, `tags_skill`, `tags_knowledge`, `source` in {tech, house}). The code maps SkillSpan `tags_knowledge` to `HARD_SKILL` and `tags_skill` to `SOFT_SKILL` (an assumption baked into `train/build_verifier_dataset.py::load_skillspan_split`, per `docs/INFERENCE_AND_EVALS.md` section 2.0). Dataset licence recorded in `MODEL_CARD.md`: CC BY 4.0 (to be confirmed with the authors before any release).

**Table T1. SkillSpan splits as used here.** Rows, gold-bearing rows and span counts are from `docs/INFERENCE_AND_EVALS.md` section 2.0 (computed by that audit with `load_skillspan_split`); word counts, unique-text counts and Mistral-tokenizer token counts were recomputed for this draft (CPU only).

| Split | Rows (docs) | Rows with at least 1 gold span | Share of rows with no gold span (derived) | Gold spans | HARD | SOFT | Words | Mistral tokens of the text | Unique texts | Rows from tech / house |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 4,800 | 1,584 | 67.0% | 5,190 | 2,969 | 2,221 | 93,453 | 124,441 | 3,952 | 3,133 / 1,667 |
| dev | 3,174 | 914 | 71.2% | 2,163 | 1,093 | 1,070 | 40,057 | 53,701 | 2,611 | 2,156 / 1,018 |
| test | 3,569 | 974 | 72.7% | 2,265 | 1,174 | 1,091 | 42,786 | 56,854 | 2,670 | 2,286 / 1,283 |
| all | 11,543 | 3,472 | | 9,618 | 5,236 | 4,382 | 176,296 | 234,996 | | |

(The "all" row is derived by summing the rows above. The total labelled corpus is therefore about 0.23 million tokens: this matters for section 5 and for `pretraining.md`.)

Evaluation population. Every reported extraction number for the Mistral adapters, the wrapper variants and the ensemble is computed on the **974 test rows that have at least one gold span** (2,265 gold spans), not on all 3,569 test rows (`docs/INFERENCE_AND_EVALS.md` section 0, item 6; `reports/skillllm_v2/qlora_loop_summary.md`). The 2,595 gold-free test rows, on which a deployed extractor would produce false positives, are excluded from scoring. Training, in contrast, sees about 67% empty-answer rows (derived, Table T1). Dev selection in the QLoRA loop likewise uses gold-bearing dev rows only (`configs/skillllm_v2_loop_qlora.yaml`: `eval.dev_only_with_gold: true`, `dev_empty_frac: 0.0`).

Repetition across splits. `reports/skillllm_v2/qlora_loop_summary.md` ("Leakage and protocol checks") states that 177 train and 169 dev normalised texts also occur among the test rows, mostly short strings, and that there are 920 unique texts among the 974 gold-bearing test rows. I did not recompute these figures. The anchor and both loop adapters trained on the same train split, so the effect is shared by all systems.

### 1.2 SFT format and the fixed prompt

SFT files: `data/skillllm_v2/mistral_sft/{train,dev,test}.jsonl` (4,800 / 3,174 / 3,569 rows). Each example is a chat (`SYSTEM_PROMPT` plus `USER_PROMPT_TEMPLATE` from `src/skillllm_v2/extract/mistral_extractor.py`, identical to the inference prompt) whose assistant turn is the strict JSON `{"skills":[{"text": ..., "label": ...}]}` built by `train/mistral_lora_trainer.py::_target_json`.

**Table T2. Where the training tokens go (anchor SFT format, train split; recomputed on CPU).** "Fixed prompt" = system prompt (56 tokens) plus user template with the text slot empty (240 tokens) plus an approximate 8 tokens of chat-control markup (the 8 is my approximation, not measured).

| Component | Tokens per epoch | Share (derived) |
|---|---:|---:|
| Fixed prompt (304 per example x 4,800) | 1,459,200 | 86.4% |
| Input text | 124,441 | 7.4% |
| Answer JSON | 105,267 | 6.2% |
| Total | 1,688,908 | 100% (351.9 tokens per example) |

The fixed prompt is therefore about 86% of all training tokens, consistent with the "about 85%" in `docs/loop_training.md`. A compact prompt now exists in the code (`SYSTEM_PROMPT_SHORT`, `USER_PROMPT_TEMPLATE_SHORT`): I measured about 77 tokens including the same 8-token markup approximation, which would give 599,308 tokens per epoch (2.8x fewer; derived). Its quality effect is **pending** (section 3.5); adapters trained with the original prompt must keep using the original prompt.

---

## 2. The 80-epoch anchor ("Raw Mistral LoRA")

### 2.1 Configuration

Code: `train/train_mistral_lora.py` launching `train/mistral_lora_trainer.py` (HF `Trainer`, PEFT) with `torchrun`, 2 processes. Effective hyper-parameters are those saved with the adapter, `artifacts/skillllm_v2/mistral_lora/model/mistral_lora_train_config.yaml` (checked directly for this draft): base `models/mistral-small-3.1-24b-instruct-2503`, 4-bit (`use_4bit: true`), LoRA r=16, alpha=32, dropout 0.05, target modules the language-model attention and MLP projections, per-device batch 1, gradient accumulation 32 (effective batch 64 on 2 GPUs), learning rate 1.2e-4, warm-up ratio 0.05, weight decay 0.01, `max_seq_length` 512 (longer examples truncated), gradient checkpointing, seed 42, 80 epochs, `save_total_limit` 2. 80 epochs correspond to 6,000 optimizer steps (`checkpoint-6000/trainer_state.json`: `global_step 6000`, `max_steps 6000`).

Loss is computed over **all tokens including the fixed prompt and the input text** (`docs/INFERENCE_AND_EVALS.md` section 2.2; `docs/loop_training.md`). The anchor's objective therefore includes language-model loss on the 124,441 input-text tokens at every epoch, which the loop runs (assistant-token loss only) do not have. No experiment isolates this difference (section 6).

Numerical precision: `configs/skillllm_v2_training_80ep_2gpu.yaml` lists `torch_dtype: float16` and `mixed_precision: fp16`, while the audit states "bf16 compute (bf16 if supported, else fp16)". The saved effective config does not record a dtype, so which one ran is **not verified** by me.

### 2.2 Compute

**Table T3. Anchor training compute.** Source: `artifacts/skillllm_v2/mistral_lora/model/training_summary.json` (the task brief cited `artifacts/skillllm_v2/mistral_lora/training_summary.json`; the file is in the `model/` subfolder).

| Quantity | Value | Note |
|---|---|---|
| `train_runtime` | 324,217.8 s = 90.06 h | wall clock, includes the 80 per-epoch dev evaluations |
| `world_size` | 2 (RTX 3090, 24 GB class) | `torchrun`, `gpu_component_availability.md` |
| GPU-hours | **180.1** | 90.06 h x 2 (derived) |
| `train_samples_per_second` | 1.184 | |
| `total_flos` | 1.857e19 | HF estimate |
| `train_loss` (average over run) | 0.1431 | dominated by the easily memorised fixed prompt |
| final `eval_loss` (dev, 3,174 rows) | 0.3974 | `eval_runtime` 700.1 s |
| Tokens processed (derived) | about 135 million | 80 x 1,688,908 (Table T2), approximate |
| Throughput (derived) | about 417 tokens/s on 2 GPUs, about 208 tokens/s per GPU | 1.184 x 351.9; includes evaluation time, so training-only throughput is higher |

The audit estimates that per-epoch evaluation consumed about 15.6 h of the 90 h (700 s x 80; **derived estimate, not recorded**). Peak GPU memory for this run is **not recorded** anywhere I read.

### 2.3 Dev loss and the over-training question

From `artifacts/skillllm_v2/mistral_lora/model/checkpoint-6000/trainer_state.json` (80 per-epoch evaluations, read directly):

**Table T4. Dev loss (all-token cross-entropy over the 3,174 dev rows).**

| Epoch | 1 | 3 (minimum) | 10 | 76 | 80 (shipped adapter) |
|---|---:|---:|---:|---:|---:|
| Dev loss | 0.2439 | **0.2340** | 0.3087 | 0.3971 | 0.3974 |

`best_model_checkpoint` is `None`: no best-checkpoint selection was applied and the shipped adapter is the epoch-80 weights, whose dev loss is 0.163 above the minimum (derived: 0.3974 - 0.2340), a 70% increase. Interpretation, carefully bounded:

- Dev loss is an all-token loss in which the memorised fixed prompt (86% of tokens) contributes almost nothing and the answer and text tokens contribute the rest, so the rise reflects fitting of train-specific content.
- **Dev F1 was never computed per epoch**, and the epoch-3 checkpoint was not kept (only `checkpoint-5925` and `checkpoint-6000` exist). The loss curve is therefore not proof that extraction F1 degraded; it shows that the model overfits the cross-entropy objective. The indirect evidence that long training is not needed is the loop result in section 3 (a rank-8 adapter after about 0.26 epoch reaches test F1 within 1.4 points of the anchor).

### 2.4 Two identical runs and the seed noise floor

The 80-epoch training was run twice with the same configuration and seed 42 (`docs/INFERENCE_AND_EVALS.md` section 2.2; logs in `logs/`):

**Table T5. Two runs of the same recipe.**

| Run | Dates | `train_runtime` | `train_loss` | Test exact-span F1 | token F1 | SOFT F1 | HARD F1 | Where the number lives |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 2026-05-28 to 06-01 | 3.237e5 s | 0.1428 | 0.675034 | 0.797798 | 0.609792 | 0.731138 | only in `logs/raw_mistral_eval_2gpu_20260603_081829.log`; adapter overwritten |
| 2 (on disk) | 2026-07-09 to 07-13 | 3.242e5 s | 0.1431 | **0.670069** | 0.787886 | 0.604046 | 0.726450 | `artifacts/skillllm_v2/mistral_lora/metrics.json`, `reports/skillllm_v2/raw_mistral_lora_metrics.json` |

The exact-F1 difference is 0.004965 (derived), about 0.005. Both runs share seed, data and config, so this is the practical noise floor from one nominally deterministic recipe (GPU non-determinism, data-order and kernel effects): **at least one such 0.005 swing must be allowed for in any single-run comparison**. It is larger than every wrapper effect (at most +0.0024) and about a quarter of the ensemble effect (+0.0198). Two runs do not give a variance estimate (n = 2); they give a lower bound on what a "different seed" could do. Run 1's adapter is gone, and the gate configuration and several docstrings still quote run 1's anchor (inconsistency I1 in `evals.md` Table E5).

---

## 3. The QLoRA loop (low-compute replacement)

### 3.1 Method

Code `train/loop_trainer.py`, config `configs/skillllm_v2_loop_qlora.yaml`, launcher `scripts/train_mistral_qlora_loop.sh`, method note `docs/loop_training.md`, tests `tests/skillllm_v2/test_loop_trainer.py` (not executed by me). In words: instead of one long run, train in short **rounds** of 640 examples (about 0.13 epoch of the 4,800-row train split) with a cosine learning-rate schedule inside each round and a 0.85 decay of the peak rate across rounds; after each round, generate on a held-out **dev subset** and score exact-span F1, keep the best-by-dev adapter, and **mine hard examples** from a rotating 192-row probe of the *train* split (misses, spurious spans, boundary and label errors weighted 1.0, 1.0, 0.75, 0.75; rows with score at least 0.5 enter a hard pool, retired once fixed). The next round mixes 35% hard rows (repeated at most 3 times) with fresh unseen rows; the share of gold-free rows ramps from 10% to 25% (positives first, then hard negatives against false positives). Dev rows are never mined, so dev F1 remains a selection signal not contaminated by training reuse (though it is reused for selection; see 4.3).

**Table T6. Settings common to both runs** (`configs/skillllm_v2_loop_qlora.yaml` and each run's `resolved_config.json`, per the audit).

| Setting | Value |
|---|---|
| Base / quantisation | Mistral-Small-3.1-24B-Instruct-2503, 4-bit NF4, double quantisation, bf16 compute |
| Optimiser, batch | `paged_adamw_8bit`, micro-batch 2 x accumulation 8 = 16 |
| Learning rate | 1.5e-4, x0.85 per round (floor ratio 0.1), warm-up ratio 0.1 |
| Sequence handling | `max_seq_length` 768, longer examples **dropped, not truncated** |
| Loss | assistant tokens only (prompt and input text masked) |
| Round size / hard fraction | 640 examples / 35%, repeat cap 3 |
| Mining probe | 192 train rows per round, 30% gold-free, hard pool max 400 |
| Dev | gold-bearing dev rows only, metric exact-span F1; generation `max_new_tokens` 384 during selection (test evaluation uses 512) |
| Budget | hard cap 5.0 GPU-h and 2.5 M tokens; training stops at 90% of the cap |
| Seed | 42 |

### 3.2 The two runs

**Table T7. QLoRA loop v1 and v2.** Sources: `reports/skillllm_v2/qlora_loop_summary.md`, `reports/skillllm_v2/qlora_loop_v2_summary.md`, `artifacts/skillllm_v2/qlora_loop{,_v2}/loop_summary.json` (verified directly for stop reason, rounds, best round, GPU-hours, tokens).

| | Loop v1 (`artifacts/skillllm_v2/qlora_loop/best`) | Loop v2 (`artifacts/skillllm_v2/qlora_loop_v2/best`) |
|---|---|---|
| LoRA r / alpha (trainable parameters) | 8 / 16 (about 46 M) | 16 / 32 (92.41 M per `logs/qlora_loop_v2_launch.log`) |
| Dev subset | 160 gold-bearing dev rows | 400 gold-bearing dev rows |
| Patience / max rounds | 2 / 8 | 3 / 6 |
| Rounds run, stop reason | 4, `plateau` | 6, `max_rounds` (patience never fired) |
| Dev exact-F1 by round | 0.6730, 0.7052, 0.7020, 0.6965 | 0.6584, 0.6652, 0.6913, 0.7029, 0.6772, 0.7005 |
| Selected round (index from 0) | 1, dev 0.7052 | 3, dev 0.7029 |
| Training GPU-hours | 1.591 (5,728 s) | 3.173 (11,424 s) |
| Share of the anchor's 180.1 GPU-h (derived) | 0.88% | 1.76% |
| GPU-h to reach the selected adapter | 0.76 (0.42%) | 2.09 (1.16%) |
| Training tokens | 945,329 | 1,419,175 |
| Training tokens vs anchor's about 135 M (derived) | 0.70% | 1.05% |
| Examples behind the selected adapter | 1,258 (about 0.26 epoch; per v1 report) | up to 4 x 640 = 2,560 (about 0.53 epoch; **derived upper bound**, not stated in the v2 report) |
| Test-set generation cost (974 docs) | 3,524.4 s (3.62 s per doc, derived) + 16.6 s load, peak allocated 16.13 GB | 4,303.1 s (4.42 s per doc, derived) + 20.4 s load, peak allocated 16.32 GB |

The only configuration differences between v1 and v2 are rank/alpha, dev-subset size, patience and round cap (audit diff of `resolved_config.json`), so a v1-to-v2 change cannot be attributed to any single factor. Training peak memory: `docs/loop_training.md` states "measured peak 16.4 GB allocated" while `reports/skillllm_v2/qlora_loop_summary.md` states "17.9 GB allocated per round" for v1; these disagree (the latter was itself flagged by the audit as not re-derived from a log). The v1 report's single-GPU wall time equals GPU-hours (1 GPU).

Two things the numbers hide. First, v2 was stopped by the round cap while dev was still fluctuating (0.6772 at round 4, 0.7005 at round 5): the plateau claim of v1 does not transfer, and the curve beyond round 5 is unknown. Second, the loop uses a different loss mask, rank (v1), batch size, learning-rate schedule, sequence-length rule and gold-free-row share than the anchor (Table T8). A "loop versus 80 epochs" comparison is a comparison of two recipes, not of one factor.

**Table T8. What differs between the anchor and the loop (none of these is ablated).**

| Factor | Anchor | Loop |
|---|---|---|
| Training length | 80 epochs (6,000 steps, batch 64) | about 0.26 epoch (v1) / up to about 0.53 epoch (v2, derived), batch 16 |
| Loss | all tokens (prompt, text, answer) | assistant tokens only |
| Rank | 16 | 8 (v1), 16 (v2) |
| Over-long examples | truncated at 512 | dropped above 768 |
| Gold-free rows | natural share, about 67% | curriculum 10% to 25% |
| Hard-example mining | none | yes |
| Checkpoint choice | last epoch | best by dev F1 |
| GPUs | 2 | 1 |

### 3.3 Hard-example mining and budget: what is and is not shown

The code implements the mining and the budget guard and the unit tests exist, but **no experiment compares the loop with and without mining**, with and without prompt-token masking, or with a random-order baseline of the same token budget. The claim "mining helps" is therefore not supported; the supported claim is "this recipe, as a whole, reaches within about 1 point of the anchor at about 1 to 2% of the GPU-hours". The 5.0 GPU-h and 2.5 M-token caps were never binding (v2 stopped at 3.17 GPU-h and 1.42 M tokens).

### 3.4 Differences between the evaluation of the anchor and the loop

Same prompt, greedy decoding, `max_new_tokens` 512, one document at a time, same 974 test rows and scorer; only the adapter path differs. Not identical: the anchor was generated with two visible GPUs (`device_map=auto` split) and the loop adapters on GPU1 only, which can change 4-bit floating-point results slightly; I did not re-run the anchor, so any effect is unquantified (the loop reports expect it to be tiny).

### 3.5 Pending: GPU experiments 1 to 3 (`docs/GPU_EXPERIMENTS_LOG.md`)

A separate GPU agent is running a pre-declared experiment series under a **total cap of 8 GPU-hours across the new experiments** (GPU1 only, one job at a time; the v2 loop's 3.17 GPU-h and its 1.20 GPU-h test evaluation are stated not to count against the cap). Success criteria were written before any GPU work (log dated 2026-10-22). I only read the log and configs; **every result below is pending**.

**Table T13. Pre-declared experiments (pending).**

| Experiment | What | Pre-declared success criterion | Status when last read (08:56 UTC) |
|---|---|---|---|
| 1. Short prompt (loop v3) | v2 recipe (r=16, 400-row dev, patience 3, 6 rounds, seed 42, same dev subset) with a compact prompt; dev batch 16 instead of 8 (throughput only); cap 3.2 GPU-h; output `artifacts/skillllm_v2/qlora_loop_v3_shortprompt`; test run once on the dev-selected adapter into `qlora_loop_v3_eval` | (1) test exact-F1 point estimate at least 0.6514 (within 0.01 of v2's 0.6614), CI of v3 minus v2 reported, and equivalence called only if the formal non-inferiority at margin 0.01 passes; (2) training-step GPU-h per round at least 2x lower (v2 round 0: 0.197 GPU-h); (3) test generation latency per document at least 1.5x lower than v2's 4.42 s. The log states the prior that (1) and (2) are plausible and (3) is doubtful because decoding dominates | Step 1a started 08:54 (guardrails launcher, log `logs/qlora_loop_v3_launch.log`). **Interim, round 0 training steps only** (read at 09:00): 40 steps, 76,685 tokens, 0.095 GPU-h, against v2 round 0's 233,499 tokens and 0.197 GPU-h (2.1x lower; 3.0x fewer tokens; derived). Dev scores, later rounds and test results are pending |
| 2. Ensemble distillation / self-training (loop v4) | Pseudo-label up to 2,400 chunks drawn from 600 of the 1,832 job postings that survive an overlap filter against SkillSpan (55 of the 1,887 postings removed; `reports/skillllm_v2/qlora_loop_v4_corpus_counts.json`), keeping spans predicted by at least 2 of 3 adapters; chunks with no predicted span are confident negatives (capped at 25% of the pool); warm start from v2 best (learning rate 7e-5, equal to v2's last-round rate; baseline dev evaluation on; cap 2.4 GPU-h per the log), 1:1 gold to pseudo rows per round, at most 5 rounds, output `qlora_loop_v4_distill` | (A) v4 above the anchor's 0.6701 with a paired 95% CI excluding 0, or (B) point estimate at least 0.6799 (within 0.01 of the 0.6899 ensemble) at 1x inference cost; precision, recall versus v2 and the ensemble also reported | Corpus preparation done (CPU); training not started |
| 3. Log-probability confidence | `record_logprobs` option stores per-span token log-probabilities during the Experiment 1 and 2 test generation; analysis of AUROC, AURC, ECE and Brier for exp(mean log-prob), exp(sum), min log-prob and vote count against the constant 0.86 | Descriptive: is the AUROC of mean log-prob above 0.5 with a CI excluding 0.5, and how does it compare to vote count | not started |

Notes for the committee.

- The log pre-declares the success criteria before the runs, which is the right protocol. With 974 test documents the CI half-width is about 0.018 and the MDE about 0.03, as the log itself states; a formal non-inferiority proof for Experiment 1 at a 0.01 margin would only pass if v3 were better than v2 by about 0.009.
- Experiment 2 uses the **at least 2 of 3** vote (precision 0.722, recall 0.661 on test), not the 3-of-3 rule (precision 0.802) discussed in `pretraining.md`; roughly 28% of its silver spans are therefore expected to be wrong, and it uses the original prompt for pseudo-labelling because the three teachers were trained with it. These are design choices, not errors; the risk is noted in `pretraining.md` section 7.
- **Discrepancy 1 (prompt).** The log's plan gives the short prompt as system "Extract skill spans." and a user template starting "Output {...}". `src/skillllm_v2/extract/mistral_extractor.py` (read at 08:5x) defines `SYSTEM_PROMPT_SHORT = "Extract skill spans from text. Return JSON only."` and a different user template, about 77 fixed tokens by my count (including my 8-token markup approximation). The log's own measurement is 41 fixed tokens. Which text the running job used depends on the code at launch; the run's `resolved_config.json` and logs must be checked before the result is reported.
- **Discrepancy 2 (token counts).** The log reports 382 tokens per example for the original prompt (fixed prompt 78.5% of tokens; text 32 and target 50 tokens) and 124 for the short prompt (3.08x fewer), from `loop_trainer --plan --count-tokens`. My recomputation (Table T2) gives 351.9 tokens per example with 86.4% fixed prompt (text 26 and answer 22 tokens on average); `docs/loop_training.md` says "about 85%". I have not identified the cause (probably a different target serialisation or a subset of rows). Use "about 80 to 86% of tokens" until reconciled.
- **Budget.** Experiments 1 and 2 plus their evaluations (about 1 GPU-h per 974-document test pass without batching) can reach the 8 GPU-h cap without leaving room for replicate seeds.

---

## 4. Held-out results and the three-adapter ensemble

All values on the 974 gold-bearing SkillSpan test rows, pooled exact-span matching, one training seed per adapter. Paired document-level bootstrap with 10,000 resamples (seed 20260710) and sign-flip permutation; unit of resampling is the document (`reports/skillllm_v2/significance_*.md`). The anchor's own 95% intervals (2,000-resample bootstrap) are in `reports/skillllm_v2/eval_suite/REPORT.md`: exact-span F1 0.6701 [0.6464, 0.6944].

### 4.1 Systems

**Table T9. Test results (974 documents).** Sources: `reports/skillllm_v2/ensemble_eval.md` (pooled F1 table), `qlora_loop_summary.md`, `qlora_loop_v2_summary.md`.

| System | Exact P | Exact R | Exact F1 | Token F1 (character-level) | Partial F1 | HARD F1 | SOFT F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anchor (80 ep, r=16; run 2) | 0.6735 | 0.6667 | 0.6701 | 0.7879 | 0.8138 | 0.7265 | 0.6040 |
| Loop v1 (r=8) | 0.6864 | 0.6291 | 0.6565 | 0.7846 | 0.8040 | 0.7046 | 0.5996 |
| Loop v2 (r=16) | 0.6720 | 0.6512 | 0.6614 | 0.7957 | 0.8170 | 0.7159 | 0.5989 |
| Majority vote, at least 2 of 3 | 0.7220 | 0.6605 | **0.6899** | 0.7962 | 0.8213 | 0.7473 | 0.6231 |
| Unanimous (3 of 3) | 0.8018 | 0.5055 | 0.6201 | 0.6583 | 0.6986 | 0.6890 | 0.5387 |
| Union (at least 1 of 3) | 0.5871 | 0.7810 | 0.6703 | 0.7993 | 0.7689 | 0.7087 | 0.6257 |

"Token F1" is character-overlap F1 requiring the label to match (`evals.md` section 1), not word-level F1 and not the SkillSpan official scorer.

### 4.2 Paired comparisons

**Table T10. Paired differences in exact-span F1 (test, 974 documents).** Raw p are bootstrap / permutation. "Holm-5" is the Holm correction over the five metrics within the comparison, as reported in the cited files.

| Comparison | Delta | 95% CI | Raw p | Holm-5 | Doc wins / losses / ties | Minimum detectable effect | Source |
|---|---:|---|---|---|---|---|---|
| Loop v1 minus anchor | -0.0135 | [-0.0342, +0.0071] | 0.201 / 0.199 | 0.80 | 180 / 191 / 603 | about 0.030 | `reports/skillllm_v2/significance_qlora_vs_anchor.md` |
| Loop v2 minus anchor | -0.0086 | [-0.0286, +0.0114] | 0.403 / 0.397 | 1.00 | 165 / 151 / 658 | about 0.029 | `reports/skillllm_v2/qlora_loop_v2_significance_vs_anchor.md` |
| Loop v2 minus v1 | +0.0049 | [-0.0133, +0.0232] | 0.602 / 0.611 | 1.00 | 155 / 137 / 682 | about 0.026 | `reports/skillllm_v2/qlora_loop_v2_significance_vs_v1.md` |
| **Majority ensemble minus anchor** | **+0.0198** | **[+0.0041, +0.0356]** | **0.0162 / 0.0146** | **0.0810 / 0.0728** | 148 / 91 / 735 (sign test p 0.0003, uncorrected) | not stated | `reports/skillllm_v2/ensemble_eval.md` |
| Unanimous minus anchor (descriptive) | -0.0500 | [-0.0670, -0.0333] | 0.0002 / 0.0000 | 0.0010 / 0.0000 | 90 / 230 / 654 | | same |
| Union minus anchor (descriptive) | +0.0003 | [-0.0123, +0.0135] | 0.9611 / 0.9717 | 0.9611 / 0.9717 | 196 / 189 / 589 | | same |

Other metrics for the ensemble against the anchor (same file): token F1 +0.0083 [-0.0036, +0.0203]; partial +0.0075 [-0.0047, +0.0203]; HARD +0.0209 [-0.0035, +0.0457]; SOFT +0.0191 [+0.0009, +0.0378] (raw p 0.0398 / 0.0428; Holm-5 0.1592 / 0.1711). Loop v2 versus v1 partial-span F1: +0.0131 [-0.0003, +0.0264], raw p 0.057, Holm-5 0.28; a McNemar test on perfectly extracted documents gives p = 0.024 uncorrected, which the v2 report itself says not to lean on.

### 4.3 Reading the results

**Single adapters.** Both loop adapters are below the anchor on exact-span F1 (-1.35 and -0.86 points), with every metric's CI containing zero. At 974 documents the minimum detectable effect is about 3 points, so the data neither show a difference nor exclude a deficit of about 3 points. This is "no significant difference detected", not equivalence; a non-inferiority claim would need a margin of at least about 3 points (stated in both loop reports). A back-of-envelope scaling (MDE is proportional to 1/sqrt(n); **derived, not computed from data**) says that resolving a 1-point difference would need on the order of nine times as many test documents, which SkillSpan does not have.

**Ensemble.** Majority voting raises precision to 0.722 (anchor 0.6735) at essentially unchanged recall (0.6605 against 0.6667) and gains 1.98 exact-F1 points. The raw p-values (0.016 and 0.015) fall below 0.05 and the CI excludes zero, but after Holm correction over five metrics the exact-F1 p is 0.081 / 0.073 (not below 0.05). Document-level wins exceed losses by 148 to 91. The defensible wording is a consistent, directionally positive gain of about 2 points that is **suggestive and not confirmed** under the pre-declared correction, with all caveats of a single split and single seeds. Three further caveats:

1. **Forking path.** `scripts/ensemble_eval.py` documents that the voting rule (at least 2 of 3) was fixed before looking at test results and that unanimity and union are descriptive. But the idea of ensembling the three adapters arose after the single-adapter test results (both below the anchor) were known; the report does not claim otherwise. The Holm family (five metrics) does not account for that choice.
2. **Correlated members.** The adapters share the base model and the training data, so their errors are correlated and the vote gain is bounded; two of the three members were themselves selected on dev subsets of the same dev split.
3. **Cost.** The ensemble needs three generation passes per document (about 3x latency; section 5 of `inference.md`).

**Selection bias.** Dev exact-F1 of the selected adapter overstates test F1: v1 0.7052 on 160 dev rows against 0.6565 on test (-0.0487, derived; the report says 4.8 to 4.9 points); v2 0.7029 on 400 dev rows against 0.6614 (-0.0415, derived; the report says 4.2). Part of this is the maximum over several noisy dev scores (dev standard error about 0.03 at 160 rows and about 0.02 at 400 rows, per the reports), and part may be that the dev subset is easier than the test set; the anchor was not scored on the loop dev subsets, so the two cannot be separated. Consequence: **dev F1 values must never be quoted as test performance**, and no loop-versus-anchor statement may use the dev numbers.

### 4.4 Vote count as a confidence signal

The anchor and loop adapters emit a constant confidence of 0.86 for every span (AUROC 0.5; `reports/skillllm_v2/eval_suite/REPORT.md`: raw-anchor ECE 0.186, AUROC 0.5). The number of adapters predicting the same exact (start, end, label) is a monotone alternative (`reports/skillllm_v2/ensemble_vote_confidence.md`, `scripts/vote_precision.py`):

**Table T11. Precision by vote count (test, 974 documents).**

| Adapters agreeing | Spans | Correct | Precision |
|---|---:|---:|---:|
| 1 of 3 | 941 | 273 | 0.2901 |
| 2 of 3 | 644 | 351 | 0.5450 |
| 3 of 3 | 1,428 | 1,145 | 0.8018 |

Operating points (keep spans with at least v votes): v=3 P 0.802 / R 0.506 / F1 0.620; v=2 P 0.722 / R 0.661 / F1 0.690; v=1 P 0.587 / R 0.781 / F1 0.670. **Caveat:** the table is estimated on the same test split it describes (the only split with predictions from all three adapters), so it shows that the signal exists and is monotone, not a calibrated, held-out probability; the library offers `fit_vote_calibration` to re-fit on other data (`docs/LIBRARY.md` section 6). Held-out calibration (an ECE for the vote-based confidence) has not been measured.

---

## 5. What can and cannot be claimed

**Table T12. Claims about training.**

| Claim | Status | Evidence / reason |
|---|---|---|
| "A rank-8 or rank-16 QLoRA loop adapter reaches test exact-F1 within about 1.4 points (v1) and 0.9 points (v2) of the 80-epoch anchor at 0.9% and 1.8% of its GPU-hours, with no significant difference detected." | **Supported**, with the CI and MDE stated | Tables T7, T9, T10; single seed, single split; MDE about 3 points |
| "The loop is as good as the anchor" / "better than the anchor" | **Not supported** | All CIs include zero and allow a deficit of about 3 points; point estimates are below the anchor |
| "Hard-example mining (or prompt masking, or the curriculum) is responsible for the efficiency" | **Not supported** | No ablation (Table T8) |
| "80 epochs over-trains" | **Partly supported**: dev loss rises 70% from epoch 3 to 80 | No per-epoch dev F1; epoch-3 checkpoint not kept |
| "v2 is better than v1" | **Not supported** | Delta +0.0049, CI [-0.0133, +0.0232]; several settings changed at once |
| "The majority-vote ensemble improves exact-F1 by about 2 points" | **Suggestive, not confirmed** | Raw p 0.016; Holm-5 0.081; single seeds; forking path; correlated members |
| "Vote count is a usable confidence signal (precision 0.29 / 0.55 / 0.80)" | **Supported as a signal, not as a calibrated probability** | Estimated on the test split itself |
| "Wrapper stages improve extraction" | **Not supported beyond cleanup** | At most +0.0024, below the 0.005 seed noise floor (`evals.md`) |
| "Training with the compact prompt matches the long prompt" | **Pending** | v3 not finished (3.5) |

Recommended thesis wording for the compute result: "At about 1 to 2% of the training GPU-hours, a quantised iterative LoRA schedule produced adapters whose exact-span F1 on the held-out SkillSpan test split was 0.9 to 1.4 points below the 80-epoch anchor, with no statistically detectable difference (95% CI of the paired difference [-3.4, +0.7] and [-2.9, +1.1] points; n = 974 documents; one seed; minimum detectable effect about 3 points)."

## 6. Threats to validity

1. **One seed per recipe.** Two identical anchor runs differ by 0.005 exact-F1 (Table T5); loop runs have no replicate. Seed variance is unmeasured; none of the CIs include it.
2. **One test split, partly repeated text.** Single SkillSpan test split; 169 to 177 repeated normalised texts across splits (quoted, not recomputed). The test set was also used to compare variants, select the ensemble operating point, estimate the vote table, and (in the wrapper) to accept variants via a gate (`evals.md`).
3. **Evaluation population.** Only gold-bearing test rows (974 of 3,569); false positives on skill-free text unmeasured. Deployment text is resumes and free text, not SkillSpan job-posting sentences; resume performance is unevaluated.
4. **Metric definitions.** Exact-span F1 depends on character offsets that the system reconstructs by string matching (first unused match of the model's returned string), a source of error independent of what the model knows; the "token F1" is character-based; `evals.md` Table E1.
5. **Recipe confounds.** Anchor and loop differ in six or more factors (Table T8); conclusions are about recipes.
6. **Selection effects.** Dev-based best-round choice (max of 4 or 6 noisy scores); dev subsets differ between v1 (160) and v2 (400); the anchor was never scored on them.
7. **Hardware.** Anchor evaluated on two visible GPUs, loop adapters on one; floating-point differences unquantified.
8. **Provenance gaps.** Run-1 adapter overwritten; its metrics exist only in a log; the training-precision setting and anchor peak memory are not recorded; loop training peak memory is reported inconsistently (16.4 vs 17.9 GB).
9. **Licence and data governance.** The base model and SkillSpan licences are noted as to be verified (`MODEL_CARD.md`); nothing has been published.

## 7. Reproduction commands

Run from `/home/923873155/Sem 4 Research/SkillLLM_v2_stage2`. These are the commands named in the repository's scripts and reports; I did not execute any of them for this draft (the sandbox has no GPU use, and the 24B model is not run).

```bash
# data
bash scripts/build_skillspan_bio_dataset.sh && bash scripts/build_mistral_sft_data.sh   # SkillSpan -> BIO and SFT jsonl (see scripts for exact outputs)

# 80-epoch anchor (2 GPUs, about 90 h wall); evaluation on the 974 test rows
bash scripts/run_skillllm_v2_80epoch_2gpu_pipeline.sh          # writes configs/skillllm_v2_training_80ep_2gpu.yaml, trains, evaluates
bash scripts/run_raw_mistral_eval_2gpu.sh                      # (re-)evaluate the anchor adapter only

# QLoRA loop v1 (defaults in configs/skillllm_v2_loop_qlora.yaml)
bash scripts/train_mistral_qlora_loop.sh --plan                # no GPU: data sizes and round-0 mix
bash scripts/train_mistral_qlora_loop.sh                       # real run, GPU1, resumable via loop_state.json
bash scripts/run_qlora_loop_test_eval.sh                       # test generation + metrics for the selected adapter

# QLoRA loop v2: overrides reconstructed from reports/skillllm_v2/qlora_loop_v2_summary.md ("Training run");
# verify against logs/qlora_loop_20261021_231012.log before quoting as the exact command
bash scripts/train_mistral_qlora_loop.sh --set lora.r=16 --set lora.alpha=32 --set eval.dev_size=400 \
     --set loop.patience=3 --set loop.max_rounds=6 --set output.dir=artifacts/skillllm_v2/qlora_loop_v2
bash scripts/run_qlora_loop_v2_test_eval.sh

# statistics and ensemble (CPU, minutes)
bash scripts/run_significance_qlora.sh && bash scripts/run_significance_qlora_v2.sh
PYTHONPATH=src python scripts/ensemble_eval.py                 # reports/skillllm_v2/ensemble_eval.{md,json}
PYTHONPATH=src python scripts/vote_precision.py                # vote-count precision table
bash scripts/run_eval_suite.sh                                 # module evaluation suite, about 5 min, CPU
```

A GPU 24 GB-class card with at least 19 GB free is required for the loop (the trainer refuses to load otherwise); 4-bit inference peaks at about 16.3 GB allocated (v2 test evaluation).

## 8. Items I could not trace or that disagree

- The task brief's path `artifacts/skillllm_v2/mistral_lora/training_summary.json` does not exist; the file is `artifacts/skillllm_v2/mistral_lora/model/training_summary.json`.
- Anchor numeric precision (fp16 in the YAML versus "bf16 if supported" in the audit): unverified.
- Loop training peak memory 16.4 GB (`docs/loop_training.md`) versus 17.9 GB (`reports/skillllm_v2/qlora_loop_summary.md`).
- `docs/loop_training.md` cites the comparability numbers "0.6701/0.6726"; 0.6726 matches no current file (current full-wrapper value is 0.6719, boundary 0.6724).
- The 177 / 169 repeated-text counts are quoted from the loop report and were not recomputed.
- Examples seen by the v2 selected adapter (about 2,560) is a derived upper bound.
- The 8-token chat-markup constant in Table T2 is my approximation.
- Token counts per example and the short-prompt text disagree between my recomputation and `docs/GPU_EXPERIMENTS_LOG.md` (section 3.5, Discrepancies 1 and 2).
