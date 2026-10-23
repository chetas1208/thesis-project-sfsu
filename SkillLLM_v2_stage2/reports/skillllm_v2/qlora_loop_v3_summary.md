# Experiment 1: short-prompt QLoRA loop (v3) - held-out result

Question: can a compact prompt (same output schema/label set) cut training/inference cost without losing held-out accuracy versus v2?
Short answer: **No (criteria failed).** Training got 2.1x cheaper per round, but held-out exact-span F1 fell 1.8 points versus v2 (CI [-3.6, +0.1]) and
generation latency per doc did not improve (1.10x).

## Prompt and token breakdown (exact tokenizer counts, `loop_trainer --plan --count-tokens`, round-0 mix of 640 examples)

| | original | short |
|---|---|---|
| fixed prompt tokens (empty text) | 300 | 41 |
| mean input-text tokens | 32.4 | 32.4 |
| mean prompt tokens | 331.6 | 73.4 |
| mean target-JSON tokens (+eos) | 50.5 | 50.5 |
| mean total tokens / example | 382.1 | 123.9 (3.08x fewer) |
| fixed prompt share of tokens | 78.5% | 33.1% |

(The briefed "~85%" is 78.5% on this data; the 50-token target JSON and the 32-token text are untouched, which caps the saving at ~3x.)
Short prompt: system `Extract skill spans.`; user `Output {"skills":[{"text":"<exact span>","label":"HARD_SKILL|SOFT_SKILL"}]}` + `Text:` + text. Chosen once, not tuned.
Option: `prompt_version` / `prompt.version` (default `original`, byte-identical to before; tests in `tests/skillllm_v2/test_prompt_version_and_logprobs.py`).

## Training (config `configs/skillllm_v2_loop_qlora_v3_shortprompt.yaml`; v2 hyperparameters, eval.batch_size 16 instead of 8)

| Round | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| v3 dev exact-span F1 (400 docs) | 0.6789 | **0.6904** | 0.6838 | 0.6785 | 0.6855 |
| v2 dev (same 400 docs, orig. prompt) | 0.6584 | 0.6652 | 0.6913 | 0.7029 | 0.6772 (r5 0.7005) |
| v3 training-step seconds | 341 | 344 | 331 | 335 | 343 |
| v3 total round seconds | 939 | 968 | 928 | 972 | 996 |

Stopped by plateau after 5 rounds (best round 1; patience 3). Loop GPU-h 1.333 (v2: 3.17 for 6 rounds); tokens 78k/round vs v2 ~237k.
Per round: training step 338 s (0.094 GPU-h) vs v2 ~710 s (0.197 GPU-h, v2 round 0 log) = **2.1x lower**; whole round (train + dev eval + mining probe) ~961 s vs ~1,906 s = 1.98x
(part of this is the larger generation batch, not the prompt). Peak GPU memory 18.1 GB.
Dev selection was a max over 5 noisy scores (SE ~0.02); v3's dev curve is flat (0.679-0.690), v2's rose to 0.703 later.

## Held-out test (974 docs, adapter = round 1, run once, identical scoring path, short prompt)

| | anchor | v2 | v3 | v3 - v2 [95% CI] | v3 - anchor [95% CI] |
|---|---|---|---|---|---|
| exact-span F1 | 0.6701 | 0.6614 | **0.6439** | -0.0176 [-0.0356, +0.0008] | -0.0262 [-0.0462, -0.0057] |
| exact P / R | 0.6735 / 0.6667 | 0.6720 / 0.6512 | 0.6751 / 0.6155 | | |
| token F1 | 0.7879 | 0.7957 | 0.7701 | -0.0256 [-0.0427, -0.0082] | -0.0178 [-0.0342, -0.0013] |
| partial F1 | 0.8138 | 0.8170 | 0.8042 | -0.0129 [-0.0267, +0.0008] | -0.0097 |
| HARD F1 | 0.7265 | 0.7159 | 0.7047 | -0.0112 | -0.0218 |
| SOFT F1 | 0.6040 | 0.5989 | 0.5694 | -0.0296 [-0.0570, -0.0013] | -0.0347 [-0.0624, -0.0065] |

p (paired bootstrap, exact-span): v3 vs v2 0.061 (Holm over the 2 exact-span tests 0.184); v3 vs anchor 0.0126 (Holm 0.025). Token F1 v3 vs v2: p 0.0046, Holm-5 0.023.
Doc level v3 vs v2: wins 107, losses 161, ties 706 (sign p = 0.0012); perfect docs v3-only 50 vs v2-only 88 (McNemar p = 0.0015). v3 lost recall (-3.6 points vs v2), precision unchanged.
Reading: v3 is probably worse than v2 (token F1 and doc-level tests agree; the headline exact-span CI just touches zero) and is nominally significantly worse than the anchor on exact-span.

## Success criteria (pre-declared in `docs/GPU_EXPERIMENTS_LOG.md`)

1. Held-out exact-span F1 within 0.01 of v2: **not met** (-0.0176; CI lower bound -0.0356, so formal non-inferiority at 0.01 is also not shown).
2. >= 2x lower train GPU-h per round: **met on the training-step measure (2.1x)**; total round cost 1.98x (marginally under).
3. >= 1.5x lower generation latency per doc: **not met**. 974 test docs: 3,920 s = 4.02 s/doc (mean; median 3.08 s) vs v2 4,303 s = 4.42 s/doc -> 1.10x. Generation is decode-bound (the ~50-token JSON answer is unchanged), the
   prompt only shortens the prefill. v3 timing includes the log-prob recording overhead (small) and any guardrails-pause wall time.
**Verdict: Experiment 1 failed.** The prompt is not free: the removed rules/negative instructions probably carried annotation conventions (e.g. what not to extract) that the 640x5 examples did not fully replace.

Caveats: one seed, one dev subset, one split; v3 and v2 differ also in eval batch size (dev only) and number of rounds; best round chosen on a flat dev curve; latency from one run each (v2 timed in an earlier session, GPU shared with another job).

## Files
- Code/config: `src/skillllm_v2/extract/mistral_extractor.py` (`PROMPT_VERSIONS`, `build_prompt_messages`), `src/skillllm_v2/train/{loop_trainer,build_mistral_sft_data,mistral_lora_trainer}.py`, `configs/skillllm_v2_loop_qlora_v3_shortprompt.yaml`, `scripts/run_loop_v3_shortprompt.sh`, `scripts/eval_adapter_test.py`, `scripts/run_qlora_loop_v3_test_eval.sh`, `scripts/significance_v3_v4.py`
- Artifacts: `artifacts/skillllm_v2/qlora_loop_v3_shortprompt/` (best/, loop_summary.json), `artifacts/skillllm_v2/qlora_loop_v3_eval/{predictions.jsonl,metrics.json,timing_segments.jsonl}`
- Reports: `reports/skillllm_v2/qlora_loop_v3_{summary.md,significance.json,significance.md,test_metrics.json}`, `logprob_confidence_v3.{md,json}`
