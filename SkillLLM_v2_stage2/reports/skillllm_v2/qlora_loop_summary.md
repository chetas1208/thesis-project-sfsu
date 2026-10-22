# QLoRA loop vs 80-epoch anchor: held-out test result

Question: is the quantized iterative loop adapter (`artifacts/skillllm_v2/qlora_loop/best`, round 1, r=8) at least as good as the 80-epoch Raw Mistral LoRA anchor (r=16) on the held-out SkillSpan test split, at about 1% of the compute?

Short answer: on test it is slightly lower on every metric, but none of the differences is statistically distinguishable from zero. The test data neither show that the loop is better nor establish that it is as good: the exact-span 95% CI is [-3.4, +0.7] points. The compute saving is real. The dev gain (0.673 -> 0.705) did not carry over to test.

## Test results (974 SkillSpan test docs with at least one gold span, identical extraction and scoring path)

| Metric | Anchor (80 ep, r=16) | QLoRA loop best (r=8) | Delta | 95% CI (paired doc bootstrap) | p boot / p perm (raw) | Holm (5 metrics) |
|---|---|---|---|---|---|---|
| exact-span F1 | 0.670069 | 0.656531 | -0.0135 | [-0.0342, +0.0071] | 0.201 / 0.199 | 0.80 / 0.80 |
| token F1 | 0.787886 | 0.784563 | -0.0033 | [-0.0198, +0.0133] | 0.692 / 0.696 | 1.00 / 1.00 |
| partial-span F1 | 0.813845 | 0.803962 | -0.0099 | [-0.0256, +0.0059] | 0.216 / 0.215 | 0.80 / 0.80 |
| HARD_SKILL F1 | 0.726450 | 0.704632 | -0.0218 | [-0.0517, +0.0079] | 0.154 / 0.152 | 0.77 / 0.76 |
| SOFT_SKILL F1 | 0.604046 | 0.599598 | -0.0044 | [-0.0319, +0.0235] | 0.762 / 0.758 | 1.00 / 1.00 |

- Exact-span precision / recall: anchor 0.6735 / 0.6667, loop 0.6864 / 0.6291. The loop is more precise and misses more spans (lower recall).
- Document level (doc exact-span F1): loop wins 180, loses 191, ties 603 (sign test p = 0.60). Perfectly extracted docs: loop-only 97, anchor-only 114 (McNemar p = 0.27).
- Minimum detectable exact-span delta at this sample size (alpha 0.05, 80% power): about 0.030. A true difference of about 1 point is therefore not resolvable on 974 documents.
- Full tables: `significance_qlora_vs_anchor.{json,md}`. The anchor's 0.670069 was reproduced from its saved predictions (`artifacts/skillllm_v2/mistral_lora/predictions.jsonl`) before running anything, and pooled-count F1s match `evaluate_extraction` and each stored metrics.json to 1e-6.

## Compute and memory

| | 80-epoch anchor | QLoRA loop |
|---|---|---|
| Training wall time | 324,218 s = 90.06 h (`training_summary.json`, `train_runtime`) | 5,733 s = 1.59 h (4 rounds; `loop_state.json`) |
| GPUs | 2 (world size 2) | 1 (GPU1) |
| GPU-hours | 180.1 | 1.59 (guardrails-billed: 1.591) |
| Ratio | 1x | 0.88% of the anchor |
| GPU-hours to reach the selected adapter | n/a | rounds 0-1 = 2,746 s = 0.76 GPU-h (0.42%); rounds 2-3 were the plateau check |
| Training examples seen | 80 epochs x 4,800 = 384,000 | 4 x ~630 = ~2,500 (selected adapter: 1,258 examples, about 0.26 epoch) |
| Trainable params | r=16 LoRA | r=8 LoRA (46 M) |
| Peak training memory | not recorded in the available logs (unverified) | 17.9 GB allocated per round, single 3090 |

Both are 4-bit NF4 QLoRA (the anchor config also has `use_4bit: true`), so "quantized" is not what differs. What differs is training length (about 0.26 epoch vs 80 epochs), rank, loss masking and hard-example mining.

Test-eval cost of this run: 974 docs in 3,524 s generation + 17 s load on one GPU (about 0.98 GPU-h). Peak allocated 16.1 GB, reserved 18.5 GB. The anchor's own test eval took 4,257 s wall with `CUDA_VISIBLE_DEVICES=0,1`.

## Reading the result honestly

- Test point estimates are all negative for the loop, with the largest drop on HARD_SKILL (-2.2 points) and the smallest on token F1 (-0.3 points). With 974 docs and one seed, none is significant (smallest raw p = 0.15; no metric survives Holm). "No significant difference" is not "equivalent": the exact-span CI still allows a 3.4-point deficit, and the MDE is 3 points. A non-inferiority claim would only hold for a margin of about 3.5 points or more.
- The dev gain was selection-biased and small. 0.673 -> 0.705 is round 0 -> round 1 of the same loop on 160 dev docs (SE roughly 0.03), and round 1 was picked as the maximum of four noisy dev scores (0.673, 0.705, 0.702, 0.696). The curve is flat within noise across rounds 1-3. Test F1 for the selected adapter (0.657) is 4.8 points below its dev score. The anchor was not scored on this dev subset, so the dev numbers do not support a loop-vs-anchor comparison, and the test comparison is the only one that does.
- Plausible cause of the small recall deficit (hypothesis, not tested): the selected adapter has seen about a quarter of one pass over train, whereas the anchor had 80 passes. The loop trades recall for precision.
- One training seed, one split, one adapter selection. Variance across seeds is unmeasured, so a different loop seed could land above or below the anchor.

Verdict: for about 1% of the compute (0.9% for the full loop, 0.4% to reach the chosen adapter) the loop gives a model whose test F1 is within about 1.4 points of the anchor (exact span), with statistically indistinguishable performance on all five metrics. That supports "comparable at about 1% of the compute", and it does not support "better" or "at least as good". If a thesis claim is needed, state it as "no significant difference detected (exact-span delta -1.4 points, 95% CI [-3.4, +0.7], n = 974, single seed)". Closing the gap would need a longer or second run, for example more rounds with the plateau criterion relaxed, or multiple seeds.

## Leakage and protocol checks

- Dev subset vs test: the loop's 160 dev docs are ids `dev-*` from SkillSpan `dev.json`; test is `test.json`. They are disjoint by split, and 0 of the 160 dev-subset texts (whitespace/case-normalised) appear in any test doc. Mining probes are train-split only, and test was never used for selection or tuning in the loop.
- Caveat applying to both systems equally: some SkillSpan texts repeat across splits (177 train and 169 full-dev normalised texts also occur among test docs, mostly short strings; 920 unique texts among the 974 test docs). Not specific to the loop; the anchor trained on the same train split.
- Eval path identical to the anchor: `configs/skillllm_v2_training_80ep_2gpu.yaml` for dataset root, `model_path` and `use_4bit`; `MistralExtractor` with the same prompt, greedy decoding, `max_new_tokens=512` (the config does not set it, so the default is used), one doc at a time, test docs filtered to `gold_spans != []`, then `evaluate_extraction`. Only `adapter_path` changed.
- Not identical: the anchor eval ran with 2 visible GPUs (`device_map=auto` split), the loop eval on GPU1 only. 4-bit kernels or split points can change floating-point results slightly. I did not re-run the anchor, so any resulting difference is unquantified (expected to be tiny).

## Files

- Predictions and metrics: `artifacts/skillllm_v2/qlora_loop_eval/{predictions.jsonl,metrics.json,timing_segments.jsonl}`; `reports/skillllm_v2/qlora_loop_test_metrics.json` (same format as `raw_mistral_lora_metrics.json`)
- Significance: `reports/skillllm_v2/significance_qlora_vs_anchor.{json,md}`, produced by `scripts/run_significance_qlora.sh` (calls `scripts/significance_qlora.py`, which reuses `skillllm_v2.eval.significance` unchanged)
- Eval code: `scripts/eval_qlora_loop_test.py`, `scripts/run_qlora_loop_test_eval.sh` (resumable: incremental jsonl, skips finished docs); log `logs/qlora_loop_test_eval.log`
