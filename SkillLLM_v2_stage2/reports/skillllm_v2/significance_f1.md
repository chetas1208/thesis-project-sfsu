# Statistical significance of extraction F1 deltas (anchor vs safe-wrapper variants)

Paired document-level bootstrap (10,000 resamples, seed 20260710) and paired sign-flip permutation test on 974 SkillSpan test documents. Micro-F1 is recomputed from pooled TP/FP/FN of every resample. Holm correction over 15 tests (3 variants x 5 metrics), alpha = 0.05. A delta is called significant only if BOTH the bootstrap and the permutation Holm-adjusted p-values are below alpha. Delta = variant minus Raw Mistral LoRA anchor.

Reproduction check: pooled-count F1s match `evaluate_extraction` and the F1s stored in each artifact's `metrics.json` to within 1e-06 for every system and metric (asserted before any test).

Note: tests use the per-document predictions currently on disk (inputs and hashes at the bottom). Their exact-F1 values equal the current `safe_wrapper_summary.json` / `money_table.md`; if you quote older figures for the GLiNER+Taxonomy or Full CAG/Graph variants (e.g. 0.670222 / 0.672602), they come from an earlier artifact version that is not on disk and are NOT what was tested here.

## Verdicts (primary metric: exact-span F1)

- boundary_refiner: exact-span F1 delta +0.0024 (95% CI [+0.0009, +0.0042]); Holm-adjusted p bootstrap=0.003, permutation=0.029 -> SIGNIFICANT after Holm
- mistral_gliner_taxonomy: exact-span F1 delta -0.0006 (95% CI [-0.0023, +0.0009]); Holm-adjusted p bootstrap=1.000, permutation=1.000 -> NOT significant after Holm
- full_cag_graph: exact-span F1 delta +0.0018 (95% CI [-0.0006, +0.0042]); Holm-adjusted p bootstrap=0.939, permutation=0.990 -> NOT significant after Holm

## Exact-span F1 summary

| Variant | Anchor F1 | Variant F1 | Delta | 95% CI | p boot (raw) | p perm (raw) | p boot (Holm) | p perm (Holm) | MDE (80% power) | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| boundary_refiner | 0.670069 | 0.672444 | +0.002376 | [+0.000893, +0.004232] | 0.0002 | 0.0020 | 0.0030 | 0.0293 | 0.0024 | significant |
| mistral_gliner_taxonomy | 0.670069 | 0.669478 | -0.000590 | [-0.002289, +0.000939] | 0.5099 | 0.6133 | 1.0000 | 1.0000 | 0.0023 | not significant |
| full_cag_graph | 0.670069 | 0.671854 | +0.001785 | [-0.000566, +0.004225] | 0.1342 | 0.1237 | 0.9393 | 0.9897 | 0.0034 | not significant |

## All metrics

| Variant | Metric | Anchor | Variant | Delta (observed) | Boot mean | 95% CI | p boot | p perm | Holm boot | Holm perm | MDE | Sig. after Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| boundary_refiner | exact_span_f1 | 0.670069 | 0.672444 | +0.002376 | +0.002375 | [+0.000893, +0.004232] | 0.0002 | 0.0020 | 0.0030 | 0.0293 | 0.0024 | yes |
| boundary_refiner | token_f1 | 0.787886 | 0.788000 | +0.000113 | +0.000113 | [+0.000000, +0.000274] | 0.1032 | 0.2500 | 0.8255 | 1.0000 | 0.0002 | no |
| boundary_refiner | partial_span_f1 | 0.813845 | 0.816444 | +0.002599 | +0.002598 | [+0.001012, +0.004527] | 0.0002 | 0.0020 | 0.0030 | 0.0293 | 0.0025 | yes |
| boundary_refiner | hard_skill_f1 | 0.726450 | 0.729518 | +0.003068 | +0.003067 | [+0.000596, +0.006253] | 0.0142 | 0.0625 | 0.1794 | 0.7500 | 0.0041 | no |
| boundary_refiner | soft_skill_f1 | 0.604046 | 0.605505 | +0.001458 | +0.001455 | [+0.000295, +0.002891] | 0.0138 | 0.0625 | 0.1794 | 0.7500 | 0.0018 | no |
| mistral_gliner_taxonomy | exact_span_f1 | 0.670069 | 0.669478 | -0.000590 | -0.000597 | [-0.002289, +0.000939] | 0.5099 | 0.6133 | 1.0000 | 1.0000 | 0.0023 | no |
| mistral_gliner_taxonomy | token_f1 | 0.787886 | 0.787829 | -0.000057 | -0.000066 | [-0.000625, +0.000559] | 0.8135 | 0.8125 | 1.0000 | 1.0000 | 0.0008 | no |
| mistral_gliner_taxonomy | partial_span_f1 | 0.813845 | 0.814206 | +0.000361 | +0.000346 | [-0.001577, +0.002382] | 0.7645 | 0.7656 | 1.0000 | 1.0000 | 0.0028 | no |
| mistral_gliner_taxonomy | hard_skill_f1 | 0.726450 | 0.726898 | +0.000448 | +0.000440 | [-0.002443, +0.003184] | 0.7269 | 0.7500 | 1.0000 | 1.0000 | 0.0040 | no |
| mistral_gliner_taxonomy | soft_skill_f1 | 0.604046 | 0.602595 | -0.001451 | -0.001461 | [-0.002852, -0.000294] | 0.0144 | 0.0625 | 0.1794 | 0.7500 | 0.0018 | no |
| full_cag_graph | exact_span_f1 | 0.670069 | 0.671854 | +0.001785 | +0.001779 | [-0.000566, +0.004225] | 0.1342 | 0.1237 | 0.9393 | 0.9897 | 0.0034 | no |
| full_cag_graph | token_f1 | 0.787886 | 0.787942 | +0.000056 | +0.000059 | [-0.000521, +0.000700] | 0.8823 | 0.8516 | 1.0000 | 1.0000 | 0.0009 | no |
| full_cag_graph | partial_span_f1 | 0.813845 | 0.816807 | +0.002962 | +0.002954 | [+0.000401, +0.005727] | 0.0242 | 0.0281 | 0.2420 | 0.3647 | 0.0038 | no |
| full_cag_graph | hard_skill_f1 | 0.726450 | 0.729975 | +0.003525 | +0.003505 | [-0.000437, +0.007663] | 0.0834 | 0.0781 | 0.7505 | 0.7500 | 0.0058 | no |
| full_cag_graph | soft_skill_f1 | 0.604046 | 0.604046 | +0.000000 | +0.000003 | [-0.001783, +0.001782] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0026 | no |

## Permutation test details

| Variant | Metric | Discordant docs | Method |
|---|---|---|---|
| boundary_refiner | exact_span_f1 | 10 | exact enumeration |
| boundary_refiner | token_f1 | 3 | exact enumeration |
| boundary_refiner | partial_span_f1 | 10 | exact enumeration |
| boundary_refiner | hard_skill_f1 | 5 | exact enumeration |
| boundary_refiner | soft_skill_f1 | 5 | exact enumeration |
| mistral_gliner_taxonomy | exact_span_f1 | 9 | exact enumeration |
| mistral_gliner_taxonomy | token_f1 | 6 | exact enumeration |
| mistral_gliner_taxonomy | partial_span_f1 | 10 | exact enumeration |
| mistral_gliner_taxonomy | hard_skill_f1 | 4 | exact enumeration |
| mistral_gliner_taxonomy | soft_skill_f1 | 5 | exact enumeration |
| full_cag_graph | exact_span_f1 | 19 | exact enumeration |
| full_cag_graph | token_f1 | 9 | exact enumeration |
| full_cag_graph | partial_span_f1 | 20 | exact enumeration |
| full_cag_graph | hard_skill_f1 | 9 | exact enumeration |
| full_cag_graph | soft_skill_f1 | 10 | exact enumeration |

## Document-level win / loss / tie (doc exact-span F1) and McNemar

| Variant | Wins | Losses | Ties | Sign-test p | Perfect docs: variant-only | anchor-only | McNemar exact p |
|---|---|---|---|---|---|---|---|
| boundary_refiner | 8 | 0 | 966 | 0.0078 | 4 | 0 | 0.1250 |
| mistral_gliner_taxonomy | 3 | 4 | 967 | 1.0000 | 0 | 1 | 1.0000 |
| full_cag_graph | 11 | 4 | 959 | 0.1185 | 4 | 1 | 0.3750 |

## Minimal detectable effect at this sample size

- boundary_refiner: paired bootstrap SD of the exact-F1 delta = 0.00086; MDE (two-sided alpha 0.05, 80% power) = 0.0024; with Holm-level alpha 0.05/15 = 0.0032.
- mistral_gliner_taxonomy: paired bootstrap SD of the exact-F1 delta = 0.00083; MDE (two-sided alpha 0.05, 80% power) = 0.0023; with Holm-level alpha 0.05/15 = 0.0031.
- full_cag_graph: paired bootstrap SD of the exact-F1 delta = 0.00122; MDE (two-sided alpha 0.05, 80% power) = 0.0034; with Holm-level alpha 0.05/15 = 0.0046.

## Can the thesis claim these deltas?

- **boundary_refiner**: statistically reliable but tiny: +0.0024 exact-F1, 95% CI [+0.0009, +0.0042]. It rests on only 10 of 974 documents that differ at all (8 better, 0 worse). Safe wording: "a small, consistent improvement in span boundaries on the SkillSpan test set (a handful of documents, no regressions)"; do not describe it as a meaningful accuracy gain.
- **mistral_gliner_taxonomy**: cannot be claimed. Delta -0.0006, 95% CI [-0.0023, +0.0009] contains 0; 3 documents better vs 4 worse. Treat as no measurable difference from the anchor (non-regression, not improvement).
- **full_cag_graph**: cannot be claimed. Delta +0.0018, 95% CI [-0.0006, +0.0042] contains 0; 11 documents better vs 4 worse. Treat as no measurable difference from the anchor (non-regression, not improvement).

**On a +0.0025 delta specifically.** At this sample size the paired standard error of an exact-F1 delta for the full CAG/Graph variant is about 0.0012, so a +0.0025 delta is roughly 2.1 standard errors from zero (approximate raw two-sided p about 0.040 under a normal approximation; this is a back-of-envelope figure, not a rerun of the test on different predictions). That would not survive Holm correction over 15 tests (or over just the 3 variants), and the MDE at 80% power is 0.0034, i.e. larger than 0.0025. A +0.0025 gain for the full wrapper should therefore not be presented as a real improvement; report it with its confidence interval and say that the study is not powered to detect effects below that size.

**Caveats that apply to every row.** (1) Resampling is over the 974 test documents of one split and one trained anchor model; it does not capture variance across training seeds, so even a significant delta is evidence about this model and test set only. (2) The variants are deterministic post-processing of the same Mistral predictions, so most documents are unchanged (ties dominate), which is why the tests are driven by a few documents and the exact permutation p-values are coarse (minimum attainable raw p is 2/2^d for d discordant documents). (3) Holm over 15 correlated tests is conservative; the per-metric Holm columns (`p_holm_within_metric` in the JSON) are looser. (4) Effects of this magnitude (about 0.2 F1 points) are practically negligible even when statistically detectable. (5) If the wrapper code or artifacts are regenerated, rerun `scripts/run_significance_test.sh`: the numbers above are tied to the input hashes listed below.

Inputs (sha256 prefix): anchor: `artifacts/skillllm_v2/mistral_lora/predictions.jsonl` (ac5917d0267670b0); boundary_refiner: `artifacts/skillllm_v2/safe_wrapper/boundary_refiner/candidate_predictions.jsonl` (39c34a24645f65d8); mistral_gliner_taxonomy: `artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/candidate_predictions.jsonl` (1ffef118b4661578); full_cag_graph: `artifacts/skillllm_v2/safe_wrapper/full_cag_graph/candidate_predictions.jsonl` (945e748c0ba0bcce)
