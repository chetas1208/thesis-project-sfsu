# QLoRA loop v2 (r=16, 400-doc dev, patience 3, up to 6 rounds): held-out test result

Question: does the second QLoRA-loop run (v2: `artifacts/skillllm_v2/qlora_loop_v2/best`, round 3, r=16) improve on the first run (v1: r=8, round 1) and is it at least as good as the 80-epoch Raw Mistral LoRA anchor (r=16)?

Short answer: v2 is not distinguishable from the anchor and not distinguishable from v1 on any metric after multiple-comparison correction. Point estimates put v2 slightly above v1 on 4 of 5 metrics and slightly below the anchor on exact-span F1 (-0.9 points) and the two skill-type F1s, slightly above it on token and partial-span F1. This does not establish that v2 beats v1, nor that it matches the anchor.

## Training run

Config `configs/skillllm_v2_loop_qlora.yaml` with overrides `lora.r=16 lora.alpha=32 eval.dev_size=400 loop.patience=3 loop.max_rounds=6 output.dir=artifacts/skillllm_v2/qlora_loop_v2` (single GPU1 job, one uninterrupted process, no relaunch needed; the session restart happened during the test eval, which is resumable and had already finished).

| Round | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Dev exact-span F1 (400 docs) | 0.6584 | 0.6652 | 0.6913 | **0.7029** | 0.6772 | 0.7005 |
| Cumulative GPU-h | 0.501 | 1.011 | 1.536 | 2.093 | 2.628 | 3.173 |

- Stop reason: `max_rounds` (6 rounds). The patience-3 criterion never fired: the best round (3) was followed by two non-improving rounds. Training was cut by the round cap, not by a plateau, so the curve beyond round 5 is unknown.
- Best round 3: dev exact-span P/R/F1 = 0.6925 / 0.7135 / 0.7029, token F1 0.8156, partial F1 0.8413, HARD 0.7467, SOFT 0.6539.
- Dev SE at 400 docs is roughly 0.02. Round 3 vs round 5 (0.7029 vs 0.7005) and round 3 vs round 4 are within noise; selecting round 3 is a max over six noisy scores. Unlike v1, v2 shows an upward trend over rounds 0-3 (+4.5 points), but the 160-doc v1 dev and 400-doc v2 dev are different subsets, so dev numbers are not comparable across runs.

## Compute

| | 80-epoch anchor | v1 | v2 |
|---|---|---|---|
| Training GPU-hours | 180.1 | 1.59 | 3.17 (`loop_summary.json`) |
| % of anchor | 100% | 0.88% | 1.76% |
| GPU-h to reach the selected adapter | n/a | 0.76 (0.42%) | 2.09 (1.16%) |
| Train tokens | n/a | n/a | 1,419,175 (budget 2.5 M) |

Test eval of v2: 974 docs, 4,303 s generation + 20 s load on GPU1 (about 1.20 GPU-h), peak allocated 16.3 GB. Including the eval, v2 end-to-end is about 4.4 GPU-h (about 2.4% of the anchor's training alone; the anchor's own eval cost is not included in its 180.1).

## Test results (974 SkillSpan test docs with at least one gold span; identical extraction/scoring path to the anchor and v1)

| Metric | Anchor | v1 | v2 | v2 - anchor [95% CI] | v2 - v1 [95% CI] |
|---|---|---|---|---|---|
| exact-span F1 | 0.6701 | 0.6565 | 0.6614 | -0.0086 [-0.0286, +0.0114] | +0.0049 [-0.0133, +0.0232] |
| token F1 | 0.7879 | 0.7846 | 0.7957 | +0.0078 [-0.0084, +0.0235] | +0.0111 [-0.0046, +0.0267] |
| partial-span F1 | 0.8138 | 0.8040 | 0.8170 | +0.0032 [-0.0122, +0.0187] | +0.0131 [-0.0003, +0.0264] |
| HARD_SKILL F1 | 0.7265 | 0.7046 | 0.7159 | -0.0105 [-0.0402, +0.0193] | +0.0113 [-0.0125, +0.0347] |
| SOFT_SKILL F1 | 0.6040 | 0.5996 | 0.5989 | -0.0051 [-0.0299, +0.0202] | -0.0007 [-0.0273, +0.0263] |

p-values (raw paired bootstrap / permutation; Holm over the 5 metrics within the comparison; Holm over all 10 tests):

| Metric | v2 vs anchor raw | Holm-5 | Holm-10 | v2 vs v1 raw | Holm-5 | Holm-10 |
|---|---|---|---|---|---|---|
| exact-span F1 | 0.403 / 0.397 | 1.00 | 1.00 | 0.602 / 0.611 | 1.00 | 1.00 |
| token F1 | 0.335 / 0.343 | 1.00 | 1.00 | 0.159 / 0.159 | 0.64 | 1.00 |
| partial-span F1 | 0.691 / 0.689 | 1.00 | 1.00 | 0.057 / 0.058 | 0.28 / 0.29 | 0.57 / 0.58 |
| HARD_SKILL F1 | 0.477 / 0.485 | 1.00 | 1.00 | 0.352 / 0.359 | 1.00 | 1.00 |
| SOFT_SKILL F1 | 0.696 / 0.691 | 1.00 | 1.00 | 0.968 / 0.966 | 1.00 | 1.00 |

- v2 exact-span precision / recall: 0.6720 / 0.6512 (v1 0.6864 / 0.6291; anchor 0.6735 / 0.6667). Versus v1, v2 recovered some recall (+2.2 points) and lost some precision (-1.4); versus the anchor recall is still 1.5 points lower.
- Document level (doc exact-span F1), v2 vs anchor: wins 165, losses 151, ties 658 (sign test p = 0.46); perfectly extracted docs v2-only 93 vs anchor-only 80 (McNemar p = 0.36).
- Document level, v2 vs v1: wins 155, losses 137, ties 682 (sign test p = 0.32); perfectly extracted docs v2-only 98 vs v1-only 68 (McNemar p = 0.024, uncorrected, secondary, one of several doc-level tests, not adjusted; it is the only nominally significant result in the set and I would not lean on it).
- Minimum detectable effect (alpha 0.05, 80% power): about 0.029 for exact-span vs anchor and 0.026 vs v1. A true difference of about 1 point is not resolvable with 974 documents.
- Reproduction: the pooled-count F1s from saved predictions match `evaluate_extraction` and each system's stored metrics.json to 1e-6 for anchor, v1 and v2. 10,000 resamples, seed 20260710, resampling unit = document.

## Verdict

- Is v2 better than v1? Directionally yes on exact, token, partial and HARD F1 (+0.5 to +1.3 points), flat on SOFT, but no difference is significant (smallest raw p = 0.057 on partial-span F1; nothing below 0.28 after Holm over 5 metrics). The data are consistent with a real gain of about 1 point and also with zero. The only signal pointing to a v1-to-v2 improvement is the nominal McNemar p = 0.024 on perfectly-extracted docs, which does not survive any reasonable multiplicity correction.
- Is v2 at least as good as the anchor? Not shown. Exact-span, the headline metric, is 0.9 points lower with CI [-2.9, +1.1]; token and partial F1 are slightly higher. All five CIs include zero and the CIs are about 3 points wide, so the supportable statement is "no significant difference detected, within roughly 3 points either way", not "equivalent" and not "better". A non-inferiority claim would only hold for a margin of about 3 points or more on exact-span.
- The compute comparison holds: v2 uses 1.8% of the anchor's training GPU-hours (1.2% to reach the selected adapter), double v1's cost.
- Dev-to-test shrinkage: the selected adapter scored 0.7029 dev exact-span F1 and 0.6614 on test (-4.2 points), the same pattern as v1 (0.705 dev, 0.657 test). Dev scores therefore overstate test performance, consistent with selection on a noisy dev signal, and the apparent dev improvement trend in v2 translated into at most a small test gain over v1.

## What remains uncertain

- One training seed per run; between-seed variance is unmeasured. v1 vs v2 differ in several things at once (rank 8 vs 16, dev set 160 vs 400, patience, number of rounds, selected round), so the +0.5 to +1.3 point difference cannot be attributed to any single change.
- One split (SkillSpan test); no cross-validation or second dataset.
- The adapter was selected on dev (max over 6 noisy scores); training was cut by `max_rounds` at round 5 with dev still fluctuating (0.7005 in the last round), so a longer run might do better or worse.
- Anchor was evaluated with 2 visible GPUs (`device_map=auto`) and v1/v2 on GPU1 only; the effect of differing 4-bit kernel/split placement was not quantified (expected tiny).
- The loop's dev/test leakage checks from the v1 report (disjoint ids; repeated short texts across splits affect all systems equally) were not re-run for the 400-doc v2 dev subset; they are expected to carry over since both use `dev.json` and the same train split, but this was not verified.

## Files

- Training: `artifacts/skillllm_v2/qlora_loop_v2/{loop_summary.json,loop_state.json,best/}`; logs `logs/qlora_loop_v2_launch.log`, `logs/qlora_loop_20261021_231012.log`
- Test eval: `artifacts/skillllm_v2/qlora_loop_v2_eval/{predictions.jsonl,metrics.json,timing_segments.jsonl}`; `reports/skillllm_v2/qlora_loop_v2_test_metrics.json`; log `logs/qlora_loop_v2_test_eval.log`
- Significance: `reports/skillllm_v2/qlora_loop_v2_significance_vs_anchor.{json,md}`, `reports/skillllm_v2/qlora_loop_v2_significance_vs_v1.{json,md}`
- Code: `scripts/run_qlora_loop_v2_test_eval.sh` (sets env, then calls `scripts/eval_qlora_loop_test.py`, now reading `QLORA_ADAPTER`, `QLORA_EVAL_OUT`, `QLORA_REPORT_METRICS`, `QLORA_RUN_LABEL` with v1 defaults unchanged), `scripts/significance_qlora_v2.py`, `scripts/run_significance_qlora_v2.sh`
