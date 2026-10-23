# Experiment 2: ensemble distillation / self-training (v4) - held-out result

Question: can ONE adapter (1x inference cost) absorb the 3-adapter majority ensemble (0.6899) by training on gold SkillSpan train docs plus unlabeled in-domain chunks pseudo-labelled by the ensemble rule?
Short answer: **Success criterion B met (within 0.01 of the ensemble), criterion A not met.** v4 reaches 0.6846 exact-span F1 (ensemble 0.6899, anchor 0.6701, v2 0.6614). The gain over the anchor is not significant (+0.0145, CI [-0.0044, +0.0330]); the gain over its own warm start v2 is nominally significant (+0.0231) but **cannot be attributed to the pseudo-labels** (see caveats), and the dev curve never beat the warm start.

## Unlabeled corpus (counts only; text is private, kept outside git/reports)
`reports/skillllm_v2/qlora_loop_v4_corpus_counts.json`: 1,887 job postings; **55 removed** for overlap with SkillSpan train/dev/test text (rule: >= 50% of a >=8-token SkillSpan sentence's 8-gram shingles occur in the posting); 1,832 left; 600 sampled; 2,400 chunks (<= 4 per posting, line/sentence-sized, mean ~14 words); 34 chunks equal to a SkillSpan dev/test sentence were dropped beforehand.
Pseudo-labelling (`reports/skillllm_v2/qlora_loop_v4_pseudo_counts.json`): anchor, v1, v2 (all trained with the ORIGINAL prompt, so the original prompt was used; Exp 1 failed anyway), greedy, 7,200 generations, 0 parse failures. Distinct predicted spans by votes 1/2/3 = 932 / 653 / 924. Kept (>= 2 votes) 1,564 spans (892 HARD, 672 SOFT). 664 positive chunks; 86 chunks excluded (only 1-vote spans); 1,650 confident-negative chunks of which 221 used (<= 25% of the pool) -> **885 pseudo rows (75% positive)**. **367 of 664 positive chunks (55%) also contained 1-vote spans that were dropped** from the target, i.e. pseudo targets systematically omit some real skills (recall-noise source).

## Training (`configs/skillllm_v2_loop_qlora_v4_distill.yaml`)
Warm start v2 best (r=16), same loop trainer, 640 examples/round = 320 gold (hard-pool mining on gold train probes only) + 320 pseudo (without replacement; the 885-row pool is seen ~1.4x over 4 rounds), lr 7e-5 (pre-declared = v2's last-round LR), original prompt, dev subset/metric as v2. Planned 5 rounds / cap 2.4 GPU-h; an OOM (a foreign process held ~3.4 GB of GPU1) killed round 2's mining probe, I resumed with cap 2.0 GPU-h and eval batch 8, so the run stopped at 4 rounds by the GPU-h budget.

| | baseline (v2 best) | r0 | r1 | r2 | r3 |
|---|---|---|---|---|---|
| dev exact-span F1 (400 docs) | 0.7061 | 0.7031 | 0.6994 | **0.7052** | 0.7034 |
| dev P / R | 0.696 / 0.717 | 0.703 / 0.703 | 0.715 / 0.684 | 0.718 / 0.693 | 0.696 / 0.711 |

**The dev curve never beat the warm start** (best v4 round 0.7052 vs 0.7061; differences are far inside the dev SE ~0.02). The pre-declared selection rule picked round 2.

## Held-out test (974 docs, round 2, run once; original prompt; same scoring path)

| System | exact P | exact R | exact F1 | token F1 | partial F1 | HARD F1 | SOFT F1 |
|---|---|---|---|---|---|---|---|
| anchor | 0.6735 | 0.6667 | 0.6701 | 0.7879 | 0.8138 | 0.7265 | 0.6040 |
| v2 (warm start) | 0.6720 | 0.6512 | 0.6614 | 0.7957 | 0.8170 | 0.7159 | 0.5989 |
| ensemble (>=2/3) | 0.7220 | 0.6605 | 0.6899 | 0.7962 | 0.8213 | 0.7473 | 0.6231 |
| **v4** | **0.7059** | **0.6645** | **0.6846** | 0.8086 | 0.8365 | 0.7364 | 0.6275 |

Paired document bootstrap (10,000, seed 20261023) / permutation (`reports/skillllm_v2/qlora_loop_v4_significance.{md,json}`); delta = v4 minus reference:

| vs | exact-span delta [95% CI] | p boot / perm | Holm (3 exact-span tests) | token F1 delta [CI] | partial F1 delta [CI] |
|---|---|---|---|---|---|
| anchor | +0.0145 [-0.0044, +0.0330] | 0.138 / 0.132 | 0.276 / 0.265 | +0.0207 [+0.0050, +0.0367] (Holm-5 0.038) | +0.0226 [+0.0081, +0.0371] (Holm-5 0.013) |
| v2 | +0.0231 [+0.0054, +0.0403] | 0.0088 / 0.0086 | 0.026 / 0.026 | +0.0129 [-0.0010, +0.0265] | +0.0194 [+0.0062, +0.0322] |
| ensemble | -0.0053 [-0.0220, +0.0108] | 0.52 / 0.54 | 0.52 / 0.54 | +0.0124 [-0.0005, +0.0248] | +0.0152 [+0.0024, +0.0272] |

Doc level (exact-span F1): vs anchor wins 179 / losses 135 (sign p 0.015; perfect docs 106 vs 67, McNemar 0.0037); vs v2 130 / 93 (sign p 0.016); vs ensemble 118 / 122 (sign p 0.85). MDE (80% power) ~0.024-0.027.

Precision/recall: vs v2, precision +3.4 points (0.672 -> 0.706) and recall +1.3 (0.651 -> 0.664); vs ensemble precision -1.6, recall +0.4; vs anchor precision +3.2, recall -0.2. **Recall did not drop**: the 1-vote omissions did not visibly hurt recall at the test level, while the precision gain is what the voting targets teach (spans that several adapters agree on are the precise ones).

## Success criteria (pre-declared)
- A: v4 beats anchor with paired CI excluding 0 - **not met** (+0.0145, CI includes 0; p 0.138). Token/partial F1 vs anchor do exclude 0 (secondary metrics).
- B: v4 within 0.01 of the ensemble (0.6899) at 1x inference cost - **met on the point estimate** (-0.0053; CI [-0.022, +0.011] is too wide to claim equivalence formally).
**Verdict: partially successful, with the attribution unresolved.** v4 is the best single adapter we have (0.6846) and roughly matches the ensemble, but the evidence does not show that distillation (rather than simply more training) caused it.

## Caveats (important)
1. **No control.** v4 = v2 + 4 further rounds (~2,560 more training examples, half gold). The same extra training on gold alone was not run (budget), so the +0.023 over v2 may be "more gold training / a lucky checkpoint" rather than pseudo-label distillation. The dev curve (no gain over the warm start) points the same way: dev and test disagree, and dev (SE ~0.02, max of 4 noisy scores) is the weaker signal either way.
2. The anchor was selected by nothing; v2/v4 were dev-selected. One seed per system, one split; v4's CI vs the anchor includes 0.
3. Pseudo-label noise: 55% of positive pseudo chunks omit at least one span another adapter found; labels inherit the ensemble's correlated errors (same base model/training data).
4. Run history: guardrails/GPU contention (RED, a foreign process on GPU1) forced a ~2.5 h queue wait and one OOM; round 2 was redone (~0.4 GPU-h lost); resume used eval batch 8 (vs 16) and cap 2.0 instead of 2.4 GPU-h, and 4 rounds instead of 5.
5. v4 test generation latency 4.58 s/doc (mean; logprobs on) vs v2 4.42: single-adapter inference cost, no change.

## Files
Code/config: `scripts/{prepare_pseudo_corpus.py,pseudo_label_generate.py,run_pseudo_label_generate.sh,build_pseudo_sft.py,run_loop_v4_distill.sh,eval_adapter_test.py,run_qlora_loop_v4_test_eval.sh,significance_v3_v4.py}`, `configs/skillllm_v2_loop_qlora_v4_distill.yaml`, `src/skillllm_v2/train/loop_trainer.py` (`extra_frac`, `data.extra_train_file`).
Artifacts: `artifacts/skillllm_v2/qlora_loop_v4_distill/{best/,loop_summary.json,loop_state.json,pseudo_pool/}` (pseudo_pool holds PRIVATE posting text; artifacts is a symlink outside git), `artifacts/skillllm_v2/qlora_loop_v4_eval/`.
Reports: `reports/skillllm_v2/qlora_loop_v4_{summary.md,significance.md,significance.json,test_metrics.json,corpus_counts.json,pseudo_counts.json}`.
