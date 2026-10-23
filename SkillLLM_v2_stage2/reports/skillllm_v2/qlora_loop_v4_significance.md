# v4: paired significance on the held-out test (974 gold-bearing SkillSpan docs)

Paired document bootstrap (10,000 resamples, seed 20261023) + sign-flip permutation; Holm over 5 metrics within each comparison and over the 3 exact-span tests of this experiment. Delta = first minus second. One training seed per system.

| System | exact P | exact R | exact F1 | token F1 | partial F1 | HARD F1 | SOFT F1 |
|---|---|---|---|---|---|---|---|
| Raw Mistral LoRA anchor | 0.6735 | 0.6667 | 0.6701 | 0.7879 | 0.8138 | 0.7265 | 0.6040 |
| QLoRA-loop v1 | 0.6864 | 0.6291 | 0.6565 | 0.7846 | 0.8040 | 0.7046 | 0.5996 |
| QLoRA-loop v2 | 0.6720 | 0.6512 | 0.6614 | 0.7957 | 0.8170 | 0.7159 | 0.5989 |
| v4 distilled (single adapter) | 0.7059 | 0.6645 | 0.6846 | 0.8086 | 0.8365 | 0.7364 | 0.6275 |
| 3-adapter majority ensemble (>=2/3) | 0.7220 | 0.6605 | 0.6899 | 0.7962 | 0.8213 | 0.7473 | 0.6231 |

## v4 distilled (single adapter) minus Raw Mistral LoRA anchor

| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6701 | 0.6846 | +0.0145 | [-0.0044, +0.0330] | 0.1380 | 0.1324 | 0.2760 | 0.2801 | 0.0267 |
| token_f1 | 0.7879 | 0.8086 | +0.0207 | [+0.0050, +0.0367] | 0.0096 | 0.0103 | 0.0384 | 0.0413 | 0.0225 |
| partial_span_f1 | 0.8138 | 0.8365 | +0.0226 | [+0.0081, +0.0371] | 0.0026 | 0.0021 | 0.0130 | 0.0103 | 0.0207 |
| hard_skill_f1 | 0.7265 | 0.7364 | +0.0100 | [-0.0148, +0.0341] | 0.4420 | 0.4301 | 0.4420 | 0.4301 | 0.0353 |
| soft_skill_f1 | 0.6040 | 0.6275 | +0.0235 | [-0.0039, +0.0506] | 0.0906 | 0.0934 | 0.2718 | 0.2801 | 0.0388 |

Exact-span F1 Holm over the family: boot 0.2760 / perm 0.2648.
Doc level: v4 wins 179, losses 135, ties 660 (sign p = 0.0151); perfect docs v4-only 106 vs anchor-only 67 (McNemar p = 0.0037).

## v4 distilled (single adapter) minus 3-adapter majority ensemble (>=2/3)

| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6899 | 0.6846 | -0.0053 | [-0.0220, +0.0108] | 0.5215 | 0.5358 | 1.0000 | 1.0000 | 0.0237 |
| token_f1 | 0.7962 | 0.8086 | +0.0124 | [-0.0005, +0.0248] | 0.0600 | 0.0589 | 0.2400 | 0.2355 | 0.0181 |
| partial_span_f1 | 0.8213 | 0.8365 | +0.0152 | [+0.0024, +0.0272] | 0.0182 | 0.0165 | 0.0910 | 0.0826 | 0.0177 |
| hard_skill_f1 | 0.7473 | 0.7364 | -0.0109 | [-0.0338, +0.0106] | 0.3264 | 0.3393 | 0.9791 | 1.0000 | 0.0318 |
| soft_skill_f1 | 0.6231 | 0.6275 | +0.0044 | [-0.0180, +0.0274] | 0.7343 | 0.7166 | 1.0000 | 1.0000 | 0.0330 |

Exact-span F1 Holm over the family: boot 0.5215 / perm 0.5358.
Doc level: v4 wins 118, losses 122, ties 734 (sign p = 0.8465); perfect docs v4-only 67 vs ensemble-only 50 (McNemar p = 0.1388).

## v4 distilled (single adapter) minus QLoRA-loop v2

| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6614 | 0.6846 | +0.0231 | [+0.0054, +0.0403] | 0.0088 | 0.0086 | 0.0352 | 0.0342 | 0.0249 |
| token_f1 | 0.7957 | 0.8086 | +0.0129 | [-0.0010, +0.0265] | 0.0702 | 0.0678 | 0.1404 | 0.1356 | 0.0197 |
| partial_span_f1 | 0.8170 | 0.8365 | +0.0194 | [+0.0062, +0.0322] | 0.0040 | 0.0036 | 0.0200 | 0.0179 | 0.0187 |
| hard_skill_f1 | 0.7159 | 0.7364 | +0.0205 | [-0.0033, +0.0437] | 0.0898 | 0.0854 | 0.1404 | 0.1356 | 0.0334 |
| soft_skill_f1 | 0.5989 | 0.6275 | +0.0286 | [+0.0056, +0.0512] | 0.0170 | 0.0148 | 0.0510 | 0.0445 | 0.0328 |

Exact-span F1 Holm over the family: boot 0.0264 / perm 0.0257.
Doc level: v4 wins 130, losses 93, ties 751 (sign p = 0.0157); perfect docs v4-only 72 vs v2-only 46 (McNemar p = 0.0210).

## Pre-declared criteria

```
{
 "criterion_A_beats_anchor_ci_excludes_0": false,
 "A_delta": 0.014488871050359675,
 "A_ci95": [
  -0.004416246706719123,
  0.03302877533720225
 ],
 "criterion_B_within_0.01_of_ensemble_point": true,
 "B_delta_v4_minus_ensemble": -0.005320142766131353,
 "B_ci95": [
  -0.021973972878324928,
  0.01079085674774357
 ]
}
```
