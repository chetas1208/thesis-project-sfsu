# v3: paired significance on the held-out test (974 gold-bearing SkillSpan docs)

Paired document bootstrap (10,000 resamples, seed 20261023) + sign-flip permutation; Holm over 5 metrics within each comparison and over the 2 exact-span tests of this experiment. Delta = first minus second. One training seed per system.

| System | exact P | exact R | exact F1 | token F1 | partial F1 | HARD F1 | SOFT F1 |
|---|---|---|---|---|---|---|---|
| Raw Mistral LoRA anchor | 0.6735 | 0.6667 | 0.6701 | 0.7879 | 0.8138 | 0.7265 | 0.6040 |
| QLoRA-loop v1 | 0.6864 | 0.6291 | 0.6565 | 0.7846 | 0.8040 | 0.7046 | 0.5996 |
| QLoRA-loop v2 | 0.6720 | 0.6512 | 0.6614 | 0.7957 | 0.8170 | 0.7159 | 0.5989 |
| v3 short prompt | 0.6751 | 0.6155 | 0.6439 | 0.7701 | 0.8042 | 0.7047 | 0.5694 |
| 3-adapter majority ensemble (>=2/3) | 0.7220 | 0.6605 | 0.6899 | 0.7962 | 0.8213 | 0.7473 | 0.6231 |

## v3 short prompt minus QLoRA-loop v2

| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6614 | 0.6439 | -0.0176 | [-0.0356, +0.0008] | 0.0614 | 0.0567 | 0.1842 | 0.1702 | 0.0258 |
| token_f1 | 0.7957 | 0.7701 | -0.0256 | [-0.0427, -0.0082] | 0.0046 | 0.0032 | 0.0230 | 0.0160 | 0.0246 |
| partial_span_f1 | 0.8170 | 0.8042 | -0.0129 | [-0.0267, +0.0008] | 0.0640 | 0.0654 | 0.1842 | 0.1702 | 0.0196 |
| hard_skill_f1 | 0.7159 | 0.7047 | -0.0112 | [-0.0328, +0.0110] | 0.3218 | 0.3175 | 0.3218 | 0.3175 | 0.0314 |
| soft_skill_f1 | 0.5989 | 0.5694 | -0.0296 | [-0.0570, -0.0013] | 0.0398 | 0.0361 | 0.1592 | 0.1445 | 0.0394 |

Exact-span F1 Holm over the family: boot 0.0614 / perm 0.0567.
Doc level: v3 wins 107, losses 161, ties 706 (sign p = 0.0012); perfect docs v3-only 50 vs v2-only 88 (McNemar p = 0.0015).

## v3 short prompt minus Raw Mistral LoRA anchor

| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6701 | 0.6439 | -0.0262 | [-0.0462, -0.0057] | 0.0126 | 0.0119 | 0.0630 | 0.0595 | 0.0291 |
| token_f1 | 0.7879 | 0.7701 | -0.0178 | [-0.0342, -0.0013] | 0.0342 | 0.0333 | 0.1026 | 0.1000 | 0.0235 |
| partial_span_f1 | 0.8138 | 0.8042 | -0.0097 | [-0.0251, +0.0057] | 0.2186 | 0.2190 | 0.3064 | 0.2983 | 0.0219 |
| hard_skill_f1 | 0.7265 | 0.7047 | -0.0218 | [-0.0510, +0.0077] | 0.1532 | 0.1492 | 0.3064 | 0.2983 | 0.0422 |
| soft_skill_f1 | 0.6040 | 0.5694 | -0.0347 | [-0.0624, -0.0065] | 0.0156 | 0.0163 | 0.0630 | 0.0654 | 0.0402 |

Exact-span F1 Holm over the family: boot 0.0252 / perm 0.0238.
Doc level: v3 wins 153, losses 205, ties 616 (sign p = 0.0069); perfect docs v3-only 88 vs anchor-only 113 (McNemar p = 0.0902).

## Pre-declared criteria

```
{
 "criterion_1_point_within_0.01_of_v2": false,
 "criterion_1_formal_noninferiority_ci_low_gt_minus_0.01": false,
 "delta_v3_minus_v2": -0.01755506995722822,
 "ci95": [
  -0.0356165570820853,
  0.0008283346845336379
 ]
}
```
