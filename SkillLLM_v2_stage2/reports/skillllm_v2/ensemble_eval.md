# Span-voting ensemble of the three adapters (held-out test)

Pre-declared rule: keep a span (start, end, label) if >= 2 of 3 systems (anchor, QLoRA-loop v1, v2) predict it. The primary and only selection candidate is **majority(>=2/3)**; unanimity and union are descriptive. 974 SkillSpan test docs, paired bootstrap (10,000 resamples) + sign-flip permutation, Holm over 5 metrics.

## Pooled F1 per system

| System | exact P | exact R | exact F1 | token F1 | partial F1 | HARD F1 | SOFT F1 |
|---|---|---|---|---|---|---|---|
| anchor | 0.6735 | 0.6667 | 0.6701 | 0.7879 | 0.8138 | 0.7265 | 0.6040 |
| v1 | 0.6864 | 0.6291 | 0.6565 | 0.7846 | 0.8040 | 0.7046 | 0.5996 |
| v2 | 0.6720 | 0.6512 | 0.6614 | 0.7957 | 0.8170 | 0.7159 | 0.5989 |
| majority(>=2/3) | 0.7220 | 0.6605 | 0.6899 | 0.7962 | 0.8213 | 0.7473 | 0.6231 |
| unanimous(3/3) | 0.8018 | 0.5055 | 0.6201 | 0.6583 | 0.6986 | 0.6890 | 0.5387 |
| union(>=1/3) | 0.5871 | 0.7810 | 0.6703 | 0.7993 | 0.7689 | 0.7087 | 0.6257 |

## majority(>=2/3) minus anchor  (PRIMARY)

| Metric | Anchor | Ensemble | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm |
|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6701 | 0.6899 | +0.0198 | [+0.0041, +0.0356] | 0.0162 | 0.0146 | 0.0810 | 0.0728 |
| token_f1 | 0.7879 | 0.7962 | +0.0083 | [-0.0036, +0.0203] | 0.1748 | 0.1715 | 0.3496 | 0.3429 |
| partial_span_f1 | 0.8138 | 0.8213 | +0.0075 | [-0.0047, +0.0203] | 0.2482 | 0.2457 | 0.3496 | 0.3429 |
| hard_skill_f1 | 0.7265 | 0.7473 | +0.0209 | [-0.0035, +0.0457] | 0.0966 | 0.0936 | 0.2898 | 0.2808 |
| soft_skill_f1 | 0.6040 | 0.6231 | +0.0191 | [+0.0009, +0.0378] | 0.0398 | 0.0428 | 0.1592 | 0.1711 |

Doc-level exact-span F1: ensemble wins 148, losses 91, ties 735 (sign test p = 0.0003).

## unanimous(3/3) minus anchor  (descriptive)

| Metric | Anchor | Ensemble | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm |
|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6701 | 0.6201 | -0.0500 | [-0.0670, -0.0333] | 0.0002 | 0.0000 | 0.0010 | 0.0000 |
| token_f1 | 0.7879 | 0.6583 | -0.1296 | [-0.1488, -0.1106] | 0.0002 | 0.0000 | 0.0010 | 0.0000 |
| partial_span_f1 | 0.8138 | 0.6986 | -0.1152 | [-0.1331, -0.0980] | 0.0002 | 0.0000 | 0.0010 | 0.0000 |
| hard_skill_f1 | 0.7265 | 0.6890 | -0.0375 | [-0.0635, -0.0123] | 0.0038 | 0.0049 | 0.0038 | 0.0049 |
| soft_skill_f1 | 0.6040 | 0.5387 | -0.0654 | [-0.0879, -0.0427] | 0.0002 | 0.0000 | 0.0010 | 0.0000 |

Doc-level exact-span F1: ensemble wins 90, losses 230, ties 654 (sign test p = 0.0000).

## union(>=1/3) minus anchor  (descriptive)

| Metric | Anchor | Ensemble | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm |
|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.6701 | 0.6703 | +0.0003 | [-0.0123, +0.0135] | 0.9611 | 0.9717 | 0.9611 | 0.9717 |
| token_f1 | 0.7879 | 0.7993 | +0.0114 | [-0.0006, +0.0238] | 0.0642 | 0.0630 | 0.1926 | 0.1872 |
| partial_span_f1 | 0.8138 | 0.7689 | -0.0450 | [-0.0567, -0.0329] | 0.0002 | 0.0000 | 0.0010 | 0.0000 |
| hard_skill_f1 | 0.7265 | 0.7087 | -0.0177 | [-0.0359, +0.0014] | 0.0690 | 0.0624 | 0.1926 | 0.1872 |
| soft_skill_f1 | 0.6040 | 0.6257 | +0.0217 | [+0.0048, +0.0397] | 0.0116 | 0.0105 | 0.0464 | 0.0421 |

Doc-level exact-span F1: ensemble wins 196, losses 189, ties 589 (sign test p = 0.7598).

## Caveats

- The three adapters share the same base model and training data, so their errors are correlated; voting gains are bounded by that.
- One test split, one seed per adapter. The rule was fixed in advance, but only one rule is a valid test; the others are context.
- Ensembling costs 3x inference (three adapters over one 4-bit base, adapters can be hot-swapped but each needs its own generation pass).
