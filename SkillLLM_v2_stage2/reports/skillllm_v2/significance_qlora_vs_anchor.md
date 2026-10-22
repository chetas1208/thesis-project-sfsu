# QLoRA-loop best adapter vs Raw Mistral LoRA anchor: paired significance (held-out test)

qlora_loop_best (round 1, r=8) minus Raw Mistral LoRA anchor (80 epochs, r=16). Paired document bootstrap (10,000 resamples, seed 20260710) and paired sign-flip permutation on 974 SkillSpan test docs (same docs/gold filter as the anchor). Delta = new minus anchor. Pooled-count F1s reproduce evaluate_extraction and each system's stored metrics.json to 1e-06. Holm correction is across the 5 metrics only. One training seed, one split.

| Metric | Anchor | New | Delta | 95% CI | p boot | p perm | Holm boot | Holm perm | MDE (80% power) |
|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.670069 | 0.656531 | -0.013538 | [-0.034229, +0.007060] | 0.2006 | 0.1992 | 0.8023 | 0.7968 | 0.0296 |
| token_f1 | 0.787886 | 0.784563 | -0.003323 | [-0.019789, +0.013336] | 0.6921 | 0.6955 | 1.0000 | 1.0000 | 0.0238 |
| partial_span_f1 | 0.813845 | 0.803962 | -0.009883 | [-0.025590, +0.005874] | 0.2160 | 0.2151 | 0.8023 | 0.7968 | 0.0224 |
| hard_skill_f1 | 0.726450 | 0.704632 | -0.021818 | [-0.051663, +0.007904] | 0.1544 | 0.1524 | 0.7719 | 0.7621 | 0.0427 |
| soft_skill_f1 | 0.604046 | 0.599598 | -0.004449 | [-0.031873, +0.023524] | 0.7617 | 0.7580 | 1.0000 | 1.0000 | 0.0398 |

## Permutation details

| Metric | Discordant docs | Method |
|---|---|---|
| exact_span_f1 | 412 | seeded Monte Carlo (200000 sign-flips, +1 smoothed) |
| token_f1 | 447 | seeded Monte Carlo (200000 sign-flips, +1 smoothed) |
| partial_span_f1 | 354 | seeded Monte Carlo (200000 sign-flips, +1 smoothed) |
| hard_skill_f1 | 260 | seeded Monte Carlo (200000 sign-flips, +1 smoothed) |
| soft_skill_f1 | 262 | seeded Monte Carlo (200000 sign-flips, +1 smoothed) |

## Document-level (doc exact-span F1)

New wins 180, losses 191, ties 603 of 974 docs (sign-test p = 0.6037). Perfectly extracted docs: new-only 97, anchor-only 114 (McNemar exact p = 0.2706).

Inputs (sha256 prefix): anchor: `artifacts/skillllm_v2/mistral_lora/predictions.jsonl` (ac5917d0267670b0); qlora_loop: `artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl` (b3c3589033644702)
