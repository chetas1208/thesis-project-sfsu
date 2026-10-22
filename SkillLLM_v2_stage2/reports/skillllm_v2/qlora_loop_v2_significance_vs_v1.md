# QLoRA-loop v2 (r=16) vs QLoRA-loop v1 (r=8, round 1): paired significance (held-out test)

Delta = v2 minus v1. Paired document bootstrap (10,000 resamples, seed 20260710) and paired sign-flip permutation on 974 SkillSpan test docs (docs with >=1 gold span). Pooled-count F1s reproduce evaluate_extraction and stored metrics.json to 1e-06. Holm: within this comparison (5 metrics) and across all 10 tests (2 comparisons x 5 metrics). One training seed per system, one split.

| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | Holm10 boot | Holm10 perm | MDE (80% power) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.656531 | 0.661435 | +0.004904 | [-0.013330, +0.023217] | 0.6017 | 0.6106 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0262 |
| token_f1 | 0.784563 | 0.795709 | +0.011146 | [-0.004553, +0.026666] | 0.1590 | 0.1591 | 0.6359 | 0.6363 | 1.0000 | 1.0000 | 0.0223 |
| partial_span_f1 | 0.803962 | 0.817040 | +0.013078 | [-0.000347, +0.026448] | 0.0568 | 0.0578 | 0.2840 | 0.2890 | 0.5679 | 0.5779 | 0.0190 |
| hard_skill_f1 | 0.704632 | 0.715904 | +0.011272 | [-0.012452, +0.034660] | 0.3524 | 0.3590 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0337 |
| soft_skill_f1 | 0.599598 | 0.598941 | -0.000657 | [-0.027286, +0.026256] | 0.9675 | 0.9663 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0386 |

## Document-level (doc exact-span F1)

v2 wins 155, losses 137, ties 682 of 974 docs (sign-test p = 0.3198). Perfectly extracted docs: v2-only 98, v1-only 68 (McNemar exact p = 0.0241).

Inputs (sha256 prefix): anchor: `artifacts/skillllm_v2/mistral_lora/predictions.jsonl` (ac5917d0267670b0); v1: `artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl` (b3c3589033644702); v2: `artifacts/skillllm_v2/qlora_loop_v2_eval/predictions.jsonl` (e4d305ddda358a6c)
