# QLoRA-loop v2 (r=16) vs Raw Mistral LoRA anchor (80 epochs, r=16): paired significance (held-out test)

Delta = v2 minus anchor. Paired document bootstrap (10,000 resamples, seed 20260710) and paired sign-flip permutation on 974 SkillSpan test docs (docs with >=1 gold span). Pooled-count F1s reproduce evaluate_extraction and stored metrics.json to 1e-06. Holm: within this comparison (5 metrics) and across all 10 tests (2 comparisons x 5 metrics). One training seed per system, one split.

| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | Holm10 boot | Holm10 perm | MDE (80% power) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_span_f1 | 0.670069 | 0.661435 | -0.008634 | [-0.028600, +0.011404] | 0.4034 | 0.3971 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0287 |
| token_f1 | 0.787886 | 0.795709 | +0.007822 | [-0.008363, +0.023463] | 0.3350 | 0.3425 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0229 |
| partial_span_f1 | 0.813845 | 0.817040 | +0.003195 | [-0.012196, +0.018713] | 0.6913 | 0.6886 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0222 |
| hard_skill_f1 | 0.726450 | 0.715904 | -0.010546 | [-0.040224, +0.019294] | 0.4774 | 0.4851 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0423 |
| soft_skill_f1 | 0.604046 | 0.598941 | -0.005105 | [-0.029938, +0.020158] | 0.6955 | 0.6913 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0355 |

## Document-level (doc exact-span F1)

v2 wins 165, losses 151, ties 658 of 974 docs (sign-test p = 0.4646). Perfectly extracted docs: v2-only 93, anchor-only 80 (McNemar exact p = 0.3616).

Inputs (sha256 prefix): anchor: `artifacts/skillllm_v2/mistral_lora/predictions.jsonl` (ac5917d0267670b0); v1: `artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl` (b3c3589033644702); v2: `artifacts/skillllm_v2/qlora_loop_v2_eval/predictions.jsonl` (e4d305ddda358a6c)
