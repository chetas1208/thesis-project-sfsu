# QLoRA-loop best adapter: held-out test metrics (974 SkillSpan test docs)

| Metric | Anchor (Raw Mistral LoRA, 80 ep) | QLoRA loop best |
|---|---|---|
| exact_span_f1 | 0.670069 | 0.656531 |
| exact_span_precision / recall | 0.673506 / 0.666667 | 0.686416 / 0.629139 |
| partial_span_f1 | 0.813845 | 0.803962 |
| token_f1 | 0.787886 | 0.784563 |
| hard_skill_f1 | 0.726450 | 0.704632 |
| soft_skill_f1 | 0.604046 | 0.599598 |

Source: `qlora_loop_test_metrics.json`. See `qlora_loop_summary.md` and `significance_qlora_vs_anchor.md`.
