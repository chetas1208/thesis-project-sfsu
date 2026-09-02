# SkillLLM_v2_stage2 Wrapper Non-Regression Gate

Each variant's extraction metrics are compared against the Raw Mistral LoRA anchor on the same split/sample. `accept` means the variant's extraction output is used downstream; `fallback_to_anchor` means the wrapper falls back to the unmodified Raw Mistral predictions for extraction (taxonomy/job/course outputs are still produced from the wrapper's enriched candidates).

| Variant | Decision | Improved | Exact F1 (anchor -> candidate) | Exact floor | Soft F1 (anchor -> candidate) | Soft floor |
|---|---|---|---|---|---|---|
| boundary_refiner | accept | False | 0.666667 -> 0.666667 | 0.656633 | 0.640000 -> 0.640000 | 0.610000 |
| mistral_gliner_taxonomy | accept | False | 0.666667 -> 0.666667 | 0.656633 | 0.640000 -> 0.640000 | 0.610000 |
| full_cag_graph | accept | False | 0.666667 -> 0.666667 | 0.656633 | 0.640000 -> 0.640000 | 0.610000 |

## Reasons

- **boundary_refiner**: boundary_refiner: exact_span_f1 0.666667 is below the improvement target (0.675034) but stays within the non-regression floor (>= 0.656633); soft_skill_f1 0.640000 stays within its guardrail (>= 0.610000) -- accepted as non-regressive.
- **mistral_gliner_taxonomy**: mistral_gliner_taxonomy: exact_span_f1 0.666667 is below the improvement target (0.675034) but stays within the non-regression floor (>= 0.656633); soft_skill_f1 0.640000 stays within its guardrail (>= 0.610000) -- accepted as non-regressive.
- **full_cag_graph**: full_cag_graph: exact_span_f1 0.666667 is below the improvement target (0.675034) but stays within the non-regression floor (>= 0.656633); soft_skill_f1 0.640000 stays within its guardrail (>= 0.610000) -- accepted as non-regressive.
