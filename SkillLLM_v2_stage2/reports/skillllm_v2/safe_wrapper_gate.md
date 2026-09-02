# SkillLLM_v2_stage2 Wrapper Non-Regression Gate

Each variant's extraction metrics are compared against the Raw Mistral LoRA anchor on the same split/sample. `accept` means the variant's extraction output is used downstream; `fallback_to_anchor` means the wrapper falls back to the unmodified Raw Mistral predictions for extraction (taxonomy/job/course outputs are still produced from the wrapper's enriched candidates).

| Variant | Decision | Improved | Exact F1 (anchor -> candidate) | Exact floor | Soft F1 (anchor -> candidate) | Soft floor |
|---|---|---|---|---|---|---|
| boundary_refiner | accept | False | 0.670069 -> 0.671111 | 0.660035 | 0.604046 -> 0.605505 | 0.574046 |
| mistral_gliner_taxonomy | accept | False | 0.670069 -> 0.670062 | 0.660035 | 0.604046 -> 0.602305 | 0.574046 |
| full_cag_graph | accept | False | 0.670069 -> 0.671102 | 0.660035 | 0.604046 -> 0.603755 | 0.574046 |

## Reasons

- **boundary_refiner**: boundary_refiner: exact_span_f1 0.671111 is below the improvement target (0.675034) but stays within the non-regression floor (>= 0.660035); soft_skill_f1 0.605505 stays within its guardrail (>= 0.574046) -- accepted as non-regressive.
- **mistral_gliner_taxonomy**: mistral_gliner_taxonomy: exact_span_f1 0.670062 is below the improvement target (0.675034) but stays within the non-regression floor (>= 0.660035); soft_skill_f1 0.602305 stays within its guardrail (>= 0.574046) -- accepted as non-regressive.
- **full_cag_graph**: full_cag_graph: exact_span_f1 0.671102 is below the improvement target (0.675034) but stays within the non-regression floor (>= 0.660035); soft_skill_f1 0.603755 stays within its guardrail (>= 0.574046) -- accepted as non-regressive.
