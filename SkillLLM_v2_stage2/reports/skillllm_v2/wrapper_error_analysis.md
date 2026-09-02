# Wrapper Error Analysis

Dataset: `data/source/skillspan_hf` split=`test` doc_count=974

## Raw Mistral LoRA (anchor)

- exact_span_f1 = 0.675034
- token_f1 = 0.797798
- hard_skill_f1 = 0.731138
- soft_skill_f1 = 0.609792

## Stage comparison

| Stage | Exact F1 | Δ Exact F1 | Token F1 | Δ Token F1 | Hard F1 | Δ Hard F1 | Soft F1 | Δ Soft F1 | Regressions |
|---|---|---|---|---|---|---|---|---|---|
| Mistral + Boundary Refiner | 0.5704 | -0.1046 | 0.7847 | -0.0131 | 0.7086 | -0.0225 | 0.4258 | -0.1840 | 194 |
| Mistral + GLiNER + Taxonomy | 0.6515 | -0.0236 | 0.7827 | -0.0151 | 0.7181 | -0.0130 | 0.5777 | -0.0321 | 0 |
| Full CAG/Graph | 0.5616 | -0.1135 | 0.7482 | -0.0496 | 0.7069 | -0.0243 | 0.4080 | -0.2018 | 225 |

## Edit / rejection counts per stage

| Stage | kept | removed | boundary_modified | label_changed | added_by_gliner | added_by_dictionary | rejected_by_verifier | rejected_by_confidence | exact_gains | exact_losses |
|---|---|---|---|---|---|---|---|---|---|---|
| Mistral + Boundary Refiner | 1312 | 194 | 194 | 0 | 0 | 0 | 0 | 0 | 53 | 194 |
| Mistral + GLiNER + Taxonomy | 1506 | 0 | 0 | 0 | 227 | 51 | 0 | 0 | 36 | 0 |
| Full CAG/Graph | 1281 | 225 | 194 | 2 | 146 | 51 | 29 | 0 | 78 | 225 |

## Regression type breakdown

| Stage | regression_type | count |
|---|---|---|
| Full CAG/Graph | bad_boundary_edit | 194 |
| Full CAG/Graph | bad_label_change | 2 |
| Full CAG/Graph | verifier_false_rejection | 29 |
| Mistral + Boundary Refiner | bad_boundary_edit | 194 |

| Stage | module_responsible | count |
|---|---|---|
| Full CAG/Graph | boundary_refiner | 196 |
| Full CAG/Graph | span_verifier | 29 |
| Mistral + Boundary Refiner | boundary_refiner | 194 |

## Conclusion

Every downstream stage shows a negative `Δ Exact F1` and `Δ Soft F1` versus the Raw Mistral anchor (see table above). The dominant regression types and `module_responsible` values above identify which postprocessing modules are responsible for destroying spans that the 80-epoch Mistral LoRA already extracted correctly. Section 2/5/7/8/9 of the SkillLLM_v2 safe wrapper rewrite directly target these modules so that the wrapper becomes non-regressive relative to Raw Mistral.
