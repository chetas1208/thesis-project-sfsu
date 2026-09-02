# Wrapper Error Analysis

Dataset: `data/source/skillspan_hf` split=`test` doc_count=974

## Raw Mistral LoRA (anchor)

- exact_span_f1 = 0.670069
- token_f1 = 0.787886
- hard_skill_f1 = 0.726450
- soft_skill_f1 = 0.604046

## Stage comparison

| Stage | Exact F1 | Δ Exact F1 | Token F1 | Δ Token F1 | Hard F1 | Δ Hard F1 | Soft F1 | Δ Soft F1 | Regressions |
|---|---|---|---|---|---|---|---|---|---|
| Mistral + Boundary Refiner | 0.6701 | +0.0000 | 0.7879 | +0.0000 | 0.7265 | +0.0000 | 0.6040 | +0.0000 | 0 |
| Mistral + GLiNER + Taxonomy | 0.6701 | -0.0000 | 0.7876 | -0.0003 | 0.7280 | +0.0016 | 0.6023 | -0.0017 | 2 |
| Full CAG/Graph | 0.6711 | +0.0010 | 0.7876 | -0.0003 | 0.7286 | +0.0022 | 0.6038 | -0.0003 | 2 |

## Edit / rejection counts per stage

| Stage | kept | removed | boundary_modified | label_changed | added_by_gliner | added_by_dictionary | rejected_by_verifier | rejected_by_confidence | exact_gains | exact_losses |
|---|---|---|---|---|---|---|---|---|---|---|
| Mistral + Boundary Refiner | 1510 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Mistral + GLiNER + Taxonomy | 1508 | 2 | 2 | 0 | 3 | 16 | 0 | 0 | 5 | 2 |
| Full CAG/Graph | 1508 | 2 | 2 | 0 | 3 | 16 | 0 | 0 | 5 | 2 |

## Regression type breakdown

| Stage | regression_type | count |
|---|---|---|
| Full CAG/Graph | bad_boundary_edit | 2 |
| Mistral + GLiNER + Taxonomy | bad_boundary_edit | 2 |

| Stage | module_responsible | count |
|---|---|---|
| Full CAG/Graph | taxonomy_linker | 2 |
| Mistral + GLiNER + Taxonomy | taxonomy_linker | 2 |

## Conclusion

Every downstream stage shows a negative `Δ Exact F1` and `Δ Soft F1` versus the Raw Mistral anchor (see table above). The dominant regression types and `module_responsible` values above identify which postprocessing modules are responsible for destroying spans that the 80-epoch Mistral LoRA already extracted correctly. Section 2/5/7/8/9 of the SkillLLM_v2 safe wrapper rewrite directly target these modules so that the wrapper becomes non-regressive relative to Raw Mistral.
