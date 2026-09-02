# Metric Sanity Report

Metrics below are read from local artifacts only. Missing values are not imputed.

| System | Exact F1 | Token F1 | SOFT F1 | Status |
|---|---:|---:|---:|---|
| BERT baseline | 0.059150 | 0.274922 | 0.018796 | present |
| GLiNER | 0.075833 | 0.114409 | 0.096571 | present |
| Raw Mistral LoRA | 0.670069 | 0.787886 | 0.604046 | present |
| Old Boundary Refiner | 0.670069 | 0.787886 | 0.604046 | present |
| Old Mistral + GLiNER + Taxonomy | 0.670062 | 0.787575 | 0.602305 | present |
| Old Full CAG/Graph | 0.671102 | 0.787575 | 0.603755 | present |
| SkillLLM_v2 Safe Wrapper | 0.671102 | 0.787886 | 0.603755 | present |
