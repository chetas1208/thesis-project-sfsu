# SkillLLM_v2 Evaluation Plan

The money table systems are:

- BERT baseline
- GLiNER
- Raw Mistral LoRA
- Mistral + Boundary Refiner
- Mistral + GLiNER + Taxonomy
- SkillLLM_v2 Full CAG/Graph

The required money-table columns are exact span F1, token F1, soft skill F1, linking accuracy, job match nDCG@10, and course recommendation P@5. Missing metrics are written as `NA`; metrics are never fabricated.

Extraction metrics are computed on SkillSpan. Linking metrics are computed when canonical labels exist or otherwise reported as retrieval coverage with notes. Job and course ranking metrics require labeled relevance data; when unavailable, demo outputs are generated and ranking metric cells remain `NA`.
