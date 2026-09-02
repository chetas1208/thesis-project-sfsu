# SkillLLM_v2 Artifact Index

## Status and logs

- `artifacts/skillllm_v2/status/*.success.json` - present
- `artifacts/skillllm_v2/status/*.failed.json` - missing
- `logs/latest_stage2_80epoch_2gpu.log` - present
- `logs/latest_stage2_80epoch_2gpu.pid` - present
- `reports/skillllm_v2/gpu_component_availability.json` - present
- `reports/skillllm_v2/gpu_component_availability.md` - present

## BERT baseline

- `artifacts/skillllm_v2/bert_baseline/model/model.json` - present
- `artifacts/skillllm_v2/bert_baseline/predictions.jsonl` - present
- `artifacts/skillllm_v2/bert_baseline/metrics.json` - present
- `reports/skillllm_v2/bert_baseline_metrics.json` - present

## GLiNER

- `artifacts/skillllm_v2/gliner_baseline/model/config.json` - present
- `artifacts/skillllm_v2/gliner_baseline/predictions.jsonl` - present
- `artifacts/skillllm_v2/gliner_baseline/metrics.json` - present
- `reports/skillllm_v2/gliner_baseline_metrics.json` - present

## Raw Mistral LoRA

- `artifacts/skillllm_v2/mistral_lora/model/` - present
- `artifacts/skillllm_v2/mistral_lora/model/training_summary.json` - present
- `artifacts/skillllm_v2/mistral_lora/model/adapter_status.json` - present
- `artifacts/skillllm_v2/mistral_lora/raw_outputs.jsonl` - present
- `artifacts/skillllm_v2/mistral_lora/predictions.jsonl` - present
- `artifacts/skillllm_v2/mistral_lora/metrics.json` - present
- `reports/skillllm_v2/raw_mistral_lora_metrics.json` - present

## Mistral + Boundary Refiner

- `artifacts/skillllm_v2/mistral_boundary/predictions.jsonl` - present
- `artifacts/skillllm_v2/mistral_boundary/metrics.json` - present
- `reports/skillllm_v2/mistral_boundary_metrics.json` - present

## Mistral + GLiNER + Taxonomy

- `artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl` - present
- `artifacts/skillllm_v2/mistral_gliner_taxonomy/linked_skills.jsonl` - present
- `artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json` - present
- `reports/skillllm_v2/mistral_gliner_taxonomy_metrics.json` - present

## SkillLLM_v2 Full CAG/Graph

- `artifacts/skillllm_v2/full_cag_graph/predictions.jsonl` - present
- `artifacts/skillllm_v2/full_cag_graph/linked_skills.jsonl` - present
- `artifacts/skillllm_v2/full_cag_graph/job_matches.jsonl` - present
- `artifacts/skillllm_v2/full_cag_graph/course_recommendations.jsonl` - present
- `artifacts/skillllm_v2/full_cag_graph/metrics.json` - present
- `reports/skillllm_v2/full_cag_graph_metrics.json` - present

## Verifier and linker

- `data/skillllm_v2/verifier_train.jsonl` - present
- `data/skillllm_v2/verifier_dev.jsonl` - present
- `artifacts/skillllm_v2/span_verifier/model/` - present
- `artifacts/skillllm_v2/span_verifier/metrics.json` - present
- `data/skillllm_v2/linker_train.jsonl` - present
- `data/skillllm_v2/linker_dev.jsonl` - present
- `artifacts/skillllm_v2/taxonomy_reranker/metrics.json` - present

## Final reports

- `reports/skillllm_v2/money_table.csv` - present
- `reports/skillllm_v2/money_table.md` - present
- `reports/skillllm_v2/extraction_ablation.csv` - present
- `reports/skillllm_v2/extraction_ablation.md` - present
- `reports/skillllm_v2/matching_eval.csv` - present
- `reports/skillllm_v2/course_recommendation_eval.csv` - present
- `reports/skillllm_v2/thesis_results_summary.md` - present
- `reports/skillllm_v2/error_analysis.md` - present
