# SkillLLM_v2 Gated Pipeline

The Stage-2 pipeline is artifact-driven. Every stage writes a success marker to `artifacts/skillllm_v2/status/{stage}.success.json` or a failure marker to `artifacts/skillllm_v2/status/{stage}.failed.json`.

## Stage Order

1. setup_stage2_repo
2. legacy_cleanup_check
3. schema_smoke
4. component_unit_tests
5. build_skillspan_bio_dataset
6. train_bert_baseline
7. eval_bert_baseline
8. train_or_run_gliner_baseline
9. eval_gliner_baseline
10. build_mistral_sft_data
11. train_or_verify_mistral_lora
12. eval_raw_mistral_lora
13. eval_mistral_boundary_refiner
14. extraction_component_smoke
15. merge_refine_verify_smoke
16. taxonomy_linking_smoke
17. eval_mistral_gliner_taxonomy
18. build_verifier_dataset
19. train_span_verifier
20. verifier_eval_smoke
21. build_linker_dataset
22. train_taxonomy_reranker
23. graph_smoke
24. matching_recommendation_smoke
25. eval_full_cag_graph
26. extraction_ablation_full
27. graph_matching_eval_full
28. money_table_generation
29. end_to_end_demo_smoke
30. final_report_generation
31. master_merge_readiness_check

## Commands

Run all smoke gates:

```bash
bash scripts/run_skillllm_v2_all_smokes.sh --config configs/skillllm_v2_smoke.yaml
```

Run the full pipeline:

```bash
bash scripts/run_skillllm_v2_pipeline.sh --config configs/skillllm_v2_training.yaml --mode full
```

Run one stage:

```bash
bash scripts/run_skillllm_v2_stage.sh --stage train_bert_baseline --config configs/skillllm_v2_training.yaml --mode full
```

## Resume And Failure Recovery

The runner skips completed stages when required outputs and success markers exist. Use `--force` to rerun all selected stages or `--force-stage STAGE_NAME` for one stage. Failed stages stop the pipeline immediately.
