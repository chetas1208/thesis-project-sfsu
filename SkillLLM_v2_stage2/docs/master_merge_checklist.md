# Master Merge Checklist

- [ ] Original `SkillLLM_v2` master repo remains untouched during Stage-2 implementation.
- [ ] Stage-2 work is on `feature/skillllm-v2-cag-wrapper`.
- [ ] No active Qwen/Gemma/Llama experiment configs remain in `configs/`, `scripts/`, or `src/skillllm_v2/`.
- [ ] No old retry scripts remain active.
- [ ] `docs/legacy_benchmark_summary.md` documents benchmark evidence.
- [ ] `docs/removed_artifacts_manifest.md` documents removed or excluded artifacts.
- [ ] `bash scripts/setup_skillllm_v2_stage2.sh` passes.
- [ ] `bash scripts/run_skillllm_v2_smoke.sh 5` passes.
- [ ] `PYTHONPATH=src pytest tests/` passes.
- [ ] `bash scripts/run_skillllm_v2_extraction_ablation.sh 5` passes.
- [ ] `bash scripts/run_skillllm_v2_pipeline.sh --config configs/skillllm_v2_smoke.yaml --mode smoke` passes.
- [ ] Money table exists with six required rows.
- [ ] Reports exist under `reports/skillllm_v2/`.
- [ ] Final diff is reviewed before copying changes back to master.
