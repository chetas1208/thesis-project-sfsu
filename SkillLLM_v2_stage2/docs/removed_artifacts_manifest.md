# Removed Or Excluded Artifacts Manifest

This manifest records files and directories removed from or excluded in the copied `SkillLLM_v2_stage2` repo. The original `SkillLLM_v2` repo remains untouched.

| Path In Original Repo | Stage-2 Action | Reason |
| --- | --- | --- |
| `models/` | Excluded from copy | Large cached model weights; configured paths remain external. |
| `results/skillspan_baselines/` | Excluded from copy | Completed legacy benchmark artifacts. Summary preserved in docs. |
| `results/skillspan_baselines_smoke_limit_5/` | Excluded from copy | Legacy smoke benchmark output. |
| `logs/` | Excluded from copy | Runtime logs from completed benchmarks and retries. |
| `.venv/` | Excluded from copy | Environment artifact, not source. |
| `.cache/` | Excluded from copy | Cache artifact, not source. |
| `.env` | Removed from staging | Local secrets/config must not be copied into the clean Stage-2 repo. |
| `Extract_jobDesc/` | Removed from staging | Separate job-description scraping experiment, not active Stage-2 code. |
| `configs/*qwen*`, `configs/*gemma*`, `configs/*llama*`, `configs/skillspan_baselines.yaml` | Removed from staging | Old active multi-model benchmark configs. |
| `scripts/run_*gemma*`, `scripts/run_*qwen*`, `scripts/run_baseline.sh`, `scripts/run_skillspan_benchmark*` | Removed from staging | Old active multi-model benchmark entrypoints. |
| `src/benchmark/`, `src/infer/`, old `src/train/`, old `src/eval/` | Removed from staging | Replaced by the final Stage-2 wrapper and focused evaluation utilities. |
| `reports/final_model_comparison.*`, `reports/README.md` | Removed from staging | Legacy multi-model reporting replaced by Stage-2 reports and benchmark summary docs. |

The legacy benchmark numbers are retained in `docs/legacy_benchmark_summary.md`.
