#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

load_local_env
ensure_cache_dirs
export_benchmark_env

python3 "${PROJECT_ROOT}/src/prepare_benchmark_data.py" --dry-run
python3 "${PROJECT_ROOT}/src/download_kaggle_sources.py" --dry-run
python3 "${PROJECT_ROOT}/src/clean_resume_sources.py" --dry-run
python3 "${PROJECT_ROOT}/src/build_annotation_queue.py" --dry-run
python3 "${PROJECT_ROOT}/src/evaluate_all.py"
python3 "${PROJECT_ROOT}/analysis/run_dataset_eda.py"
python3 "${PROJECT_ROOT}/analysis/run_benchmark_analysis.py"
bash "${PROJECT_ROOT}/scripts/train_jobbert.sh" --dry-run
bash "${PROJECT_ROOT}/scripts/train_nnose.sh" --dry-run
bash "${PROJECT_ROOT}/scripts/train_gliner.sh" --dry-run
bash "${PROJECT_ROOT}/scripts/train_skill_llm.sh" --dry-run
bash "${PROJECT_ROOT}/scripts/train_context.sh" --dry-run
bash "${PROJECT_ROOT}/scripts/train_gpt5.sh" --dry-run
