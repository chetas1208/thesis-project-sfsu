#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

load_local_env
ensure_cache_dirs
export_benchmark_env

bash "${PROJECT_ROOT}/scripts/download_kaggle_sources.sh" "$@"
bash "${PROJECT_ROOT}/scripts/clean_resume_sources.sh"
bash "${PROJECT_ROOT}/scripts/build_annotation_queue.sh"
