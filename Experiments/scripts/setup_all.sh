#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

load_local_env
ensure_cache_dirs
export_benchmark_env

bash "${PROJECT_ROOT}/scripts/setup_jobbert.sh"
bash "${PROJECT_ROOT}/scripts/setup_gliner.sh"
bash "${PROJECT_ROOT}/scripts/setup_skill_llm.sh"
bash "${PROJECT_ROOT}/scripts/setup_context.sh"
bash "${PROJECT_ROOT}/scripts/setup_openai.sh"
bash "${PROJECT_ROOT}/scripts/setup_nnose.sh"
