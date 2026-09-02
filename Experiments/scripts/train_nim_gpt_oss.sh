#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

load_local_env
ensure_cache_dirs
export_benchmark_env

if [[ " $* " == *" --dry-run "* ]]; then
  python3 "${PROJECT_ROOT}/src/run_nvidia_build.py" --model-key nim_gpt_oss_20b "$@"
  exit 0
fi

require_conda
run_in_env "skill-bench-openai" python "${PROJECT_ROOT}/src/run_nvidia_build.py" --model-key nim_gpt_oss_20b "$@"
