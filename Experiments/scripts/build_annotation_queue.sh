#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

load_local_env
ensure_cache_dirs
export_benchmark_env

python3 "${PROJECT_ROOT}/src/build_annotation_queue.py" "$@"
