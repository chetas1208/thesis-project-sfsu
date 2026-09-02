#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ENV_NAME="skill-bench-nnose"

load_local_env
ensure_cache_dirs
export_benchmark_env
require_conda

if ! conda_env_exists "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/envs/nnose.yml"
fi

run_in_env "${ENV_NAME}" pip install --upgrade pip
run_in_env "${ENV_NAME}" pip install "torch==1.10.1+cu113" -f https://download.pytorch.org/whl/torch_stable.html
run_in_env "${ENV_NAME}" pip install -r "${PROJECT_ROOT}/envs/nnose-requirements.txt"
run_in_env "${ENV_NAME}" python -c "import faiss, torch, transformers; print('nnose env ready')"
