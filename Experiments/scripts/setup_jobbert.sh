#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ENV_NAME="skill-bench-jobbert"

load_local_env
ensure_cache_dirs
export_benchmark_env
require_conda

if ! conda_env_exists "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/envs/jobbert.yml"
fi

run_in_env "${ENV_NAME}" pip install --upgrade pip
run_in_env "${ENV_NAME}" pip install \
  "torch==2.2.1" \
  "transformers==4.48.1" \
  "datasets==2.18.0" \
  "accelerate==0.29.3" \
  "evaluate==0.4.1" \
  "seqeval==1.2.2" \
  "scikit-learn==1.4.1.post1" \
  "python-dotenv==1.0.1"
run_in_env "${ENV_NAME}" python -c "import torch, transformers, datasets; print('jobbert env ready')"

