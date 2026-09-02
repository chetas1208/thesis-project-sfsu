#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ENV_NAME="skill-bench-gliner"

load_local_env
ensure_cache_dirs
export_benchmark_env
require_conda

if ! conda_env_exists "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/envs/gliner.yml"
fi

run_in_env "${ENV_NAME}" pip install --upgrade pip
run_in_env "${ENV_NAME}" pip install \
  "numpy<2" \
  "torch==2.2.1" \
  "transformers==4.57.3" \
  "huggingface_hub==0.36.2" \
  "accelerate==1.13.0" \
  "onnxruntime-gpu==1.23.2" \
  "sentencepiece==0.2.1" \
  "tqdm==4.67.3"
run_in_env "${ENV_NAME}" pip install -e "${PROJECT_ROOT}/vendor/GLiNER" --no-deps
run_in_env "${ENV_NAME}" pip install python-dotenv
run_in_env "${ENV_NAME}" python -c "import torch; from gliner import GLiNER; print('gliner env ready')"
