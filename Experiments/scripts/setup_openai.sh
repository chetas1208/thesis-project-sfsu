#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ENV_NAME="skill-bench-openai"

load_local_env
ensure_cache_dirs
export_benchmark_env
require_conda

if ! conda_env_exists "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/envs/openai.yml"
fi

run_in_env "${ENV_NAME}" pip install --upgrade pip
run_in_env "${ENV_NAME}" pip install "openai>=1.50.0" "python-dotenv==1.0.1"
run_in_env "${ENV_NAME}" python -c "from openai import OpenAI; print('openai env ready')"

