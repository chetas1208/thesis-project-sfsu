#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ENV_NAME="skill-bench-skill-llm"

load_local_env
ensure_cache_dirs
export_benchmark_env
require_conda

if ! conda_env_exists "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/envs/skill_llm.yml"
fi

run_in_env "${ENV_NAME}" pip install --upgrade pip
run_in_env "${ENV_NAME}" pip install -r "${PROJECT_ROOT}/vendor/Skill-LLM/requirements.txt"
run_in_env "${ENV_NAME}" pip install python-dotenv
run_in_env "${ENV_NAME}" python -c "import torch, pandas, spacy; print('skill-llm env ready')"

