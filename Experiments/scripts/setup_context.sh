#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ENV_NAME="skill-bench-context"
SHARED_BASE_ENV="skill-bench-jobbert"

load_local_env
ensure_cache_dirs
export_benchmark_env
require_conda

if ! conda_env_exists "${SHARED_BASE_ENV}"; then
  bash "${PROJECT_ROOT}/scripts/setup_jobbert.sh"
fi

if ! conda_env_exists "${ENV_NAME}"; then
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/envs/context.yml"
fi

CONTEXT_SITE_PACKAGES="$(
  run_in_env "${ENV_NAME}" python -c 'import site; print(site.getsitepackages()[0])'
)"
SHARED_SITE_PACKAGES="$(
  run_in_env "${SHARED_BASE_ENV}" python -c 'import site; print(site.getsitepackages()[0])'
)"

printf '%s\n' "${SHARED_SITE_PACKAGES}" > "${CONTEXT_SITE_PACKAGES}/skill_bench_jobbert_shared.pth"

run_in_env "${ENV_NAME}" pip install --no-input --no-cache-dir --no-deps "sentence-transformers==3.4.0" "pillow>=10,<12"
run_in_env "${ENV_NAME}" python -c "import sentence_transformers, torch, sklearn; print('context env ready')"
