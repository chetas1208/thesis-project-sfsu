#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_project_relative_path() {
  local path_value="$1"
  if [[ -z "${path_value}" ]]; then
    echo ""
    return
  fi
  if [[ "${path_value}" = /* ]]; then
    echo "${path_value}"
    return
  fi
  echo "${PROJECT_ROOT}/${path_value#./}"
}

load_local_env() {
  if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
  fi
}

ensure_cache_dirs() {
  mkdir -p \
    "${PROJECT_ROOT}/.cache/huggingface" \
    "${PROJECT_ROOT}/.cache/huggingface/transformers" \
    "${PROJECT_ROOT}/.cache/huggingface/hub" \
    "${PROJECT_ROOT}/.cache/pip" \
    "${PROJECT_ROOT}/.cache/openai" \
    "${PROJECT_ROOT}/.cache/tmp" \
    "${PROJECT_ROOT}/.cache/wandb" \
    "${PROJECT_ROOT}/.cache/metrics"
}

export_benchmark_env() {
  export HF_HOME="$(resolve_project_relative_path "${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}")"
  export TRANSFORMERS_CACHE="$(resolve_project_relative_path "${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}")"
  export HUGGINGFACE_HUB_CACHE="$(resolve_project_relative_path "${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}")"
  export PIP_CACHE_DIR="$(resolve_project_relative_path "${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}")"
  export TMPDIR="$(resolve_project_relative_path "${TMPDIR:-${PROJECT_ROOT}/.cache/tmp}")"
  export TMP="${TMPDIR}"
  export TEMP="${TMPDIR}"
  export WANDB_DIR="$(resolve_project_relative_path "${WANDB_DIR:-${PROJECT_ROOT}/.cache/wandb}")"
  export WANDB_CACHE_DIR="$(resolve_project_relative_path "${WANDB_CACHE_DIR:-${PROJECT_ROOT}/.cache/wandb}")"
  export OPENAI_CACHE_DIR="$(resolve_project_relative_path "${OPENAI_CACHE_DIR:-${PROJECT_ROOT}/.cache/openai}")"
}

require_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found on PATH." >&2
    exit 1
  fi
}

conda_env_exists() {
  local env_name="$1"
  conda env list | awk '{print $1}' | grep -Fxq "${env_name}"
}

run_in_env() {
  local env_name="$1"
  shift
  conda run --no-capture-output -n "${env_name}" "$@"
}
