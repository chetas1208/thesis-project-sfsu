#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

mkdir -p "$MODELS_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD_BIN=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD_BIN=(huggingface-cli download)
else
  echo "Neither 'hf' nor 'huggingface-cli' is available on PATH." >&2
  exit 1
fi

download_model() {
  local repo_id="$1"
  local local_dir="$2"

  echo "Downloading ${repo_id} -> ${local_dir}"
  "${HF_DOWNLOAD_BIN[@]}" "$repo_id" \
    --local-dir "$local_dir" \
    --force-download=false
}

download_model "meta-llama/Llama-3.1-8B-Instruct" "$MODELS_DIR/llama-3.1-8b-instruct"
download_model "Qwen/Qwen3-14B" "$MODELS_DIR/qwen3-14b"
download_model "mistralai/Mistral-Small-3.1-24B-Instruct-2503" "$MODELS_DIR/mistral-small-3.1-24b-instruct-2503"
download_model "google/gemma-4-E4B-it" "$MODELS_DIR/gemma-4-e4b-it"
download_model "google/gemma-4-26B-A4B-it" "$MODELS_DIR/gemma-4-26b-a4b-it"
