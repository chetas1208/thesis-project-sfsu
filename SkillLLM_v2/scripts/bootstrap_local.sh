#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-0}"
BUILD_BASELINE_DATA="${BUILD_BASELINE_DATA:-1}"
BUILD_V2_DATA="${BUILD_V2_DATA:-1}"

bash "$ROOT_DIR/scripts/setup_env.sh"
source "$ROOT_DIR/.venv/bin/activate"

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  bash "$ROOT_DIR/scripts/download_models.sh"
  python "$ROOT_DIR/scripts/check_model_access.py" --models-root "$ROOT_DIR/models"
fi

if [[ "$BUILD_BASELINE_DATA" == "1" ]]; then
  bash "$ROOT_DIR/src/data_preparation.sh"
fi

if [[ "$BUILD_V2_DATA" == "1" ]]; then
  python -m src.data.build_span_json_dataset --output-dir "$ROOT_DIR/data/v2"
  python -m src.data.build_hard_examples \
    --input-file "$ROOT_DIR/data/v2/train.jsonl" \
    --output-file "$ROOT_DIR/data/v2/train_hard_augmented.jsonl"
fi

printf '\nBootstrap complete.\n'
printf 'Authentication notes: %s\n' "$ROOT_DIR/scripts/login_notes.md"
