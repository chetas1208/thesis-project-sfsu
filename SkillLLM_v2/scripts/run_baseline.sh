#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$ROOT_DIR/.venv/bin/activate"

python -m src.train.train_baseline_wrapper --config "$ROOT_DIR/configs/baseline_llama31_8b.yaml"

bash "$ROOT_DIR/scripts/run_eval_suite.sh" \
  "$ROOT_DIR/configs/baseline_llama31_8b.yaml" \
  "baseline_llama31_8b" \
  "$ROOT_DIR/outputs/baseline_llama31_8b/predictions/test_predictions.jsonl"
