#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$ROOT_DIR/.venv/bin/activate"

python -m src.train.train_sft_v2 --config "$ROOT_DIR/configs/qwen3_14b_v2.yaml"
python -m src.infer.batch_infer --config "$ROOT_DIR/configs/qwen3_14b_v2.yaml" --splits val test

bash "$ROOT_DIR/scripts/run_eval_suite.sh" \
  "$ROOT_DIR/configs/qwen3_14b_v2.yaml" \
  "qwen3_14b_v2" \
  "$ROOT_DIR/outputs/qwen3_14b_v2/predictions/test_predictions.jsonl" \
  "$ROOT_DIR/outputs/qwen3_14b_v2/predictions/raw_test.jsonl"
