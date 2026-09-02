#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$1"
EXPERIMENT_NAME="$2"
PREDICTION_FILE="$3"
RAW_FILE="${4:-}"

source "$ROOT_DIR/.venv/bin/activate"

python -m src.eval.evaluate_exact_span \
  --gold-file "$ROOT_DIR/data/v2/test.jsonl" \
  --prediction-file "$PREDICTION_FILE" \
  --output-file "$ROOT_DIR/reports/metrics/${EXPERIMENT_NAME}_exact.json" \
  --experiment-name "$EXPERIMENT_NAME"

python -m src.eval.evaluate_relaxed_span \
  --gold-file "$ROOT_DIR/data/v2/test.jsonl" \
  --prediction-file "$PREDICTION_FILE" \
  --output-file "$ROOT_DIR/reports/metrics/${EXPERIMENT_NAME}_relaxed.json" \
  --experiment-name "$EXPERIMENT_NAME"

python -m src.eval.evaluate_doc_set \
  --gold-file "$ROOT_DIR/data/v2/test.jsonl" \
  --prediction-file "$PREDICTION_FILE" \
  --output-file "$ROOT_DIR/reports/metrics/${EXPERIMENT_NAME}_doc_set.json" \
  --experiment-name "$EXPERIMENT_NAME"

python -m src.eval.classification_report \
  --gold-file "$ROOT_DIR/data/v2/test.jsonl" \
  --prediction-file "$PREDICTION_FILE" \
  --output-file "$ROOT_DIR/reports/metrics/${EXPERIMENT_NAME}_classification.json" \
  --csv-file "$ROOT_DIR/reports/tables/${EXPERIMENT_NAME}_classification.csv" \
  --experiment-name "$EXPERIMENT_NAME"

python -m src.eval.error_analysis \
  --gold-file "$ROOT_DIR/data/v2/test.jsonl" \
  --prediction-file "$PREDICTION_FILE" \
  --output-file "$ROOT_DIR/reports/errors/${EXPERIMENT_NAME}_errors.csv"

if [[ -n "$RAW_FILE" && -f "$RAW_FILE" ]]; then
  python -m src.eval.latency_benchmark \
    --raw-file "$RAW_FILE" \
    --output-file "$ROOT_DIR/reports/metrics/${EXPERIMENT_NAME}_latency.json" \
    --experiment-name "$EXPERIMENT_NAME"
fi

python -m src.eval.make_latex_tables \
  --metrics-dir "$ROOT_DIR/reports/metrics" \
  --output-dir "$ROOT_DIR/reports/tables" \
  --report-dir "$ROOT_DIR/reports"
