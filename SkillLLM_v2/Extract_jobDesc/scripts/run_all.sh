#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Starting skill-span dataset pipeline..."

bash scripts/build_silver_dataset.sh
bash scripts/prepare_gold_review.sh

echo "Validating silver dataset..."
"$PYTHON_BIN" src/validate/validate.py --input-file data/silver/skillspan_style_silver_bio.jsonl --report-file reports/validation_report.json

echo "Generating silver dataset statistics..."
"$PYTHON_BIN" src/stats/stats.py --input-file data/silver/skillspan_style_silver_bio.jsonl --json-out reports/dataset_stats.json --csv-out reports/dataset_stats.csv

echo "If data/gold_review/review_sheet.csv has corrected spans, build gold with:"
echo "  bash scripts/build_gold_dataset.sh"

echo "Pipeline completed."
