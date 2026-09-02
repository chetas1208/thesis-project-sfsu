#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p data/gold

echo "Building gold dataset from corrected review sheet..."
"$PYTHON_BIN" src/validate/build_gold.py --review-csv data/gold_review/review_sheet.csv --silver-input data/silver/skillspan_style_silver_bio.jsonl --output-dir data/gold

echo "Validating gold dataset..."
"$PYTHON_BIN" src/validate/validate.py --input-file data/gold/reviewed.jsonl --report-file reports/gold_validation_report.json
