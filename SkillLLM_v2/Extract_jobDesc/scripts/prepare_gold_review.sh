#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p data/gold_review

echo "Preparing review candidates..."
"$PYTHON_BIN" src/validate/prepare_gold.py --input-file data/silver/skillspan_style_silver_bio.jsonl --json-output data/gold_review/review_candidates.jsonl --csv-output data/gold_review/review_sheet.csv
