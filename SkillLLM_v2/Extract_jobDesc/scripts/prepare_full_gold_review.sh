#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p data/gold_review

echo "Preparing full-corpus gold review sheet..."
"$PYTHON_BIN" src/validate/prepare_gold.py \
  --input-file data/silver/skillspan_style_silver_bio.jsonl \
  --json-output data/gold_review/full_review_candidates.jsonl \
  --csv-output data/gold_review/full_review_sheet.csv \
  --mode full
