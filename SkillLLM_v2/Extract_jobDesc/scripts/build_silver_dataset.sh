#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p data/processed data/silver reports

echo "Running ingest phase..."
"$PYTHON_BIN" src/ingest/ingest.py --input-root data --output-file data/processed/jobs_ingested.jsonl

echo "Running clean phase..."
"$PYTHON_BIN" src/clean/clean.py --input-file data/processed/jobs_ingested.jsonl --output-file data/processed/jobs_cleaned.jsonl

echo "Running deduplication phase..."
"$PYTHON_BIN" src/dedup/dedup.py --input-file data/processed/jobs_cleaned.jsonl --output-file data/processed/jobs_deduped.jsonl --report-file reports/dedup_report.json

echo "Running segmentation phase..."
"$PYTHON_BIN" src/segment/segment.py --input-file data/processed/jobs_deduped.jsonl --output-file data/processed/job_sentences.jsonl

echo "Running weak labeling phase..."
"$PYTHON_BIN" src/weak_label/weak_label.py --input-file data/processed/job_sentences.jsonl --output-file data/silver/skillspan_style_silver.jsonl

echo "Running BIO conversion phase..."
"$PYTHON_BIN" src/convert/convert_to_bio.py --input-file data/silver/skillspan_style_silver.jsonl --output-file data/silver/skillspan_style_silver_bio.jsonl

echo "Creating silver train/dev/test splits..."
"$PYTHON_BIN" src/split/split.py --input-file data/silver/skillspan_style_silver_bio.jsonl --out-dir data/silver
