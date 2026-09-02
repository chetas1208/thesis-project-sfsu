#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Rebuilding splits and CoNLL exports from silver BIO data..."
"$PYTHON_BIN" src/split/split.py --input-file data/silver/skillspan_style_silver_bio.jsonl --out-dir data/silver
