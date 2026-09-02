#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" src/train/train_token_classifier.py \
  --train-file data/silver/train.jsonl \
  --eval-file data/gold/dev.jsonl \
  --test-file data/gold/test.jsonl \
  --model-name bert-base-cased \
  --output-dir outputs/silver_train_gold_eval_bert_base_cased
