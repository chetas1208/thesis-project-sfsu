#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python -m skillllm_v2.train.build_verifier_dataset \
  --dataset-path data/source/skillspan_hf \
  --output-dir data/skillllm_v2 \
  "${@}"
echo "Verifier dataset: data/skillllm_v2/verifier_train.jsonl and data/skillllm_v2/verifier_dev.jsonl"
