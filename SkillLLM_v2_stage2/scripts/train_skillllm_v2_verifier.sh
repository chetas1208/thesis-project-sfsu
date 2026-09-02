#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python -m skillllm_v2.train.train_span_verifier \
  --data-dir data/skillllm_v2 \
  --output-dir artifacts/skillllm_v2/span_verifier
echo "Verifier artifact: artifacts/skillllm_v2/span_verifier/verifier_config.json"
