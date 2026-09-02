#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python -m skillllm_v2.pipeline.pipeline_runner --stage train_taxonomy_reranker --config configs/skillllm_v2_training.yaml --mode full "$@"
