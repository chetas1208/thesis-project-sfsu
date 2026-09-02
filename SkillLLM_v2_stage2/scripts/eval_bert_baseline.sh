#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python -m skillllm_v2.pipeline.pipeline_runner --stage eval_bert_baseline --config configs/skillllm_v2_training.yaml --mode full "$@"
