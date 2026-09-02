#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python -m skillllm_v2.train.build_mistral_sft_data "$@"
