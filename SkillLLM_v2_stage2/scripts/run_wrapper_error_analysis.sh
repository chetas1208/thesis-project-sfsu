#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi

"$PYTHON_BIN" -m skillllm_v2.eval.wrapper_error_analysis \
  --output-dir reports/skillllm_v2/research_refresh "$@"
