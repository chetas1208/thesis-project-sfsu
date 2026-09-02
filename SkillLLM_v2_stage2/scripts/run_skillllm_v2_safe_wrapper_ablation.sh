#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi

"$PYTHON_BIN" -m skillllm_v2.eval.run_safe_wrapper "$@"
"$PYTHON_BIN" -m skillllm_v2.eval.money_table_generator
cp reports/skillllm_v2/money_table.csv reports/skillllm_v2/research_refresh/safe_wrapper_ablation.csv
cp reports/skillllm_v2/money_table.md reports/skillllm_v2/research_refresh/safe_wrapper_ablation.md
