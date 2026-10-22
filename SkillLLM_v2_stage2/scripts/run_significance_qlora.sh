#!/usr/bin/env bash
# Paired bootstrap + sign-flip permutation: QLoRA-loop best adapter vs Raw Mistral LoRA anchor (974 test docs). CPU only.
# Writes reports/skillllm_v2/significance_qlora_vs_anchor.{json,md}; existing significance_f1.* untouched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
"$PYTHON_BIN" scripts/significance_qlora.py \
  --n-resamples "${N_RESAMPLES:-10000}" --seed "${SEED:-20260710}" \
  --output-json reports/skillllm_v2/significance_qlora_vs_anchor.json \
  --output-md reports/skillllm_v2/significance_qlora_vs_anchor.md "$@"
