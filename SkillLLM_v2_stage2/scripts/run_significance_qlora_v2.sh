#!/usr/bin/env bash
# Paired bootstrap + permutation: QLoRA-loop v2 vs anchor and vs v1 (974 test docs). CPU only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
"$PYTHON_BIN" scripts/significance_qlora_v2.py --n-resamples "${N_RESAMPLES:-10000}" --seed "${SEED:-20260710}" "$@"
