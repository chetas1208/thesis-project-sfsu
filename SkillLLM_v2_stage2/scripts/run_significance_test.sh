#!/usr/bin/env bash
# run_significance_test.sh
#
# Paired document-bootstrap + sign-flip permutation significance tests for the
# extraction F1 deltas between the Raw Mistral LoRA anchor and the three safe
# wrapper variants (974 SkillSpan test docs). CPU only, ~10 s.
# Asserts that pooled-count F1s reproduce evaluate_extraction / metrics.json
# to 1e-6 before testing; aborts otherwise.
#
# Writes: reports/skillllm_v2/significance_f1.{json,md}
#
# Usage:
#   /home/923873155/server-guardrails/bin/guardrails run cpu -- bash scripts/run_significance_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi

"$PYTHON_BIN" -m skillllm_v2.eval.significance \
  --n-resamples "${N_RESAMPLES:-10000}" \
  --seed "${SEED:-20260710}" \
  --output-json reports/skillllm_v2/significance_f1.json \
  --output-md reports/skillllm_v2/significance_f1.md "$@"
