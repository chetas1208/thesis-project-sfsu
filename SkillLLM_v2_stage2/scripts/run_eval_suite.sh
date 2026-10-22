#!/usr/bin/env bash
# Run the SkillLLM_v2 evaluation suite on CPU under the server guardrails launcher.
#   scripts/run_eval_suite.sh                 # full run (creates the baseline on the first full run)
#   scripts/run_eval_suite.sh --smoke         # tiny subsets, writes to reports/skillllm_v2/eval_suite/smoke/
#   scripts/run_eval_suite.sh --only match,gap --seed 7
#   scripts/run_eval_suite.sh --update-baseline   # accept the current numbers as the new baseline (full run only)
# Extra arguments are passed to `python -m skillllm_v2.evals.run_all`.  Exit: 0 clean, 1 control failure / regression,
# 2 an eval module crashed, 3 bad usage.  Set GUARDRAILS=/path/to/launcher and PYTHON=/path/to/python to override.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/miniconda/bin/python}"
GUARDRAILS="${GUARDRAILS:-/home/923873155/server-guardrails/bin/guardrails}"
cd "$ROOT" || exit 3
CMD=(/usr/bin/env "PYTHONPATH=$ROOT/src" "$PYTHON" -m skillllm_v2.evals.run_all --root "$ROOT" "$@")
if [ -x "$GUARDRAILS" ]; then
  exec "$GUARDRAILS" run cpu -- "${CMD[@]}"
else
  echo "warning: guardrails launcher not found at $GUARDRAILS; running unguarded (CPU only, small)" >&2
  exec "${CMD[@]}"
fi
