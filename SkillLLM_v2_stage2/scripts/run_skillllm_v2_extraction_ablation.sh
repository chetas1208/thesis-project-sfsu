#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LIMIT="${1:-20}"
PYTHONPATH=src python - "$LIMIT" <<'PY'
import sys
from skillllm_v2.eval.ablation_runner import run_ablation
rows = run_ablation("data/source/skillspan_hf", "reports/skillllm_v2", limit=int(sys.argv[1]))
for row in rows:
    print(row["variant"], "exact_f1=", round(row["exact_span_f1"], 4), "token_f1=", round(row["token_f1"], 4))
PY
echo "Ablation report: reports/skillllm_v2/extraction_ablation.md"
