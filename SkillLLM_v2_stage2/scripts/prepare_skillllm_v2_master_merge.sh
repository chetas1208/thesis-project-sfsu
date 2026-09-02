#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if find configs scripts src/skillllm_v2 -type f \( -iname '*qwen*' -o -iname '*gemma*' -o -iname '*llama*' -o -iname '*retry*' -o -iname '*skillspan_benchmark*' \) | grep -q .; then
  echo "Active legacy model benchmark files remain:" >&2
  find configs scripts src/skillllm_v2 -type f \( -iname '*qwen*' -o -iname '*gemma*' -o -iname '*llama*' -o -iname '*retry*' -o -iname '*skillspan_benchmark*' \) >&2
  exit 1
fi

test -f docs/SKILLLM_V2_STAGE2_PLAN.md
test -f docs/SKILLLM_V2_ARCHITECTURE.md
test -f docs/legacy_benchmark_summary.md
test -f docs/removed_artifacts_manifest.md
test -f reports/skillllm_v2/money_table.csv
test -f reports/skillllm_v2/money_table.md
test -f artifacts/skillllm_v2/bert_baseline/metrics.json
test -f artifacts/skillllm_v2/gliner_baseline/metrics.json
test -f artifacts/skillllm_v2/mistral_lora/metrics.json
test -f artifacts/skillllm_v2/mistral_boundary/metrics.json
test -f artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json
test -f artifacts/skillllm_v2/full_cag_graph/metrics.json

PYTHONPATH=src pytest tests/skillllm_v2

echo "Merge readiness checks passed."
