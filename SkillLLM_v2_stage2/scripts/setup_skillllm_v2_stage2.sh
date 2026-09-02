#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/skillllm_v2 data/skillllm_v2 reports/skillllm_v2 logs

test -d src/skillllm_v2
test -f configs/skillllm_v2_stage2.yaml
test -f configs/skillllm_v2_smoke.yaml
test -f configs/skillllm_v2_training.yaml
test -f data/source/skillspan_hf/train.json
test -f data/source/skillspan_hf/dev.json
test -f data/source/skillspan_hf/test.json

if find configs scripts src/skillllm_v2 -type f \( -iname '*qwen*' -o -iname '*gemma*' -o -iname '*llama*' -o -iname '*skillspan_benchmark*' \) | grep -q .; then
  echo "Old active multi-model configs/scripts are still present." >&2
  find configs scripts src/skillllm_v2 -type f \( -iname '*qwen*' -o -iname '*gemma*' -o -iname '*llama*' -o -iname '*skillspan_benchmark*' \) >&2
  exit 1
fi

PYTHONPATH=src python - <<'PY'
import importlib
for name in ["skillllm_v2", "yaml"]:
    importlib.import_module(name)
print("Python imports OK")
PY

ADAPTER_PATH="${SKILLLM_V2_MISTRAL_ADAPTER:-artifacts/mistral-small-3.1-24b-skillspan-lora}"
if [[ ! -e "$ADAPTER_PATH" ]]; then
  echo "Warning: Mistral adapter not found at $ADAPTER_PATH. Smoke mode will use deterministic fallback extraction." >&2
fi

echo "Stage-2 setup validation passed."
