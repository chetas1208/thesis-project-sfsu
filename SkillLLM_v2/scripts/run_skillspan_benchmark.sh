#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/skillspan_baselines.yaml}"
DATASET_PATH="${2:-$ROOT_DIR/data/skillspan}"
OUTPUT_DIR="${3:-/usr/data/models/skillllm-v2/results/skillspan_baselines}"
LOG_DIR="$ROOT_DIR/logs"
MAIN_LOG="$LOG_DIR/skillspan_benchmark.log"
MODELS=(
  "gemma-4-26b-a4b-it"
  "gemma-4-e4b-it"
  "llama-3.1-8b-instruct"
  "mistral-small-3.1-24b-instruct-2503"
  "qwen3-14b"
)

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Critical setup error: missing config at $CONFIG_PATH" | tee -a "$MAIN_LOG"
  exit 1
fi

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
  echo "Critical setup error: missing virtual environment at $ROOT_DIR/.venv" | tee -a "$MAIN_LOG"
  exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"
cd "$ROOT_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOCAL_PYTHON_HEADERS="$ROOT_DIR/.cache/apt_python_headers/extracted/usr/include"
if [[ -d "$LOCAL_PYTHON_HEADERS" ]]; then
  export C_INCLUDE_PATH="$LOCAL_PYTHON_HEADERS:$LOCAL_PYTHON_HEADERS/python3.12:$LOCAL_PYTHON_HEADERS/x86_64-linux-gnu/python3.12:${C_INCLUDE_PATH:-}"
fi

python - <<'PY' >> "$MAIN_LOG" 2>&1
import sys
try:
    import tokenizers
    import transformers
except Exception as exc:
    raise SystemExit(f"Critical setup error: {exc}")

def parse(version):
    parts = []
    for item in version.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts)

if parse(transformers.__version__) < (4, 51, 0):
    raise SystemExit(f"Critical setup error: transformers {transformers.__version__} is too old.")
if parse(tokenizers.__version__) < (0, 21, 0):
    raise SystemExit(f"Critical setup error: tokenizers {tokenizers.__version__} is too old.")
PY
if [[ $? -ne 0 ]]; then
  exit 1
fi

echo "Starting SkillSpan benchmark at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$MAIN_LOG"

for model in "${MODELS[@]}"; do
  echo "Running model: $model" | tee -a "$MAIN_LOG"
  if python scripts/run_skillspan_benchmark.py \
    --config "$CONFIG_PATH" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --model "$model" >> "$MAIN_LOG" 2>&1; then
    echo "Completed model: $model" | tee -a "$MAIN_LOG"
  else
    echo "Model failed and benchmark will continue: $model" | tee -a "$MAIN_LOG"
  fi
done

echo "Final leaderboard: $OUTPUT_DIR/leaderboard.csv" | tee -a "$MAIN_LOG"
