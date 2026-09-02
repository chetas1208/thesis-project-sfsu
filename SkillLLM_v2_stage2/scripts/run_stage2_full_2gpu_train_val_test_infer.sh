#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python venv executable: $PYTHON_BIN" >&2
  exit 1
fi
TEST_PYTHON_BIN="${TEST_PYTHON_BIN:-/opt/miniconda/bin/python}"
if [ ! -x "$TEST_PYTHON_BIN" ]; then
  TEST_PYTHON_BIN="$PYTHON_BIN"
fi

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export OMP_NUM_THREADS=8

RUN_ID="${RUN_ID:-stage2_full_train_val_test_infer_2gpu_$(date +%Y%m%d_%H%M%S)}"

echo "============================================================"
echo "SkillLLM_v2 full train/val/test/inference 2x3090 run"
echo "Run ID: ${RUN_ID}"
echo "Repo: $(pwd)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Python: ${PYTHON_BIN}"
echo "Test Python: ${TEST_PYTHON_BIN}"
echo "============================================================"

echo "[0/8] PyTorch CUDA enforcement: exactly 2x RTX 3090"
"$PYTHON_BIN" -c 'import torch; print("torch.cuda.is_available:", torch.cuda.is_available(), flush=True); print("torch.cuda.device_count:", torch.cuda.device_count(), flush=True); print("cuda_devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)'

"$PYTHON_BIN" -m skillllm_v2.pipeline.gpu_preflight \
  --config configs/skillllm_v2_training_80ep_2gpu.yaml \
  --require-two-gpus

echo "[1/8] Unit tests before smoke"
if [ "${SKIP_BACKGROUND_PYTEST:-0}" = "1" ]; then
  echo "Skipping background pytest because it was already run in foreground."
else
  "$TEST_PYTHON_BIN" -m pytest tests/skillllm_v2/ -q -s
fi

echo "[2/8] Smoke gates"
"$PYTHON_BIN" -m skillllm_v2.pipeline.pipeline_runner \
  --config configs/skillllm_v2_smoke.yaml \
  --mode smoke \
  --force

echo "[3/8] Full gated training/evaluation pipeline"
"$PYTHON_BIN" -m skillllm_v2.pipeline.pipeline_runner \
  --config configs/skillllm_v2_training_80ep_2gpu.yaml \
  --mode full \
  --force

echo "[4/8] Validation/test report generation"
"$PYTHON_BIN" -m skillllm_v2.pipeline.pipeline_runner \
  --stage final_report_generation \
  --config configs/skillllm_v2_training_80ep_2gpu.yaml \
  --mode full \
  --force

echo "[5/8] Inference: safe wrapper over cached/produced Mistral predictions"
"$PYTHON_BIN" -m skillllm_v2.eval.run_safe_wrapper
"$PYTHON_BIN" -m skillllm_v2.eval.money_table_generator

echo "[6/8] Research refresh report without retraining"
"$PYTHON_BIN" -m skillllm_v2.research_refresh --skip-safe-wrapper

echo "[7/8] Final tests"
if [ "${SKIP_BACKGROUND_PYTEST:-0}" = "1" ]; then
  echo "Skipping background pytest because it was already run in foreground."
else
  "$TEST_PYTHON_BIN" -m pytest tests/skillllm_v2/ -q -s
fi

echo "[8/8] Final artifact check"
find artifacts/skillllm_v2/status -maxdepth 1 -name "*.failed.json" -print
FAILED_COUNT=$(find artifacts/skillllm_v2/status -maxdepth 1 -name "*.failed.json" | wc -l)
echo "Failed markers: ${FAILED_COUNT}"
if [ "${FAILED_COUNT}" -ne 0 ]; then
  exit 1
fi

ls -lh reports/skillllm_v2 | sed -n "1,120p"
echo "Completed: $(date)"
