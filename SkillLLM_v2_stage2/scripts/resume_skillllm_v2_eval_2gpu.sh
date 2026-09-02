#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
TRAIN_CFG="${TRAIN_CFG:-configs/skillllm_v2_training_80ep_2gpu.yaml}"
START_STAGE="${START_STAGE:-eval_raw_mistral_lora}"
RUN_ID="${RUN_ID:-stage2_resume_eval_2gpu_$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_DISABLED=true

mkdir -p logs artifacts/skillllm_v2/status reports/skillllm_v2

echo "============================================================"
echo "SkillLLM_v2 Stage-2 resume from ${START_STAGE}"
echo "Run ID: ${RUN_ID}"
echo "Repo: ${REPO_ROOT}"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Training stage is skipped; using existing adapter artifacts/skillllm_v2/mistral_lora/model"
echo "============================================================"

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("Expected 2 visible CUDA GPUs for resume eval")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
PY

"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
  --config "${TRAIN_CFG}" \
  --mode full \
  --stage "${START_STAGE}" \
  --force

"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
  --config "${TRAIN_CFG}" \
  --mode full \
  --start-at eval_mistral_boundary_refiner \
  --force

"${PYTHON_BIN}" - <<'PY'
from skillllm_v2.pipeline.smoke_runner import _write_artifact_index

_write_artifact_index()
PY

echo "============================================================"
echo "Resume run completed: $(date)"
echo "Metrics index: reports/skillllm_v2/artifact_index.md"
echo "Money table: reports/skillllm_v2/money_table.md"
echo "============================================================"
