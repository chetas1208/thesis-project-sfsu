#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
TRAIN_CFG="${TRAIN_CFG:-configs/skillllm_v2_training_80ep_2gpu.yaml}"
RUN_ID="${RUN_ID:-raw_mistral_eval_2gpu_$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs artifacts/skillllm_v2/status reports/skillllm_v2

echo "============================================================"
echo "SkillLLM_v2 Raw Mistral LoRA full eval"
echo "Run ID: ${RUN_ID}"
echo "Repo: ${REPO_ROOT}"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Config: ${TRAIN_CFG}"
echo "============================================================"

"${PYTHON_BIN}" - <<'PY'
import time

import torch

from skillllm_v2.pipeline.artifact_tracker import ArtifactTracker, utc_now
from skillllm_v2.pipeline.smoke_runner import eval_raw_mistral_lora

config_path = "configs/skillllm_v2_training_80ep_2gpu.yaml"
stage = "eval_raw_mistral_lora"
system_name = "Raw Mistral LoRA"
outputs = ["artifacts/skillllm_v2/mistral_lora/metrics.json"]

print("torch.cuda.is_available:", torch.cuda.is_available(), flush=True)
print("torch.cuda.device_count:", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise SystemExit("Expected 2 visible CUDA GPUs for raw Mistral eval")
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)

tracker = ArtifactTracker("artifacts/skillllm_v2")
started_at = utc_now()
started_time = time.perf_counter()
try:
    metrics = eval_raw_mistral_lora(config_path, "full", limit=None)
    tracker.write_status(stage, system_name, "success", started_at, started_time, config_path, [], outputs, metrics)
    print("Raw Mistral full eval metrics:", metrics, flush=True)
except Exception as exc:
    tracker.write_status(stage, system_name, "failed", started_at, started_time, config_path, [], outputs, error=str(exc))
    raise
PY

echo "============================================================"
echo "Raw Mistral LoRA full eval completed: $(date)"
echo "Metrics: artifacts/skillllm_v2/mistral_lora/metrics.json"
echo "Predictions: artifacts/skillllm_v2/mistral_lora/predictions.jsonl"
echo "============================================================"
