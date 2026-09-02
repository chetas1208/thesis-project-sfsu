#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs artifacts/skillllm_v2/status reports/skillllm_v2
RUN_ID="stage2_80epoch_2gpu_$(date +%Y%m%d_%H%M%S)"

nohup bash -lc '
set -euo pipefail

cd "'"$ROOT"'"

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export WANDB_DISABLED=true
export PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python)"
fi
export TEST_PYTHON_BIN="${TEST_PYTHON_BIN:-$(command -v python)}"

RUN_ID="'"$RUN_ID"'"
TRAIN_CFG="configs/skillllm_v2_training_80ep_2gpu.yaml"
BASE_CFG="configs/skillllm_v2_training.yaml"

echo "============================================================"
echo "SkillLLM_v2 Stage-2 gated 80-epoch 2-GPU run"
echo "Run ID: ${RUN_ID}"
echo "Repo: $(pwd)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "TEST_PYTHON_BIN=${TEST_PYTHON_BIN}"
echo "============================================================"

echo "[0/9] GPU visibility check"
nvidia-smi || true

"${PYTHON_BIN}" - <<'"'"'PY'"'"'
import torch

print("torch.cuda.is_available:", torch.cuda.is_available())
print("torch.cuda.device_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Stop before training.")

if torch.cuda.device_count() < 2:
    raise SystemExit("Expected 2 visible GPUs, but PyTorch sees fewer than 2.")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY

echo "[1/9] Creating 80-epoch 2-GPU config: ${TRAIN_CFG}"

"${PYTHON_BIN}" - <<'"'"'PY'"'"'
from pathlib import Path
import yaml

base = Path("configs/skillllm_v2_training.yaml")
out = Path("configs/skillllm_v2_training_80ep_2gpu.yaml")

if not base.exists():
    raise FileNotFoundError(f"Missing base config: {base}")

with base.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def ensure_dict(obj, key):
    if key not in obj or not isinstance(obj[key], dict):
        obj[key] = {}
    return obj[key]

def set_epoch_keys(d):
    d["epochs"] = 80
    d["num_epochs"] = 80
    d["num_train_epochs"] = 80
    d["max_epochs"] = 80

def set_gpu_keys(d):
    d["cuda_visible_devices"] = "0,1"
    d["num_gpus"] = 2
    d["world_size"] = 2
    d["distributed"] = True
    d["distributed_backend"] = "ddp"
    d["use_ddp"] = True
    d["use_accelerate"] = True
    d["accelerate_num_processes"] = 2
    d["device_map"] = "auto"
    d["torch_dtype"] = "float16"
    d["mixed_precision"] = "fp16"
    d["gradient_checkpointing"] = True

cfg["run_id"] = "stage2_80epoch_2gpu"
cfg["mode"] = "full"

hardware = ensure_dict(cfg, "hardware")
set_gpu_keys(hardware)

distributed = ensure_dict(cfg, "distributed")
distributed["launcher"] = "torchrun"
distributed["num_processes"] = 2
distributed["cuda_visible_devices"] = "0,1"

device = ensure_dict(cfg, "device")
device["prefer_gpu"] = True
device["cuda_visible_devices"] = "0,1"

training = ensure_dict(cfg, "training")
set_epoch_keys(training)
set_gpu_keys(training)
training["full_run"] = True
training["force_training"] = True
training["train_mistral_lora"] = True

model = ensure_dict(cfg, "model")
model["train_mistral_lora"] = True
model["allow_smoke_fallback"] = False

for section in [
    "bert",
    "bert_baseline",
    "gliner",
    "gliner_baseline",
    "mistral",
    "mistral_lora",
    "span_verifier",
    "verifier",
    "taxonomy_reranker",
    "linker",
    "reranker",
]:
    sec = ensure_dict(cfg, section)
    set_epoch_keys(sec)
    set_gpu_keys(sec)

ensure_dict(cfg, "mistral_lora")["train_mistral_lora"] = True
ensure_dict(cfg, "mistral")["train_mistral_lora"] = True

# Fit Mistral-Small-3.1-24B within 2x RTX 3090 (24 GiB each).
# max_seq_length 512 saves ~4x activation memory vs 2048.
# lora_r 16 keeps trainable params small; compensate with more grad accum steps.
for _k in ("mistral_lora", "mistral"):
    _s = ensure_dict(cfg, _k)
    _s["max_seq_length"] = 512
    _s["lora_r"] = 16
    _s["lora_alpha"] = 32
    _s["batch_size"] = 1
    _s["eval_batch_size"] = 1
    _s["gradient_accumulation_steps"] = 32

stages = ensure_dict(cfg, "stages")
for stage in [
    "train_bert_baseline",
    "train_or_run_gliner_baseline",
    "train_or_verify_mistral_lora",
    "train_span_verifier",
    "train_taxonomy_reranker",
]:
    st = ensure_dict(stages, stage)
    set_epoch_keys(st)
    set_gpu_keys(st)
stages["train_or_verify_mistral_lora"]["train_mistral_lora"] = True

with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote {out}")
PY

echo "[2/9] GPU component availability preflight"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.gpu_preflight \
  --config "${TRAIN_CFG}" \
  --require-two-gpus

echo "[3/9] Running pytest before smoke gates"
"${TEST_PYTHON_BIN}" -m pytest tests/skillllm_v2/ -x

echo "[4/9] Running all component smoke gates"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
  --config configs/skillllm_v2_smoke.yaml \
  --mode smoke

echo "[5/9] Running full gated 80-epoch 2-GPU training/evaluation pipeline"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
  --config "${TRAIN_CFG}" \
  --mode full \
  --force

echo "[6/9] Generating final money table"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
  --stage money_table_generation \
  --config "${TRAIN_CFG}" \
  --mode full \
  --force

echo "[7/9] Running merge readiness check"
PYTHONPATH=src "${TEST_PYTHON_BIN}" -m pytest tests/skillllm_v2/ -x

echo "[8/9] Final artifact check"
ls -lh reports/skillllm_v2/ || true

echo "[9/9] Failed status marker check"
FAILED_COUNT=$(find artifacts/skillllm_v2/status -maxdepth 1 -name "*.failed.json" | wc -l)
echo "Failed markers: ${FAILED_COUNT}"
find artifacts/skillllm_v2/status -maxdepth 1 -name "*.failed.json" -print

if [ "${FAILED_COUNT}" -ne 0 ]; then
  echo "One or more pipeline stages failed. Check artifacts/skillllm_v2/status/*.failed.json"
  exit 1
fi

echo "============================================================"
echo "SkillLLM_v2 Stage-2 80-epoch 2-GPU run completed"
echo "Finished: $(date)"
echo "============================================================"
' > "logs/${RUN_ID}.log" 2>&1 &

echo $! | tee "logs/${RUN_ID}.pid"
ln -sfn "${RUN_ID}.pid" logs/latest_stage2_80epoch_2gpu.pid
ln -sfn "${RUN_ID}.log" logs/latest_stage2_80epoch_2gpu.log
echo "Log: logs/${RUN_ID}.log"
