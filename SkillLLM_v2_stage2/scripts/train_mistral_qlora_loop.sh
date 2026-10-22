#!/usr/bin/env bash
# Low-compute QLoRA loop training for Mistral-Small-3.1-24B, GPU1 only, through server-guardrails.
#   bash scripts/train_mistral_qlora_loop.sh                # real run (resumes automatically)
#   bash scripts/train_mistral_qlora_loop.sh --plan         # print round plan, no GPU
#   bash scripts/train_mistral_qlora_loop.sh --smoke        # 1 tiny round (20 docs)
#   bash scripts/train_mistral_qlora_loop.sh --set budget.max_gpu_hours=2 --set loop.max_rounds=4
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GUARDRAILS_BIN="${GUARDRAILS_BIN:-/home/923873155/server-guardrails/bin/guardrails}"
GUARDRAILS_PROFILE="${GUARDRAILS_PROFILE:-/home/923873155/server-guardrails/configs/profiles/gpu-workstation.toml}"
PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
CONFIG="${LOOP_CONFIG:-configs/skillllm_v2_loop_qlora.yaml}"
GPU_INDEX="${LOOP_GPU:-1}"
JOB_RAM_GB="${LOOP_RAM_GB:-16}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="$(command -v python)"
fi
mkdir -p logs

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_DISABLED=true
export OMP_NUM_THREADS=4

# The guardrails launcher may change the working directory, hence the explicit cd above.
# --plan only reads data: no GPU, no guardrails slot needed.
for arg in "$@"; do
  if [ "$arg" = "--plan" ]; then
    exec "${PYTHON_BIN}" -m skillllm_v2.train.loop_trainer --config "${CONFIG}" "$@"
  fi
done

# Only the chosen card is visible; never touch the other user's GPU0 process.
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

LOG="logs/qlora_loop_$(date +%Y%m%d_%H%M%S).log"
echo "[loop] gpu=${GPU_INDEX} config=${CONFIG} log=${LOG}"
"${GUARDRAILS_BIN}" -c "${GUARDRAILS_PROFILE}" status
"${GUARDRAILS_BIN}" -c "${GUARDRAILS_PROFILE}" run --ram "${JOB_RAM_GB}" "gpu${GPU_INDEX}" -- \
  bash -c 'cd "$0" && exec "$@"' "${ROOT}" \
  "${PYTHON_BIN}" -m skillllm_v2.train.loop_trainer --config "${CONFIG}" "$@" 2>&1 | tee -a "${LOG}"
