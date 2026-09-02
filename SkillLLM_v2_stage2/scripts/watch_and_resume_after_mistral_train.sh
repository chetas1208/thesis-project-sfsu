#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs artifacts/skillllm_v2/status reports/skillllm_v2

RUN_ID="${RUN_ID:-stage2_downstream_after_train_2gpu_$(date +%Y%m%d_%H%M%S)}"
TRAIN_PID="${TRAIN_PID:-}"
if [ -z "${TRAIN_PID}" ] && [ -f "logs/latest_stage2_full_2gpu.pid" ]; then
  TRAIN_PID="$(cat logs/latest_stage2_full_2gpu.pid)"
fi

if [ -z "${TRAIN_PID}" ]; then
  echo "FATAL: TRAIN_PID is not set and logs/latest_stage2_full_2gpu.pid is missing."
  exit 1
fi

export PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_DISABLED=true

LOG_FILE="logs/${RUN_ID}.log"
PID_FILE="logs/${RUN_ID}.pid"
echo "$$" > "${PID_FILE}"
ln -sfn "${RUN_ID}.log" logs/latest_stage2_downstream_after_train.log
ln -sfn "${RUN_ID}.pid" logs/latest_stage2_downstream_after_train.pid

exec > "${LOG_FILE}" 2>&1

echo "============================================================"
echo "SkillLLM_v2 downstream watcher"
echo "Run ID: ${RUN_ID}"
echo "Repo: ${REPO_ROOT}"
echo "Watching training PID: ${TRAIN_PID}"
echo "Started: $(date)"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "FATAL: PYTHON_BIN is not executable: ${PYTHON_BIN}"
  exit 1
fi

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  echo "$(date): training PID ${TRAIN_PID} still running; waiting 120s"
  sleep 120
done

echo "$(date): training PID ${TRAIN_PID} finished or disappeared."
echo "Starting validation/test/inference/report resume."
nvidia-smi || true

bash scripts/resume_skillllm_v2_eval_2gpu.sh
"${PYTHON_BIN}" -m skillllm_v2.eval.run_safe_wrapper
"${PYTHON_BIN}" -m skillllm_v2.eval.money_table_generator
"${PYTHON_BIN}" -m skillllm_v2.research_refresh --skip-safe-wrapper

echo "============================================================"
echo "Downstream watcher completed: $(date)"
echo "============================================================"
