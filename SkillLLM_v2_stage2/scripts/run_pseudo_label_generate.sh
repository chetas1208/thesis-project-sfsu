#!/usr/bin/env bash
# Experiment 2 step 1: pseudo-label generation with anchor/v1/v2 adapters. GPU1 only; resumable (re-run the same command).
# Launch: setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu1 -- bash "<abs>/scripts/run_pseudo_label_generate.sh" > logs/pseudo_label_generate.log 2>&1 &
set -euo pipefail
REPO_ROOT="/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"
cd "${REPO_ROOT}"
PYTHON_BIN="/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" scripts/pseudo_label_generate.py
echo "end $(date)"
