#!/usr/bin/env bash
# Experiment 1: QLoRA loop with the short prompt (v3). GPU1 only; resumes automatically from loop_state.json.
# Launch: setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu1 -- bash "<abs path>/scripts/run_loop_v3_shortprompt.sh" > logs/qlora_loop_v3_launch.log 2>&1 &
set -euo pipefail
REPO_ROOT="/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"
cd "${REPO_ROOT}"
PYTHON_BIN="/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export WANDB_DISABLED=true
export OMP_NUM_THREADS=4
echo "start $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -m skillllm_v2.train.loop_trainer --config configs/skillllm_v2_loop_qlora_v3_shortprompt.yaml "$@"
echo "end $(date)"
