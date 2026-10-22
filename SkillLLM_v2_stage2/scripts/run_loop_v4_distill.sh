#!/usr/bin/env bash
# Experiment 2: distillation/self-training loop (v4), warm start from v2 best. GPU1 only; resumes automatically.
# Needs artifacts/skillllm_v2/qlora_loop_v4_distill/pseudo_pool/pseudo_train.jsonl (scripts/build_pseudo_sft.py).
# Launch: setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu1 -- bash "<abs path>/scripts/run_loop_v4_distill.sh" > logs/qlora_loop_v4_launch.log 2>&1 &
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
"${PYTHON_BIN}" -m skillllm_v2.train.loop_trainer --config configs/skillllm_v2_loop_qlora_v4_distill.yaml "$@"
echo "end $(date)"
