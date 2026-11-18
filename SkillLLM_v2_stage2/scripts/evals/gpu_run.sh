#!/usr/bin/env bash
# Run one anchor inference stage on one physical GPU (A3 research eval).  Usage:
#   bash scripts/evals/gpu_run.sh <gpu_index> <args for scripts/evals/gpu_anchor_infer.py ...>
# Launch detached through the guardrails admission controller, e.g.
#   setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu0 -- \
#     bash "<ROOT>/scripts/evals/gpu_run.sh" 0 --job anchor_test_logprobs --split test --subset gold --logprobs \
#     > "<ROOT>/logs/evals_gpu/anchor_test_logprobs.log" 2>&1 &
set -euo pipefail
REPO_ROOT="/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"
cd "${REPO_ROOT}"
GPU="$1"; shift
PYTHON_BIN="/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python"
export PYTHONPATH="${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "start $(date -u +%FT%TZ) gpu=${GPU} args=$*"
"${PYTHON_BIN}" scripts/evals/gpu_anchor_infer.py "$@"
echo "end $(date -u +%FT%TZ)"
