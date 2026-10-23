#!/usr/bin/env bash
# Tiny GPU smoke (8 test docs) of the logprob-recording eval path with the v2 best adapter (original prompt); output to a scratch dir.
set -euo pipefail
REPO_ROOT="/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src" TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 WANDB_DISABLED=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QLORA_ADAPTER="artifacts/skillllm_v2/qlora_loop_v2/best" QLORA_EVAL_OUT="artifacts/skillllm_v2/logprob_smoke" QLORA_PROMPT_VERSION=original QLORA_LOGPROBS=1 LIMIT=8
echo "start $(date)"
"/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python" scripts/eval_adapter_test.py
echo "end $(date)"
