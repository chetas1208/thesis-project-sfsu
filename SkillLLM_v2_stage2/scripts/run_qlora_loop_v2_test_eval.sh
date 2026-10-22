#!/usr/bin/env bash
# Resumable test-set eval of the QLoRA-loop v2 best adapter. GPU1 only.
# Launch: /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu1 -- bash "<abs path>/scripts/run_qlora_loop_v2_test_eval.sh"
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
export QLORA_ADAPTER="artifacts/skillllm_v2/qlora_loop_v2/best"
export QLORA_EVAL_OUT="artifacts/skillllm_v2/qlora_loop_v2_eval"
export QLORA_REPORT_METRICS="reports/skillllm_v2/qlora_loop_v2_test_metrics.json"
export QLORA_RUN_LABEL="v2, r=16, best round per loop_summary.json"
echo "start $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" scripts/eval_qlora_loop_test.py
echo "end $(date)"
