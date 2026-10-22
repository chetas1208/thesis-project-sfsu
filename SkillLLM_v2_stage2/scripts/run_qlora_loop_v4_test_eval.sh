#!/usr/bin/env bash
# Resumable held-out test eval of the QLoRA-loop v4 (distilled) best adapter. GPU1 only. Run ONCE after the loop has finished.
# Launch: setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu1 -- bash "<abs path>/scripts/run_qlora_loop_v4_test_eval.sh" > logs/qlora_loop_v4_test_eval.log 2>&1 &
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
export QLORA_ADAPTER="artifacts/skillllm_v2/qlora_loop_v4_distill/best"
export QLORA_EVAL_OUT="artifacts/skillllm_v2/qlora_loop_v4_eval"
export QLORA_REPORT_METRICS="reports/skillllm_v2/qlora_loop_v4_test_metrics.json"
export QLORA_RUN_LABEL="v4 distilled, r=16, dev-selected best round per loop_summary.json"
export QLORA_PROMPT_VERSION="original"
export QLORA_LOGPROBS="1"
echo "start $(date) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" scripts/eval_adapter_test.py
echo "end $(date)"
