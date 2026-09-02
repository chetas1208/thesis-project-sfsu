#!/usr/bin/env bash
# generate_skillllm_v2_dev_predictions.sh
#
# Inference-only pass with the existing 80-epoch Raw Mistral LoRA adapter
# (artifacts/skillllm_v2/mistral_lora/model) over the SkillSpan dev split
# (914 docs with gold spans), producing
# artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl +
# artifacts/skillllm_v2/mistral_lora/metrics_dev.json.
#
# This is the dev-set prediction cache consumed by:
#   - scripts/tune_skillllm_v2_boundary_refiner.sh
#   - scripts/tune_skillllm_v2_candidate_merger.sh
#   - scripts/tune_skillllm_v2_confidence_calibrator.sh
#
# SAFETY: does NOT retrain Mistral. Loads the trained adapter only; if it is
# missing, skillllm_v2.eval.generate_split_predictions fails loudly rather
# than silently falling back.
#
# Usage:
#   bash scripts/generate_skillllm_v2_dev_predictions.sh [--limit N]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_DISABLED=true

PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
fi

CFG="${CFG:-configs/skillllm_v2_training_80ep_2gpu.yaml}"

if [ ! -f "artifacts/skillllm_v2/mistral_lora/model/adapter_config.json" ]; then
    echo "FATAL: artifacts/skillllm_v2/mistral_lora/model/adapter_config.json missing."
    echo "       The 80-epoch Mistral LoRA adapter must already be trained."
    exit 1
fi

echo "============================================================"
echo "SkillLLM_v2_stage2: dev-set Raw Mistral LoRA predictions (inference only)"
echo "Config  : ${CFG}"
echo "Started : $(date)"
echo "============================================================"

"${PYTHON_BIN}" -m skillllm_v2.eval.generate_split_predictions \
    --config "${CFG}" \
    --split dev \
    "$@"

echo "============================================================"
echo "Done: $(date)"
echo "Predictions: artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl"
echo "Metrics    : artifacts/skillllm_v2/mistral_lora/metrics_dev.json"
echo "============================================================"
