#!/usr/bin/env bash
# tune_skillllm_v2_boundary_refiner.sh
#
# Dev-set grid search over skillllm_v2.verify.boundary_selector.BoundarySelector
# configuration (see src/skillllm_v2/train/tune_boundary_refiner.py). Selects
# the configuration that maximizes dev Exact Span F1 subject to the Soft
# Skill F1 guardrail, falling back to a no-op config (every Raw Mistral span
# preserved unchanged) if nothing beats the no-op baseline.
#
# Requires artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl, produced
# by scripts/generate_skillllm_v2_dev_predictions.sh.
#
# CPU-only, no Mistral inference or retraining.
#
# Writes:
#   artifacts/skillllm_v2/boundary_refiner/tuned_config.yaml
#   artifacts/skillllm_v2/boundary_refiner/dev_tuning_results.csv
#   reports/skillllm_v2/boundary_refiner_tuning.md
#
# Usage:
#   bash scripts/tune_skillllm_v2_boundary_refiner.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
fi

DEV_PREDICTIONS="artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl"
if [ ! -f "${DEV_PREDICTIONS}" ]; then
    echo "FATAL: ${DEV_PREDICTIONS} missing."
    echo "       Run scripts/generate_skillllm_v2_dev_predictions.sh first."
    exit 1
fi

echo "============================================================"
echo "SkillLLM_v2_stage2: boundary refiner dev-set tuning"
echo "Started : $(date)"
echo "============================================================"

"${PYTHON_BIN}" -m skillllm_v2.train.tune_boundary_refiner "$@"

echo "============================================================"
echo "Done: $(date)"
echo "Tuned config: artifacts/skillllm_v2/boundary_refiner/tuned_config.yaml"
echo "Report      : reports/skillllm_v2/boundary_refiner_tuning.md"
echo "============================================================"
