#!/usr/bin/env bash
# tune_skillllm_v2_candidate_merger.sh
#
# Dev-set grid search over the "add-only" policy in
# skillllm_v2.extract.candidate_merger.CandidateMerger (see
# src/skillllm_v2/train/tune_candidate_merger.py). The merger never removes a
# candidate whose metadata.sources includes "mistral"; this tuning only
# decides how permissive the merger is about ADDING brand-new
# GLiNER/dictionary-only spans on top of the cached Raw Mistral LoRA dev
# predictions. Falls back to a no-op (mistral-only passthrough) config if
# nothing beats that baseline on Exact Span F1 within the Soft Skill F1
# guardrail.
#
# Requires artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl, produced
# by scripts/generate_skillllm_v2_dev_predictions.sh.
#
# CPU-only (runs GLiNER + dictionary extractors to build a dev candidate
# cache); no Mistral inference or retraining.
#
# Writes:
#   artifacts/skillllm_v2/candidate_merger/dev_candidates_cache.jsonl
#   artifacts/skillllm_v2/candidate_merger/tuned_config.yaml
#   artifacts/skillllm_v2/candidate_merger/dev_tuning_results.csv
#   reports/skillllm_v2/candidate_merger_tuning.md
#
# Usage:
#   bash scripts/tune_skillllm_v2_candidate_merger.sh

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
echo "SkillLLM_v2_stage2: candidate merger dev-set tuning"
echo "Started : $(date)"
echo "============================================================"

"${PYTHON_BIN}" -m skillllm_v2.train.tune_candidate_merger "$@"

echo "============================================================"
echo "Done: $(date)"
echo "Tuned config: artifacts/skillllm_v2/candidate_merger/tuned_config.yaml"
echo "Report      : reports/skillllm_v2/candidate_merger_tuning.md"
echo "============================================================"
