#!/usr/bin/env bash
# tune_skillllm_v2_confidence_calibrator.sh
#
# Dev-set grid search over skillllm_v2.verify.confidence_calibrator.ConfidenceCalibrator
# thresholds (see src/skillllm_v2/train/tune_confidence_calibrator.py). The
# calibrator never rejects a candidate whose metadata.sources intersects
# protect_anchor_sources (default {"mistral"}); this tuning only controls how
# aggressively brand-new GLiNER/dictionary-only additions are filtered after
# CandidateMerger and BoundaryRefiner. Falls back to a no-op (ALWAYS_ACCEPT
# for all four thresholds) config if nothing beats that baseline on Exact
# Span F1 within the Soft Skill F1 guardrail.
#
# Run AFTER scripts/tune_skillllm_v2_boundary_refiner.sh and
# scripts/tune_skillllm_v2_candidate_merger.sh, since the dev pipeline
# replayed here is:
#   merge (tuned merger config, if present)
#   -> refine (tuned boundary config, if present)
#   -> verify (source-aware, not tuned)
#   -> calibrate (grid-searched here)
#
# Requires artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl, produced
# by scripts/generate_skillllm_v2_dev_predictions.sh. Reuses
# artifacts/skillllm_v2/candidate_merger/dev_candidates_cache.jsonl if
# present (rebuilds it otherwise).
#
# CPU-only; no Mistral inference or retraining.
#
# Writes:
#   artifacts/skillllm_v2/confidence_calibrator/tuned_config.yaml
#   artifacts/skillllm_v2/confidence_calibrator/dev_tuning_results.csv
#   reports/skillllm_v2/confidence_calibrator_tuning.md
#
# Usage:
#   bash scripts/tune_skillllm_v2_confidence_calibrator.sh

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
echo "SkillLLM_v2_stage2: confidence calibrator dev-set tuning"
echo "Started : $(date)"
echo "============================================================"

"${PYTHON_BIN}" -m skillllm_v2.train.tune_confidence_calibrator "$@"

echo "============================================================"
echo "Done: $(date)"
echo "Tuned config: artifacts/skillllm_v2/confidence_calibrator/tuned_config.yaml"
echo "Report      : reports/skillllm_v2/confidence_calibrator_tuning.md"
echo "============================================================"
