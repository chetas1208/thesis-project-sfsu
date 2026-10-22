#!/usr/bin/env bash
# run_skillllm_v2_safe_wrapper.sh
#
# Runs skillllm_v2.eval.run_safe_wrapper over the full SkillSpan test split
# (974 docs), gating each wrapper variant (boundary_refiner,
# mistral_gliner_taxonomy, full_cag_graph) against the cached 80-epoch Raw
# Mistral LoRA anchor predictions
# (artifacts/skillllm_v2/mistral_lora/predictions.jsonl) via
# skillllm_v2.eval.non_regression_gate and
# configs/skillllm_v2_wrapper_guardrails.yaml.
#
# SAFETY: does NOT run Mistral inference. Only CPU components (GLiNER,
# dictionary extractor, CandidateMerger, BoundaryRefiner, SpanVerifier,
# ConfidenceCalibrator, TaxonomyLinker) run on top of the cached anchor.
#
# Requires artifacts/skillllm_v2/mistral_lora/predictions.jsonl (974 rows),
# produced by scripts/eval_raw_mistral_lora.sh / scripts/run_raw_mistral_eval_2gpu.sh.
#
# Writes:
#   artifacts/skillllm_v2/safe_wrapper/<variant>/{predictions,candidate_predictions,metrics}.json
#   artifacts/skillllm_v2/safe_wrapper/<variant>/linked_candidates.jsonl (taxonomy variants)
#   artifacts/skillllm_v2/safe_wrapper/gate_results.json
#   reports/skillllm_v2/safe_wrapper_gate.md
#   reports/skillllm_v2/safe_wrapper_summary.json
#
# Usage:
#   bash scripts/run_skillllm_v2_safe_wrapper.sh [--limit N]

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

ANCHOR_PREDICTIONS="artifacts/skillllm_v2/mistral_lora/predictions.jsonl"
if [ ! -f "${ANCHOR_PREDICTIONS}" ]; then
    echo "FATAL: ${ANCHOR_PREDICTIONS} missing."
    echo "       Run scripts/eval_raw_mistral_lora.sh (or run_raw_mistral_eval_2gpu.sh) first."
    exit 1
fi

echo "============================================================"
echo "SkillLLM_v2_stage2: safe wrapper (non-regression-gated extraction)"
echo "Anchor  : ${ANCHOR_PREDICTIONS}"
echo "Started : $(date)"
echo "============================================================"

"${PYTHON_BIN}" -m skillllm_v2.eval.run_safe_wrapper "$@"

echo "============================================================"
echo "Done: $(date)"
echo "Gate report: reports/skillllm_v2/safe_wrapper_gate.md"
echo "Summary    : reports/skillllm_v2/safe_wrapper_summary.json"
echo "============================================================"
