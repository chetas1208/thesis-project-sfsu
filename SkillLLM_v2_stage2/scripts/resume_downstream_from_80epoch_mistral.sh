#!/usr/bin/env bash
# resume_downstream_from_80epoch_mistral.sh
#
# Re-runs every pipeline stage from eval_mistral_boundary_refiner onward using
# the 80-epoch Mistral LoRA predictions already in:
#   artifacts/skillllm_v2/mistral_lora/predictions.jsonl  (Jun 3, 80-epoch)
#
# SAFETY: does NOT retrain Mistral.  The trained adapter and cached predictions
# are treated as the source of truth.
#
# Flags used:
#   --invalidate-downstream eval_mistral_boundary_refiner
#       Deletes stale .success.json markers from that stage onward so the
#       pipeline won't skip them.
#   --start-at eval_mistral_boundary_refiner
#       Begins execution at that stage (skips training stages).
#   --force
#       Forces re-run even if a stale success marker happens to remain.
#   --mode full
#       Full dataset evaluation (974 docs with gold spans).
#
# Produces (updated/regenerated):
#   artifacts/skillllm_v2/mistral_boundary/{predictions,metrics}.json
#   artifacts/skillllm_v2/mistral_gliner_taxonomy/{predictions,metrics}.json
#   artifacts/skillllm_v2/full_cag_graph/{predictions,metrics}.json
#   reports/skillllm_v2/money_table.{csv,md}
#   reports/skillllm_v2/thesis_results_summary.md

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export WANDB_DISABLED=true
export PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
fi

CFG="configs/skillllm_v2_training_80ep_2gpu.yaml"

echo "========================================================"
echo "SkillLLM_v2: downstream resume from 80-epoch Mistral"
echo "Config  : ${CFG}"
echo "Started : $(date)"
echo "PYTHON  : ${PYTHON_BIN}"
echo "========================================================"

# 0. Confirm the 80-epoch predictions anchor exists and is recent.
echo ""
echo "[0] Checking upstream anchor..."
if [ ! -f "artifacts/skillllm_v2/mistral_lora/predictions.jsonl" ]; then
    echo "ERROR: artifacts/skillllm_v2/mistral_lora/predictions.jsonl is missing."
    echo "       Cannot proceed without 80-epoch Mistral predictions."
    exit 1
fi
ls -lh artifacts/skillllm_v2/mistral_lora/predictions.jsonl
wc -l   artifacts/skillllm_v2/mistral_lora/predictions.jsonl

# 1. Run prediction provenance check (informational — non-fatal at this stage).
echo ""
echo "[1] Prediction provenance check..."
"${PYTHON_BIN}" -m skillllm_v2.eval.prediction_provenance_check --fix-hint || true

# 2. Smoke-test individual components (fast, no GPU needed for most).
echo ""
echo "[2] Component smoke tests..."
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
    --config configs/skillllm_v2_smoke.yaml \
    --mode smoke \
    --start-at eval_mistral_boundary_refiner \
    --stop-after taxonomy_linking_smoke \
    --force

# 3. Invalidate stale markers and run full downstream pipeline.
echo ""
echo "[3] Invalidating stale markers and running full downstream pipeline..."
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
    --config "${CFG}" \
    --mode full \
    --invalidate-downstream eval_mistral_boundary_refiner \
    --start-at eval_mistral_boundary_refiner \
    --force

# 4. Metric sanity check.
echo ""
echo "[4] Metric sanity check..."
"${PYTHON_BIN}" -m skillllm_v2.eval.metric_sanity --strict

# 5. Provenance re-check (should now pass).
echo ""
echo "[5] Final provenance check..."
"${PYTHON_BIN}" -m skillllm_v2.eval.prediction_provenance_check

# 6. Print the money table.
echo ""
echo "[6] Money table:"
cat reports/skillllm_v2/money_table.md 2>/dev/null || echo "(money_table.md not found)"

echo ""
echo "========================================================"
echo "Downstream resume complete: $(date)"
echo "========================================================"
