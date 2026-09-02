#!/usr/bin/env bash
# nohup_resume_downstream_from_80epoch_mistral.sh
#
# Wraps resume_downstream_from_80epoch_mistral.sh in nohup with timestamped
# logging, PID tracking, and a final pass/fail summary.
#
# Usage (recommended):
#   nohup bash scripts/nohup_resume_downstream_from_80epoch_mistral.sh \
#     > logs/resume_downstream_bootstrap.log 2>&1 &
#
# Or let the script manage nohup itself (foreground call):
#   bash scripts/nohup_resume_downstream_from_80epoch_mistral.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

RUN_ID="resume_downstream_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${RUN_ID}.log"
PID_FILE="logs/${RUN_ID}.pid"

export PYTHON_BIN="${PYTHON_BIN:-/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python)"
fi

echo "========================================================"
echo "SkillLLM_v2 downstream resume (nohup wrapper)"
echo "Run ID  : ${RUN_ID}"
echo "Log     : ${LOG_FILE}"
echo "Repo    : ${ROOT}"
echo "Started : $(date)"
echo "========================================================"

nohup bash -lc '
set -euo pipefail
cd "'"${ROOT}"'"

export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export WANDB_DISABLED=true
export PYTHON_BIN="'"${PYTHON_BIN}"'"

CFG="configs/skillllm_v2_training_80ep_2gpu.yaml"
RUN_ID="'"${RUN_ID}"'"

echo "========================================================"
echo "SkillLLM_v2: downstream resume from 80-epoch Mistral"
echo "Run ID  : ${RUN_ID}"
echo "Config  : ${CFG}"
echo "Started : $(date)"
echo "PYTHON  : ${PYTHON_BIN}"
echo "CUDA    : ${CUDA_VISIBLE_DEVICES}"
echo "========================================================"

# ──────────────────────────────────────────────────────────────
# STEP 0: Sanity-check GPU availability
# ──────────────────────────────────────────────────────────────
echo ""
echo "[0] GPU check"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader || true
"${PYTHON_BIN}" - <<'"'"'PY'"'"'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  free={free/1e9:.1f}GB / total={total/1e9:.1f}GB")
PY

# ──────────────────────────────────────────────────────────────
# STEP 1: Confirm upstream anchor (80-epoch predictions)
# ──────────────────────────────────────────────────────────────
echo ""
echo "[1] Upstream anchor check"
if [ ! -f "artifacts/skillllm_v2/mistral_lora/predictions.jsonl" ]; then
    echo "FATAL: artifacts/skillllm_v2/mistral_lora/predictions.jsonl missing"
    exit 1
fi
ls -lh artifacts/skillllm_v2/mistral_lora/predictions.jsonl
PRED_COUNT=$(wc -l < artifacts/skillllm_v2/mistral_lora/predictions.jsonl)
echo "  prediction rows: ${PRED_COUNT}"
if [ "${PRED_COUNT}" -lt 900 ]; then
    echo "FATAL: Expected ~974 prediction rows, got ${PRED_COUNT}. File may be truncated."
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# STEP 2: Diagnose stale artifacts (before fix)
# ──────────────────────────────────────────────────────────────
echo ""
echo "[2] Stale artifact diagnosis (before)"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.diagnose_stale_artifacts || true

# ──────────────────────────────────────────────────────────────
# STEP 3: Smoke downstream components (fast, mostly CPU)
# ──────────────────────────────────────────────────────────────
echo ""
echo "[3] Smoke downstream components"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
    --config configs/skillllm_v2_smoke.yaml \
    --mode smoke \
    --start-at eval_mistral_boundary_refiner \
    --stop-after taxonomy_linking_smoke \
    --force

# ──────────────────────────────────────────────────────────────
# STEP 4: Full downstream pipeline (eval_mistral_boundary_refiner → end)
#         --invalidate-downstream removes stale markers first.
#         --force forces re-run even if any residual markers exist.
# ──────────────────────────────────────────────────────────────
echo ""
echo "[4] Full downstream pipeline"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
    --config "${CFG}" \
    --mode full \
    --invalidate-downstream eval_mistral_boundary_refiner \
    --start-at eval_mistral_boundary_refiner \
    --force

# ──────────────────────────────────────────────────────────────
# STEP 5: Re-generate money table explicitly (belt-and-suspenders)
# ──────────────────────────────────────────────────────────────
echo ""
echo "[5] Money table regeneration"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
    --config "${CFG}" \
    --stage money_table_generation \
    --mode full \
    --force

# ──────────────────────────────────────────────────────────────
# STEP 6: Final report regeneration
# ──────────────────────────────────────────────────────────────
echo ""
echo "[6] Final report regeneration"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.pipeline_runner \
    --config "${CFG}" \
    --stage final_report_generation \
    --mode full \
    --force

# ──────────────────────────────────────────────────────────────
# STEP 7: Metric sanity check
# ──────────────────────────────────────────────────────────────
echo ""
echo "[7] Metric sanity check"
"${PYTHON_BIN}" -m skillllm_v2.eval.metric_sanity --strict

# ──────────────────────────────────────────────────────────────
# STEP 8: Prediction provenance check
# ──────────────────────────────────────────────────────────────
echo ""
echo "[8] Prediction provenance check"
"${PYTHON_BIN}" -m skillllm_v2.eval.prediction_provenance_check

# ──────────────────────────────────────────────────────────────
# STEP 9: Stale artifact re-diagnosis (should be clean)
# ──────────────────────────────────────────────────────────────
echo ""
echo "[9] Stale artifact diagnosis (after)"
"${PYTHON_BIN}" -m skillllm_v2.pipeline.diagnose_stale_artifacts || true

# ──────────────────────────────────────────────────────────────
# STEP 10: Print final money table
# ──────────────────────────────────────────────────────────────
echo ""
echo "[10] Money table"
cat reports/skillllm_v2/money_table.md 2>/dev/null || echo "(missing)"
echo ""
echo "Thesis summary:"
cat reports/skillllm_v2/thesis_results_summary.md 2>/dev/null || echo "(missing)"

# ──────────────────────────────────────────────────────────────
# STEP 11: Failed marker check
# ──────────────────────────────────────────────────────────────
echo ""
echo "[11] Failed marker check"
FAILED_COUNT=$(find artifacts/skillllm_v2/status -maxdepth 1 -name "*.failed.json" 2>/dev/null | wc -l)
echo "  Failed markers: ${FAILED_COUNT}"
if [ "${FAILED_COUNT}" -ne 0 ]; then
    find artifacts/skillllm_v2/status -maxdepth 1 -name "*.failed.json" -print
    echo "WARNING: Some stages failed. Review logs."
fi

echo ""
echo "========================================================"
echo "Downstream resume completed: $(date)"
echo "Run ID: ${RUN_ID}"
echo "========================================================"
' > "${LOG_FILE}" 2>&1 &

NOHUP_PID=$!
echo "${NOHUP_PID}" > "${PID_FILE}"
ln -sfn "${RUN_ID}.log" logs/latest_resume_downstream.log
ln -sfn "${RUN_ID}.pid" logs/latest_resume_downstream.pid

echo "Launched PID ${NOHUP_PID}"
echo "Log : ${LOG_FILE}"
echo "Tail: tail -f ${LOG_FILE}"
