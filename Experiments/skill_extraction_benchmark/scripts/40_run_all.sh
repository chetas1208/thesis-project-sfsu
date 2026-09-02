#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Master orchestration script for the 5-model skill extraction benchmark
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "============================================================"
echo " Starting Skill Extraction Benchmark"
echo "============================================================"

# Ensure working directory is project root
cd "$(dirname "$0")/.."

: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this benchmark}"
export HF_TOKEN
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo ""
echo "=== Phase 2: Data Pipeline ==="
# Download and export
python scripts/00_download_skillspan.py
python scripts/01_prepare_canonical.py
python scripts/02_export_nnose.py
python scripts/03_export_gliner.py
python scripts/04_export_skillllm.py
python scripts/05_export_context.py
python scripts/06_export_jobbert.py
python scripts/07_export_gemma4.py

echo ""
echo "=== Phase 3: Training ==="
# Train all models sequentially
python scripts/10_train_nnose.py
python scripts/11_train_gliner.py
python scripts/12_train_skillllm.py
python scripts/13_train_context.py
python scripts/14_train_jobbert.py
python scripts/15_train_gemma4.py

echo ""
echo "=== Phase 4: Inference ==="
# Run prediction
python scripts/20_predict_nnose.py
python scripts/21_predict_gliner.py
python scripts/22_predict_skillllm.py
python scripts/23_predict_context.py
python scripts/24_predict_jobbert.py
python scripts/25_predict_gemma4.py

echo ""
echo "=== Phase 5: Evaluation ==="
# Compute metrics and generate reports
python scripts/30_evaluate_span_metrics.py
python scripts/31_evaluate_document_metrics.py
python scripts/32_build_comparison_table.py

echo ""
echo "============================================================"
echo " Benchmark Complete! See outputs/reports/comparison.md"
echo "============================================================"
