#!/bin/bash
set -e
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "=== Phase 4: Inference (Resuming from Skill-LLM) ==="
source venv/bin/activate
python scripts/22_predict_skillllm.py
python scripts/23_predict_context.py
python scripts/24_predict_jobbert.py
python scripts/25_predict_gemma4.py

echo "=== Phase 5: Evaluation ==="
python scripts/30_evaluate_metrics.py
python scripts/31_evaluate_document_metrics.py
python scripts/32_build_comparison_table.py

echo "=== BENCHMARK RESUME COMPLETE ==="
