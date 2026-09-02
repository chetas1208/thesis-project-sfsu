#!/bin/bash
set -e
export PYTHONPATH="$(pwd):$PYTHONPATH"
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this benchmark}"
export HF_TOKEN

echo "=== Resuming Benchmark from Phase 3: Training Skill-LLM ==="
source venv/bin/activate

echo "Running 12_train_skillllm.py..."
python scripts/12_train_skillllm.py

echo "Running 13_train_context.py..."
python scripts/13_train_context.py

echo "Running 15_train_gemma4.py..."
python scripts/15_train_gemma4.py

echo "=== Phase 4: Inference ==="
python scripts/20_predict_nnose.py
python scripts/21_predict_gliner.py
python scripts/22_predict_skillllm.py
python scripts/23_predict_context.py
python scripts/24_predict_jobbert.py
python scripts/25_predict_gemma4.py

echo "=== Phase 5: Evaluation ==="
python scripts/30_evaluate_metrics.py
python scripts/31_evaluate_document_metrics.py
python scripts/32_build_comparison_table.py

echo "=== BENCHMARK RESUME COMPLETE ==="
