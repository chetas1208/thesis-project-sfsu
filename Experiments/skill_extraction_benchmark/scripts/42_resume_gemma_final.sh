#!/bin/bash
set -e
export PYTHONPATH="$(pwd):$PYTHONPATH"
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this benchmark}"
export HF_TOKEN

echo "=== Resuming Benchmark from Phase 3: Training Gemma 4 ==="
source venv/bin/activate

echo "Running 15_train_gemma4.py..."
python -m torch.distributed.run --nproc_per_node 2 scripts/15_train_gemma4.py

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
