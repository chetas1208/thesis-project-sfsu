#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="/opt/miniconda/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=========================================================="
echo "SkillLLM_v2 Course Recommendation Pipeline (November 1 MVP)"
echo "=========================================================="
echo "Catalogue: inputs/courses/course_catalog.jsonl"
echo "Output: reports/skillllm_v2/demo/course_recommendations.json"

PYTHONPATH=src "$PYTHON_BIN" src/skillllm_v2/pipeline/course_recommendation_pipeline.py \
    --catalog inputs/courses/course_catalog.jsonl \
    --out reports/skillllm_v2/demo/course_recommendations.json \
    --top_k 5

echo "=========================================================="
echo "Pipeline execution completed successfully."
echo "=========================================================="
