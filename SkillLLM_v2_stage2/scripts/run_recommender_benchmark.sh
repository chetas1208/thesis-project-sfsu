#!/usr/bin/env bash
# Rebuild the synthetic large catalogue, the independent relevance labels and the recommender benchmark report.
# Heavy-ish (loads the full pipeline + MiniLM); run through the server guardrails on shared machines.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="/opt/miniconda/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"
export PYTHONPATH=src

"$PYTHON_BIN" scripts/build_course_catalog_large.py            # inputs/courses/course_catalog_large.jsonl (SYNTHETIC)
"$PYTHON_BIN" -m skillllm_v2.eval.build_relevance_labels       # benchmark_scenarios.jsonl, relevance_labels.jsonl, human_review_sample.csv
"$PYTHON_BIN" -m skillllm_v2.eval.recommender_benchmark        # reports/skillllm_v2/course_recommendation_benchmark.{md,json}, course_recommendation_eval.csv
