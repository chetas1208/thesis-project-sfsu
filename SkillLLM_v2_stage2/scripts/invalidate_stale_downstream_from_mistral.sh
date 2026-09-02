#!/usr/bin/env bash
# invalidate_stale_downstream_from_mistral.sh
#
# Deletes success markers for every stage from eval_mistral_boundary_refiner
# onward.  Safe to run multiple times (idempotent).  Does NOT delete artifact
# data files — only the .success.json markers that gate pipeline re-runs.
#
# After this script, run:
#   bash scripts/resume_downstream_from_80epoch_mistral.sh
# or:
#   nohup bash scripts/nohup_resume_downstream_from_80epoch_mistral.sh ...

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STATUS_DIR="artifacts/skillllm_v2/status"

STAGES=(
    eval_mistral_boundary_refiner
    extraction_component_smoke
    merge_refine_verify_smoke
    taxonomy_linking_smoke
    eval_mistral_gliner_taxonomy
    build_verifier_dataset
    train_span_verifier
    verifier_eval_smoke
    build_linker_dataset
    train_taxonomy_reranker
    graph_smoke
    matching_recommendation_smoke
    eval_full_cag_graph
    extraction_ablation_full
    graph_matching_eval_full
    money_table_generation
    end_to_end_demo_smoke
    final_report_generation
    master_merge_readiness_check
)

echo "Invalidating downstream success markers (from eval_mistral_boundary_refiner onward)"
echo "Status dir: ${STATUS_DIR}"
echo ""

removed=0
for stage in "${STAGES[@]}"; do
    marker="${STATUS_DIR}/${stage}.success.json"
    if [ -f "$marker" ]; then
        rm "$marker"
        echo "  removed: ${marker}"
        removed=$((removed + 1))
    else
        echo "  absent:  ${marker}"
    fi
done

echo ""
echo "Done. Removed ${removed} marker(s)."
echo ""
echo "Verify upstream anchor:"
ls -lh artifacts/skillllm_v2/mistral_lora/predictions.jsonl 2>/dev/null \
    && echo "  (this file is the 80-epoch model output — downstream will use it)" \
    || echo "  WARNING: mistral_lora/predictions.jsonl missing!"
echo ""
echo "Next step:"
echo "  nohup bash scripts/nohup_resume_downstream_from_80epoch_mistral.sh \\"
echo "    > logs/resume_downstream_bootstrap.log 2>&1 &"
