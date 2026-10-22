#!/usr/bin/env bash
# End-to-end SkillLLM_v2 inference: profile (+ job) -> extraction -> gate -> profile/graph/match -> courses -> explanation.
#
# Usage:
#   scripts/run_skillllm_v2_inference.sh [--profile FILE] [--job FILE] [--out OUT.json] [--no-mistral] [pipeline flags...]
#
# Defaults run the demo inputs with the FLAGGED deterministic stub anchor (--no-mistral): CPU only, no GPU, no model
# weights, a few seconds. Pass --mistral to use the real Mistral anchor (anchor.mode=auto); that is heavy (24B, 4-bit),
# so the script then re-launches itself through the shared-server guardrails (cpu class unless GUARD_CLASS=gpuN).
#   (relative paths resolve from the repo root)
#   --guard   force the guardrails launcher even for the stub path
#   Other flags (--skip, --stages, --courses null|adapter, --target-job, --top-k, --config, --anchor) pass through.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="/opt/miniconda/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"
GUARDRAILS="/home/923873155/server-guardrails/bin/guardrails"

PROFILE="inputs/demo/user_profile.json"
JOB="inputs/demo/jobs.jsonl"
OUT="reports/skillllm_v2/demo/inference_result.json"
STUB=1
GUARD=0
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="$2"; shift 2 ;;
        --job) JOB="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --no-mistral) STUB=1; shift ;;
        --mistral) STUB=0; GUARD=1; shift ;;
        --guard) GUARD=1; shift ;;
        --no-guard-child) GUARD=0; shift ;;
        *) PASS+=("$1"); shift ;;
    esac
done

if [ "$GUARD" = "1" ] && [ -z "${SKILLLLM_INFERENCE_GUARDED:-}" ] && [ -x "$GUARDRAILS" ]; then
    ARGS=(--profile "$PROFILE" --job "$JOB" --out "$OUT")
    [ "$STUB" = "1" ] && ARGS+=(--no-mistral) || ARGS+=(--mistral)
    export SKILLLLM_INFERENCE_GUARDED=1
    exec "$GUARDRAILS" run "${GUARD_CLASS:-cpu}" -- bash "$ROOT/scripts/run_skillllm_v2_inference.sh" "${ARGS[@]}" ${PASS[@]+"${PASS[@]}"}
fi

mkdir -p "$(dirname "$OUT")"
FLAGS=()
[ "$STUB" = "1" ] && FLAGS+=(--no-mistral)
PYTHONPATH=src "$PYTHON_BIN" -m skillllm_v2.pipeline.inference \
    --profile "$PROFILE" --job "$JOB" --out "$OUT" ${FLAGS[@]+"${FLAGS[@]}"} ${PASS[@]+"${PASS[@]}"}
echo "Inference result: $OUT"
