#!/usr/bin/env bash
# Run the layered research evaluation (L01-L12) in dependency order, then the leaderboard and the evaluation card.
#
#   scripts/evals/run_research_evals.sh                  # full CPU run (GPU outputs are consumed if their DONE.json exists)
#   scripts/evals/run_research_evals.sh --smoke          # tiny subsets -> <layer>/smoke/, never quoted as results
#   scripts/evals/run_research_evals.sh --only L05,L10   # selected layers (+ leaderboard + card unless --no-summary)
#   scripts/evals/run_research_evals.sh --from L08       # resume after a failure
#   scripts/evals/run_research_evals.sh --strict         # a missing layer module is an error instead of a skip
#   scripts/evals/run_research_evals.sh --list           # show the plan and which modules exist
# Options: --seed N (default 42), --n-boot N (default: each module's own default), --no-guardrails, --no-summary.
# Per-module extra arguments: environment variable ARGS_<module>, e.g. ARGS_robustness="--cpu-only".
# Every module runs through the server-guardrails CPU launcher (heavy-job admission control). GPU jobs (anchor re-inference,
# fair baselines, robustness generation) are NOT launched here: they live in scripts/evals/gpu_*.sh and write DONE.json.
# Exit: 0 all requested steps passed (skips listed), 1 a step failed (message names the layer and log), 2 bad usage.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/opt/miniconda/bin/python}"
GUARDRAILS="${GUARDRAILS:-/home/923873155/server-guardrails/bin/guardrails}"
RESEARCH="$ROOT/src/skillllm_v2/evals/research"

# layer : module   (dependency order: producers first, L11 after L03/L06 samples, L10 after all per_unit_scores)
PLAN=(
  "L01:extraction_full"
  "L02:generalization"
  "L03:robustness"
  "L04:calibration_selective"
  "L05:wrapper_contribution"
  "L06:linking_eval"
  "L07:compaction"
  "L08:graph_retrieval"
  "L09:course_rec"
  "L12:systems"
  "L11:human_eval_samples"
  "L10:significance"
)

SMOKE=0; SEED=42; NBOOT=""; ONLY=""; FROM=""; STRICT=0; USE_GR=1; SUMMARY=1; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --seed) SEED="${2:?--seed needs a value}"; shift ;;
    --n-boot) NBOOT="${2:?--n-boot needs a value}"; shift ;;
    --only) ONLY="${2:?--only needs e.g. L01,L05}"; shift ;;
    --from) FROM="${2:?--from needs e.g. L05}"; shift ;;
    --strict) STRICT=1 ;;
    --no-guardrails) USE_GR=0 ;;
    --no-summary) SUMMARY=0 ;;
    --list) LIST=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "run_research_evals: unknown option '$1' (see --help)" >&2; exit 2 ;;
  esac
  shift
done

cd "$ROOT" || exit 2
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="$ROOT/logs/evals_research/$STAMP$([ $SMOKE = 1 ] && echo _smoke)"
mkdir -p "$LOGDIR"
SUMMARY_FILE="$LOGDIR/SUMMARY.txt"

launch() {  # launch <logfile> <python args...>
  local log="$1"; shift
  local cmd=(/usr/bin/env "PYTHONPATH=$ROOT/src" "$PYTHON" "$@")
  if [ $USE_GR = 1 ] && [ -x "$GUARDRAILS" ]; then
    "$GUARDRAILS" run cpu -- "${cmd[@]}" >"$log" 2>&1
  else
    [ $USE_GR = 1 ] && echo "warning: guardrails launcher not found at $GUARDRAILS; running unguarded" >&2
    "${cmd[@]}" >"$log" 2>&1
  fi
}

selected() {  # selected <layer>
  local layer="$1"
  if [ -n "$ONLY" ]; then
    case ",$ONLY," in *",$layer,"*) return 0 ;; *) return 1 ;; esac
  fi
  return 0
}

echo "research evals: root=$ROOT smoke=$SMOKE seed=$SEED n_boot=${NBOOT:-module default} logs=$LOGDIR" | tee "$SUMMARY_FILE"
started=0; [ -z "$FROM" ] && started=1
for entry in "${PLAN[@]}"; do
  layer="${entry%%:*}"; module="${entry#*:}"
  [ "$layer" = "$FROM" ] && started=1
  [ $started = 1 ] || { echo "skip   $layer ($module): before --from $FROM" | tee -a "$SUMMARY_FILE"; continue; }
  selected "$layer" || continue
  if [ ! -f "$RESEARCH/$module.py" ]; then
    msg="$layer ($module): module src/skillllm_v2/evals/research/$module.py not present"
    if [ $STRICT = 1 ]; then
      echo "FAILED $msg (--strict)" | tee -a "$SUMMARY_FILE"; exit 1
    fi
    echo "skip   $msg - layer will show as 'missing' in the leaderboard" | tee -a "$SUMMARY_FILE"
    continue
  fi
  [ $LIST = 1 ] && { echo "plan   $layer -> python -m skillllm_v2.evals.research.$module" | tee -a "$SUMMARY_FILE"; continue; }
  args=(-m "skillllm_v2.evals.research.$module" --seed "$SEED")
  [ -n "$NBOOT" ] && args+=(--n-boot "$NBOOT")
  [ $SMOKE = 1 ] && args+=(--smoke)
  extra_var="ARGS_$module"; extra="${!extra_var:-}"
  # shellcheck disable=SC2206
  [ -n "$extra" ] && args+=($extra)
  log="$LOGDIR/${layer}_${module}.log"
  t0=$(date +%s)
  echo "run    $layer: python ${args[*]}" | tee -a "$SUMMARY_FILE"
  launch "$log" "${args[@]}"
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then
    {
      echo "FAILED $layer ($module) exit=$rc after ${dt}s"
      echo "       log: $log"
      echo "       last lines:"; tail -n 15 "$log" | sed 's/^/       | /'
      echo "       fix the cause, then resume with: scripts/evals/run_research_evals.sh --from $layer$([ $SMOKE = 1 ] && echo ' --smoke')"
    } | tee -a "$SUMMARY_FILE" >&2
    exit 1
  fi
  echo "ok     $layer (${dt}s)" | tee -a "$SUMMARY_FILE"
done
[ $LIST = 1 ] && exit 0

if [ $SUMMARY = 1 ]; then
  lb_args=(-m skillllm_v2.evals.research.leaderboard)
  [ $SMOKE = 1 ] && lb_args+=(--smoke) || lb_args+=(--chapter)
  echo "run    leaderboard: python ${lb_args[*]}" | tee -a "$SUMMARY_FILE"
  launch "$LOGDIR/leaderboard.log" "${lb_args[@]}" || { echo "FAILED leaderboard - log: $LOGDIR/leaderboard.log" | tee -a "$SUMMARY_FILE" >&2; tail -n 15 "$LOGDIR/leaderboard.log" >&2; exit 1; }
  tail -n 1 "$LOGDIR/leaderboard.log" | tee -a "$SUMMARY_FILE"
  card_args=(-m skillllm_v2.evals.research.repro_card)
  [ $SMOKE = 1 ] && card_args+=(--smoke)
  echo "run    repro card: python ${card_args[*]}" | tee -a "$SUMMARY_FILE"
  launch "$LOGDIR/repro_card.log" "${card_args[@]}" || { echo "FAILED repro card - log: $LOGDIR/repro_card.log" | tee -a "$SUMMARY_FILE" >&2; tail -n 15 "$LOGDIR/repro_card.log" >&2; exit 1; }
  tail -n 1 "$LOGDIR/repro_card.log" | tee -a "$SUMMARY_FILE"
fi
echo "done   all requested steps passed; summary $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
exit 0
