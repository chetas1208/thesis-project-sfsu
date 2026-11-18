#!/usr/bin/env bash
# A3 per-GPU job queue for the research evaluation. Every stage is resumable and is skipped when its DONE.json exists,
# so the whole queue can simply be relaunched after an interruption. Stages that depend on inputs prepared on CPU wait for
# a READY marker (bounded wait) and are skipped (logged) if it never appears.
#   GPU0: J1 anchor_test_logprobs -> J3 anchor_test_nospan shard0 -> J5 bert_fair (jobbert) -> J4 robustness shard0 -> J6 xdata shard0
#   GPU1: J2 anchor_dev_logprobs  -> J3 anchor_test_nospan shard1 -> J5 bert_fair (bert-base-cased) -> J4 robustness shard1 -> J6 xdata shard1
# Launch (one queue per GPU, each holds that GPU's guardrails slot):
#   setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu0 -- bash "<ROOT>/scripts/evals/gpu_queue.sh" 0 > "<ROOT>/logs/evals_gpu/queue_gpu0.log" 2>&1 &
#   setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 16 gpu1 -- bash "<ROOT>/scripts/evals/gpu_queue.sh" 1 > "<ROOT>/logs/evals_gpu/queue_gpu1.log" 2>&1 &
set -uo pipefail
REPO_ROOT="/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"
cd "${REPO_ROOT}"
GPU="$1"
OUT="artifacts/skillllm_v2/evals_gpu"
L03="inputs/evals/research/L03_robustness"
PYTHON_BIN="/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python"
ts() { date -u +%FT%TZ; }
stage() {  # stage <name> <done-file> <cmd...>
  local name="$1" done="$2"; shift 2
  if [ -f "${done}" ]; then echo "$(ts) skip ${name} (${done} exists)"; return 0; fi
  echo "$(ts) BEGIN ${name}"
  "$@" > "logs/evals_gpu/${name}.gpu${GPU}.log" 2>&1
  local rc=$?
  echo "$(ts) END ${name} rc=${rc}"
}
wait_for() {  # wait_for <file> <max_minutes>
  local f="$1" max="$2" waited=0
  while [ ! -f "${f}" ] && [ "${waited}" -lt "${max}" ]; do sleep 60; waited=$((waited+1)); done
  [ -f "${f}" ]
}
infer() { bash scripts/evals/gpu_run.sh "${GPU}" "$@"; }

if [ "${GPU}" = "0" ]; then
  stage J1_anchor_test_logprobs "${OUT}/anchor_test_logprobs/DONE.json" infer --job anchor_test_logprobs --split test --subset gold --logprobs
  SHARD=0; BERT_MODEL="jjzha/jobbert-base-cased"
else
  stage J2_anchor_dev_logprobs "${OUT}/anchor_dev_logprobs/DONE.json" infer --job anchor_dev_logprobs --split dev --subset gold --logprobs
  SHARD=1; BERT_MODEL="bert-base-cased"
fi
stage "J3_anchor_test_nospan_shard${SHARD}" "${OUT}/anchor_test_nospan/shards/shard${SHARD}of2/DONE.json" \
  infer --job anchor_test_nospan --split test --subset nogold --logprobs --shard "${SHARD}" --num-shards 2

if wait_for "${OUT}/bert_fair/READY_TO_TRAIN" 240; then
  stage "J5_bert_fair_${SHARD}" "${OUT}/bert_fair/DONE_${BERT_MODEL//\//__}.json" \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${REPO_ROOT}/src" TOKENIZERS_PARALLELISM=false \
        HF_HOME=/usr/data/models/skillllm-v2-stage2/hf_cache_a3 \
        "${PYTHON_BIN}" scripts/evals/gpu_bert_fair.py --model "${BERT_MODEL}" --seeds 42 43 44
else
  echo "$(ts) J5 not ready after wait; skipped"
fi

if wait_for "${L03}/READY" 300; then
  stage "J4_robustness_anchor_shard${SHARD}" "${OUT}/robustness_anchor/shards/shard${SHARD}of2/DONE.json" \
    infer --job robustness_anchor --input "${L03}/perturbed_docs.jsonl" --logprobs --shard "${SHARD}" --num-shards 2
else
  echo "$(ts) J4 inputs not ready after wait; skipped"
fi

if wait_for "${L03}/READY_XDATA" 240; then
  stage "J6_xdata_anchor_shard${SHARD}" "${OUT}/xdata_anchor/shards/shard${SHARD}of2/DONE.json" \
    infer --job xdata_anchor --input "${L03}/xdata_docs.jsonl" --logprobs --shard "${SHARD}" --num-shards 2
else
  echo "$(ts) J6 inputs not ready after wait; skipped"
fi
echo "$(ts) queue gpu${GPU} finished"
