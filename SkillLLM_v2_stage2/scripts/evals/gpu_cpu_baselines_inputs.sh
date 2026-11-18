#!/usr/bin/env bash
# A3: CPU baselines on the L03 inputs (perturbed docs + cross-dataset docs). Systems needing GPU outputs first
# (bert_fair -> J5 SELECTED.json; safe_wrapper -> J4/J6 anchor predictions) are skipped until those exist; rerunning
# this script later fills them in (finished systems are skipped via their DONE.json).
# Launch: setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 6 cpu -- bash "<ROOT>/scripts/evals/gpu_cpu_baselines_inputs.sh" > "<ROOT>/logs/evals_gpu/a3_cpu_baselines_inputs.log" 2>&1 &
set -uo pipefail
REPO_ROOT="/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"
cd "${REPO_ROOT}"
PY="/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="" HF_HOME=/usr/data/models/skillllm-v2-stage2/hf_cache_a3 TOKENIZERS_PARALLELISM=false PYTHONPATH="${REPO_ROOT}/src"
L03="inputs/evals/research/L03_robustness"
OUT="artifacts/skillllm_v2/evals_gpu"
SYS="token_lookup gliner_zeroshot"
[ -f "${OUT}/bert_fair/SELECTED.json" ] && SYS="${SYS} bert_fair"
"${PY}" scripts/evals/gpu_baselines_infer.py --job robustness --input "${L03}/perturbed_docs.jsonl" --systems ${SYS} --device cpu
"${PY}" scripts/evals/gpu_baselines_infer.py --job xdata --input "${L03}/xdata_docs.jsonl" --systems ${SYS} --device cpu
if [ -f "${OUT}/robustness_anchor/DONE.json" ]; then
  "${PY}" scripts/evals/gpu_baselines_infer.py --job robustness --input "${L03}/perturbed_docs.jsonl" --systems safe_wrapper \
     --anchor-predictions "${OUT}/robustness_anchor/predictions.jsonl" --device cpu
fi
if [ -f "${OUT}/xdata_anchor/DONE.json" ]; then
  "${PY}" scripts/evals/gpu_baselines_infer.py --job xdata --input "${L03}/xdata_docs.jsonl" --systems safe_wrapper \
     --anchor-predictions "${OUT}/xdata_anchor/predictions.jsonl" --device cpu
fi
if [ -f "${OUT}/bert_fair/SELECTED.json" ]; then
  "${PY}" scripts/evals/gpu_baselines_infer.py --job test974 --split test --subset gold --systems bert_fair --device cpu
fi
echo "done $(date -u +%FT%TZ)"
