#!/usr/bin/env bash
# B1: peak CPU RSS of each CPU system, one FRESH process per system, 100 test docs (RSS saturates after model load).
# Output: artifacts/skillllm_v2/evals_gpu/smoke/rss_<system>/DONE.json (process_peak_rss_gb_so_far). Read by research/systems.py.
# Launch: setsid nohup /home/923873155/server-guardrails/bin/guardrails run --ram 6 cpu -- bash scripts/evals/gpu_cpu_rss.sh > logs/evals_gpu/b1_cpu_rss.log 2>&1 &
set -uo pipefail
REPO_ROOT="/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"
cd "${REPO_ROOT}"
PY="/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python"
export CUDA_VISIBLE_DEVICES="" HF_HOME=/usr/data/models/skillllm-v2-stage2/hf_cache_a3 TOKENIZERS_PARALLELISM=false PYTHONPATH="${REPO_ROOT}/src"
for s in token_lookup gliner_zeroshot bert_fair; do
  "${PY}" scripts/evals/gpu_baselines_infer.py --job rss --systems "$s" --device cpu --smoke --limit 100 --force
done
"${PY}" scripts/evals/gpu_baselines_infer.py --job rss --systems safe_wrapper --device cpu --smoke --limit 100 --force \
  --anchor-predictions artifacts/skillllm_v2/mistral_lora/predictions.jsonl
echo "done $(date -u +%FT%TZ)"
