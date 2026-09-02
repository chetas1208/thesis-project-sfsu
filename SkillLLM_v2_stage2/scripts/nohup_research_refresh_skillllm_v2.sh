#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs reports/skillllm_v2/research_refresh artifacts/skillllm_v2/research_refresh docs/thesis
echo "$$" > logs/latest_research_refresh_skillllm_v2.pid
exec > logs/latest_research_refresh_skillllm_v2.log 2>&1

echo "============================================================"
echo "SkillLLM_v2 research refresh"
echo "Repo: $ROOT"
echo "Started: $(date)"
echo "PID: $$"
echo "============================================================"

if [ -f "/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/activate"
fi

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

echo "[1/6] Environment"
python --version
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu_{i}:", torch.cuda.get_device_name(i))
except Exception as exc:
    print("torch_check_error:", exc)
PY

echo "[2/6] Wrapper error analysis"
python -m skillllm_v2.eval.wrapper_error_analysis \
  --output-dir reports/skillllm_v2/research_refresh

echo "[3/6] Safe wrapper and gated money table"
python -m skillllm_v2.eval.run_safe_wrapper --disable-gliner
python -m skillllm_v2.eval.money_table_generator

echo "[4/6] Unit tests"
pytest tests/skillllm_v2/ -q

echo "[5/6] Research refresh reports and LaTeX"
python -m skillllm_v2.research_refresh

echo "[6/6] Artifact check"
ls -lh reports/skillllm_v2/research_refresh docs/thesis artifacts/skillllm_v2/context || true

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
