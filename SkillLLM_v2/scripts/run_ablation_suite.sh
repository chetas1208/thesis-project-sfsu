#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$ROOT_DIR/.venv/bin/activate"

run_v2_ablation() {
  local experiment_name="$1"
  shift

  python -m src.train.train_sft_v2 --config "$ROOT_DIR/configs/qwen3_14b_v2.yaml" \
    --override "experiment_name=${experiment_name}" \
    --override "output_dir=outputs/${experiment_name}" \
    "$@"

  python -m src.infer.batch_infer --config "$ROOT_DIR/configs/qwen3_14b_v2.yaml" \
    --override "experiment_name=${experiment_name}" \
    --override "output_dir=outputs/${experiment_name}" \
    "$@" \
    --splits test

  bash "$ROOT_DIR/scripts/run_eval_suite.sh" \
    "$ROOT_DIR/configs/qwen3_14b_v2.yaml" \
    "$experiment_name" \
    "$ROOT_DIR/outputs/${experiment_name}/predictions/test_predictions.jsonl" \
    "$ROOT_DIR/outputs/${experiment_name}/predictions/raw_test.jsonl"
}

python -m src.train.train_baseline_wrapper --config "$ROOT_DIR/configs/baseline_llama31_8b.yaml" \
  --override "experiment_name=ablation_a_original_baseline" \
  --override "output_dir=outputs/ablation_a_original_baseline"
bash "$ROOT_DIR/scripts/run_eval_suite.sh" \
  "$ROOT_DIR/configs/baseline_llama31_8b.yaml" \
  "ablation_a_original_baseline" \
  "$ROOT_DIR/outputs/ablation_a_original_baseline/predictions/test_predictions.jsonl"

python -m src.train.train_baseline_wrapper --config "$ROOT_DIR/configs/qwen3_14b_v2.yaml" \
  --override "experiment_name=ablation_b_new_backbone_only" \
  --override "output_dir=outputs/ablation_b_new_backbone_only"
bash "$ROOT_DIR/scripts/run_eval_suite.sh" \
  "$ROOT_DIR/configs/qwen3_14b_v2.yaml" \
  "ablation_b_new_backbone_only" \
  "$ROOT_DIR/outputs/ablation_b_new_backbone_only/predictions/test_predictions.jsonl"

run_v2_ablation \
  "ablation_c_span_json_target" \
  --override "prompt_template_file=src/prompts/skillspan_span_prompt_relaxed.txt" \
  --override "fewshot_prompt_file=" \
  --override "use_fewshot_for_inference=false" \
  --override "use_span_recovery=false"

run_v2_ablation \
  "ablation_d_exact_copy_prompt" \
  --override "use_span_recovery=false"

run_v2_ablation \
  "ablation_e_span_recovery" \
  --override "use_span_recovery=true"

run_v2_ablation \
  "ablation_f_hard_augmentation" \
  --override "use_span_recovery=true" \
  --override "train_file=data/v2/train_hard_augmented.jsonl"

python -m src.eval.ablation_summary \
  --metrics-dir "$ROOT_DIR/reports/metrics" \
  --output-dir "$ROOT_DIR/reports/tables"
