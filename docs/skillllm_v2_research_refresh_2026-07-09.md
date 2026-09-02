# SkillLLM_v2 Full 2x3090 Pipeline Run

Date: 2026-07-09

Project:

`/home/923873155/Sem 4 Research/SkillLLM_v2_stage2`

## Current Run

The full Stage-2 pipeline has been started with only GPUs `0,1` visible.

Active training process:

`logs/latest_stage2_full_2gpu.pid`

Current observed training PID:

`1003528`

Active training log:

`logs/latest_stage2_full_2gpu.log`

The active Mistral LoRA training process is running through `torchrun` with two workers. Each worker is using one RTX 3090.

Downstream watcher:

`logs/latest_stage2_downstream_after_train.pid`

Current observed downstream watcher PID:

`1005254`

Downstream watcher log:

`logs/latest_stage2_downstream_after_train.log`

Downstream watcher script:

`scripts/watch_and_resume_after_mistral_train.sh`

The downstream watcher waits for the Mistral training PID to finish, then resumes validation/test/inference/report stages with `CUDA_VISIBLE_DEVICES=0,1`.

## Commands To Monitor

```bash
cd "/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"

cat logs/latest_stage2_full_2gpu.pid
ps -p "$(cat logs/latest_stage2_full_2gpu.pid)" -o pid,ppid,sid,stat,etime,pcpu,pmem,cmd
nvidia-smi
tail -f logs/latest_stage2_full_2gpu.log
```

Monitor downstream resume:

```bash
cd "/home/923873155/Sem 4 Research/SkillLLM_v2_stage2"

cat logs/latest_stage2_downstream_after_train.pid
ps -p "$(cat logs/latest_stage2_downstream_after_train.pid)" -o pid,ppid,sid,stat,etime,pcpu,pmem,cmd
tail -f logs/latest_stage2_downstream_after_train.log
```

## Expected Outputs

Training artifacts:

- `artifacts/skillllm_v2/mistral_lora/model/`
- `artifacts/skillllm_v2/mistral_lora/raw_outputs.jsonl`
- `artifacts/skillllm_v2/mistral_lora/predictions.jsonl`
- `artifacts/skillllm_v2/mistral_lora/metrics.json`

Downstream artifacts:

- `artifacts/skillllm_v2/mistral_boundary/metrics.json`
- `artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json`
- `artifacts/skillllm_v2/full_cag_graph/metrics.json`
- `artifacts/skillllm_v2/safe_wrapper/metrics.json`
- `reports/skillllm_v2/money_table.csv`
- `reports/skillllm_v2/money_table.md`
- `reports/skillllm_v2/thesis_results_summary.md`

## Notes

- Do not start another full Mistral training run while the active training PID is alive.
- GPU visibility is constrained through `CUDA_VISIBLE_DEVICES=0,1`.
- Metrics should only be trusted after the corresponding `metrics.json` files are regenerated.
