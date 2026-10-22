# Low-compute loop training (quantized iterative QLoRA)

Replaces the brute-force run (`configs/skillllm_v2_training_80ep_2gpu.yaml`: 80 epochs, 2 GPUs, ~90 GPU-hours x2, r=16, loss on prompt tokens too) with short rounds, hard-example mining, and a hard compute cap.

Files: `src/skillllm_v2/train/loop_trainer.py`, `configs/skillllm_v2_loop_qlora.yaml`, `scripts/train_mistral_qlora_loop.sh`, `tests/skillllm_v2/test_loop_trainer.py`.

## Method

| Piece | Choice | Why |
|---|---|---|
| Base | Mistral-Small-3.1-24B, 4-bit NF4, double-quant, bf16 compute | ~14.6 GB resident, fits one 3090 beside nothing else |
| Adapter | LoRA r=8, alpha=16, all attn+MLP linears (46 M params) | small, cheap to checkpoint per round |
| Memory | gradient checkpointing, micro-batch 2 x accum 8, paged 8-bit AdamW | measured peak 16.4 GB allocated |
| Loss | assistant tokens only (prompt masked), examples >768 tokens dropped not truncated | the old trainer also trained on the ~300-token fixed prompt |
| Round | 640 examples (~0.13 epoch), cosine LR inside the round, x0.85 LR per round | many cheap checkpoints instead of one long run |
| Eval | fixed 160-doc dev subset (gold-bearing docs), `evaluate_extraction`, metric `exact_span_f1` | same metric as the wrapper tuning; best adapter copied to `<out>/best` |
| Mining | after each round, generate on a 192-doc rotating TRAIN probe (30% empty docs); classify each doc's errors as missed / spurious / boundary / label; keep docs with score>=0.5 in a hard pool (retired once fixed) | dev is never mined, so dev F1 stays a clean selection signal |
| Next mix | 35% hard docs (repeat = round(score), cap 3) + fresh unseen docs; empty-doc share ramps 10%->25% | curriculum: learn positives first, then hard negatives against false positives |
| Stop | plateau (patience 2, min_delta 0.002), `max_rounds`, or budget | |
| Budget | `max_gpu_hours` 5.0 and `max_train_tokens` 2.5 M, hard. Training stops at 90% of GPU-hours so the final eval fits; a round that is predicted to exceed the cap is not started. SIGSTOP pauses from guardrails are billed at most 120 s per event | |
| Resume | `loop_state.json` (atomic write) is committed only after a round is trained, evaluated and mined. Re-running the same command resumes from the last completed round's adapter; a crashed round is redone | |

Outputs in `artifacts/skillllm_v2/qlora_loop/`: `best/` (use as `adapter_path`), `rounds/round_NN/{adapter,mix_manifest.json,dev_predictions.jsonl,hard_examples.jsonl,round_summary.json}`, `loop_state.json`, `loop_summary.json`. It never touches the existing 80-epoch adapter.

## Run

```bash
bash scripts/train_mistral_qlora_loop.sh --plan      # no GPU: data sizes + round-0 mix
bash scripts/train_mistral_qlora_loop.sh --smoke     # 1 tiny round, 20 docs (verified)
bash scripts/train_mistral_qlora_loop.sh             # real run, GPU1, resumes automatically
bash scripts/train_mistral_qlora_loop.sh --set budget.max_gpu_hours=2 --set loop.max_rounds=4
```

The script pins `CUDA_VISIBLE_DEVICES=1`, runs through `guardrails run --ram 16 gpu1`, and uses the project venv (`SkillLLM_v2/.venv`, the only env that has torch + peft + bitsandbytes together). The trainer refuses to load if the card has < 19 GB free.

Estimated cost per round on one 3090 (from the smoke: ~0.8 examples/s, ~350 tokens/example, ~1.2 s/doc generation): train ~13 min + dev eval and mining probe ~7 min = ~20 min, ~0.33 GPU-h. A full 8-round run is ~2.7 GPU-h and ~2.8 M tokens, so the 2.5 M-token cap is the binding limit (~7 rounds) unless raised. Compare ~180 GPU-h for the 80-epoch run. These are extrapolations from a 12-example round; re-check after round 0.

## Would a smaller base be a better trade?

Local candidates (siblings of `models/`, which only holds a symlink to the 24B): `llama-3.1-8b-instruct`, `qwen3-14b`, `gemma-4-e4b-it`. There is no 7B Mistral on disk.

- Llama-3.1-8B NF4 is ~5.5 GB and roughly 3x cheaper per token (~7 min/round), so it can fit loop hyperparameter search (hard_frac, round size, patience) in about an hour; it still does not fit the ~4 GB free on GPU0.
- Span extraction is mostly copy-with-boundaries; the 8B gap after SFT is plausibly a few F1 points, but that is an untested hypothesis here.
- The project, the wrapper, `MistralExtractor` and the thesis claims are all pinned to Mistral-Small-3.1; swapping the final model would break comparability with the 0.6701/0.6726 numbers.

Recommendation: keep the 24B for the reported result (the loop already makes it ~60x cheaper than before) and use Llama-3.1-8B only as an optional pilot for hyperparameters:

```bash
bash scripts/train_mistral_qlora_loop.sh \
  --set model.model_path="/home/923873155/Sem 4 Research/SkillLLM_v2/models/llama-3.1-8b-instruct" \
  --set 'lora.target_modules=.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' \
  --set model.min_free_gb=9 --set output.dir=artifacts/skillllm_v2/qlora_loop_llama8b
```
(Not run. Tokens/prompt template are shared, so no code change is needed in principle.)

The biggest remaining lever is the fixed ~300-token extraction prompt (~85% of every training token). A shorter prompt would cut compute ~5x but must be mirrored in `MistralExtractor`, so it is left as a separate, deliberate change.
