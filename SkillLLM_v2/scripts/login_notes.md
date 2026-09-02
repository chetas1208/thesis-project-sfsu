# Local Authentication Notes

## Hugging Face

Run:

```bash
hf auth login
```

Use a token with access to any gated models you plan to download. For this project that usually means:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-4-E4B-it`
- `google/gemma-4-26B-A4B-it`

If Hugging Face reports a gated access error, open the model page in the browser, accept the license, and retry the download.

## Model License Checks

Before downloading, confirm you can open each model page while logged in:

- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/Qwen/Qwen3-14B
- https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
- https://huggingface.co/google/gemma-4-E4B-it
- https://huggingface.co/google/gemma-4-26B-A4B-it

## Weights & Biases

WandB is optional. Log in only if you want remote run tracking.

```bash
wandb login
```

If you skip it, the new training pipeline falls back to local-only logging.
