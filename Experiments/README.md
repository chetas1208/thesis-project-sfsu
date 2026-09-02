# Skill Extraction Benchmark

This workspace is a dataset-ready benchmark scaffold for the skill-extraction part of the thesis. It does not train anything by default. Once a dataset is dropped into `data/raw/`, the benchmark can normalize it into one canonical schema, export model-specific inputs, run the selected models, and aggregate exact span metrics into one summary table.

## Models Included

- `NNOSE` via the upstream [jjzha/nnose](https://github.com/jjzha/nnose) repo
- `Skill-LLM` via the upstream [herandy/Skill-LLM](https://github.com/herandy/Skill-LLM) repo
- `GLiNER` via the upstream [urchade/GLiNER](https://github.com/urchade/GLiNER) repo
- `JobBERT` Hugging Face token-classification baseline using `jjzha/jobbert_skill_extraction`
- `GPT-5.4` OpenAI prompt baseline using structured outputs
- `ConTeXT` runnable auxiliary retrieval baseline using `TechWolf/ConTeXT-Skill-Extraction-base`

## Layout

- `vendor/`: upstream repos cloned locally
- `data/raw/`: place the incoming dataset here
- `data/annotation/`: annotation queue + prompt templates for raw resume sources
- `data/canonical/`: normalized benchmark splits (`train/dev/test.jsonl`)
- `data/model_inputs/`: per-model exported datasets
- `results/`: predictions, metrics, checkpoints, benchmark summaries, and analysis outputs
- `scripts/`: shell entrypoints for setup, training/inference, evaluation, and smoke tests
- `src/`: dataset normalization, model wrappers, adapters, and evaluator
- `analysis/`: script-first dataset EDA and benchmark analysis
- `.cache/`: centralized Hugging Face, pip, WandB, and OpenAI caches

## Canonical Dataset Schema

Each canonical record is JSON with:

```json
{
  "id": "train-000001",
  "text": "Built APIs in Python and FastAPI.",
  "tokens": [
    {"text": "Built", "start": 0, "end": 4, "ws": true},
    {"text": "APIs", "start": 6, "end": 9, "ws": true}
  ],
  "spans": [
    {"start": 14, "end": 19, "label": "SKILL", "text": "Python"}
  ]
}
```

Offsets are inclusive. By default all labels are collapsed to `SKILL` for apples-to-apples benchmarking.

## Secrets

Create local secrets in `.env`. This scaffold loads secrets from environment variables only and does not embed keys in scripts or configs.

Relevant variables:

- `HF_TOKEN`
- `OPENAI_API_KEY`
- `WANDB_API_KEY`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

An example file is provided at `.env.example`.

## Typical Flow

1. `bash scripts/setup_all.sh`
2. `bash scripts/download_kaggle_sources.sh`
3. `bash scripts/clean_resume_sources.sh`
4. `bash scripts/build_annotation_queue.sh`
5. Annotate into the canonical resume schema and save into `data/raw/annotated/`
6. `bash scripts/prepare_data.sh`
7. Run model wrappers as needed:
   - `bash scripts/train_jobbert.sh --mode full`
   - `bash scripts/train_nnose.sh --mode full`
   - `bash scripts/train_gliner.sh --mode full`
   - `bash scripts/train_skill_llm.sh --mode full`
   - `bash scripts/train_context.sh --mode full`
   - `bash scripts/train_gpt5.sh --split test`
8. `bash scripts/evaluate_all.sh`
9. Analysis:
   - `bash scripts/run_dataset_eda.sh`
   - `bash scripts/run_benchmark_analysis.sh`

## Notes

- `ConTeXT` is evaluated separately as a retrieval model and written to `results/retrieval_summary.csv`; it is intentionally kept out of the exact-span leaderboard in `results/summary.csv`.
- `Skill-LLM` assumes your Hugging Face account can access `meta-llama/Meta-Llama-3-8B-Instruct`.
- `GPT-5.4` is configured as a prompt baseline only.
