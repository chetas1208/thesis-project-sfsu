# Skill Extraction Benchmark

This is the runnable benchmark harness for comparing skill-extraction approaches on one canonical corpus. It supports token-classification, span, retrieval, and generative baselines while keeping data preparation, inference, and evaluation in separate stages.

## Models compared

The repository includes adapters or launchers for:

1. **NNOSE** — kNN-augmented token classification.
2. **GLiNER** — zero-shot/few-shot span extraction.
3. **Skill-LLM** — instruction-tuned LLM extraction.
4. **ConTeXT** — dense retrieval, reported separately from exact-span extraction.
5. **JobBERT** — token-classification baseline.
6. **Gemma 4** — generative JSON extraction with exact string-to-source-span recovery.

An OpenAI structured-output prompt baseline is also available where the local configuration and account access permit it. Check the active script/config before comparing a run; availability is not the same as a completed result.

## What the pipeline does

- Normalizes source annotations into a canonical JSONL schema.
- Exports model-specific inputs without changing the source offsets.
- Runs selected model wrappers and records latency/resource metadata.
- Computes span-level precision, recall, and F1 plus text/document summaries.
- Produces machine-readable metrics and a comparison report under `outputs/`.

Generative systems must return JSON skill strings that are mapped back to exact source spans. A generated skill that cannot be aligned to the source is not counted as a valid exact-span prediction.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env  # edit locally; .env is ignored
```

Use environment variables for credentials. For example, export `HF_TOKEN` in the shell before running a Hugging Face-backed model; do not place a real token in `.env` committed to Git, source files, or notebooks.

## Run

The complete orchestrator is:

```bash
export HF_TOKEN='your-token-in-the-environment'
bash scripts/40_run_all.sh
```

It downloads the SkillSpan source, creates canonical data, exports six model formats, trains the configured models, runs inference, and builds the comparison tables. For stage-by-stage work, use the numbered scripts:

```bash
python scripts/00_download_skillspan.py
python scripts/01_prepare_canonical.py
python scripts/02_export_nnose.py
python scripts/03_export_gliner.py
python scripts/04_export_skillllm.py
python scripts/05_export_context.py
python scripts/06_export_jobbert.py
python scripts/07_export_gemma4.py

# Optional model download, training, inference, and evaluation stages
python scripts/08_download_models.py
python scripts/30_evaluate_span_metrics.py
python scripts/31_evaluate_document_metrics.py
python scripts/32_build_comparison_table.py
```

Model-specific training and prediction launchers are numbered `10`–`15` and `20`–`25`. Inspect their configuration and required local checkpoints before starting a long run. Keep GPU jobs in a durable session and retain the command/config alongside the result for reproducibility.

## Outputs and evidence

Generated models, predictions, metrics, reports, caches, and checkpoints live under `outputs/` or `.cache/` and are ignored by Git. The benchmark’s comparison table is an exact-span leaderboard, while ConTeXT retrieval metrics belong in its separate retrieval summary. Do not combine those values or present a smoke run as a quality/promotion result.

See [`../../README.md`](../../README.md) for the project map, [`BENCHMARK_GUIDE.md`](BENCHMARK_GUIDE.md) for operational details, and [`../../docs/thesis/codebase_audit.md`](../../docs/thesis/codebase_audit.md) for the thesis evidence boundary.
