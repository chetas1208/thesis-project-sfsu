# Experiments and Benchmark Lab

This directory contains the independent benchmark and data-preparation work that supports the thesis. It is a research harness, not the production-facing Stage-2 pipeline: it normalizes an incoming corpus, exports model-specific inputs, runs selected baselines, and aggregates comparable span metrics.

## Scope

The benchmark can compare:

- NNOSE, Skill-LLM, GLiNER, and JobBERT extraction baselines.
- ConTeXT as a retrieval-oriented auxiliary baseline.
- An OpenAI structured-output prompt baseline when credentials and access are available.

The exact model set and protocol are defined by the scripts and configuration under [`skill_extraction_benchmark/`](skill_extraction_benchmark/). Do not interpret a benchmark result as a Stage-2 promotion result without checking the thesis audit and the metric definition.

## Layout

```text
skill_extraction_benchmark/
├── data/raw/          downloaded source data (review before sharing)
├── data/exports/      canonical and model-specific JSON/JSONL exports
├── configs/           model and global training settings
├── scripts/            numbered setup, training, inference, and evaluation stages
├── src/                schema, normalization, wrappers, and metrics
├── outputs/            local models, predictions, metrics, and reports (ignored)
└── .cache/             local dependency and service caches (ignored)
```

Canonical records use character offsets with `start` inclusive and `end` exclusive. Labels are normally collapsed to `SKILL` for apples-to-apples span evaluation; preserve the original annotation when the experiment needs it.

## Typical flow

Run commands from `Experiments/skill_extraction_benchmark/` in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env  # fill locally; do not commit .env

python scripts/00_download_skillspan.py
python scripts/01_prepare_canonical.py
python scripts/02_export_nnose.py
python scripts/03_export_gliner.py
python scripts/04_export_skillllm.py
python scripts/05_export_context.py
python scripts/06_export_jobbert.py
python scripts/07_export_gemma4.py
```

The full orchestration entry point is `bash scripts/40_run_all.sh`. It also runs training, inference, and evaluation, and requires `HF_TOKEN` in the environment. Use a durable session and verify the resulting report before making claims. The numbered scripts can be run individually when debugging a single stage or model.

## Credentials and artifacts

Credentials are read from environment variables such as `HF_TOKEN`, `OPENAI_API_KEY`, `WANDB_API_KEY`, `KAGGLE_USERNAME`, and `KAGGLE_KEY`. The checked-in `.env.example` contains placeholders only. Keep raw resumes, scraped job records, model weights, checkpoints, logs, caches, and generated results local; the root `.gitignore` excludes these classes of artifacts.

See the benchmark-specific [README](skill_extraction_benchmark/README.md), the [repository guide](../README.md), and [`docs/thesis/codebase_audit.md`](../docs/thesis/codebase_audit.md) for the supported boundaries and evidence rules.
