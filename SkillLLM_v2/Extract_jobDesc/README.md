# Job Description Extractor

This companion tool collects full job descriptions from URLs listed in SimplifyJobs GitHub repositories. It is separate from the SkillLLM extraction and matching pipeline: its output is source material for later experiments, not a benchmark result by itself.

## Overview

The asynchronous Python pipeline reads Markdown tables from GitHub, follows application URLs, and uses Playwright to render JavaScript-heavy pages. It scores likely “Show More”, “Description”, and “Qualifications” controls, then re-checks the visible page text after each interaction.

Optionally, NVIDIA Nemotron can provide fallback navigation reasoning, clean noisy scraped text, and extract structured sections such as salary and qualifications. The deterministic heuristics remain usable without an LLM.

## Features

- Async Playwright support for SPAs and common ATS platforms.
- Checkpointing to `data/raw_jobs.csv`; use `--force` to reprocess attempted URLs.
- Deterministic candidate scoring and bounded interaction attempts.
- Optional LLM integration behind explicit configuration and the `--use-llm` flag.

## Setup

Run these commands from this directory:

```bash
cd SkillLLM_v2/Extract_jobDesc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
playwright install chromium
```

For optional LLM features, copy `.env.example` to a local `.env` and set `LLM_BASE_URL` and `LLM_API_KEY`. The `.env` file is ignored; never commit a real key or paste one into a command, issue, or log.

## Usage

```bash
# Scrape New Grad positions
python main.py --source newgrad

# Scrape internships with bounded concurrency for a trial run
python main.py --source internships --limit 10 --concurrency 5

# Enable optional LLM reasoning/cleanup
python main.py --source all --use-llm
```

Raw job descriptions, browser downloads, checkpoints, and generated structured output can contain private or licensed material. Keep them local and review the root `.gitignore` before sharing an artifact.

## Architecture notes

`discover_description` first validates visible text using length and keyword thresholds. If the page is incomplete, it scores DOM candidates such as buttons and tab elements, clicks the best candidates one at a time, and stops when the text resembles a full description. Greenhouse, Lever, and similar ATS pages generally work through Playwright; domain-specific shadow-DOM or iframe handling belongs in the extractor implementation.

When `--use-llm` is enabled and several tabs are ambiguous, `LLMHelper` receives a small page-context snippet and candidate list, then returns the best candidate index. It can also post-process text, but the raw scrape remains the provenance record.

Every attempted URL is appended to `data/raw_jobs.csv`. Subsequent runs skip attempted URLs unless `--force` is supplied, so preserve the checkpoint when resuming a collection job.
