# Job Description Extractor

A robust, asynchronous Python pipeline to extract full job descriptions from URLs found in SimplifyJobs GitHub repositories.

## Overview

This tool reads Markdown tables directly from GitHub, follows the application URLs, and uses Playwright to render JavaScript-heavy pages. It employs smart heuristics to click through "Show More" buttons or specific tabs (like "Job Description" or "Qualifications") if the initial page text does not resemble a full job description.

Optionally, it integrates with NVIDIA Nemotron to provide:
1. Fallback reasoning when navigating ambiguous tab structures.
2. Cleaning up noisy, scraped text.
3. Extracting structured sections (salary, qualifications, etc.) as JSON.

## Features

- **Async Playwright**: Fast and handles SPAs / ATS platforms well.
- **Resilience**: Checkpoints progress to `data/raw_jobs.csv`. Pass `--force` to re-process.
- **Fallback Heuristics**: Deterministic scoring identifies the best buttons to click to reveal hidden job details.
- **Pluggable LLM**: Works fine offline using regex/heuristics, but gets smarter with an LLM.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```
3. Configure environment variables (optional, for LLM):
   ```bash
   cp .env.example .env
   # Edit .env with your LLM_BASE_URL and LLM_API_KEY
   ```

## Usage

Run the main pipeline:
```bash
# Scrape New Grad positions
python main.py --source newgrad

# Scrape Internships with concurrency of 5 and limit to 10 for testing
python main.py --source internships --limit 10 --concurrency 5

# Enable LLM reasoning
python main.py --source all --use-llm
```

## Architecture Notes

### Tab Discovery & Interaction
Pages often hide job descriptions behind "Read More" accordions or separate tabs. The `discover_description` function:
1. Validates visible text against a length & keyword threshold (`is_valid_job_description`).
2. If invalid, it queries the DOM for clickable elements (`button`, `div[role='tab']`, etc.).
3. It scores these elements based on text content (e.g., "Description" = +10 pts).
4. Clicks the highest scored candidates one by one, pausing and re-evaluating the visible text each time.

### Nemotron LLM Usage
When `--use-llm` is enabled and multiple ambiguous tabs exist, the `LLMHelper` sends a snippet of the page context and the available buttons to the model. The model replies with the integer index of the best button to click. It also post-processes the scraped text for cleanliness and structure.

### Resuming & Checkpoints
Every processed URL is immediately appended to `data/raw_jobs.csv`. Upon subsequent runs, `load_checkpoints` reads this file and skips any previously attempted URL (regardless of success/failure) unless `--force` is supplied.

### Extending Job Boards
Currently, ATS platforms like Greenhouse and Lever are handled naturally because Playwright renders their SPAs. If a specific ATS uses shadow DOM or complex iFrames, you can extend the `candidate_selectors` in `extractor.py` to pierce the shadow DOM, or add domain-specific logic inside `process_job`.
