#!/usr/bin/env python3
"""
Job Description Extractor — CLI entry point.

Usage:
    python main.py --source newgrad --limit 5
    python main.py --source internships --concurrency 5
    python main.py --source all --use-llm --debug
    python main.py --source newgrad --force          # re-process everything
"""

import argparse
import asyncio
import os
import logging

from dotenv import load_dotenv

from scraper.github_sources import get_jobs
from scraper.browser import BrowserManager
from scraper.extractor import process_job, INTER_PAGE_DELAY
from scraper.llm_helper import LLMHelper
from scraper.models import ScrapingStatus
from scraper.utils import (
    setup_logging,
    load_checkpoints,
    save_result_csv,
    save_description_jsonl,
)

# ── Output paths ──────────────────────────────────────────────────────────────
RAW_CSV = "data/raw_jobs.csv"
FAILED_CSV = "data/failed_jobs.csv"
JSONL_OUT = "data/job_descriptions.jsonl"


async def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract job descriptions from SimplifyJobs GitHub repos",
    )
    parser.add_argument(
        "--source",
        choices=["newgrad", "internships", "speedyapply", "intern_intl", "newgrad_intl", "all"],
        default="newgrad",
        help="Which repo(s) to scrape",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max jobs to process")
    parser.add_argument(
        "--headless",
        choices=["true", "false"],
        default="true",
        help="Run browser headless",
    )
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel pages")
    parser.add_argument("--force", action="store_true", help="Ignore checkpoints")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM reasoning")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logger = setup_logging(debug=args.debug)

    if args.use_llm:
        os.environ["USE_LLM"] = "true"
    llm = LLMHelper()

    # ── 1. Fetch job links from GitHub ────────────────────────────────────
    logger.info("Fetching jobs from GitHub …")
    jobs = get_jobs(args.source)
    if not jobs:
        logger.info("No jobs found.  Exiting.")
        return

    # ── 2. Apply checkpoints / limits ─────────────────────────────────────
    if not args.force:
        done = load_checkpoints(RAW_CSV)
        before = len(jobs)
        jobs = [j for j in jobs if j.job_url not in done]
        logger.info("Checkpoint filter: %d → %d remaining", before, len(jobs))

    if args.limit:
        jobs = jobs[: args.limit]
        logger.info("Limit applied: processing %d jobs", len(jobs))

    if not jobs:
        logger.info("Nothing left to process.  Exiting.")
        return

    # ── 3. Process jobs concurrently ──────────────────────────────────────
    headless = args.headless == "true"
    sem = asyncio.Semaphore(args.concurrency)

    async def _worker(job, ctx):
        async with sem:
            logger.info("▶  %s — %s", job.company, job.role_title)
            result = await process_job(job, ctx, llm)

            # Persist immediately (checkpoint)
            save_result_csv(result, RAW_CSV)
            if result.status == ScrapingStatus.SUCCESS:
                save_description_jsonl(result, JSONL_OUT)
                logger.info("   ✓  success  (%d chars)", len(result.job_description_text or ""))
            else:
                save_result_csv(result, FAILED_CSV)
                logger.warning(
                    "   ✗  %s — %s", result.status.value, result.failure_reason
                )

            # Small delay to respect rate limits
            await asyncio.sleep(INTER_PAGE_DELAY)
            return result

    logger.info("Launching browser (headless=%s, concurrency=%d) …", headless, args.concurrency)
    async with BrowserManager(headless=headless) as mgr:
        ctx = await mgr.new_context()
        tasks = [_worker(job, ctx) for job in jobs]
        results = await asyncio.gather(*tasks)

    successes = sum(1 for r in results if r.status == ScrapingStatus.SUCCESS)
    logger.info(
        "Done.  %d/%d succeeded.  Output: %s, %s",
        successes, len(results), RAW_CSV, JSONL_OUT,
    )


if __name__ == "__main__":
    asyncio.run(main())
