"""
Core extraction logic.

For each job listing the pipeline:
1. Navigates Playwright to the job URL.
2. Waits for content to settle.
3. Checks whether the visible text already satisfies `is_valid_job_description`.
4. If not, discovers clickable candidates (tabs, buttons, accordions,
   "Show more" links) scores them, and clicks the best ones in order.
5. After each click it re-checks the visible text.
6. Optionally asks the LLM to pick the best button if deterministic
   scoring is tied or ambiguous.
7. Returns a `JobExtractionResult`.
"""

import asyncio
import logging
import os
from typing import Optional

from bs4 import BeautifulSoup
from playwright.async_api import Page, BrowserContext, TimeoutError as PwTimeout

from .models import JobListing, JobExtractionResult, ScrapingStatus
from .detectors import is_valid_job_description, score_candidate_element
from .llm_helper import LLMHelper

logger = logging.getLogger("scraper.extractor")

# Maximum number of click-and-check cycles per page
MAX_CLICK_ATTEMPTS = 5
# Pause between rate-limited requests (seconds)
INTER_PAGE_DELAY = 1.5


# ── Public API ────────────────────────────────────────────────────────────────


async def process_job(
    job: JobListing,
    context: BrowserContext,
    llm: LLMHelper,
    *,
    data_dir: str = "data",
) -> JobExtractionResult:
    """Scrapes a single job URL and returns the extraction result."""
    page = await context.new_page()
    result = JobExtractionResult(
        source_repo=job.source_repo,
        company=job.company,
        role_title=job.role_title,
        location=job.location,
        job_url=job.job_url,
        final_landing_url=job.job_url,
        status=ScrapingStatus.ERROR,
    )

    try:
        resp = await page.goto(job.job_url, timeout=30_000, wait_until="domcontentloaded")
        result.final_landing_url = page.url

        if resp is None or resp.status >= 400:
            result.status = ScrapingStatus.INVALID_PAGE
            result.failure_reason = f"HTTP {resp.status if resp else 'no response'}"
            return result

        # Give JS frameworks a moment to hydrate
        try:
            await page.wait_for_load_state("networkidle", timeout=10_000)
        except PwTimeout:
            pass  # some SPAs never fully idle

        discovery = await _discover_description(page, llm)

        if discovery["found"]:
            raw_text: str = discovery["text"]

            # Optional LLM post-processing
            if llm.enabled:
                raw_text = await llm.clean_extracted_text(raw_text)
                result.structured_sections = await llm.structure_job_description(raw_text)

            result.job_description_text = raw_text
            result.status = ScrapingStatus.SUCCESS
        else:
            result.status = ScrapingStatus.DESCRIPTION_NOT_FOUND
            result.failure_reason = "heuristics exhausted"
            # Save a debug screenshot
            _save_screenshot(page, job, data_dir)

    except PwTimeout:
        result.status = ScrapingStatus.TIMEOUT
        result.failure_reason = "page load timed out"
    except Exception as exc:
        result.status = ScrapingStatus.ERROR
        result.failure_reason = str(exc)[:300]
    finally:
        await page.close()

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _extract_visible_text(page: Page) -> str:
    """Strips scripts/styles and returns the body text."""
    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


async def _discover_description(page: Page, llm: LLMHelper) -> dict:
    """
    Tab-discovery strategy:
    1. Read current visible text — if it's valid, return immediately.
    2. Collect clickable candidates (buttons, tabs, accordions, "show more"
       links, <summary> elements).
    3. Score each candidate deterministically via `score_candidate_element`.
    4. (Optional) ask the LLM to re-rank if scoring is tied.
    5. Click candidates in best-first order, re-check after each click.
    6. Give up after MAX_CLICK_ATTEMPTS.
    """
    # Step 1 — check the page as-is
    text = await _extract_visible_text(page)
    if is_valid_job_description(text):
        return {"found": True, "text": text}

    # Step 2 — gather clickable candidates
    # This CSS selector targets common interactive patterns on ATS pages.
    selector = (
        "button, "
        "[role='tab'], "
        "[role='button'], "
        "a.show-more, a.read-more, "
        "summary, "
        "details > summary, "
        "[data-testid*='description'], "
        "[class*='show-more'], "
        "[class*='read-more'], "
        "[class*='expand']"
    )
    try:
        elements = await page.query_selector_all(selector)
    except Exception:
        elements = []

    candidates = []
    for idx, el in enumerate(elements):
        try:
            if not await el.is_visible():
                continue
            label = (await el.inner_text()).strip()
            if not label or len(label) > 60:
                continue
            tag = await el.evaluate("e => e.tagName.toLowerCase()")
            score = score_candidate_element(label)
            if score > 0:
                candidates.append(
                    {"idx": idx, "element": el, "text": label, "tag": tag, "score": score}
                )
        except Exception:
            pass

    if not candidates:
        return {"found": False, "text": text}

    # Sort best-first
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Step 4 — optional LLM re-rank
    if llm.enabled and len(candidates) > 1:
        llm_cands = [{"index": i, "text": c["text"], "tag": c["tag"]} for i, c in enumerate(candidates)]
        best_idx = await llm.choose_next_action(llm_cands, text[:500])
        if best_idx is not None and 0 <= best_idx < len(candidates):
            chosen = candidates.pop(best_idx)
            candidates.insert(0, chosen)
            logger.info("LLM re-ranked: chose '%s'", chosen["text"])

    # Step 5 — click and check
    for attempt, cand in enumerate(candidates):
        if attempt >= MAX_CLICK_ATTEMPTS:
            break
        logger.debug("  clicking '%s' (score=%d)", cand["text"], cand["score"])
        try:
            await cand["element"].click(timeout=3_000)
            await page.wait_for_timeout(1_200)

            text = await _extract_visible_text(page)
            if is_valid_job_description(text):
                return {"found": True, "text": text}
        except Exception as exc:
            logger.debug("  click failed: %s", exc)

    return {"found": False, "text": text}


def _save_screenshot(page: Page, job: JobListing, data_dir: str) -> None:
    """Best-effort screenshot for debugging failed extractions."""
    try:
        safe = f"{job.company}_{job.role_title}"[:60].replace(" ", "_").replace("/", "_")
        path = os.path.join(data_dir, f"failed_{safe}.png")
        # fire-and-forget — don't await in finally
        asyncio.ensure_future(page.screenshot(path=path, full_page=True))
    except Exception:
        pass
