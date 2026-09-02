"""
Optional LLM integration (NVIDIA Nemotron via OpenAI-compatible API).

The pipeline works fully without an LLM.  When enabled (--use-llm flag or
USE_LLM=true in .env), three capabilities are unlocked:

1. choose_next_action  – picks the best tab/button when heuristic scoring
   produces a tie.
2. clean_extracted_text – cleans up scraped HTML noise.
3. structure_job_description – extracts structured fields as JSON.

Configuration (environment variables):
    LLM_BASE_URL  – e.g. http://localhost:8000/v1
    LLM_API_KEY   – any string accepted by the endpoint
    LLM_MODEL     – default: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("scraper.llm_helper")

try:
    from openai import AsyncOpenAI
    _OPENAI_OK = True
except ImportError:
    _OPENAI_OK = False


class LLMHelper:
    """Thin async wrapper around an OpenAI-compatible chat endpoint."""

    def __init__(self) -> None:
        self.enabled: bool = os.getenv("USE_LLM", "false").lower() == "true"
        self._client: Optional["AsyncOpenAI"] = None
        self.model: str = os.getenv(
            "LLM_MODEL",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        )

        if self.enabled:
            if not _OPENAI_OK:
                logger.error("openai package not installed — LLM disabled")
                self.enabled = False
            else:
                base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
                api_key = os.getenv("LLM_API_KEY", "dummy")
                self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
                logger.info("LLM enabled: model=%s  base_url=%s", self.model, base_url)

    # ── 1. Navigation reasoning ──────────────────────────────────────────

    async def choose_next_action(
        self,
        candidates: list[dict],
        page_context: str,
    ) -> Optional[int]:
        """
        Given a list of clickable candidates (each with 'index', 'text', 'tag')
        and a snippet of the current page text, return the integer index of the
        best candidate to click.  Returns None on failure.
        """
        if not self.enabled or not self._client:
            return None

        prompt = (
            "You are a web-scraping assistant.  The goal is to find the full "
            "job description on a careers page.\n"
            f"Page snippet: {page_context[:500]}\n\n"
            "Clickable candidates:\n"
        )
        for c in candidates:
            prompt += f"  [{c['index']}] <{c['tag']}> \"{c['text']}\"\n"
        prompt += "\nRespond with ONLY the integer index of the best candidate."

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            digits = "".join(filter(str.isdigit, resp.choices[0].message.content))
            return int(digits) if digits else None
        except Exception as exc:
            logger.warning("LLM choose_next_action failed: %s", exc)
            return None

    # ── 2. Text cleanup ──────────────────────────────────────────────────

    async def clean_extracted_text(self, text: str) -> str:
        """Cleans noisy scraped text using the LLM."""
        if not self.enabled or not self._client:
            return text

        prompt = (
            "Clean up the following job description text.  "
            "Remove navigation menus, cookie banners, and fix broken sentences.  "
            "Preserve all original information.\n\n" + text[:6000]
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM clean_extracted_text failed: %s", exc)
            return text

    # ── 3. Structured extraction ─────────────────────────────────────────

    async def structure_job_description(self, text: str) -> dict:
        """Extracts structured fields (responsibilities, qualifications, …)."""
        if not self.enabled or not self._client:
            return {}

        prompt = (
            "Extract the following fields from the job description as JSON:\n"
            "  responsibilities  (list[str])\n"
            "  qualifications    (list[str])\n"
            "  preferred_qualifications (list[str])\n"
            "  salary            (str | null)\n"
            "  visa_sponsorship  (bool | null)\n"
            "  employment_type   (str | null)\n\n"
            "Job Description:\n" + text[:8000]
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            logger.warning("LLM structure_job_description failed: %s", exc)
            return {}
