"""
Playwright browser lifecycle management.

Provides an async context manager that creates a Chromium browser instance
with anti-detection flags and realistic viewport / user-agent settings.
"""

import asyncio
import logging
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger("scraper.browser")


class BrowserManager:
    """Async context manager for a single Playwright browser instance."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._pw = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self) -> "BrowserManager":
        self._pw = await async_playwright().start()
        # Try launching chromium.  The headless_shell variant sometimes
        # fails due to missing libasound; fall back to the full browser.
        launch_kwargs = dict(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-gpu",
            ],
        )
        try:
            self._browser = await self._pw.chromium.launch(**launch_kwargs)
        except Exception:
            logger.warning("Default launch failed, retrying with chromium path…")
            import glob, os
            # Find the full Chromium binary that Playwright downloaded
            candidates = glob.glob(
                os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")
            )
            if candidates:
                launch_kwargs["executable_path"] = candidates[0]
                self._browser = await self._pw.chromium.launch(**launch_kwargs)
            else:
                raise
        logger.info("Browser launched (headless=%s)", self.headless)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        logger.info("Browser closed")

    async def new_context(self) -> BrowserContext:
        """Creates a fresh browser context with realistic fingerprints."""
        ctx = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            ignore_https_errors=True,
        )
        return ctx
