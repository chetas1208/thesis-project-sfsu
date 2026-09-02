"""
GitHub README parser for SimplifyJobs repositories.

The READMEs use HTML <table> elements (not markdown pipe-tables).
Each <tr> row has 5 <td> cells:
  0: Company  (with <a> link to simplify.jobs company page)
  1: Role     (plain text)
  2: Location (plain text, sometimes with <br>)
  3: Application link(s) — contains an <a> with the actual ATS link
     and optionally a second <a> pointing to simplify.jobs
  4: Age      (e.g., "0d", "1d", etc.)

Closed listings have a 🔒 emoji in the Application column.
"""

import re
import logging
from typing import List

import requests
from bs4 import BeautifulSoup

from .models import JobListing

logger = logging.getLogger("scraper.github_sources")

# Raw URLs for each source.  We use the 'dev' branch which is the most current.
SOURCES = {
    "newgrad": "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/README.md",
    "internships": "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "speedyapply": "https://raw.githubusercontent.com/speedyapply/2026-SWE-College-Jobs/main/README.md",
    "intern_intl": "https://raw.githubusercontent.com/speedyapply/2026-AI-College-Jobs/main/INTERN_INTL.md",
    "newgrad_intl": "https://raw.githubusercontent.com/speedyapply/2026-AI-College-Jobs/main/NEW_GRAD_INTL.md",
}


def fetch_markdown(url: str) -> str:
    """Fetches raw README markdown from a GitHub URL."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return ""


def parse_html_tables(raw_md: str, source_name: str) -> List[JobListing]:
    """
    Parses all HTML <table> elements inside the raw markdown and extracts
    job listings.  Skips closed (🔒) positions.
    """
    jobs: List[JobListing] = []

    # BeautifulSoup can handle the mixed markdown + HTML just fine —
    # it will ignore the markdown and only pick up the HTML tags.
    soup = BeautifulSoup(raw_md, "html.parser")

    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue  # header rows use <th>, or malformed row

        # --- Company ---
        company_cell = cells[0]
        company = company_cell.get_text(strip=True)
        # Strip common emoji prefixes
        company = re.sub(r"^[🔥🛂🇺🇸🔒🎓\s]+", "", company).strip()

        # --- Role ---
        role = cells[1].get_text(strip=True)

        # --- Location ---
        location = cells[2].get_text(separator=", ", strip=True)

        # --- Application link ---
        app_cell = cells[3]
        cell_text = app_cell.get_text()

        # Skip closed listings (🔒 emoji present)
        if "🔒" in cell_text:
            continue

        # Find the first <a> that points to an actual job board / ATS,
        # NOT a simplify.jobs convenience link.
        job_url = ""
        for a_tag in app_cell.find_all("a", href=True):
            href = a_tag["href"]
            # Skip simplify.jobs "Easy Apply" links and image-only links
            if "simplify.jobs/p/" in href:
                continue
            job_url = href
            break  # take the first non-simplify link

        if not job_url:
            continue  # no usable link

        # Strip tracking params for dedup but keep original for navigation
        jobs.append(
            JobListing(
                source_repo=source_name,
                company=company,
                role_title=role,
                location=location,
                job_url=job_url,
            )
        )

    return jobs


def parse_markdown_tables(raw_md: str, source_name: str) -> List[JobListing]:
    """
    Parses Markdown tables (e.g. | Company | Role | Location | ...)
    """
    jobs: List[JobListing] = []
    
    for line in raw_md.split('\n'):
        line = line.strip()
        if not line.startswith('|') or not line.endswith('|'):
            continue
            
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 5:
            continue
            
        if 'Company' in cells[0] and 'Position' in cells[1]:
            continue
        if '---' in cells[0]:
            continue
            
        # Parse Company (cell 0)
        company_soup = BeautifulSoup(cells[0], "html.parser")
        company = company_soup.get_text(strip=True)
        if not company:
            company = cells[0]
        company = re.sub(r"^[🔥🛂🇺🇸🔒🎓\s]+", "", company).strip()
        
        role = cells[1]
        location = cells[2]
        
        # Link is usually in cell 4 or 3. Search backwards through cells.
        job_url = ""
        for cell in cells[3:]:
            cell_soup = BeautifulSoup(cell, "html.parser")
            for a_tag in cell_soup.find_all("a", href=True):
                href = a_tag["href"]
                if "simplify.jobs" in href or "speedyapply.com" in href:
                    continue
                job_url = href
                break
            if job_url:
                break
                
            # Fallback to markdown link format
            match = re.search(r'\]\((https?://[^\)]+)\)', cell)
            if match:
                href = match.group(1)
                if "simplify.jobs" not in href and "speedyapply.com" not in href:
                    job_url = href
                    break

        if not job_url:
            continue
            
        jobs.append(
            JobListing(
                source_repo=source_name,
                company=company,
                role_title=role,
                location=location,
                job_url=job_url,
            )
        )
    return jobs


def get_jobs(source: str) -> List[JobListing]:
    """
    Fetches and parses job listings for the requested source(s).
    source: "newgrad" | "internships" | "all"
    Returns de-duplicated list of JobListing objects.
    """
    if source == "all":
        targets = list(SOURCES.items())
    elif source in SOURCES:
        targets = [(source, SOURCES[source])]
    else:
        logger.error("Unknown source: %s", source)
        return []

    all_jobs: List[JobListing] = []
    for name, url in targets:
        logger.info("Fetching markdown for '%s' …", name)
        raw_md = fetch_markdown(url)
        if raw_md:
            parsed = parse_html_tables(raw_md, name)
            parsed.extend(parse_markdown_tables(raw_md, name))
            logger.info("Found %d active jobs in '%s'", len(parsed), name)
            all_jobs.extend(parsed)

    # De-duplicate by canonical URL (strip utm params)
    seen: dict[str, JobListing] = {}
    for job in all_jobs:
        canon = _canonical_url(job.job_url)
        if canon not in seen:
            seen[canon] = job
    deduped = list(seen.values())
    logger.info("Total unique jobs after dedup: %d", len(deduped))
    return deduped


def _canonical_url(url: str) -> str:
    """Strips common tracking query parameters for dedup comparison."""
    # Remove everything after utm_source / ref= for comparison
    return re.split(r"[?&](utm_|ref=)", url)[0]
