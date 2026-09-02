"""
Heuristic detectors for job description validation and tab-candidate scoring.

All detection is deterministic — no LLM calls.  The LLM fallback lives in
llm_helper.py and is invoked only from extractor.py when heuristics are
insufficient.
"""

import re
from typing import List

# ── Keywords that signal a real job description ───────────────────────────────

JD_SECTION_KEYWORDS: List[str] = [
    r"responsibilit",
    r"what you.{0,10}do",
    r"duties",
    r"your role",
    r"qualifications?",
    r"requirements?",
    r"what we.{0,10}looking for",
    r"preferred qualifications?",
    r"nice to have",
    r"bonus points",
    r"about the role",
    r"about this job",
    r"about the team",
    r"about the company",
    r"the team",
    r"basic qualifications?",
    r"minimum qualifications?",
    r"skills",
    r"experience",
    r"who you are",
    r"the opportunity",
    r"job summary",
    r"position summary",
    r"key accountabilit",
]

# Keywords that tend to appear in *boilerplate only* pages
EEO_KEYWORDS = [
    "equal opportunity",
    "sexual orientation",
    "gender identity",
    "marital status",
    "veteran status",
    "affirmative action",
]

# ── Clickable-candidate labels worth trying ───────────────────────────────────

_HIGH_PRIORITY_LABELS = [
    "description",
    "job description",
    "about the role",
    "overview",
    "details",
    "read more",
    "show more",
    "see more",
    "view full description",
    "full description",
    "expand",
]

_MEDIUM_PRIORITY_LABELS = [
    "responsibilities",
    "qualifications",
    "requirements",
    "what you'll do",
    "duties",
    "about this job",
    "about the job",
    "the role",
    "skills",
]


# ── Public API ────────────────────────────────────────────────────────────────


def is_valid_job_description(text: str) -> bool:
    """
    Returns True when *text* looks like a genuine, reasonably complete
    job description — not just cookie banners, nav chrome, or EEO boilerplate.
    """
    if not text:
        return False

    cleaned = _normalise(text)

    # 1. Minimum length
    if len(cleaned) < 400:
        return False

    # 2. Keyword hits
    hits = sum(1 for kw in JD_SECTION_KEYWORDS if re.search(kw, cleaned))
    if hits < 2 and len(cleaned) < 1500:
        return False

    # 3. Reject pages that are *only* EEO boilerplate
    eeo_hits = sum(cleaned.count(kw) for kw in EEO_KEYWORDS)
    if eeo_hits >= 3 and len(cleaned) < 1500:
        return False

    return True


def score_candidate_element(label: str) -> int:
    """
    Scores a clickable element's visible text to decide click priority.
    Higher = more promising for revealing a job description.
    """
    if not label:
        return 0
    label = label.strip().lower()

    if any(p in label for p in _HIGH_PRIORITY_LABELS):
        return 10
    if any(p in label for p in _MEDIUM_PRIORITY_LABELS):
        return 5
    return 0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
