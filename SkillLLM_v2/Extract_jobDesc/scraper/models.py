"""
Data models used across the scraper package.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ScrapingStatus(str, Enum):
    """Outcome codes for a single job extraction attempt."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    DESCRIPTION_NOT_FOUND = "description_not_found"
    INVALID_PAGE = "invalid_page"
    EXTERNAL_REDIRECT_FAILURE = "external_redirect_failure"
    EMPTY_CONTENT = "empty_content"
    ERROR = "error"


@dataclass
class JobListing:
    """A job link parsed from a GitHub README table."""
    source_repo: str
    company: str
    role_title: str
    location: str
    job_url: str


@dataclass
class JobExtractionResult:
    """Complete extraction result for one job."""
    source_repo: str
    company: str
    role_title: str
    location: str
    job_url: str
    final_landing_url: str
    status: ScrapingStatus
    job_description_text: Optional[str] = None
    structured_sections: dict = field(default_factory=dict)
    failure_reason: Optional[str] = None
