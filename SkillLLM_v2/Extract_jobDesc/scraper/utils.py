"""
Logging, checkpointing, and incremental I/O utilities.
"""

import csv
import json
import logging
import os
from typing import Set


def setup_logging(log_file: str = "logs/extract.log", debug: bool = False) -> logging.Logger:
    """Configures the root 'scraper' logger with console + file handlers."""
    logger = logging.getLogger("scraper")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_checkpoints(csv_path: str) -> Set[str]:
    """Returns the set of job URLs already recorded in *csv_path*."""
    urls: Set[str] = set()
    if not os.path.exists(csv_path):
        return urls
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if "job_url" in row:
                    urls.add(row["job_url"])
    except Exception as exc:
        logging.getLogger("scraper").warning("checkpoint read failed: %s", exc)
    return urls


def save_result_csv(result, csv_path: str) -> None:
    """Appends one row to the CSV (creates file + header if needed)."""
    is_new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "source_repo", "company", "role_title", "location",
        "job_url", "final_landing_url", "status", "failure_reason",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "source_repo": result.source_repo,
            "company": result.company,
            "role_title": result.role_title,
            "location": result.location,
            "job_url": result.job_url,
            "final_landing_url": result.final_landing_url,
            "status": result.status.value,
            "failure_reason": result.failure_reason or "",
        })


def save_description_jsonl(result, jsonl_path: str) -> None:
    """Appends a successful extraction to the JSONL file."""
    if not result.job_description_text:
        return
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        record = {
            "job_url": result.job_url,
            "final_landing_url": result.final_landing_url,
            "company": result.company,
            "role_title": result.role_title,
            "location": result.location,
            "job_description_text": result.job_description_text,
            "structured_sections": result.structured_sections,
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
