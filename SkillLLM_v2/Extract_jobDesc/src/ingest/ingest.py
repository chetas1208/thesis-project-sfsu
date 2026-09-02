import argparse
import csv
import glob
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional


SUPPORTED_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt", ".html", ".htm"}
SKIP_DIRS = {"processed", "silver", "gold", "gold_review", "reports", "__pycache__"}

TEXT_KEYS = (
    "full_description_raw",
    "job_description_text",
    "description",
    "description_text",
    "raw_text",
    "cleaned_text",
    "content",
    "body",
    "html",
    "text",
)

URL_KEYS = ("job_url", "final_landing_url", "url", "link")
TITLE_KEYS = ("role_title", "title", "job_title", "position")
COMPANY_KEYS = ("company", "company_name", "employer")
LOCATION_KEYS = ("location", "job_location", "city")
DATE_KEYS = ("posted_date", "date_posted", "posted_at", "created_at")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_space(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def pick_value(record: Dict[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return normalize_space(str(value))
    return ""


def extract_text(record: Dict[str, object]) -> str:
    for key in TEXT_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def stable_job_id(url: str, company: str, title: str, source_hint: str) -> str:
    base = "||".join(
        [
            normalize_space(url).lower(),
            normalize_space(company).lower(),
            normalize_space(title).lower(),
            normalize_space(source_hint).lower(),
        ]
    )
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"job_{digest}"


def canonical_identity(job: Dict[str, object]) -> str:
    for key in ("url", "final_landing_url"):
        value = normalize_space(str(job.get(key, ""))).lower()
        if value:
            return f"url::{value}"

    explicit_job_id = normalize_space(str(job.get("job_id", ""))).lower()
    if explicit_job_id:
        return f"job_id::{explicit_job_id}"

    title = normalize_space(str(job.get("title", ""))).lower()
    company = normalize_space(str(job.get("company", ""))).lower()
    location = normalize_space(str(job.get("location", ""))).lower()
    return f"meta::{company}::{title}::{location}"


def row_to_job(record: Dict[str, object], source_path: str) -> Dict[str, object]:
    url = pick_value(record, URL_KEYS)
    final_landing_url = normalize_space(str(record.get("final_landing_url", "")))
    company = pick_value(record, COMPANY_KEYS)
    title = pick_value(record, TITLE_KEYS)
    location = pick_value(record, LOCATION_KEYS)
    posted_date = pick_value(record, DATE_KEYS)
    raw_text = extract_text(record)
    source_repo = normalize_space(str(record.get("source_repo", "")))

    source_hint = source_path if raw_text else f"{source_path}:{url}:{company}:{title}"
    job_id = normalize_space(str(record.get("job_id", ""))) or stable_job_id(
        url or final_landing_url,
        company,
        title,
        source_hint,
    )

    return {
        "job_id": job_id,
        "title": title,
        "company": company,
        "location": location,
        "url": url or final_landing_url,
        "final_landing_url": final_landing_url,
        "posted_date": posted_date,
        "raw_text": raw_text,
        "source_repo": source_repo,
        "source_file": os.path.relpath(source_path),
        "source_format": os.path.splitext(source_path)[1].lstrip(".").lower(),
    }


def iter_json_records(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = [payload]

    for record in payload:
        if isinstance(record, dict):
            yield record


def iter_jsonl_records(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def iter_csv_records(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def iter_text_record(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw_text = handle.read()

    filename = os.path.splitext(os.path.basename(path))[0]
    yield {
        "job_id": filename,
        "title": filename.replace("_", " ").replace("-", " ").strip(),
        "full_description_raw": raw_text,
    }


def iter_records(path: str) -> Iterable[Dict[str, object]]:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".json":
        yield from iter_json_records(path)
    elif extension == ".jsonl":
        yield from iter_jsonl_records(path)
    elif extension == ".csv":
        yield from iter_csv_records(path)
    elif extension in {".txt", ".html", ".htm"}:
        yield from iter_text_record(path)


def job_score(job: Dict[str, object]) -> int:
    score = len(str(job.get("raw_text", "")))
    for field in ("title", "company", "location", "url", "posted_date"):
        if normalize_space(str(job.get(field, ""))):
            score += 25
    return score


def merge_jobs(left: Dict[str, object], right: Dict[str, object]) -> Dict[str, object]:
    merged = dict(left)

    for field in ("title", "company", "location", "url", "final_landing_url", "posted_date", "source_repo"):
        if not normalize_space(str(merged.get(field, ""))):
            merged[field] = right.get(field, "")

    if len(str(right.get("raw_text", ""))) > len(str(merged.get("raw_text", ""))):
        merged["raw_text"] = right.get("raw_text", "")
        merged["source_file"] = right.get("source_file", merged.get("source_file", ""))
        merged["source_format"] = right.get("source_format", merged.get("source_format", ""))

    if job_score(right) > job_score(merged):
        for field in ("title", "company", "location", "url", "final_landing_url", "posted_date"):
            if normalize_space(str(right.get(field, ""))):
                merged[field] = right.get(field, "")

    return merged


def discover_input_files(input_root: str) -> List[str]:
    data_root = os.path.abspath(input_root)
    discovered: List[str] = []

    for path in glob.glob(os.path.join(data_root, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(path)[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue

        rel_parts = os.path.relpath(path, data_root).split(os.sep)
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue

        discovered.append(path)

    return sorted(set(discovered))


def run_ingest(input_root: str, output_file: str) -> Dict[str, int]:
    ensure_parent(output_file)
    merged_jobs: Dict[str, Dict[str, object]] = {}
    stats = {
        "files_seen": 0,
        "records_seen": 0,
        "records_with_text": 0,
        "jobs_without_text_dropped": 0,
        "jobs_written": 0,
    }

    for path in discover_input_files(input_root):
        stats["files_seen"] += 1
        for record in iter_records(path):
            stats["records_seen"] += 1
            job = row_to_job(record, path)
            if normalize_space(str(job.get("raw_text", ""))):
                stats["records_with_text"] += 1

            identity = canonical_identity(job)
            if identity in merged_jobs:
                merged_jobs[identity] = merge_jobs(merged_jobs[identity], job)
            else:
                merged_jobs[identity] = job

    jobs = []
    for job in merged_jobs.values():
        if normalize_space(str(job.get("raw_text", ""))):
            jobs.append(job)
        else:
            stats["jobs_without_text_dropped"] += 1
    jobs.sort(key=lambda item: (item.get("job_id", ""), item.get("company", ""), item.get("title", "")))
    stats["jobs_written"] = len(jobs)

    with open(output_file, "w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw job-posting inputs into JSONL.")
    parser.add_argument("--input-root", default="data", help="Directory to scan for raw source files.")
    parser.add_argument("--output-file", default="data/processed/jobs_ingested.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_ingest(args.input_root, args.output_file)
    print(json.dumps(summary, indent=2))
