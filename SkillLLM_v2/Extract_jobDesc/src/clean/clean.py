import argparse
import html
import json
import os
import re
from typing import Dict, List

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


LINE_NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"privacy policy",
        r"cookie consent",
        r"accept(?:ed)? file types?",
        r"upload file",
        r"attach(?:ments?)?",
        r"enter manually",
        r"powered by",
        r"cover letter",
        r"resume/?cv",
        r"apply now",
        r"submit application",
        r"google drive",
        r"dropbox",
        r"locate me",
        r"application (?:instructions|question)",
        r"indicates a required field",
        r"first name",
        r"last name",
        r"preferred first name",
        r"country phone",
        r"autofill with",
        r"create alert",
        r"additional attachments",
        r"security vulnerability disclosure",
        r"e-verify",
        r"equal opportunity employer",
        r"we are an equal opportunity employer",
        r"all qualified applicants will receive consideration",
        r"reasonable accommodation",
        r"pay transparency",
        r"candidate privacy notice",
        r"employment (?:eligibility|verification)",
        r"criminal histories",
        r"veteran status",
        r"gender identity",
        r"sexual orientation",
    )
]

COUNTRY_PICKER_PATTERN = re.compile(r"(?:\+\d{1,4}\s+){4,}")
HTML_MARKER_PATTERN = re.compile(r"<[^>]+>")
MULTISPACE_PATTERN = re.compile(r"[ \t]+")
BLANKLINE_PATTERN = re.compile(r"\n{3,}")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def looks_like_html(text: str) -> bool:
    return bool(HTML_MARKER_PATTERN.search(text))


def strip_html(text: str) -> str:
    if not looks_like_html(text):
        return text

    if BeautifulSoup is not None:
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        return soup.get_text("\n")

    text = re.sub(r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return html.unescape(text)


def normalize_line(line: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", line).strip()


def is_probable_noise(line: str) -> bool:
    compact = normalize_line(line)
    if not compact:
        return True

    if COUNTRY_PICKER_PATTERN.search(compact):
        return True

    lower = compact.lower()
    if len(compact) <= 2 and compact not in {"C#", "C++"}:
        return True

    if compact.count("*") >= 2 and len(compact.split()) <= 8:
        return True

    if re.fullmatch(r"[\W_]+", compact):
        return True

    for pattern in LINE_NOISE_PATTERNS:
        if pattern.search(lower):
            return True

    return False


def normalize_raw_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_html(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = BLANKLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    normalized = normalize_raw_text(text)
    if not normalized:
        return ""

    cleaned_lines: List[str] = []
    previous = ""

    for raw_line in normalized.split("\n"):
        line = normalize_line(raw_line)
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        if is_probable_noise(line):
            continue

        if line == previous:
            continue

        cleaned_lines.append(line)
        previous = line

    paragraphs: List[str] = []
    buffer: List[str] = []
    for line in cleaned_lines:
        if line == "":
            if buffer:
                paragraphs.append(" ".join(buffer).strip())
                buffer = []
            continue
        buffer.append(line)

    if buffer:
        paragraphs.append(" ".join(buffer).strip())

    filtered_paragraphs = []
    for paragraph in paragraphs:
        if is_probable_noise(paragraph):
            continue
        filtered_paragraphs.append(paragraph)

    text = "\n".join(paragraph for paragraph in filtered_paragraphs if paragraph)
    text = re.sub(r" ?([,;:.!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = BLANKLINE_PATTERN.sub("\n\n", text)
    return text.strip()


def run_clean(input_file: str, output_file: str) -> Dict[str, int]:
    ensure_parent(output_file)
    stats = {
        "jobs_seen": 0,
        "jobs_written": 0,
        "jobs_empty_after_cleaning": 0,
    }

    with open(input_file, "r", encoding="utf-8") as source, open(
        output_file, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue

            stats["jobs_seen"] += 1
            job = json.loads(line)
            raw_text = str(job.get("raw_text", ""))
            cleaned_text = clean_text(raw_text)
            if not cleaned_text:
                stats["jobs_empty_after_cleaning"] += 1

            record = {
                "job_id": job.get("job_id", ""),
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "url": job.get("url", ""),
                "final_landing_url": job.get("final_landing_url", ""),
                "posted_date": job.get("posted_date", ""),
                "source_repo": job.get("source_repo", ""),
                "source_file": job.get("source_file", ""),
                "source_format": job.get("source_format", ""),
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["jobs_written"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw job descriptions.")
    parser.add_argument("--input-file", default="data/processed/jobs_ingested.jsonl")
    parser.add_argument("--output-file", default="data/processed/jobs_cleaned.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_clean(args.input_file, args.output_file)
    print(json.dumps(summary, indent=2))
