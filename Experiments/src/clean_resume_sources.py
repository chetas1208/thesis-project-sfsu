import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List

from build_annotation_queue import (
    ID_KEYS,
    ROLE_KEYS,
    TEXT_KEYS,
    build_sections,
    combine_section_text,
    discover_source_files,
    infer_source,
    load_source_records,
    normalize_sections,
    _get_value,
    _stringify,
)
from common import ensure_dir, load_config, load_env_file, resolve_path, write_json, write_jsonl


SECTION_ALIASES = {
    "summary": {"summary", "professional summary", "profile", "objective", "career objective"},
    "skills": {"skills", "technical skills", "key skills", "core skills", "competencies"},
    "experience": {
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "career history",
    },
    "projects": {"projects", "project experience", "academic projects", "personal projects"},
    "education": {"education", "academic background", "qualifications"},
    "certifications": {"certifications", "certificates", "licenses", "license"},
    "achievements": {"achievements", "awards", "accomplishments"},
    "languages": {"languages", "language"},
}

PAGE_NOISE_PATTERNS = (
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*resume\s*$", re.IGNORECASE),
)


def normalized_line_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def is_heading(line: str) -> str | None:
    normalized = normalized_line_key(line.rstrip(":"))
    if not normalized:
        return None
    if len(normalized.split()) > 4:
        return None
    for section_name, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section_name
    return None


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if all(pattern.match(stripped) for pattern in (re.compile(r"^[_\-\s|]{3,}$"),)):
        return True
    return any(pattern.match(stripped) for pattern in PAGE_NOISE_PATTERNS)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    cleaned_lines = []
    blank_streak = 0
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if is_noise_line(line):
            continue
        if not line:
            blank_streak += 1
            if blank_streak <= 1:
                cleaned_lines.append("")
            continue
        blank_streak = 0
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text


def detect_sections_from_text(text: str) -> List[Dict]:
    sections: List[Dict] = []
    current_name = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_name, current_lines
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"section_name": current_name or "resume", "text": body})
        current_name = None
        current_lines = []

    for line in text.splitlines():
        heading_name = is_heading(line)
        if heading_name:
            flush()
            current_name = heading_name
            continue
        if line.strip():
            current_lines.append(line)
        elif current_lines:
            current_lines.append("")
    flush()
    return sections


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clean_record(record: Dict, source_file: Path, running_index: int) -> Dict | None:
    sections = normalize_sections(record.get("sections")) or build_sections(record)
    raw_text = _stringify(_get_value(record, TEXT_KEYS)) or combine_section_text(sections)
    if not raw_text:
        return None

    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        return None

    cleaned_sections = []
    for section in sections:
        section_text = clean_text(section.get("text", ""))
        if section_text:
            cleaned_sections.append(
                {
                    "section_name": str(section["section_name"]).casefold().replace(" ", "_"),
                    "text": section_text,
                }
            )
    if not cleaned_sections:
        cleaned_sections = detect_sections_from_text(cleaned_text)
    if not cleaned_sections:
        cleaned_sections = [{"section_name": "resume", "text": cleaned_text}]

    resume_id = _stringify(_get_value(record, ID_KEYS)) or f"{source_file.stem}_{running_index:06d}"
    role_hint = _stringify(_get_value(record, ROLE_KEYS)) or None
    source_name = infer_source(source_file, record)

    return {
        "resume_id": resume_id,
        "source": source_name,
        "role_hint": role_hint,
        "text": cleaned_text,
        "sections": cleaned_sections,
        "raw_metadata": {
            "input_file": str(source_file),
            "record_index": running_index,
            "text_hash": text_hash(cleaned_text),
            "original_text_length": len(raw_text),
            "cleaned_text_length": len(cleaned_text),
        },
    }


def summarize(cleaned_records: Iterable[Dict], source_files: List[Path], duplicate_count: int, skipped_count: int) -> Dict:
    records = list(cleaned_records)
    source_counts: Dict[str, int] = {}
    for record in records:
        source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
    return {
        "source_file_count": len(source_files),
        "cleaned_record_count": len(records),
        "duplicate_record_count": duplicate_count,
        "skipped_record_count": skipped_count,
        "source_counts": source_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and deduplicate raw resume sources for annotation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-dir", default="data/raw/source")
    parser.add_argument("--output", default="data/raw/cleaned/cleaned_resumes.jsonl")
    parser.add_argument("--summary-output", default="data/raw/cleaned/cleaning_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    del config

    input_dir = resolve_path(args.input_dir)
    output_path = resolve_path(args.output)
    summary_output = resolve_path(args.summary_output)

    source_files = discover_source_files(input_dir)
    if not source_files:
        print(f"No raw source files found under {input_dir}")
        return

    cleaned_records = []
    seen_hashes = set()
    duplicate_count = 0
    skipped_count = 0
    running_index = 1

    for source_file in source_files:
        for record in load_source_records(source_file):
            cleaned = clean_record(record, source_file, running_index)
            running_index += 1
            if cleaned is None:
                skipped_count += 1
                continue
            record_hash = cleaned["raw_metadata"]["text_hash"]
            if record_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(record_hash)
            cleaned_records.append(cleaned)
            if args.limit is not None and len(cleaned_records) >= args.limit:
                break
        if args.limit is not None and len(cleaned_records) >= args.limit:
            break

    summary = summarize(cleaned_records, source_files, duplicate_count, skipped_count)

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        if cleaned_records:
            example = cleaned_records[0]
            print(
                f"Example: resume_id={example['resume_id']} source={example['source']} "
                f"sections={len(example['sections'])}"
            )
        return

    ensure_dir(output_path.parent)
    write_jsonl(output_path, cleaned_records)
    write_json(summary_output, summary)
    print(f"Cleaned records written to {output_path}")
    print(f"Cleaning summary written to {summary_output}")


if __name__ == "__main__":
    main()
