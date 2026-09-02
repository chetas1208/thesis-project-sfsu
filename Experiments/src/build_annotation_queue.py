import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from common import ensure_dir, load_config, load_env_file, read_json_records, resolve_path, write_jsonl


TEXT_KEYS = (
    "resume_str",
    "resume_text",
    "resume",
    "text",
    "content",
    "raw_text",
)
ROLE_KEYS = ("role_hint", "category", "role", "target_role", "job_title", "title")
ID_KEYS = ("resume_id", "resumeid", "id", "uuid")
SECTION_FIELD_MAP = {
    "summary": ("summary", "profile", "objective"),
    "skills": ("skills", "technical_skills", "key_skills"),
    "experience": ("experience", "work_experience", "professional_experience"),
    "projects": ("projects", "project_experience"),
    "education": ("education", "academic_background"),
    "certifications": ("certifications", "certification", "licenses"),
}


def _casefold_mapping(record: Dict) -> Dict[str, str]:
    return {str(key).casefold(): key for key in record.keys()}


def _get_value(record: Dict, candidates: Iterable[str]):
    lowered = _casefold_mapping(record)
    for candidate in candidates:
        key = lowered.get(candidate.casefold())
        if key is None:
            continue
        value = record.get(key)
        if value in (None, ""):
            continue
        return value
    return None


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip()).strip()
    return str(value).strip()


def normalize_sections(value) -> List[Dict]:
    if not value:
        return []
    normalized = []
    if isinstance(value, dict):
        iterable = [{"section_name": key, "text": item} for key, item in value.items()]
    elif isinstance(value, list):
        iterable = value
    else:
        return []

    for section in iterable:
        if isinstance(section, str):
            section_name = "resume"
            text = section
        elif isinstance(section, dict):
            section_name = _stringify(
                _get_value(section, ("section_name", "name", "title", "label"))
            ) or "resume"
            text = _stringify(_get_value(section, ("text", "content", "body", "value")))
        else:
            continue
        if not text:
            continue
        normalized.append({"section_name": section_name.casefold().replace(" ", "_"), "text": text})
    return normalized


def infer_source(path: Path, record: Dict) -> str:
    explicit = _get_value(record, ("source", "dataset_source"))
    if explicit:
        return _stringify(explicit)
    lowered_parts = "/".join(part.casefold() for part in path.parts)
    if "rayyankauchali0" in lowered_parts:
        return "kaggle_rayyankauchali0"
    if "snehaanbhawal" in lowered_parts:
        return "kaggle_snehaanbhawal"
    return path.stem


def build_sections(record: Dict) -> List[Dict]:
    preserved_sections = normalize_sections(record.get("sections"))
    if preserved_sections:
        return preserved_sections

    sections = []
    for section_name, candidate_keys in SECTION_FIELD_MAP.items():
        value = _stringify(_get_value(record, candidate_keys))
        if value:
            sections.append({"section_name": section_name, "text": value})

    if sections:
        return sections

    raw_text = _stringify(_get_value(record, TEXT_KEYS))
    if raw_text:
        return [{"section_name": "resume", "text": raw_text}]
    return []


def combine_section_text(sections: List[Dict]) -> str:
    chunks = []
    for section in sections:
        title = str(section["section_name"]).replace("_", " ").title()
        body = section["text"].strip()
        if body:
            chunks.append(f"{title}\n{body}")
    return "\n\n".join(chunks).strip()


def load_csv_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_source_records(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        return load_csv_records(path)
    return read_json_records(path)


def discover_source_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        return []
    files = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        files.append(path)
    return files


def default_input_dir(explicit_input: str | None = None) -> Path:
    if explicit_input:
        return resolve_path(explicit_input)
    cleaned_dir = resolve_path("data/raw/cleaned")
    if discover_source_files(cleaned_dir):
        return cleaned_dir
    return resolve_path("data/raw/source")


def queue_record(record: Dict, source_file: Path, running_index: int) -> Dict | None:
    sections = build_sections(record)
    text = _stringify(_get_value(record, TEXT_KEYS)) or combine_section_text(sections)
    if not text:
        return None

    resume_id = _stringify(_get_value(record, ID_KEYS)) or f"{source_file.stem}_{running_index:06d}"
    role_hint = _stringify(_get_value(record, ROLE_KEYS)) or None
    raw_metadata = dict(record.get("raw_metadata") or {})
    raw_metadata.setdefault("input_file", str(source_file))
    raw_metadata.setdefault("record_index", running_index)

    return {
        "resume_id": resume_id,
        "source": infer_source(source_file, record),
        "role_hint": role_hint,
        "text": text,
        "sections": sections,
        "raw_metadata": raw_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an annotation queue from raw resume files.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Defaults to data/raw/cleaned when present, otherwise data/raw/source",
    )
    parser.add_argument("--output", default=None, help="Defaults to data/annotation/annotation_queue.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    input_dir = default_input_dir(args.input_dir)
    output_path = resolve_path(args.output or "data/annotation/annotation_queue.jsonl")

    source_files = discover_source_files(input_dir)
    if not source_files:
        print(f"No raw source files found under {input_dir}")
        return

    queue = []
    running_index = 1
    for source_file in source_files:
        for record in load_source_records(source_file):
            normalized = queue_record(record, source_file, running_index)
            running_index += 1
            if normalized is None:
                continue
            queue.append(normalized)
            if args.limit is not None and len(queue) >= args.limit:
                break
        if args.limit is not None and len(queue) >= args.limit:
            break

    if args.dry_run:
        print(f"Discovered {len(source_files)} source files under {input_dir}")
        print(f"Built {len(queue)} annotation queue records")
        if queue:
            example = queue[0]
            print(
                f"Example: resume_id={example['resume_id']} source={example['source']} "
                f"role_hint={example['role_hint'] or 'null'}"
            )
        return

    ensure_dir(output_path.parent)
    write_jsonl(output_path, queue)
    print(f"Annotation queue written to {output_path}")


if __name__ == "__main__":
    main()
