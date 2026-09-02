import argparse
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from common import ensure_dir, load_config, load_env_file, read_json_records, resolve_path, write_json, write_jsonl


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[./#-][A-Za-z0-9]+)*\+*|[^\w\s]", re.UNICODE)
DELIMITER_PATTERN = re.compile(r"[,\n;|]+|\s{2,}")
SKILL_SECTION_NAMES = {"skills", "technical_skills", "core_competencies", "highlights"}
SKILL_HEADINGS = {
    "skills",
    "technical skills",
    "core competencies",
    "competencies",
    "highlights",
    "key skills",
    "technology",
    "technologies",
    "tools",
}
STOP_HEADINGS = {
    "summary",
    "objective",
    "experience",
    "work experience",
    "professional experience",
    "work history",
    "employment history",
    "education",
    "projects",
    "certifications",
    "accomplishments",
    "awards",
    "languages",
}


def normalize_phrase(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-,:")
    return text


def normalize_heading(text: str) -> str:
    normalized = normalize_phrase(text).casefold()
    normalized = re.sub(r"[^a-z0-9+#./ -]+", "", normalized)
    return normalized.strip()


def is_valid_skill_phrase(text: str) -> bool:
    normalized = normalize_phrase(text)
    if not normalized:
        return False
    if len(normalized) < 2 or len(normalized) > 48:
        return False
    if not re.search(r"[A-Za-z]", normalized):
        return False
    if normalized.count("@") or "http" in normalized.casefold():
        return False
    words = normalized.split()
    if len(words) > 6:
        return False
    return True


def split_skill_blob(text: str) -> List[str]:
    candidates = []
    for part in DELIMITER_PATTERN.split(text):
        normalized = normalize_phrase(part)
        if is_valid_skill_phrase(normalized):
            candidates.append(normalized)
    return candidates


def build_primary_skill_resources(source_path: Path) -> tuple[Dict[str, List[str]], List[str]]:
    explicit_map: Dict[str, List[str]] = {}
    global_terms: Counter[str] = Counter()
    for record in read_json_records(source_path):
        resume_id = str(record.get("ResumeID") or record.get("resume_id") or "")
        skill_terms = []
        for term in split_skill_blob(str(record.get("Skills") or "")):
            lowered = term.casefold()
            skill_terms.append(term)
            global_terms[lowered] += 1
        if resume_id and skill_terms:
            explicit_map[resume_id] = sorted(dict.fromkeys(skill_terms), key=lambda value: (-len(value), value.casefold()))
    global_skill_list = sorted(
        {term for term, count in global_terms.items() if count >= 2},
        key=lambda value: (-len(value), value),
    )
    return explicit_map, global_skill_list


def extract_resume_specific_terms(record: Dict, explicit_map: Dict[str, List[str]]) -> List[str]:
    resume_id = str(record.get("resume_id") or "")
    candidates = list(explicit_map.get(resume_id, []))

    for section in record.get("sections", []):
        section_name = str(section.get("section_name") or "").casefold()
        if section_name in SKILL_SECTION_NAMES:
            candidates.extend(split_skill_blob(str(section.get("text") or "")))

    lines = [line.strip() for line in str(record.get("text") or "").splitlines() if line.strip()]
    collect = False
    for line in lines:
        heading = normalize_heading(line.rstrip(":"))
        if heading in STOP_HEADINGS:
            collect = False
            continue
        if heading in SKILL_HEADINGS:
            collect = True
            continue
        if collect:
            if len(line.split()) <= 4 and normalize_heading(line) in STOP_HEADINGS:
                collect = False
                continue
            candidates.extend(split_skill_blob(line))

    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_phrase(candidate)
        if not is_valid_skill_phrase(normalized):
            continue
        lowered = normalized.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    deduped.sort(key=lambda value: (-len(value), value.casefold()))
    return deduped


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    current_start = None
    for index, char in enumerate(text):
        if current_start is None and not char.isspace():
            current_start = index
        boundary = False
        boundary_end = index + 1
        if char == "\n":
            boundary = True
            boundary_end = index
        elif char in ".!?;" and (index + 1 == len(text) or text[index + 1].isspace()):
            boundary = True
            boundary_end = index + 1

        if boundary and current_start is not None:
            start = current_start
            end = boundary_end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append((start, end))
            current_start = None

    if current_start is not None:
        start = current_start
        end = len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def tokenize_sentence(text: str) -> List[Dict]:
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        start = match.start()
        end = match.end() - 1
        ws = match.end() < len(text) and text[match.end()].isspace()
        tokens.append({"text": match.group(0), "start": start, "end": end, "ws": ws})
    return tokens


def compile_skill_patterns(skills: Sequence[str]) -> List[Tuple[str, re.Pattern]]:
    patterns = []
    for skill in skills:
        normalized = normalize_phrase(skill)
        if not is_valid_skill_phrase(normalized):
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(normalized)}(?![A-Za-z0-9])", re.IGNORECASE)
        patterns.append((normalized, pattern))
    patterns.sort(key=lambda item: (-len(item[0]), item[0].casefold()))
    return patterns


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def match_skill_spans(text: str, patterns: Sequence[Tuple[str, re.Pattern]]) -> List[Dict]:
    spans: List[Dict] = []
    occupied: List[tuple[int, int]] = []
    for _, pattern in patterns:
        for match in pattern.finditer(text):
            candidate = (match.start(), match.end() - 1)
            if any(spans_overlap(candidate, interval) for interval in occupied):
                continue
            occupied.append(candidate)
            spans.append(
                {
                    "start": candidate[0],
                    "end": candidate[1],
                    "label": "SKILL",
                    "text": text[candidate[0] : candidate[1] + 1],
                }
            )
    spans.sort(key=lambda item: (item["start"], item["end"]))
    return spans


def build_sentence_records(
    record: Dict,
    global_patterns: Sequence[Tuple[str, re.Pattern]],
    explicit_map: Dict[str, List[str]],
    negative_ratio: float,
    seed: int,
) -> List[Dict]:
    text = str(record.get("text") or "")
    if not text:
        return []

    local_terms = extract_resume_specific_terms(record, explicit_map)
    local_patterns = compile_skill_patterns(local_terms)
    combined_patterns = list(local_patterns)
    local_terms_lower = {term.casefold() for term, _ in local_patterns}
    for term, pattern in global_patterns:
        if term.casefold() in local_terms_lower:
            continue
        combined_patterns.append((term, pattern))

    positives = []
    negatives = []
    sentence_index = 1
    for start, end in sentence_spans(text):
        sentence_text = text[start:end]
        if len(sentence_text) < 12:
            continue
        tokens = tokenize_sentence(sentence_text)
        if not tokens:
            continue
        spans = match_skill_spans(sentence_text, combined_patterns)
        sentence_record = {
            "id": f"{record['resume_id']}_s{sentence_index}",
            "resume_id": record["resume_id"],
            "source": record.get("source"),
            "role_hint": record.get("role_hint"),
            "section_name": None,
            "text": sentence_text,
            "tokens": tokens,
            "spans": spans,
            "normalized_skills": [
                {
                    "text": span["text"],
                    "canonical_name": span["text"],
                    "taxonomy": None,
                    "skill_id": None,
                }
                for span in spans
            ],
        }
        sentence_index += 1
        if spans:
            positives.append(sentence_record)
        else:
            negatives.append(sentence_record)

    if not positives:
        return []

    negative_cap = min(len(negatives), max(1, int(len(positives) * negative_ratio)))
    rng = random.Random(f"{seed}:{record['resume_id']}")
    sampled_negatives = rng.sample(negatives, negative_cap) if negative_cap and negatives else []
    selected = positives + sampled_negatives
    selected.sort(key=lambda item: item["id"])
    return selected


def split_by_resume(records: Sequence[Dict], seed: int, ratios: Dict[str, float]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(record["resume_id"], []).append(record)

    resume_ids = sorted(grouped)
    random.Random(seed).shuffle(resume_ids)
    total = len(resume_ids)
    train_count = int(total * ratios["train"])
    dev_count = int(total * ratios["dev"])
    if total >= 3:
        train_count = max(1, train_count)
        dev_count = max(1, dev_count)
    test_count = total - train_count - dev_count
    if total >= 3 and test_count < 1:
        test_count = 1
        if train_count > dev_count and train_count > 1:
            train_count -= 1
        elif dev_count > 1:
            dev_count -= 1

    train_ids = set(resume_ids[:train_count])
    dev_ids = set(resume_ids[train_count : train_count + dev_count])
    test_ids = set(resume_ids[train_count + dev_count :])

    splits = {"train": [], "dev": [], "test": []}
    for resume_id, rows in grouped.items():
        split_name = "test"
        if resume_id in train_ids:
            split_name = "train"
        elif resume_id in dev_ids:
            split_name = "dev"
        elif resume_id in test_ids:
            split_name = "test"
        splits[split_name].extend(sorted(rows, key=lambda item: item["id"]))
    return splits


def summarize(records: Sequence[Dict], splits: Dict[str, List[Dict]]) -> Dict:
    source_counter = Counter(record.get("source") or "unknown" for record in records)
    role_counter = Counter(record.get("role_hint") or "unknown" for record in records)
    positive_counter = Counter(record.get("source") or "unknown" for record in records if record.get("spans"))
    split_summary = {}
    for split_name, rows in splits.items():
        split_summary[split_name] = {
            "sentence_count": len(rows),
            "resume_count": len({row["resume_id"] for row in rows}),
            "positive_sentence_count": sum(1 for row in rows if row.get("spans")),
            "skill_span_count": sum(len(row.get("spans", [])) for row in rows),
        }
    return {
        "sentence_count": len(records),
        "resume_count": len({record["resume_id"] for record in records}),
        "positive_sentence_count": sum(1 for record in records if record.get("spans")),
        "skill_span_count": sum(len(record.get("spans", [])) for record in records),
        "source_counts": dict(source_counter),
        "positive_source_counts": dict(positive_counter),
        "top_roles": role_counter.most_common(25),
        "splits": split_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate silver-labeled train/dev/test benchmark splits from Kaggle resumes.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default="data/raw/cleaned/cleaned_resumes.jsonl")
    parser.add_argument("--primary-source", default="data/raw/source/rayyankauchali0_resume_dataset/resumes_dataset.jsonl")
    parser.add_argument("--output-dir", default="data/raw/annotated")
    parser.add_argument("--negative-ratio", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    seed = int(config["seed"])

    cleaned_records = read_json_records(resolve_path(args.input))
    if args.limit is not None:
        cleaned_records = cleaned_records[: args.limit]

    explicit_map, global_terms = build_primary_skill_resources(resolve_path(args.primary_source))
    global_patterns = compile_skill_patterns(global_terms)

    silver_records = []
    for record in cleaned_records:
        silver_records.extend(
            build_sentence_records(
                record=record,
                global_patterns=global_patterns,
                explicit_map=explicit_map,
                negative_ratio=args.negative_ratio,
                seed=seed,
            )
        )

    if not silver_records:
        raise ValueError("No silver-labeled sentence records were generated.")

    splits = split_by_resume(silver_records, seed=seed, ratios=config["split_ratios"])
    summary = summarize(silver_records, splits)

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        example = next((record for record in silver_records if record.get("spans")), silver_records[0])
        print(
            json.dumps(
                {
                    "id": example["id"],
                    "source": example["source"],
                    "role_hint": example["role_hint"],
                    "text": example["text"],
                    "spans": example["spans"],
                },
                indent=2,
            )
        )
        return

    output_dir = ensure_dir(resolve_path(args.output_dir))
    for split_name, rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)
    write_json(output_dir / "silver_summary.json", summary)
    print(f"Silver annotations written under {output_dir}")


if __name__ == "__main__":
    main()
