import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_for_hash(text: str) -> str:
    tokens = TOKEN_PATTERN.findall(text.lower())
    return " ".join(tokens)


def shingled_tokens(text: str, size: int = 5) -> set:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + size]) for index in range(0, len(tokens) - size + 1)}


def jaccard_similarity(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def metadata_score(job: Dict[str, object]) -> int:
    score = len(str(job.get("cleaned_text", "")))
    for field in ("title", "company", "location", "url", "posted_date", "raw_text"):
        if str(job.get(field, "")).strip():
            score += 20
    return score


def pick_canonical(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    return max(candidates, key=metadata_score)


def serialize_group(reason: str, canonical: Dict[str, object], duplicates: List[Dict[str, object]], similarity: float = 1.0) -> Dict[str, object]:
    return {
        "reason": reason,
        "similarity": round(similarity, 4),
        "canonical_job_id": canonical.get("job_id", ""),
        "canonical_title": canonical.get("title", ""),
        "canonical_company": canonical.get("company", ""),
        "duplicate_job_ids": [item.get("job_id", "") for item in duplicates],
        "duplicate_titles": [item.get("title", "") for item in duplicates],
    }


def read_jobs(path: str) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            jobs.append(json.loads(line))
    return jobs


def exact_group_key(job: Dict[str, object]) -> str:
    cleaned = str(job.get("cleaned_text", ""))
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def normalized_group_key(job: Dict[str, object]) -> str:
    cleaned = str(job.get("cleaned_text", ""))
    return hashlib.md5(normalize_for_hash(cleaned).encode("utf-8")).hexdigest()


def run_dedup(input_file: str, output_file: str, report_file: str, near_threshold: float = 0.92) -> Dict[str, object]:
    ensure_parent(output_file)
    ensure_parent(report_file)

    jobs = read_jobs(input_file)
    duplicate_groups: List[Dict[str, object]] = []
    exact_removed = 0
    normalized_removed = 0
    near_removed = 0

    grouped_by_exact: Dict[str, List[Dict[str, object]]] = {}
    for job in jobs:
        grouped_by_exact.setdefault(exact_group_key(job), []).append(job)

    exact_deduped: List[Dict[str, object]] = []
    for group in grouped_by_exact.values():
        canonical = pick_canonical(group)
        duplicates = [item for item in group if item is not canonical]
        if duplicates:
            exact_removed += len(duplicates)
            duplicate_groups.append(serialize_group("exact_cleaned_text", canonical, duplicates))
        exact_deduped.append(canonical)

    grouped_by_normalized: Dict[str, List[Dict[str, object]]] = {}
    for job in exact_deduped:
        grouped_by_normalized.setdefault(normalized_group_key(job), []).append(job)

    normalized_deduped: List[Dict[str, object]] = []
    for group in grouped_by_normalized.values():
        canonical = pick_canonical(group)
        duplicates = [item for item in group if item is not canonical]
        if duplicates:
            normalized_removed += len(duplicates)
            duplicate_groups.append(serialize_group("normalized_text", canonical, duplicates))
        normalized_deduped.append(canonical)

    final_jobs: List[Dict[str, object]] = []
    seen_shingles: List[Tuple[Dict[str, object], set]] = []
    for job in sorted(normalized_deduped, key=lambda item: metadata_score(item), reverse=True):
        job_text = str(job.get("cleaned_text", ""))
        job_shingles = shingled_tokens(job_text)
        is_duplicate = False

        for canonical, shingles in seen_shingles:
            length_ratio = min(len(job_text), len(str(canonical.get("cleaned_text", "")))) / max(
                len(job_text), len(str(canonical.get("cleaned_text", ""))), 1
            )
            if length_ratio < 0.85:
                continue

            similarity = jaccard_similarity(job_shingles, shingles)
            if similarity >= near_threshold:
                duplicate_groups.append(serialize_group("near_duplicate", canonical, [job], similarity))
                near_removed += 1
                is_duplicate = True
                break

        if not is_duplicate:
            final_jobs.append(job)
            seen_shingles.append((job, job_shingles))

    final_jobs.sort(key=lambda item: item.get("job_id", ""))
    with open(output_file, "w", encoding="utf-8") as handle:
        for job in final_jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    report = {
        "original_count": len(jobs),
        "exact_duplicates_removed": exact_removed,
        "normalized_duplicates_removed": normalized_removed,
        "near_duplicates_removed": near_removed,
        "final_count": len(final_jobs),
        "near_duplicate_threshold": near_threshold,
        "duplicate_groups": duplicate_groups,
    }

    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate cleaned job descriptions.")
    parser.add_argument("--input-file", default="data/processed/jobs_cleaned.jsonl")
    parser.add_argument("--output-file", default="data/processed/jobs_deduped.jsonl")
    parser.add_argument("--report-file", default="reports/dedup_report.json")
    parser.add_argument("--near-threshold", type=float, default=0.92)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_dedup(args.input_file, args.output_file, args.report_file, args.near_threshold)
    print(json.dumps({k: v for k, v in summary.items() if k != "duplicate_groups"}, indent=2))
