import argparse
import csv
import json
import os
import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def read_records(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def average_confidence(spans: Sequence[Dict[str, object]]) -> float:
    if not spans:
        return 0.0
    return sum(float(span.get("confidence", 0.0)) for span in spans) / len(spans)


def choose_bucket(record: Dict[str, object]) -> str:
    label_mix = set(record.get("label_mix", []))
    if record["span_count"] == 0 and record["token_count"] >= 12:
        return "unlabeled_control"
    if {"hard_skill", "soft_skill"}.issubset(label_mix):
        return "hard_soft_mix"
    if record["has_rare_skill"]:
        return "rare_skill"
    if record["span_count"] >= 2 and record["avg_confidence"] >= 0.93:
        return "high_confidence_multi_skill"
    if 0 < record["avg_confidence"] < 0.90:
        return "low_confidence"
    if record["span_count"] >= 1 and record["token_count"] >= 35:
        return "long_sentence"
    return "backfill"


def select_bucket(records: List[Dict[str, object]], predicate, quota: int, used_ids: set, seed: int) -> List[Tuple[str, Dict[str, object]]]:
    candidates = [record for record in records if predicate(record) and record["id"] not in used_ids]
    random.Random(seed).shuffle(candidates)
    selected = candidates[:quota]
    for record in selected:
        used_ids.add(record["id"])
    return [(record["sampling_bucket"], record) for record in selected]


def build_review_rows(records: Iterable[Tuple[str, Dict[str, object]]], csv_output: str, json_output: str) -> None:
    ensure_parent(csv_output)
    ensure_parent(json_output)

    rows = list(records)
    with open(json_output, "w", encoding="utf-8") as json_handle:
        for bucket, record in rows:
            enriched = dict(record)
            enriched["sampling_bucket"] = bucket
            if isinstance(enriched.get("label_mix"), set):
                enriched["label_mix"] = sorted(enriched["label_mix"])
            json_handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    with open(csv_output, "w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(
            [
                "id",
                "job_id",
                "title",
                "company",
                "sentence",
                "predicted_spans",
                "predicted_labels",
                "confidence",
                "sampling_bucket",
                "review_decision",
                "corrected_spans",
                "corrected_labels",
                "notes",
            ]
        )
        for bucket, record in rows:
            spans = list(record.get("spans", []))
            writer.writerow(
                [
                    record.get("id", ""),
                    record.get("job_id", ""),
                    record.get("title", ""),
                    record.get("company", ""),
                    record.get("sentence", ""),
                    " | ".join(str(span.get("text", "")) for span in spans),
                    " | ".join(str(span.get("label", "")) for span in spans),
                    " | ".join(str(span.get("confidence", "")) for span in spans),
                    bucket,
                    "",
                    "",
                    "",
                    "",
                ]
            )


def enrich_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    skill_frequency = Counter()
    for record in records:
        for span in record.get("spans", []):
            skill_frequency[str(span.get("normalized_skill", span.get("text", "")).lower())] += 1

    enriched: List[Dict[str, object]] = []
    for record in records:
        spans = list(record.get("spans", []))
        labels = {span.get("label") for span in spans}
        record_copy = dict(record)
        record_copy["avg_confidence"] = average_confidence(spans)
        record_copy["span_count"] = len(spans)
        record_copy["token_count"] = len(record.get("tokens", []))
        record_copy["has_rare_skill"] = any(
            skill_frequency[str(span.get("normalized_skill", span.get("text", "")).lower())] <= 3 for span in spans
        )
        record_copy["sampling_bucket"] = choose_bucket(record_copy)
        record_copy["label_mix"] = labels
        enriched.append(record_copy)
    return enriched


def run_prepare_gold(
    input_file: str,
    json_output: str,
    csv_output: str,
    sample_size: int = 250,
    seed: int = 42,
    mode: str = "sampled",
) -> Dict[str, object]:
    records = read_records(input_file)
    enriched = enrich_records(records)

    if mode == "full":
        full_rows = sorted(enriched, key=lambda record: (record["sampling_bucket"], record["job_id"], record["id"]))
        build_review_rows([(record["sampling_bucket"], record) for record in full_rows], csv_output, json_output)
        bucket_counts = Counter(record["sampling_bucket"] for record in full_rows)
        return {
            "sample_size": len(full_rows),
            "bucket_counts": dict(bucket_counts),
            "mode": mode,
        }

    quotas = {
        "low_confidence": max(20, int(sample_size * 0.20)),
        "high_confidence_multi_skill": max(30, int(sample_size * 0.24)),
        "rare_skill": max(20, int(sample_size * 0.16)),
        "hard_soft_mix": max(15, int(sample_size * 0.12)),
        "long_sentence": max(20, int(sample_size * 0.16)),
        "unlabeled_control": max(20, int(sample_size * 0.12)),
    }
    total_quota = sum(quotas.values())
    if total_quota > sample_size:
        overflow = total_quota - sample_size
        quotas["unlabeled_control"] = max(10, quotas["unlabeled_control"] - overflow)

    used_ids = set()
    selections: List[Tuple[str, Dict[str, object]]] = []

    def add(bucket: str, predicate, quota_seed: int) -> None:
        count = quotas.get(bucket, 0)
        picks = select_bucket(enriched, predicate, count, used_ids, quota_seed)
        selections.extend((bucket, record) for _, record in picks)

    add("low_confidence", lambda record: 0 < record["avg_confidence"] < 0.90, seed + 1)
    add(
        "high_confidence_multi_skill",
        lambda record: record["span_count"] >= 2 and record["avg_confidence"] >= 0.93,
        seed + 2,
    )
    add("rare_skill", lambda record: record["has_rare_skill"], seed + 3)
    add(
        "hard_soft_mix",
        lambda record: {"hard_skill", "soft_skill"}.issubset(record["label_mix"]),
        seed + 4,
    )
    add(
        "long_sentence",
        lambda record: record["span_count"] >= 1 and record["token_count"] >= 35,
        seed + 5,
    )
    add("unlabeled_control", lambda record: record["span_count"] == 0 and record["token_count"] >= 12, seed + 6)

    remaining = sample_size - len(selections)
    if remaining > 0:
        leftovers = [record for record in enriched if record["id"] not in used_ids]
        leftovers.sort(key=lambda record: (-record["span_count"], -record["token_count"], record["id"]))
        for record in leftovers[:remaining]:
            selections.append(("backfill", record))
            used_ids.add(record["id"])

    build_review_rows(selections, csv_output, json_output)
    bucket_counts = Counter(bucket for bucket, _ in selections)
    return {
        "sample_size": len(selections),
        "bucket_counts": dict(bucket_counts),
        "mode": mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a high-value review subset for gold annotation.")
    parser.add_argument("--input-file", default="data/silver/skillspan_style_silver_bio.jsonl")
    parser.add_argument("--json-output", default="data/gold_review/review_candidates.jsonl")
    parser.add_argument("--csv-output", default="data/gold_review/review_sheet.csv")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["sampled", "full"], default="sampled")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_prepare_gold(
        args.input_file,
        args.json_output,
        args.csv_output,
        sample_size=args.sample_size,
        seed=args.seed,
        mode=args.mode,
    )
    print(json.dumps(summary, indent=2))
