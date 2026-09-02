import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def confidence_bucket(score: float) -> str:
    if score >= 0.95:
        return "0.95-1.00"
    if score >= 0.90:
        return "0.90-0.949"
    if score >= 0.85:
        return "0.85-0.899"
    return "<0.85"


def run_stats(input_file: str, json_out: str, csv_out: str) -> Dict[str, object]:
    ensure_parent(json_out)
    ensure_parent(csv_out)

    stats: Dict[str, object] = {
        "num_jobs": 0,
        "num_sentences": 0,
        "num_labeled_sentences": 0,
        "total_hard_skill_spans": 0,
        "total_soft_skill_spans": 0,
        "avg_spans_per_sentence": 0.0,
        "avg_spans_per_labeled_sentence": 0.0,
        "avg_sentence_length_tokens": 0.0,
    }

    job_ids = set()
    hard_skill_counts = Counter()
    soft_skill_counts = Counter()
    confidence_counts = Counter()
    total_token_length = 0
    total_spans = 0

    with open(input_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)
            job_ids.add(record.get("job_id"))
            stats["num_sentences"] += 1

            tokens = list(record.get("tokens", []))
            total_token_length += len(tokens)
            spans = list(record.get("spans", []))

            if spans:
                stats["num_labeled_sentences"] += 1
                total_spans += len(spans)

            for span in spans:
                confidence_counts[confidence_bucket(float(span.get("confidence", 0.0)))] += 1
                normalized = str(span.get("normalized_skill", span.get("text", ""))).lower()
                if span.get("label") == "hard_skill":
                    stats["total_hard_skill_spans"] += 1
                    hard_skill_counts[normalized] += 1
                elif span.get("label") == "soft_skill":
                    stats["total_soft_skill_spans"] += 1
                    soft_skill_counts[normalized] += 1

    stats["num_jobs"] = len(job_ids)
    if stats["num_sentences"]:
        stats["avg_spans_per_sentence"] = round(total_spans / stats["num_sentences"], 4)
        stats["avg_sentence_length_tokens"] = round(total_token_length / stats["num_sentences"], 4)
    if stats["num_labeled_sentences"]:
        stats["avg_spans_per_labeled_sentence"] = round(total_spans / stats["num_labeled_sentences"], 4)

    stats["most_common_hard_skills"] = hard_skill_counts.most_common(20)
    stats["most_common_soft_skills"] = soft_skill_counts.most_common(20)
    stats["confidence_distribution"] = dict(confidence_counts)

    with open(json_out, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    with open(csv_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            if isinstance(value, list):
                value = "; ".join(f"{item}:{count}" for item, count in value)
            elif isinstance(value, dict):
                value = "; ".join(f"{item}:{count}" for item, count in sorted(value.items()))
            writer.writerow([key, value])

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset statistics.")
    parser.add_argument("--input-file", default="data/silver/skillspan_style_silver_bio.jsonl")
    parser.add_argument("--json-out", default="reports/dataset_stats.json")
    parser.add_argument("--csv-out", default="reports/dataset_stats.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_stats(args.input_file, args.json_out, args.csv_out)
    print(json.dumps(summary, indent=2))
