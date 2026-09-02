import argparse
import csv
import json
import os
from typing import Dict, List, Sequence


NONE_MARKERS = {"__NONE__", "NONE", "[NONE]"}
ACCEPT_MARKERS = {"accept_predicted", "accept", "accepted"}
REJECT_MARKERS = {"reject_all", "reject", "negative"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_silver_records(path: str) -> Dict[str, Dict[str, object]]:
    records: Dict[str, Dict[str, object]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["id"])] = record
    return records


def split_pipe(value: str) -> List[str]:
    if not value.strip():
        return []
    if value.strip().upper() in NONE_MARKERS:
        return []
    return [piece.strip() for piece in value.split("|") if piece.strip()]


def align_spans(sentence: str, span_texts: Sequence[str], labels: Sequence[str]) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    cursor = 0
    for text, label in zip(span_texts, labels):
        start = sentence.find(text, cursor)
        if start == -1:
            start = sentence.find(text)
        if start == -1:
            raise ValueError(f"Could not align corrected span '{text}'")
        end = start + len(text)
        spans.append(
            {
                "text": text,
                "label": label,
                "start": start,
                "end": end,
                "confidence": 1.0,
                "source_method": "human_review",
                "normalized_skill": text.lower(),
            }
        )
        cursor = end
    return spans


def build_reviewed_records(review_csv: str, silver_records: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    gold_records: List[Dict[str, object]] = []
    with open(review_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            review_decision = row.get("review_decision", "").strip().lower()
            raw_spans = row.get("corrected_spans", "").strip()
            raw_labels = row.get("corrected_labels", "").strip()
            notes = row.get("notes", "").strip()
            reviewed = bool(raw_spans or raw_labels or notes)
            corrected_spans = split_pipe(raw_spans)
            corrected_labels = split_pipe(raw_labels)

            if review_decision in ACCEPT_MARKERS and not corrected_spans and not corrected_labels:
                corrected_spans = split_pipe(row.get("predicted_spans", ""))
                corrected_labels = split_pipe(row.get("predicted_labels", ""))
                reviewed = True

            if review_decision in REJECT_MARKERS and not raw_spans and not raw_labels:
                reviewed = True
                corrected_spans = []
                corrected_labels = []

            if not reviewed:
                continue
            if len(corrected_spans) != len(corrected_labels):
                raise ValueError(f"Mismatched corrected spans/labels for row {row.get('id')}")

            base = silver_records.get(row["id"])
            if base is None:
                raise KeyError(f"Missing silver source record for {row['id']}")

            sentence = str(base["sentence"])
            corrected = align_spans(sentence, corrected_spans, corrected_labels)

            gold_record = dict(base)
            gold_record["spans"] = corrected
            gold_records.append(gold_record)

    return gold_records


def tokenize(sentence: str) -> List[Dict[str, int]]:
    import re

    pattern = re.compile(r"C\+\+|C#|Node\.js|\.NET|A/B|[\w]+(?:[./+#-][\w]+)*|[^\w\s]", re.UNICODE)
    return [{"start": match.start(), "end": match.end(), "text": match.group()} for match in pattern.finditer(sentence)]


def add_bio_fields(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for record in records:
        sentence = str(record["sentence"])
        token_offsets = tokenize(sentence)
        tags = ["O"] * len(token_offsets)
        for span in record["spans"]:
            covered = [
                idx
                for idx, token in enumerate(token_offsets)
                if token["start"] >= int(span["start"]) and token["end"] <= int(span["end"])
            ]
            if covered:
                tags[covered[0]] = f"B-{span['label']}"
                for idx in covered[1:]:
                    tags[idx] = f"I-{span['label']}"

        enriched = dict(record)
        enriched["tokens"] = [token["text"] for token in token_offsets]
        enriched["token_offsets"] = [{"start": token["start"], "end": token["end"]} for token in token_offsets]
        enriched["ner_tags"] = tags
        output.append(enriched)
    return output


def export_jsonl(records: Sequence[Dict[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_conll(records: Sequence[Dict[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            for token, tag in zip(record.get("tokens", []), record.get("ner_tags", [])):
                handle.write(f"{token} {tag}\n")
            handle.write("\n")


def split_job_ids(records: Sequence[Dict[str, object]]) -> tuple:
    job_ids = sorted({str(record["job_id"]) for record in records})
    total = len(job_ids)
    train_end = int(total * 0.70)
    dev_end = int(total * 0.85)
    return set(job_ids[:train_end]), set(job_ids[train_end:dev_end]), set(job_ids[dev_end:])


def run_build_gold(review_csv: str, silver_input: str, output_dir: str) -> Dict[str, int]:
    ensure_dir(output_dir)
    silver_records = load_silver_records(silver_input)
    reviewed_records = add_bio_fields(build_reviewed_records(review_csv, silver_records))
    reviewed_records.sort(key=lambda record: (record["job_id"], record["id"]))

    export_jsonl(reviewed_records, os.path.join(output_dir, "reviewed.jsonl"))
    export_conll(reviewed_records, os.path.join(output_dir, "reviewed.conll"))

    train_jobs, dev_jobs, test_jobs = split_job_ids(reviewed_records)
    train_records = [record for record in reviewed_records if record["job_id"] in train_jobs]
    dev_records = [record for record in reviewed_records if record["job_id"] in dev_jobs]
    test_records = [record for record in reviewed_records if record["job_id"] in test_jobs]

    export_jsonl(train_records, os.path.join(output_dir, "train.jsonl"))
    export_jsonl(dev_records, os.path.join(output_dir, "dev.jsonl"))
    export_jsonl(test_records, os.path.join(output_dir, "test.jsonl"))
    export_conll(train_records, os.path.join(output_dir, "train.conll"))
    export_conll(dev_records, os.path.join(output_dir, "dev.conll"))
    export_conll(test_records, os.path.join(output_dir, "test.conll"))

    return {
        "reviewed_records": len(reviewed_records),
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "test_records": len(test_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold dataset splits from reviewed CSV corrections.")
    parser.add_argument("--review-csv", default="data/gold_review/review_sheet.csv")
    parser.add_argument("--silver-input", default="data/silver/skillspan_style_silver_bio.jsonl")
    parser.add_argument("--output-dir", default="data/gold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_build_gold(args.review_csv, args.silver_input, args.output_dir)
    print(json.dumps(summary, indent=2))
