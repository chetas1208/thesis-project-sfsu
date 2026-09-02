import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common import exact_span_metrics, load_config, load_env_file, read_json_records, resolve_path, write_json


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def length_bucket(record):
    token_count = len(record.get("tokens", []))
    if token_count <= 10:
        return "short"
    if token_count <= 20:
        return "medium"
    return "long"


def build_skill_frequency(train_dev_records):
    counter = Counter()
    for record in train_dev_records:
        for span in record.get("spans", []):
            counter[span["text"].strip().lower()] += 1
    return counter


def skill_bucket(skill_text, frequency_counter):
    frequency = frequency_counter.get(skill_text.strip().lower(), 0)
    if frequency >= 10:
        return "head"
    if frequency >= 3:
        return "mid"
    if frequency >= 1:
        return "tail"
    return "unseen"


def span_slice_rows(model_name, canonical_test, predictions):
    rows = []
    dimensions = {
        "section_name": lambda record: record.get("section_name") or "unknown",
        "role_hint": lambda record: record.get("role_hint") or "unknown",
        "sentence_length_bucket": length_bucket,
    }
    prediction_map = {record["id"]: record for record in predictions}
    for dimension, resolver in dimensions.items():
        grouped_gold = defaultdict(list)
        grouped_pred = defaultdict(list)
        for record in canonical_test:
            group_value = resolver(record)
            grouped_gold[group_value].append(record)
            grouped_pred[group_value].append(prediction_map.get(record["id"], {"id": record["id"], "predicted_spans": []}))
        for value, gold_records in grouped_gold.items():
            metrics = exact_span_metrics(gold_records, grouped_pred[value])
            rows.append(
                {
                    "model": model_name,
                    "dimension": dimension,
                    "value": value,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "support": len(gold_records),
                }
            )
    return rows


def span_frequency_rows(model_name, canonical_test, predictions, frequency_counter):
    prediction_map = {record["id"]: record for record in predictions}
    bucket_rows = []
    counts = defaultdict(lambda: {"matched": 0, "total": 0})
    for record in canonical_test:
        predicted = {
            (int(span["start"]), int(span["end"]), span["label"])
            for span in prediction_map.get(record["id"], {"predicted_spans": []}).get("predicted_spans", [])
        }
        for span in record.get("spans", []):
            bucket = skill_bucket(span["text"], frequency_counter)
            counts[bucket]["total"] += 1
            if (int(span["start"]), int(span["end"]), span["label"]) in predicted:
                counts[bucket]["matched"] += 1
    for bucket, payload in counts.items():
        bucket_rows.append(
            {
                "model": model_name,
                "dimension": "skill_frequency_bucket",
                "value": bucket,
                "recall": (payload["matched"] / payload["total"]) if payload["total"] else 0.0,
                "support": payload["total"],
            }
        )
    return bucket_rows


def span_error_rows(model_name, canonical_test, predictions):
    prediction_map = {record["id"]: record for record in predictions}
    false_negatives = Counter()
    false_positives = Counter()
    for record in canonical_test:
        gold = {(int(span["start"]), int(span["end"]), span["label"], span["text"]) for span in record.get("spans", [])}
        predicted = {
            (
                int(span["start"]),
                int(span["end"]),
                span["label"],
                record["text"][int(span["start"]) : int(span["end"]) + 1],
            )
            for span in prediction_map.get(record["id"], {"predicted_spans": []}).get("predicted_spans", [])
        }
        for span in gold - predicted:
            false_negatives[span[3]] += 1
        for span in predicted - gold:
            false_positives[span[3]] += 1
    rows = []
    for span_text, count in false_negatives.most_common(50):
        rows.append({"model": model_name, "error_type": "false_negative", "item": span_text, "count": count})
    for span_text, count in false_positives.most_common(50):
        rows.append({"model": model_name, "error_type": "false_positive", "item": span_text, "count": count})
    return rows


def retrieval_pair_slice_rows(model_name, pair_scores, sentence_meta):
    rows = []
    dimensions = ["section_name", "role_hint", "sentence_length_bucket"]
    for dimension in dimensions:
        grouped = defaultdict(list)
        for row in pair_scores:
            metadata = sentence_meta.get(row["sentence"], {})
            value = metadata.get(dimension) or "unknown"
            grouped[value].append(row)
        for value, group_rows in grouped.items():
            tp = sum(1 for row in group_rows if row["gold_label"] == 1 and row["predicted_label"] == 1)
            fp = sum(1 for row in group_rows if row["gold_label"] == 0 and row["predicted_label"] == 1)
            fn = sum(1 for row in group_rows if row["gold_label"] == 1 and row["predicted_label"] == 0)
            tn = sum(1 for row in group_rows if row["gold_label"] == 0 and row["predicted_label"] == 0)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            accuracy = (tp + tn) / len(group_rows) if group_rows else 0.0
            rows.append(
                {
                    "model": model_name,
                    "dimension": dimension,
                    "value": value,
                    "pair_precision": precision,
                    "pair_recall": recall,
                    "pair_f1": f1,
                    "pair_accuracy": accuracy,
                    "support": len(group_rows),
                }
            )
    return rows


def retrieval_frequency_rows(model_name, pair_scores, frequency_counter):
    grouped = defaultdict(list)
    for row in pair_scores:
        bucket = skill_bucket(row["skill"], frequency_counter)
        grouped[bucket].append(row)
    rows = []
    for bucket, group_rows in grouped.items():
        positives = [row for row in group_rows if row["gold_label"] == 1]
        if not positives:
            continue
        matched = sum(1 for row in positives if row["predicted_label"] == 1)
        rows.append(
            {
                "model": model_name,
                "dimension": "skill_frequency_bucket",
                "value": bucket,
                "pair_recall": matched / len(positives),
                "support": len(positives),
            }
        )
    return rows


def retrieval_error_rows(model_name, pair_scores):
    false_negatives = Counter(row["skill"] for row in pair_scores if row["gold_label"] == 1 and row["predicted_label"] == 0)
    false_positives = Counter(row["skill"] for row in pair_scores if row["gold_label"] == 0 and row["predicted_label"] == 1)
    rows = []
    for skill, count in false_negatives.most_common(50):
        rows.append({"model": model_name, "error_type": "false_negative", "item": skill, "count": count})
    for skill, count in false_positives.most_common(50):
        rows.append({"model": model_name, "error_type": "false_positive", "item": skill, "count": count})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run benchmark result analysis for span and retrieval models.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    results_root = resolve_path(config["paths"]["results_dir"])
    output_dir = results_root / "analysis" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_dir = resolve_path(config["paths"]["canonical_dir"])
    canonical_train = read_json_records(canonical_dir / "train.jsonl") if (canonical_dir / "train.jsonl").exists() else []
    canonical_dev = read_json_records(canonical_dir / "dev.jsonl") if (canonical_dir / "dev.jsonl").exists() else []
    canonical_test = read_json_records(canonical_dir / "test.jsonl") if (canonical_dir / "test.jsonl").exists() else []
    sentence_meta = {
        record["text"]: {
            "section_name": record.get("section_name") or "unknown",
            "role_hint": record.get("role_hint") or "unknown",
            "sentence_length_bucket": length_bucket(record),
        }
        for record in canonical_test
    }
    frequency_counter = build_skill_frequency(canonical_train + canonical_dev)

    summary_path = results_root / "summary.csv"
    retrieval_summary_path = results_root / "retrieval_summary.csv"
    summary_rows = []
    retrieval_summary_rows = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
    if retrieval_summary_path.exists():
        with retrieval_summary_path.open("r", encoding="utf-8", newline="") as handle:
            retrieval_summary_rows = list(csv.DictReader(handle))

    if summary_rows:
        write_csv(output_dir / "exact_span_leaderboard.csv", summary_rows[0].keys(), summary_rows)
    if retrieval_summary_rows:
        write_csv(output_dir / "retrieval_leaderboard.csv", retrieval_summary_rows[0].keys(), retrieval_summary_rows)

    span_slice_metrics = []
    span_frequency_metrics = []
    retrieval_slice_metrics = []
    retrieval_frequency_metrics = []
    error_rows = []

    for model_key, model_cfg in config["models"].items():
        task_type = model_cfg.get("task_type", "span_extraction")
        results_dir = resolve_path(model_cfg["results_dir"])
        if task_type == "span_extraction":
            predictions_path = results_dir / "predictions.jsonl"
            if not predictions_path.exists() or not canonical_test:
                continue
            predictions = read_json_records(predictions_path)
            span_slice_metrics.extend(span_slice_rows(model_key, canonical_test, predictions))
            span_frequency_metrics.extend(span_frequency_rows(model_key, canonical_test, predictions, frequency_counter))
            error_rows.extend(span_error_rows(model_key, canonical_test, predictions))
        else:
            pair_scores_path = results_dir / "pair_scores.jsonl"
            if not pair_scores_path.exists():
                continue
            pair_scores = read_json_records(pair_scores_path)
            retrieval_slice_metrics.extend(retrieval_pair_slice_rows(model_key, pair_scores, sentence_meta))
            retrieval_frequency_metrics.extend(retrieval_frequency_rows(model_key, pair_scores, frequency_counter))
            error_rows.extend(retrieval_error_rows(model_key, pair_scores))

    if span_slice_metrics:
        write_csv(output_dir / "span_slice_metrics.csv", span_slice_metrics[0].keys(), span_slice_metrics)
    if span_frequency_metrics:
        write_csv(output_dir / "span_frequency_metrics.csv", span_frequency_metrics[0].keys(), span_frequency_metrics)
    if retrieval_slice_metrics:
        write_csv(output_dir / "retrieval_slice_metrics.csv", retrieval_slice_metrics[0].keys(), retrieval_slice_metrics)
    if retrieval_frequency_metrics:
        write_csv(output_dir / "retrieval_frequency_metrics.csv", retrieval_frequency_metrics[0].keys(), retrieval_frequency_metrics)
    if error_rows:
        write_csv(output_dir / "top_errors.csv", error_rows[0].keys(), error_rows)

    report_lines = [
        "# Benchmark Analysis",
        "",
        f"- Span leaderboard rows: {len(summary_rows)}",
        f"- Retrieval leaderboard rows: {len(retrieval_summary_rows)}",
        f"- Canonical test records: {len(canonical_test)}",
        f"- Span slice rows: {len(span_slice_metrics)}",
        f"- Retrieval slice rows: {len(retrieval_slice_metrics)}",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(
        output_dir / "summary.json",
        {
            "span_leaderboard_rows": len(summary_rows),
            "retrieval_leaderboard_rows": len(retrieval_summary_rows),
            "canonical_test_records": len(canonical_test),
            "span_slice_rows": len(span_slice_metrics),
            "retrieval_slice_rows": len(retrieval_slice_metrics),
        },
    )
    print(f"Benchmark analysis written under {output_dir}")


if __name__ == "__main__":
    main()
