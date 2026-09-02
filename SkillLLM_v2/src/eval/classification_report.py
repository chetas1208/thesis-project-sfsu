from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.eval.common import align_records, extract_entities, overlap_iou, write_metrics


def build_pairs(gold_entities, pred_entities):
    pairs = []
    used_pred = set()
    for gold in gold_entities:
        best = None
        best_score = 0.0
        for pred_index, pred in enumerate(pred_entities):
            if pred_index in used_pred:
                continue
            score = overlap_iou(gold, pred)
            if score > best_score:
                best = pred_index
                best_score = score
        if best is None or best_score == 0.0:
            pairs.append((gold.label, "missing"))
        else:
            used_pred.add(best)
            pairs.append((gold.label, pred_entities[best].label))
    for pred_index, pred in enumerate(pred_entities):
        if pred_index not in used_pred:
            pairs.append(("none", pred.label))
    return pairs


def write_csv_report(csv_path: str, report: dict) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        row = {"label": label}
        row.update(values)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a per-label classification report.")
    parser.add_argument("--gold-file", required=True, help="Gold jsonl file.")
    parser.add_argument("--prediction-file", required=True, help="Prediction jsonl file.")
    parser.add_argument("--output-file", required=True, help="Metric output json file.")
    parser.add_argument("--csv-file", required=True, help="CSV output file.")
    parser.add_argument("--experiment-name", required=True, help="Experiment name.")
    args = parser.parse_args()

    try:
        from sklearn.metrics import classification_report as sklearn_classification_report
    except ImportError as exc:  # pragma: no cover - setup validation
        raise RuntimeError("scikit-learn must be installed before generating classification reports.") from exc

    aligned, leftovers = align_records(args.gold_file, args.prediction_file)
    y_true = []
    y_pred = []

    for gold_record, pred_record in aligned:
        for gold_label, pred_label in build_pairs(extract_entities(gold_record), extract_entities(pred_record)):
            y_true.append(gold_label)
            y_pred.append(pred_label)

    labels = sorted(set(y_true) | set(y_pred))
    report = sklearn_classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    payload = {
        "experiment_name": args.experiment_name,
        "metric_type": "classification",
        "leftover_prediction_records": len(leftovers),
        "labels": labels,
        "report": report,
    }
    write_metrics(args.output_file, payload)
    write_csv_report(args.csv_file, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
