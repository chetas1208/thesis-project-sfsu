from __future__ import annotations

import argparse
from collections import Counter

from src.eval.common import align_records, compute_prf, extract_entities, write_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate document-level set F1.")
    parser.add_argument("--gold-file", required=True, help="Gold jsonl file.")
    parser.add_argument("--prediction-file", required=True, help="Prediction jsonl file.")
    parser.add_argument("--output-file", required=True, help="Metric output json file.")
    parser.add_argument("--experiment-name", required=True, help="Experiment name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    args = parser.parse_args()

    aligned, leftovers = align_records(args.gold_file, args.prediction_file)
    counts = Counter(tp=0, predicted=0, gold=0)

    for gold_record, pred_record in aligned:
        gold_set = {(entity.label, entity.text.lower()) for entity in extract_entities(gold_record)}
        pred_set = {(entity.label, entity.text.lower()) for entity in extract_entities(pred_record)}
        counts.update(tp=len(gold_set & pred_set), predicted=len(pred_set), gold=len(gold_set))

    metrics = {
        "experiment_name": args.experiment_name,
        "metric_type": "doc_set",
        "split": args.split,
        "leftover_prediction_records": len(leftovers),
        "document_set": compute_prf(counts["tp"], counts["predicted"], counts["gold"]),
    }
    write_metrics(args.output_file, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
