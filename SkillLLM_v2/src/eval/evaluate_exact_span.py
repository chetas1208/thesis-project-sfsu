from __future__ import annotations

import argparse
from collections import Counter

from src.eval.common import align_records, compute_prf, exact_score, extract_entities, greedy_match, per_label_counter, write_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate strict exact-span precision, recall, and F1.")
    parser.add_argument("--gold-file", required=True, help="Gold jsonl file.")
    parser.add_argument("--prediction-file", required=True, help="Prediction jsonl file.")
    parser.add_argument("--output-file", required=True, help="Metric output json file.")
    parser.add_argument("--experiment-name", required=True, help="Experiment name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    args = parser.parse_args()

    aligned, leftovers = align_records(args.gold_file, args.prediction_file)
    total = Counter(tp=0, predicted=0, gold=0)
    per_label = per_label_counter()

    for gold_record, pred_record in aligned:
        gold_entities = extract_entities(gold_record)
        pred_entities = extract_entities(pred_record)
        matches, _, _ = greedy_match(gold_entities, pred_entities, exact_score)

        total.update(tp=len(matches), predicted=len(pred_entities), gold=len(gold_entities))
        for entity in gold_entities:
            per_label[entity.label].update(gold=1)
        for entity in pred_entities:
            per_label[entity.label].update(predicted=1)
        for gold_index, pred_index, _ in matches:
            per_label[gold_entities[gold_index].label].update(tp=1)

    metrics = {
        "experiment_name": args.experiment_name,
        "metric_type": "exact",
        "split": args.split,
        "leftover_prediction_records": len(leftovers),
        "overall": compute_prf(total["tp"], total["predicted"], total["gold"]),
        "per_label": {
            label: compute_prf(counter["tp"], counter["predicted"], counter["gold"])
            for label, counter in per_label.items()
        },
    }
    write_metrics(args.output_file, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
