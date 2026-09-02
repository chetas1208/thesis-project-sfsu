from __future__ import annotations

import argparse
from collections import Counter

from src.eval.common import (
    align_records,
    compute_prf,
    extract_entities,
    greedy_match,
    relaxed_score,
    text_match_score,
    write_metrics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate relaxed span and text-match metrics.")
    parser.add_argument("--gold-file", required=True, help="Gold jsonl file.")
    parser.add_argument("--prediction-file", required=True, help="Prediction jsonl file.")
    parser.add_argument("--output-file", required=True, help="Metric output json file.")
    parser.add_argument("--experiment-name", required=True, help="Experiment name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    args = parser.parse_args()

    aligned, leftovers = align_records(args.gold_file, args.prediction_file)
    relaxed_counts = Counter(tp=0, predicted=0, gold=0)
    text_counts = Counter(tp=0, predicted=0, gold=0)

    for gold_record, pred_record in aligned:
        gold_entities = extract_entities(gold_record)
        pred_entities = extract_entities(pred_record)

        relaxed_matches, _, _ = greedy_match(gold_entities, pred_entities, relaxed_score)
        text_matches, _, _ = greedy_match(gold_entities, pred_entities, text_match_score)

        relaxed_counts.update(tp=len(relaxed_matches), predicted=len(pred_entities), gold=len(gold_entities))
        text_counts.update(tp=len(text_matches), predicted=len(pred_entities), gold=len(gold_entities))

    metrics = {
        "experiment_name": args.experiment_name,
        "metric_type": "relaxed",
        "split": args.split,
        "leftover_prediction_records": len(leftovers),
        "relaxed_span": compute_prf(relaxed_counts["tp"], relaxed_counts["predicted"], relaxed_counts["gold"]),
        "text_match": compute_prf(text_counts["tp"], text_counts["predicted"], text_counts["gold"]),
    }
    write_metrics(args.output_file, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
