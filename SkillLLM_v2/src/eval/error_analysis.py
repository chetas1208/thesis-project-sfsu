from __future__ import annotations

import argparse
import csv
from collections import Counter

from src.eval.common import align_records, exact_score, extract_entities, greedy_match, overlap_size


def entity_key(entity):
    return entity.label, entity.start, entity.end, entity.text


def main() -> int:
    parser = argparse.ArgumentParser(description="Categorize extraction errors.")
    parser.add_argument("--gold-file", required=True, help="Gold jsonl file.")
    parser.add_argument("--prediction-file", required=True, help="Prediction jsonl file.")
    parser.add_argument("--output-file", required=True, help="CSV error output.")
    args = parser.parse_args()

    aligned, _ = align_records(args.gold_file, args.prediction_file)
    rows = []

    for gold_record, pred_record in aligned:
        gold_entities = extract_entities(gold_record)
        pred_entities = extract_entities(pred_record)

        duplicate_counter = Counter(entity_key(entity) for entity in pred_entities)
        for entity, count in duplicate_counter.items():
            if count > 1:
                for _ in range(count - 1):
                    rows.append(
                        {
                            "category": "duplicate_output",
                            "id": gold_record["id"],
                            "text": gold_record["text"],
                            "gold_text": "",
                            "gold_label": "",
                            "gold_start": "",
                            "gold_end": "",
                            "pred_text": entity[3],
                            "pred_label": entity[0],
                            "pred_start": entity[1],
                            "pred_end": entity[2],
                        }
                    )

        exact_matches, used_gold, used_pred = greedy_match(gold_entities, pred_entities, exact_score)
        unmatched_gold = {index for index in range(len(gold_entities)) if index not in used_gold}
        unmatched_pred = {index for index in range(len(pred_entities)) if index not in used_pred}

        for gold_index in list(unmatched_gold):
            gold = gold_entities[gold_index]
            wrong_label_match = next(
                (
                    pred_index
                    for pred_index in unmatched_pred
                    if pred_entities[pred_index].start == gold.start and pred_entities[pred_index].end == gold.end
                ),
                None,
            )
            if wrong_label_match is not None:
                pred = pred_entities[wrong_label_match]
                rows.append(
                    {
                        "category": "wrong_label",
                        "id": gold_record["id"],
                        "text": gold_record["text"],
                        "gold_text": gold.text,
                        "gold_label": gold.label,
                        "gold_start": gold.start,
                        "gold_end": gold.end,
                        "pred_text": pred.text,
                        "pred_label": pred.label,
                        "pred_start": pred.start,
                        "pred_end": pred.end,
                    }
                )
                unmatched_gold.remove(gold_index)
                unmatched_pred.remove(wrong_label_match)

        for gold_index in list(unmatched_gold):
            gold = gold_entities[gold_index]
            overlapping_preds = [
                pred_index
                for pred_index in unmatched_pred
                if pred_entities[pred_index].label == gold.label and overlap_size(gold, pred_entities[pred_index]) > 0
            ]
            if len(overlapping_preds) > 1:
                rows.append(
                    {
                        "category": "span_splitting",
                        "id": gold_record["id"],
                        "text": gold_record["text"],
                        "gold_text": gold.text,
                        "gold_label": gold.label,
                        "gold_start": gold.start,
                        "gold_end": gold.end,
                        "pred_text": " | ".join(pred_entities[index].text for index in overlapping_preds),
                        "pred_label": gold.label,
                        "pred_start": "",
                        "pred_end": "",
                    }
                )
                unmatched_gold.remove(gold_index)
                for pred_index in overlapping_preds:
                    unmatched_pred.discard(pred_index)

        for pred_index in list(unmatched_pred):
            pred = pred_entities[pred_index]
            overlapping_gold = [
                gold_index
                for gold_index in unmatched_gold
                if gold_entities[gold_index].label == pred.label and overlap_size(gold_entities[gold_index], pred) > 0
            ]
            if len(overlapping_gold) > 1:
                rows.append(
                    {
                        "category": "span_merging",
                        "id": gold_record["id"],
                        "text": gold_record["text"],
                        "gold_text": " | ".join(gold_entities[index].text for index in overlapping_gold),
                        "gold_label": pred.label,
                        "gold_start": "",
                        "gold_end": "",
                        "pred_text": pred.text,
                        "pred_label": pred.label,
                        "pred_start": pred.start,
                        "pred_end": pred.end,
                    }
                )
                unmatched_pred.remove(pred_index)
                for gold_index in overlapping_gold:
                    unmatched_gold.discard(gold_index)

        for gold_index in list(unmatched_gold):
            gold = gold_entities[gold_index]
            overlap_pred = next(
                (
                    pred_index
                    for pred_index in unmatched_pred
                    if pred_entities[pred_index].label == gold.label and overlap_size(gold, pred_entities[pred_index]) > 0
                ),
                None,
            )
            if overlap_pred is not None:
                pred = pred_entities[overlap_pred]
                rows.append(
                    {
                        "category": "boundary_drift",
                        "id": gold_record["id"],
                        "text": gold_record["text"],
                        "gold_text": gold.text,
                        "gold_label": gold.label,
                        "gold_start": gold.start,
                        "gold_end": gold.end,
                        "pred_text": pred.text,
                        "pred_label": pred.label,
                        "pred_start": pred.start,
                        "pred_end": pred.end,
                    }
                )
                unmatched_gold.remove(gold_index)
                unmatched_pred.remove(overlap_pred)

        for pred_index in unmatched_pred:
            pred = pred_entities[pred_index]
            rows.append(
                {
                    "category": "hallucinated_skill",
                    "id": gold_record["id"],
                    "text": gold_record["text"],
                    "gold_text": "",
                    "gold_label": "",
                    "gold_start": "",
                    "gold_end": "",
                    "pred_text": pred.text,
                    "pred_label": pred.label,
                    "pred_start": pred.start,
                    "pred_end": pred.end,
                }
            )

        for gold_index in unmatched_gold:
            gold = gold_entities[gold_index]
            rows.append(
                {
                    "category": f"missed_{gold.label}",
                    "id": gold_record["id"],
                    "text": gold_record["text"],
                    "gold_text": gold.text,
                    "gold_label": gold.label,
                    "gold_start": gold.start,
                    "gold_end": gold.end,
                    "pred_text": "",
                    "pred_label": "",
                    "pred_start": "",
                    "pred_end": "",
                }
            )

    fieldnames = [
        "category",
        "id",
        "text",
        "gold_text",
        "gold_label",
        "gold_start",
        "gold_end",
        "pred_text",
        "pred_label",
        "pred_start",
        "pred_end",
    ]
    with open(args.output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
