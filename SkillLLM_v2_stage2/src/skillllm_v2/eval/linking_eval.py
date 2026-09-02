from __future__ import annotations


def linking_accuracy(gold_ids: list[str], predicted_ids: list[str]) -> float:
    if not gold_ids:
        return 0.0
    return sum(g == p for g, p in zip(gold_ids, predicted_ids)) / len(gold_ids)
