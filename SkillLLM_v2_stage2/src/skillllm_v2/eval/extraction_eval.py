from __future__ import annotations

from typing import Any


def evaluate_extraction(gold: list[list[dict[str, Any]]], predicted: list[list[dict[str, Any]]]) -> dict[str, float]:
    tp = fp = fn = 0
    partial_tp = 0
    token_tp = token_fp = token_fn = 0
    label_counts: dict[str, dict[str, int]] = {}
    for gold_doc, pred_doc in zip(gold, predicted):
        gold_set = {_key(item) for item in gold_doc}
        pred_set = {_key(item) for item in pred_doc}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        partial_tp += _partial_matches(gold_doc, pred_doc)
        for label in {"HARD_SKILL", "SOFT_SKILL"}:
            label_gold = {_key(item) for item in gold_doc if item.get("label") == label}
            label_pred = {_key(item) for item in pred_doc if item.get("label") == label}
            counts = label_counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})
            counts["tp"] += len(label_gold & label_pred)
            counts["fp"] += len(label_pred - label_gold)
            counts["fn"] += len(label_gold - label_pred)
        gtoks = _tokens(gold_doc)
        ptoks = _tokens(pred_doc)
        token_tp += len(gtoks & ptoks)
        token_fp += len(ptoks - gtoks)
        token_fn += len(gtoks - ptoks)
    precision, recall, f1 = _prf(tp, fp, fn)
    token_precision, token_recall, token_f1 = _prf(token_tp, token_fp, token_fn)
    partial_precision, partial_recall, partial_f1 = _prf(partial_tp, fp + tp - partial_tp, fn + tp - partial_tp)
    per_label = {label: _prf(c["tp"], c["fp"], c["fn"]) for label, c in label_counts.items()}
    macro_f1 = sum(values[2] for values in per_label.values()) / max(1, len(per_label))
    return {
        "exact_span_precision": precision,
        "exact_span_recall": recall,
        "exact_span_f1": f1,
        "partial_span_precision": partial_precision,
        "partial_span_recall": partial_recall,
        "partial_span_f1": partial_f1,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
        "hard_skill_f1": per_label.get("HARD_SKILL", (0.0, 0.0, 0.0))[2],
        "soft_skill_f1": per_label.get("SOFT_SKILL", (0.0, 0.0, 0.0))[2],
        "macro_f1": macro_f1,
        "micro_f1": f1,
    }


def _key(item: dict[str, Any]) -> tuple[int | None, int | None, str]:
    return item.get("start"), item.get("end"), item.get("label")


def _tokens(items: list[dict[str, Any]]) -> set[tuple[int, str]]:
    output = set()
    for item in items:
        if item.get("start") is None or item.get("end") is None:
            continue
        for index in range(item["start"], item["end"]):
            output.add((index, item.get("label", "HARD_SKILL")))
    return output


def _partial_matches(gold_doc: list[dict[str, Any]], pred_doc: list[dict[str, Any]]) -> int:
    matched_gold: set[int] = set()
    count = 0
    for pred in pred_doc:
        for i, gold in enumerate(gold_doc):
            if i in matched_gold or pred.get("label") != gold.get("label"):
                continue
            if pred.get("start") is None or gold.get("start") is None:
                continue
            if max(pred["start"], gold["start"]) < min(pred["end"], gold["end"]):
                matched_gold.add(i)
                count += 1
                break
    return count


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
