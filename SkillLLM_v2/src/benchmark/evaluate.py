from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.runtime_utils import read_jsonl, write_json


BENCHMARK_LABELS = ("SKILL", "HARD_SKILL", "SOFT_SKILL")


@dataclass(frozen=True)
class Span:
    label: str
    start: int | None
    end: int | None
    text: str


def _span_from_item(item: dict[str, Any]) -> Span:
    return Span(
        label=str(item["label"]),
        start=item.get("start"),
        end=item.get("end"),
        text=str(item["text"]),
    )


def _valid_span(span: Span) -> bool:
    return span.start is not None and span.end is not None and int(span.start) < int(span.end)


def _overlap(left: Span, right: Span) -> int:
    if not (_valid_span(left) and _valid_span(right)):
        return 0
    return max(0, min(int(left.end), int(right.end)) - max(int(left.start), int(right.start)))


def _exact_match(left: Span, right: Span) -> bool:
    return (
        _valid_span(left)
        and _valid_span(right)
        and left.label == right.label
        and int(left.start) == int(right.start)
        and int(left.end) == int(right.end)
    )


def _greedy_match(gold: list[Span], pred: list[Span], mode: str) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold_span in enumerate(gold):
        for pred_index, pred_span in enumerate(pred):
            if gold_span.label != pred_span.label:
                continue
            if mode == "exact":
                score = 1.0 if _exact_match(gold_span, pred_span) else 0.0
            else:
                score = float(_overlap(gold_span, pred_span))
            if score > 0:
                candidates.append((score, gold_index, pred_index))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gold_index, pred_index in candidates:
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        matches.append((gold_index, pred_index))
    return matches


def _prf(tp: int, predicted: int, gold: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "predicted": predicted,
        "gold": gold,
    }


def _span_key(span: Span) -> tuple[Any, ...]:
    return (span.label, span.start, span.end, span.text)


def _build_predicted_token_labels(record: dict[str, Any]) -> list[str]:
    labels = ["O"] * len(record["tokens"])
    spans = [
        _span_from_item(item)
        for item in record.get("predicted_skills", [])
        if item.get("start") is not None and item.get("end") is not None
    ]
    token_offsets = record.get("token_offsets", [])
    for span in spans:
        token_indices = []
        for token_index, token_offset in enumerate(token_offsets):
            overlap = max(0, min(int(span.end), int(token_offset["end"])) - max(int(span.start), int(token_offset["start"])))
            if overlap > 0:
                token_indices.append(token_index)
        if not token_indices:
            continue
        for token_pos, token_index in enumerate(token_indices):
            labels[token_index] = f"{'B' if token_pos == 0 else 'I'}-{span.label}"
    return labels


def _micro_macro_exact(per_label: dict[str, dict[str, float]]) -> tuple[float, float]:
    usable = [values for label, values in per_label.items() if label in BENCHMARK_LABELS and values["gold"] > 0]
    if not usable:
        return 0.0, 0.0

    macro_f1 = sum(values["f1"] for values in usable) / len(usable)
    total_tp = sum(int(values["tp"]) for values in usable)
    total_predicted = sum(int(values["predicted"]) for values in usable)
    total_gold = sum(int(values["gold"]) for values in usable)
    micro_f1 = _prf(total_tp, total_predicted, total_gold)["f1"]
    return macro_f1, micro_f1


def evaluate_predictions(
    gold_file: str | Path,
    prediction_file: str | Path,
    output_file: str | Path,
    total_runtime_seconds: float,
) -> dict[str, Any]:
    gold_records = read_jsonl(gold_file)
    prediction_records = read_jsonl(prediction_file)
    prediction_by_id = {record["id"]: record for record in prediction_records}

    exact_counts = Counter(tp=0, predicted=0, gold=0)
    partial_counts = Counter(tp=0, predicted=0, gold=0)
    per_label_counts = {label: Counter(tp=0, predicted=0, gold=0) for label in BENCHMARK_LABELS}
    token_counts = Counter(tp=0, predicted=0, gold=0)

    parse_failures = 0
    total_records = 0
    total_predicted_skills = 0
    total_unaligned_predictions = 0
    latencies: list[float] = []

    for gold_record in gold_records:
        pred_record = prediction_by_id.get(
            gold_record["id"],
            {
                "id": gold_record["id"],
                "text": gold_record["text"],
                "tokens": gold_record.get("tokens", []),
                "token_offsets": gold_record.get("token_offsets", []),
                "predicted_skills": [],
                "parse_failed": False,
            },
        )
        pred_record.setdefault("tokens", gold_record.get("tokens", []))
        pred_record.setdefault("token_offsets", gold_record.get("token_offsets", []))

        total_records += 1
        if pred_record.get("parse_failed"):
            parse_failures += 1
        if pred_record.get("generation_seconds") is not None:
            latencies.append(float(pred_record["generation_seconds"]))

        gold_spans = [_span_from_item(item) for item in gold_record.get("skills", [])]
        pred_spans = [_span_from_item(item) for item in pred_record.get("predicted_skills", [])]

        exact_matches = _greedy_match(gold_spans, pred_spans, mode="exact")
        partial_matches = _greedy_match(gold_spans, pred_spans, mode="partial")

        exact_counts.update(tp=len(exact_matches), predicted=len(pred_spans), gold=len(gold_spans))
        partial_counts.update(tp=len(partial_matches), predicted=len(pred_spans), gold=len(gold_spans))

        for span in gold_spans:
            per_label_counts.setdefault(span.label, Counter(tp=0, predicted=0, gold=0)).update(gold=1)
        for span in pred_spans:
            per_label_counts.setdefault(span.label, Counter(tp=0, predicted=0, gold=0)).update(predicted=1)
            total_predicted_skills += 1
            if not _valid_span(span):
                total_unaligned_predictions += 1
        for gold_index, _ in exact_matches:
            per_label_counts[gold_spans[gold_index].label].update(tp=1)

        gold_token_labels = gold_record.get("token_labels", [])
        pred_token_labels = _build_predicted_token_labels(pred_record)
        gold_token_set = {(index, label) for index, label in enumerate(gold_token_labels) if label != "O"}
        pred_token_set = {(index, label) for index, label in enumerate(pred_token_labels) if label != "O"}
        token_counts.update(
            tp=len(gold_token_set & pred_token_set),
            predicted=len(pred_token_set),
            gold=len(gold_token_set),
        )

    per_label_metrics = {
        label: _prf(counts["tp"], counts["predicted"], counts["gold"])
        for label, counts in sorted(per_label_counts.items())
        if counts["gold"] or counts["predicted"]
    }
    macro_f1, micro_f1 = _micro_macro_exact(per_label_metrics)

    metrics = {
        "exact_span_precision": _prf(exact_counts["tp"], exact_counts["predicted"], exact_counts["gold"])["precision"],
        "exact_span_recall": _prf(exact_counts["tp"], exact_counts["predicted"], exact_counts["gold"])["recall"],
        "exact_span_f1": _prf(exact_counts["tp"], exact_counts["predicted"], exact_counts["gold"])["f1"],
        "token_precision": _prf(token_counts["tp"], token_counts["predicted"], token_counts["gold"])["precision"],
        "token_recall": _prf(token_counts["tp"], token_counts["predicted"], token_counts["gold"])["recall"],
        "token_f1": _prf(token_counts["tp"], token_counts["predicted"], token_counts["gold"])["f1"],
        "partial_span_precision": _prf(partial_counts["tp"], partial_counts["predicted"], partial_counts["gold"])["precision"],
        "partial_span_recall": _prf(partial_counts["tp"], partial_counts["predicted"], partial_counts["gold"])["recall"],
        "partial_span_f1": _prf(partial_counts["tp"], partial_counts["predicted"], partial_counts["gold"])["f1"],
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "parse_failure_rate": parse_failures / total_records if total_records else 0.0,
        "unaligned_prediction_rate": total_unaligned_predictions / total_predicted_skills if total_predicted_skills else 0.0,
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "total_runtime_minutes": total_runtime_seconds / 60.0,
        "per_label_metrics": per_label_metrics,
        "counts": {
            "samples": total_records,
            "predicted_skills": total_predicted_skills,
            "unaligned_predictions": total_unaligned_predictions,
            "parse_failures": parse_failures,
        },
    }
    write_json(output_file, metrics)
    return metrics
