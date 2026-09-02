from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.data.dataset_utils import normalize_label
from src.runtime_utils import read_jsonl, write_json


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int


def normalize_entity(item: dict[str, Any], source_text: str) -> Entity | None:
    label = normalize_label(item.get("label"))
    if label is None:
        return None
    start = item.get("start")
    end = item.get("end")
    if start is None or end is None:
        return None
    try:
        start = int(start)
        end = int(end)
    except (TypeError, ValueError):
        return None
    if start < 0 or end > len(source_text) or start >= end:
        return None

    text = item.get("text")
    if isinstance(text, str) and source_text[start:end] == text:
        entity_text = text
    elif end < len(source_text) and isinstance(text, str) and source_text[start : end + 1] == text:
        end += 1
        entity_text = text
    else:
        entity_text = source_text[start:end]

    return Entity(text=entity_text, label=label, start=start, end=end)


def extract_entities(record: dict[str, Any]) -> list[Entity]:
    source_text = record["text"]
    raw_items: list[dict[str, Any]] = []
    if isinstance(record.get("predicted_skills"), list):
        raw_items = list(record["predicted_skills"])
    elif isinstance(record.get("skills"), list):
        raw_items = list(record["skills"])
    elif isinstance(record.get("spans"), list):
        for span in record["spans"]:
            start = span.get("start")
            end = span.get("end")
            if start is None or end is None:
                continue
            raw_items.append(
                {
                    "text": source_text[int(start) : int(end) + 1],
                    "label": span.get("label"),
                    "start": int(start),
                    "end": int(end) + 1,
                }
            )

    entities = []
    for item in raw_items:
        entity = normalize_entity(item, source_text)
        if entity is not None:
            entities.append(entity)
    return sorted(entities, key=lambda entity: (entity.start, entity.end, entity.label, entity.text))


def align_records(gold_file: str, prediction_file: str) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    gold_records = read_jsonl(gold_file)
    prediction_records = read_jsonl(prediction_file)

    if prediction_records and all("id" in record for record in prediction_records):
        prediction_by_id = {record["id"]: record for record in prediction_records}
        aligned = []
        for gold in gold_records:
            prediction = prediction_by_id.pop(
                gold["id"],
                {
                    "id": gold["id"],
                    "text": gold["text"],
                    "predicted_skills": [],
                },
            )
            aligned.append((gold, prediction))
        return aligned, list(prediction_by_id.values())

    prediction_by_text: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for prediction in prediction_records:
        prediction_by_text[prediction["text"]].append(prediction)

    aligned = []
    leftovers = []
    for gold in gold_records:
        if prediction_by_text[gold["text"]]:
            aligned.append((gold, prediction_by_text[gold["text"]].popleft()))
        else:
            aligned.append((gold, {"id": gold["id"], "text": gold["text"], "predicted_skills": []}))

    for queue in prediction_by_text.values():
        leftovers.extend(list(queue))
    return aligned, leftovers


def exact_score(gold: Entity, pred: Entity) -> float:
    if gold.label == pred.label and gold.start == pred.start and gold.end == pred.end:
        return 1.0
    return 0.0


def overlap_size(gold: Entity, pred: Entity) -> int:
    return max(0, min(gold.end, pred.end) - max(gold.start, pred.start))


def overlap_iou(gold: Entity, pred: Entity) -> float:
    overlap = overlap_size(gold, pred)
    if overlap <= 0:
        return 0.0
    union = max(gold.end, pred.end) - min(gold.start, pred.start)
    return overlap / union


def relaxed_score(gold: Entity, pred: Entity) -> float:
    if gold.label != pred.label:
        return 0.0
    return overlap_iou(gold, pred)


def text_match_score(gold: Entity, pred: Entity) -> float:
    if gold.label != pred.label:
        return 0.0
    return 1.0 if gold.text.lower() == pred.text.lower() else 0.0


def greedy_match(
    gold_entities: list[Entity],
    pred_entities: list[Entity],
    scorer: Callable[[Entity, Entity], float],
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidate_pairs: list[tuple[float, int, int]] = []
    for gold_index, gold in enumerate(gold_entities):
        for pred_index, pred in enumerate(pred_entities):
            score = scorer(gold, pred)
            if score > 0:
                candidate_pairs.append((score, gold_index, pred_index))

    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    matches: list[tuple[int, int, float]] = []
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for score, gold_index, pred_index in candidate_pairs:
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        matches.append((gold_index, pred_index, score))
    return matches, used_gold, used_pred


def compute_prf(tp: int, predicted: int, gold: int) -> dict[str, float]:
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


def per_label_counter() -> dict[str, Counter]:
    return {
        "hard_skill": Counter(tp=0, predicted=0, gold=0),
        "soft_skill": Counter(tp=0, predicted=0, gold=0),
    }


def write_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)
