"""Post-processing utilities for unifying model predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.data.canonical_schema import SpanAnnotation, bio_to_spans


def unify_prediction_record(
    record_id: str,
    doc_id: str,
    tokens: List[str],
    predicted_spans: List[Dict],
    predicted_skill_texts: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Create a unified prediction dict with standard keys."""
    return {
        "id": record_id,
        "doc_id": doc_id,
        "model": model_name,
        "tokens": tokens,
        "predicted_spans": predicted_spans,
        "predicted_skill_texts": predicted_skill_texts,
    }


def write_predictions(path: Path, predictions: Sequence[Dict]) -> None:
    """Write unified predictions to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")


def load_predictions(path: Path) -> List[Dict]:
    """Load predictions from a JSONL file."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
