"""Document-level aggregation of sentence predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.data.normalizers import deduplicate_skills


def aggregate_to_documents(
    sentence_predictions: Sequence[Dict],
    gold_records: Sequence[Dict] | None = None,
) -> List[Dict]:
    """Group sentence predictions by doc_id and merge skill sets.

    Returns a list of document-level prediction dicts:
        doc_id, predicted_skill_texts, num_sentences, gold_skill_texts (if gold provided)
    """
    doc_groups: Dict[str, Dict[str, Any]] = {}

    for pred in sentence_predictions:
        doc_id = str(pred["doc_id"])
        if doc_id not in doc_groups:
            doc_groups[doc_id] = {
                "doc_id": doc_id,
                "all_pred_skills": [],
                "num_sentences": 0,
            }
        doc_groups[doc_id]["num_sentences"] += 1
        doc_groups[doc_id]["all_pred_skills"].extend(pred.get("predicted_skill_texts", []))

    # Also collect gold skills per document if available
    gold_by_doc: Dict[str, List[str]] = {}
    if gold_records:
        for rec in gold_records:
            doc_id = str(rec.get("doc_id", ""))
            if doc_id not in gold_by_doc:
                gold_by_doc[doc_id] = []
            for span in rec.get("spans", []):
                if isinstance(span, dict):
                    gold_by_doc[doc_id].append(span.get("text", ""))

    results: List[Dict] = []
    for doc_id in sorted(doc_groups.keys()):
        group = doc_groups[doc_id]
        doc_pred = {
            "doc_id": doc_id,
            "predicted_skill_texts": deduplicate_skills(group["all_pred_skills"]),
            "num_sentences": group["num_sentences"],
        }
        if doc_id in gold_by_doc:
            doc_pred["gold_skill_texts"] = deduplicate_skills(gold_by_doc[doc_id])
        results.append(doc_pred)

    return results


def write_document_predictions(path: Path, documents: Sequence[Dict]) -> None:
    """Write document-level predictions to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
