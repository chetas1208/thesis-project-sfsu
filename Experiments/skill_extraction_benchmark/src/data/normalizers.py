"""Text normalization and deduplication utilities for skill extraction."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Set


def normalize_skill_text(text: str) -> str:
    """Normalise a skill surface form for deduplication.

    Lowercases, collapses whitespace, strips leading/trailing whitespace.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


def deduplicate_skills(skills: Sequence[str]) -> List[str]:
    """Deduplicate skill texts by normalised form, preserving first occurrence."""
    seen: Set[str] = set()
    unique: List[str] = []
    for skill in skills:
        key = normalize_skill_text(skill)
        if key and key not in seen:
            seen.add(key)
            unique.append(skill)
    return sorted(unique, key=lambda s: s.lower())


def aggregate_document_skills(
    sentence_predictions: Sequence[Dict],
) -> Dict[str, Dict]:
    """Group sentence-level predictions by doc_id and merge skill sets.

    Args:
        sentence_predictions: list of dicts with keys:
            doc_id, sentence_id, predicted_skill_texts, ...

    Returns:
        dict mapping doc_id → {
            "doc_id": str,
            "predicted_skill_texts": sorted deduplicated list,
            "num_sentences": int,
        }
    """
    doc_groups: Dict[str, Dict] = {}

    for pred in sentence_predictions:
        doc_id = str(pred["doc_id"])
        if doc_id not in doc_groups:
            doc_groups[doc_id] = {
                "doc_id": doc_id,
                "all_skills": [],
                "num_sentences": 0,
            }
        doc_groups[doc_id]["num_sentences"] += 1
        doc_groups[doc_id]["all_skills"].extend(pred.get("predicted_skill_texts", []))

    result: Dict[str, Dict] = {}
    for doc_id, group in doc_groups.items():
        result[doc_id] = {
            "doc_id": doc_id,
            "predicted_skill_texts": deduplicate_skills(group["all_skills"]),
            "num_sentences": group["num_sentences"],
        }

    return result
