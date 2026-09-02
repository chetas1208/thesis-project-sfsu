from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skillllm_v2.schema import normalize_text


def contextualize_record(record: dict[str, Any], context_fields: list[str] | None = None) -> str:
    fields = context_fields or ["course_name", "title", "description", "teaches_skills", "required_skills"]
    parts: list[str] = []
    for field in fields:
        value = record.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        parts.append(f"{field}: {value}")
    return "\n".join(parts)


@dataclass(slots=True)
class RetrievalResult:
    record: dict[str, Any]
    score: float
    evidence: str


class ContextualRetriever:
    """Small BM25-like lexical retriever for smoke/demo paths.

    Dense retrieval can be plugged in later; this class gives deterministic
    contextual retrieval behavior without extra dependencies.
    """

    def __init__(self, records: list[dict[str, Any]], context_fields: list[str] | None = None) -> None:
        self.records = records
        self.contexts = [contextualize_record(record, context_fields) for record in records]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_terms = set(normalize_text(query).split())
        scored: list[RetrievalResult] = []
        for record, context in zip(self.records, self.contexts):
            terms = set(normalize_text(context).split())
            if not terms:
                continue
            overlap = len(query_terms & terms)
            score = overlap / max(1, len(query_terms))
            if score > 0:
                scored.append(RetrievalResult(record=record, score=score, evidence=context))
        scored.sort(key=lambda item: item.score, reverse=True)
        return [
            {"record": item.record, "score": item.score, "evidence": item.evidence}
            for item in scored[:top_k]
        ]
