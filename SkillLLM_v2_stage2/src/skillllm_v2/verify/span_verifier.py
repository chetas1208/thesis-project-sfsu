from __future__ import annotations

from typing import Any

from skillllm_v2.schema import normalize_text, validate_skill
from skillllm_v2.policy import ANCHOR_SOURCES, is_anchor_candidate

GENERIC_REJECTS = {
    "experience",
    "knowledge",
    "tools",
    "technologies",
    "responsibilities",
    "job",
    "degree",
}


class SpanVerifier:
    """Reject low-quality candidate spans before calibration/linking.

    The Raw Mistral LoRA anchor (and any candidate it corroborates, i.e.
    ``"mistral" in metadata.sources``) is never rejected on quality grounds
    (``generic_phrase``/``span_too_long``/``low_confidence``) -- only a
    candidate with no usable offsets is dropped, since it can never become an
    exact-span true positive anyway. The quality checks below remain in force
    for brand-new GLiNER/dictionary-only additions (already gated by
    ``CandidateMerger``'s add-only policy).
    """

    def __init__(self, min_confidence: float = 0.35, protect_anchor_sources: frozenset[str] | None = None) -> None:
        self.min_confidence = min_confidence
        self.protect_anchor_sources = frozenset(protect_anchor_sources) if protect_anchor_sources is not None else ANCHOR_SOURCES

    def verify(self, candidate: dict[str, Any]) -> dict[str, Any]:
        validate_skill(candidate)
        norm = normalize_text(candidate["surface_form"])
        is_anchor = is_anchor_candidate(candidate, self.protect_anchor_sources)
        valid = True
        reason = "accepted"
        if candidate["start"] is None or candidate["end"] is None:
            valid = False
            reason = "missing_offsets"
        elif is_anchor:
            reason = "accepted_anchor"
        elif norm in GENERIC_REJECTS:
            valid = False
            reason = "generic_phrase"
        elif len(norm.split()) > 6:
            valid = False
            reason = "span_too_long"
        elif float(candidate["confidence"]) < self.min_confidence:
            valid = False
            reason = "low_confidence"
        return {
            "is_valid_skill": valid,
            "corrected_span": candidate["surface_form"],
            "label": candidate["label"],
            "confidence": float(candidate["confidence"]),
            "reason": reason,
        }

    def filter(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for candidate in candidates:
            decision = self.verify(candidate)
            enriched = dict(candidate)
            enriched["metadata"] = dict(candidate.get("metadata") or {})
            enriched["metadata"]["verification"] = decision
            if decision["is_valid_skill"]:
                output.append(enriched)
        return output
