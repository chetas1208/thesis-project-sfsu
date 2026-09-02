"""source_aware_policy.py

Single source of truth for the "Raw Mistral LoRA is the extraction anchor"
policy shared by :class:`skillllm_v2.extract.candidate_merger.CandidateMerger`,
:class:`skillllm_v2.verify.span_verifier.SpanVerifier`, and
:class:`skillllm_v2.verify.confidence_calibrator.ConfidenceCalibrator`.

The 80-epoch Raw Mistral LoRA extractor is the wrapper's non-regression
anchor (see ``reports/skillllm_v2/wrapper_error_analysis*.md``): Exact Span F1
0.675034 on the test split, the strongest of the four extraction
configurations measured. A candidate is "anchor-derived" if its
``metadata.sources`` (or, for un-merged candidates that have not yet passed
through ``CandidateMerger``, ``[candidate["source"]]``) intersects
:data:`ANCHOR_SOURCES`. Anchor-derived candidates are:

  - never removed by ``CandidateMerger``'s add-only policy
  - never rejected on quality grounds by ``SpanVerifier``
  - never rejected on confidence grounds by ``ConfidenceCalibrator``

Non-anchor candidates (brand-new GLiNER/dictionary-only spans) pass through
additional, independently dev-tunable gates at each of those three stages
(``tune_candidate_merger.py``, ``tune_confidence_calibrator.py``).
"""
from __future__ import annotations

from typing import Any

ANCHOR_SOURCES: frozenset[str] = frozenset({"mistral"})


def candidate_sources(candidate: dict[str, Any]) -> set[str]:
    """Return the set of extractor sources that produced ``candidate``.

    Falls back to ``{candidate["source"]}`` for candidates that have not yet
    passed through ``CandidateMerger`` (and therefore have no
    ``metadata.sources`` list).
    """
    sources = candidate.get("metadata", {}).get("sources")
    if sources:
        return set(sources)
    return {candidate["source"]}


def is_anchor_candidate(candidate: dict[str, Any], anchor_sources: frozenset[str] = ANCHOR_SOURCES) -> bool:
    """True if ``candidate`` is the Raw Mistral anchor or corroborated by it."""
    return bool(candidate_sources(candidate) & anchor_sources)


def source_key(candidate: dict[str, Any]) -> str:
    """Stable, sorted ``"+"``-joined key for a candidate's source combination.

    Used to look up per-source-combination thresholds (e.g.
    ``min_confidence_new_span`` in ``CandidateMerger``) -- ``sorted()`` makes
    the key independent of extraction order, e.g. both ``{"gliner",
    "dictionary"}`` and ``{"dictionary", "gliner"}`` map to
    ``"dictionary+gliner"``.
    """
    return "+".join(sorted(candidate_sources(candidate)))
