"""Token-budget helpers for context compaction.

``count_tokens`` is a deterministic *proxy*: word and punctuation units (``\\w+|[^\\w\\s]``), the same proxy the
context evals use. Real BPE token counts are typically 1.2-1.5x larger, so a budget given here is in proxy tokens;
scale it down if you need a hard model-token limit.

``skill_importance`` ranks skill facts for budgeted compaction. Signals and why they are (not) used:

* confidence (max over supporting extractions): used, the extractor's own score.
* distinct evidence count: used, ``1 + ln(n)`` so repeated corroboration helps without dominating.
* job-target relevance: used, a skill named in the target role string gets a fixed bonus (checkable from inputs).
* source reliability: used only if the caller passes ``source_reliability`` (source -> weight); the default is a
  neutral 1.0 because this repository holds no per-source reliability measurement for these signals. Evidence
  sources are averaged, so a skill backed by one low-weight source ranks below one backed by a high-weight source.
* recency: NOT used. Evidence timestamps are ingestion time (identical for every skill), extraction output carries
  no dates, so there is no recency signal to rank on. Using ``generated_at`` would only add noise.
* HARD vs SOFT label: NOT used. Nothing in the pipeline justifies weighting one over the other; callers who want it
  can pass ``label_weights``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

from skillllm_v2.schema import normalize_text

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Deterministic token proxy: word and punctuation units."""
    return len(_TOKEN_RE.findall(text))


def json_tokens(obj: Any) -> int:
    return count_tokens(json.dumps(obj, sort_keys=True, ensure_ascii=False))


TARGET_BONUS = 0.25


def skill_importance(
    skill: Mapping[str, Any],
    *,
    target_terms: set[str] | frozenset[str] = frozenset(),
    source_reliability: Mapping[str, float] | None = None,
    label_weights: Mapping[str, float] | None = None,
) -> float:
    """Importance of one profile skill fact (higher = keep first). Deterministic, no randomness."""
    confidence = float(skill.get("confidence", 0.0))
    evidence = skill.get("evidence", []) or []
    n_distinct = len({str(e.get("text", "")) for e in evidence}) or 1
    score = confidence * (1.0 + math.log(n_distinct))
    if source_reliability:
        weights = []
        for item in evidence:
            for src in str(item.get("source", "")).split("+"):
                weights.append(float(source_reliability.get(src, 1.0)))
        if weights:
            score *= sum(weights) / len(weights)
    if label_weights:
        score *= float(label_weights.get(str(skill.get("label", "")), 1.0))
    if target_terms:
        names = {normalize_text(str(skill.get("surface_form", ""))), normalize_text(str(skill.get("canonical_skill_name") or ""))}
        if any(n and (n in target_terms or set(n.split()) <= target_terms) for n in names):
            score *= 1.0 + TARGET_BONUS
    return score
