from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable, Literal

SkillLabel = Literal["HARD_SKILL", "SOFT_SKILL"]
SkillSource = Literal["bert", "mistral", "gliner", "dictionary"]


@dataclass(slots=True)
class SkillCandidate:
    surface_form: str
    label: SkillLabel
    start: int | None
    end: int | None
    source: SkillSource
    confidence: float
    evidence: str
    canonical_skill_id: str | None = None
    canonical_skill_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_skill(candidate: dict[str, Any]) -> None:
    required = {
        "surface_form",
        "label",
        "start",
        "end",
        "source",
        "confidence",
        "evidence",
        "canonical_skill_id",
        "canonical_skill_name",
        "metadata",
    }
    missing = sorted(required.difference(candidate))
    if missing:
        raise ValueError(f"Skill candidate missing fields: {missing}")
    if candidate["label"] not in {"HARD_SKILL", "SOFT_SKILL"}:
        raise ValueError(f"Unsupported skill label: {candidate['label']}")
    if candidate["source"] not in {"bert", "mistral", "gliner", "dictionary"}:
        raise ValueError(f"Unsupported skill source: {candidate['source']}")
    start, end = candidate["start"], candidate["end"]
    if (start is None) != (end is None):
        raise ValueError("start and end must either both be set or both be null")
    if start is not None and (not isinstance(start, int) or not isinstance(end, int) or start >= end):
        raise ValueError(f"Invalid offsets: start={start}, end={end}")
    confidence = float(candidate["confidence"])
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be in [0, 1], got {confidence}")
    if not candidate["surface_form"]:
        raise ValueError("surface_form must not be empty")
    if not isinstance(candidate["metadata"], dict):
        raise ValueError("metadata must be an object")


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[\W_]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


# Graph key: like normalize_text, but symbols that are part of a skill's name survive, so "C++", "C#", "C", "F#",
# ".NET" and "Node.js" are different graph nodes. normalize_text (used for alias / taxonomy matching everywhere)
# is deliberately left unchanged: it still maps C++, C# and C to "c" and ".NET" to "net".
_GRAPH_TOKEN = re.compile(r"(?<![^\W_])\.[^\W_]+(?:\.[^\W_]*[^\W\d_][^\W_]*)*|[^\W_]+(?:\.[^\W\d_][^\W_]*)*[+#]*")


def graph_key(value: str) -> str:
    """Normalized graph node key that preserves symbol-bearing skill names.

    Kept: a trailing run of ``+`` / ``#`` on a token (c++, c#, f#), a leading ``.`` before a word (.net) and a ``.``
    between a word and an alphabetic segment (node.js, asp.net). Everything else is normalized exactly like
    :func:`normalize_text` (lowercase, other punctuation to single spaces), so names without those symbols get the
    same key from both functions and existing alias / taxonomy matching is unaffected. Version numbers ("3.9")
    are not treated as dotted names.
    """
    return " ".join(_GRAPH_TOKEN.findall(value.lower().strip()))


def find_offsets(text: str, surface: str, used: set[tuple[int, int]] | None = None) -> tuple[int | None, int | None]:
    used = used or set()
    if not surface:
        return None, None
    matches = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(surface)}\b", text)]
    if not matches:
        matches = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(surface)}\b", text, re.IGNORECASE)]
    if not matches:
        matches = [(m.start(), m.end()) for m in re.finditer(re.escape(surface), text)]
    if not matches:
        lower_text = text.lower()
        lower_surface = surface.lower()
        matches = [(m.start(), m.end()) for m in re.finditer(re.escape(lower_surface), lower_text)]
    for start, end in matches:
        if (start, end) not in used:
            return start, end
    return matches[0] if matches else (None, None)


def bio_spans(tokens: list[str], tags: list[str], label: SkillLabel, text: str) -> list[SkillCandidate]:
    spans: list[SkillCandidate] = []
    index = 0
    token_offsets: list[tuple[int, int]] = []
    for token in tokens:
        start = text.find(token, index)
        if start < 0:
            start = index
        end = start + len(token)
        token_offsets.append((start, end))
        index = end

    active_start: int | None = None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or (tag == "O" and active_start is not None):
            if active_start is not None:
                start = token_offsets[active_start][0]
                end = token_offsets[i - 1][1]
                surface = text[start:end]
                spans.append(
                    SkillCandidate(surface, label, start, end, "dictionary", 1.0, surface)
                )
            active_start = i if tag.startswith("B-") else None
        elif tag.startswith("I-"):
            if active_start is None:
                active_start = i
    return spans


def ensure_skills(candidates: Iterable[SkillCandidate | dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        item = candidate.to_dict() if isinstance(candidate, SkillCandidate) else dict(candidate)
        validate_skill(item)
        output.append(item)
    return output
