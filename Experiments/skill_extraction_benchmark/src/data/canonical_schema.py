"""Canonical dataset schema and BIO ↔ span conversion utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class SpanAnnotation:
    """A single skill span within a sentence."""

    start_token: int
    end_token: int  # inclusive
    label: str
    text: str


@dataclass
class CanonicalRecord:
    """One row in the canonical JSONL dataset (one sentence)."""

    split: str
    doc_id: str
    sentence_id: str
    tokens: List[str]
    bio_tags: List[str]
    spans: List[SpanAnnotation]
    text: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spans"] = [asdict(s) for s in self.spans]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CanonicalRecord":
        spans = [SpanAnnotation(**s) for s in d.get("spans", [])]
        return cls(
            split=d["split"],
            doc_id=str(d["doc_id"]),
            sentence_id=str(d["sentence_id"]),
            tokens=d["tokens"],
            bio_tags=d["bio_tags"],
            spans=spans,
            text=d["text"],
            source=d.get("source", ""),
        )


# ── BIO ↔ span conversion ───────────────────────────────────────────


def bio_to_spans(tokens: Sequence[str], tags: Sequence[str], label: str = "SKILL") -> List[SpanAnnotation]:
    """Convert a BIO tag sequence to a list of SpanAnnotation objects.

    Handles both plain B/I/O and prefixed B-SKILL/I-SKILL/O formats.
    """
    if len(tokens) != len(tags):
        raise ValueError(f"Token count ({len(tokens)}) != tag count ({len(tags)})")

    spans: List[SpanAnnotation] = []
    start: int | None = None
    end: int | None = None

    for idx, tag in enumerate(tags):
        # Normalise tag prefix
        prefix = tag.split("-")[0] if "-" in tag else tag

        if prefix == "B":
            # Close any open span
            if start is not None:
                spans.append(SpanAnnotation(
                    start_token=start,
                    end_token=end,  # type: ignore[arg-type]
                    label=label,
                    text=" ".join(tokens[start: end + 1]),  # type: ignore[operator]
                ))
            start = idx
            end = idx
        elif prefix == "I" and start is not None:
            end = idx
        else:
            # O tag or orphan I → close current span
            if start is not None:
                spans.append(SpanAnnotation(
                    start_token=start,
                    end_token=end,  # type: ignore[arg-type]
                    label=label,
                    text=" ".join(tokens[start: end + 1]),  # type: ignore[operator]
                ))
                start = None
                end = None

    # Close trailing span
    if start is not None:
        spans.append(SpanAnnotation(
            start_token=start,
            end_token=end,  # type: ignore[arg-type]
            label=label,
            text=" ".join(tokens[start: end + 1]),  # type: ignore[operator]
        ))

    return spans


def spans_to_bio(num_tokens: int, spans: Sequence[SpanAnnotation]) -> List[str]:
    """Inverse of bio_to_spans: produce B-SKILL / I-SKILL / O tag list."""
    tags = ["O"] * num_tokens
    for span in spans:
        if span.start_token < 0 or span.end_token >= num_tokens:
            raise ValueError(f"Span ({span.start_token}, {span.end_token}) out of bounds for {num_tokens} tokens")
        tags[span.start_token] = f"B-{span.label}"
        for i in range(span.start_token + 1, span.end_token + 1):
            tags[i] = f"I-{span.label}"
    return tags


# ── Detokenisation ───────────────────────────────────────────────────

# Tokens that should NOT have a leading space
_NO_LEFT_SPACE = {".", ",", "!", "?", ";", ":", "'", ")", "]", "}", "%", "'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "n't"}
# Tokens that should NOT have a trailing space
_NO_RIGHT_SPACE = {"(", "[", "{", "$", "#", "@"}


def detokenize(tokens: Sequence[str]) -> str:
    """Deterministic detokenizer: rebuild sentence text from token list.

    Uses heuristic rules for punctuation spacing. Designed to produce
    human-readable text, not to recover exact original whitespace.
    """
    if not tokens:
        return ""

    pieces: List[str] = [tokens[0]]
    for tok in tokens[1:]:
        if tok in _NO_LEFT_SPACE:
            pieces.append(tok)
        elif pieces and pieces[-1] in _NO_RIGHT_SPACE:
            pieces.append(tok)
        else:
            pieces.append(" ")
            pieces.append(tok)

    return "".join(pieces)


# ── I/O helpers ──────────────────────────────────────────────────────


def load_canonical_split(path: Path) -> List[CanonicalRecord]:
    """Load a canonical JSONL file into a list of CanonicalRecord."""
    if not path.exists():
        raise FileNotFoundError(f"Canonical split not found: {path}")
    records: List[CanonicalRecord] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        records.append(CanonicalRecord.from_dict(json.loads(line)))
    return records


def write_canonical_split(path: Path, records: Sequence[CanonicalRecord]) -> None:
    """Write canonical records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
