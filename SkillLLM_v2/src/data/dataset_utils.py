from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SOURCE_TO_CANONICAL_LABEL = {
    "SKILL": "soft_skill",
    "KNOWLEDGE": "hard_skill",
    "skill": "soft_skill",
    "knowledge": "hard_skill",
    "soft_skill": "soft_skill",
    "hard_skill": "hard_skill",
}

CANONICAL_TO_SOURCE_LABEL = {
    "soft_skill": "SKILL",
    "hard_skill": "KNOWLEDGE",
}

ALLOWED_LABELS = tuple(sorted(set(SOURCE_TO_CANONICAL_LABEL.values())))


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    text: str


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    return SOURCE_TO_CANONICAL_LABEL.get(label)


def detokenize(tokens: Iterable[str]) -> str:
    return " ".join(tokens)


def token_offsets(tokens: list[str]) -> list[TokenSpan]:
    offsets: list[TokenSpan] = []
    cursor = 0
    for index, token in enumerate(tokens):
        start = cursor
        end = start + len(token)
        offsets.append(TokenSpan(start=start, end=end, text=token))
        cursor = end
        if index < len(tokens) - 1:
            cursor += 1
    return offsets


def extract_spans_from_bio(tokens: list[str], tags: list[str], label: str) -> list[dict[str, object]]:
    canonical_label = normalize_label(label)
    if canonical_label is None:
        raise ValueError(f"Unsupported label: {label}")
    if len(tokens) != len(tags):
        raise ValueError("Token and tag lengths do not match.")

    text = detokenize(tokens)
    offsets = token_offsets(tokens)
    spans: list[dict[str, object]] = []
    active_start: int | None = None

    def flush(end_index: int) -> None:
        nonlocal active_start
        if active_start is None:
            return
        start_char = offsets[active_start].start
        end_char = offsets[end_index].end
        span_text = text[start_char:end_char]
        spans.append(
            {
                "text": span_text,
                "label": canonical_label,
                "start": start_char,
                "end": end_char,
                "token_start": active_start,
                "token_end": end_index,
            }
        )
        active_start = None

    for index, tag in enumerate(tags):
        if tag == "B":
            flush(index - 1)
            active_start = index
        elif tag == "I":
            if active_start is None:
                active_start = index
        else:
            flush(index - 1)

    flush(len(tags) - 1)
    return spans


def validate_skills(text: str, skills: list[dict[str, object]]) -> None:
    for skill in skills:
        start = int(skill["start"])
        end = int(skill["end"])
        if start < 0 or end > len(text) or start >= end:
            raise ValueError(f"Invalid span bounds for skill {skill}")
        if text[start:end] != skill["text"]:
            raise ValueError(
                f"Span text mismatch for {skill}: source='{text[start:end]}', annotated='{skill['text']}'"
            )
        if normalize_label(str(skill["label"])) is None:
            raise ValueError(f"Unsupported label in skill {skill}")


def build_example_from_row(row: dict[str, object], split_name: str, row_index: int) -> dict[str, object]:
    tokens = list(row["tokens"])
    text = detokenize(tokens)
    skill_spans = extract_spans_from_bio(tokens, list(row["tags_skill"]), "SKILL")
    knowledge_spans = extract_spans_from_bio(tokens, list(row["tags_knowledge"]), "KNOWLEDGE")
    combined_spans = sorted(skill_spans + knowledge_spans, key=lambda item: (item["start"], item["end"], item["label"]))
    validate_skills(text, combined_spans)

    document_id = row.get("idx", row_index)
    return {
        "id": f"{split_name}_{document_id}_{row_index}",
        "document_id": document_id,
        "source": row.get("source"),
        "text": text,
        "skills": [
            {
                "text": span["text"],
                "label": span["label"],
                "start": span["start"],
                "end": span["end"],
            }
            for span in combined_spans
        ],
    }


def load_skillspan_dataset(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - setup validation
        raise RuntimeError("datasets must be installed before building SkillSpan files.") from exc
    return load_dataset(dataset_name)


def unique_skill_texts(examples: list[dict[str, object]], label: str | None = None) -> list[str]:
    values = []
    seen = set()
    for example in examples:
        for skill in example["skills"]:
            if label is not None and skill["label"] != label:
                continue
            text = str(skill["text"])
            if text not in seen:
                seen.add(text)
                values.append(text)
    return values
