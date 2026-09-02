from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.runtime_utils import strip_markdown_fences


VALID_LABELS = {"HARD_SKILL", "SOFT_SKILL", "SKILL"}
LABEL_ALIASES = {
    "hard_skill": "HARD_SKILL",
    "soft_skill": "SOFT_SKILL",
    "skill": "SKILL",
    "knowledge": "HARD_SKILL",
}


@dataclass
class BenchmarkParseResult:
    skills: list[dict[str, Any]]
    repair_actions: list[str] = field(default_factory=list)
    rejected_items: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None
    raw_payload: Any = None

    @property
    def parse_failed(self) -> bool:
        return self.parse_error is not None


def _normalize_label(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    normalized = LABEL_ALIASES.get(label.strip(), label.strip().upper())
    if normalized in VALID_LABELS:
        return normalized
    return None


def _repair_text(text: str) -> tuple[str, list[str]]:
    repaired = strip_markdown_fences(text)
    actions: list[str] = []

    stripped_prefix = re.sub(r"^\s*json\s*", "", repaired, flags=re.IGNORECASE)
    if stripped_prefix != repaired:
        repaired = stripped_prefix
        actions.append("removed_json_prefix")

    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    for source, target in replacements.items():
        if source in repaired:
            repaired = repaired.replace(source, target)
            actions.append("normalized_quotes")

    cleaned = re.sub(r",(\s*[}\]])", r"\1", repaired)
    if cleaned != repaired:
        repaired = cleaned
        actions.append("removed_trailing_commas")
    return repaired.strip(), actions


def _decode_first_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    candidate_positions = [index for index, char in enumerate(text) if char in "[{"]
    if text and text[0] in "[{":
        candidate_positions = [0] + [index for index in candidate_positions if index != 0]

    for start_index in candidate_positions:
        try:
            payload, _ = decoder.raw_decode(text[start_index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not recover a valid JSON object or list from model output.")


def _flatten_payload(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        skills = payload.get("skills")
        if isinstance(skills, list):
            return skills
        raise ValueError('Parsed JSON object is missing a "skills" list.')
    if isinstance(payload, list):
        return payload
    raise ValueError("Parsed JSON payload must be an object or list.")


def parse_benchmark_output(raw_text: str) -> BenchmarkParseResult:
    repaired_text, repair_actions = _repair_text(raw_text)
    result = BenchmarkParseResult(skills=[], repair_actions=repair_actions)

    try:
        payload = _decode_first_json(repaired_text)
    except Exception as exc:
        result.parse_error = str(exc)
        return result

    result.raw_payload = payload
    try:
        items = _flatten_payload(payload)
    except Exception as exc:
        result.parse_error = str(exc)
        return result

    for item in items:
        if not isinstance(item, dict):
            result.rejected_items.append({"reason": "item_not_object", "item": item})
            continue
        text = item.get("text")
        label = _normalize_label(item.get("label"))
        if not isinstance(text, str) or not text.strip():
            result.rejected_items.append({"reason": "missing_text", "item": item})
            continue
        if label is None:
            result.rejected_items.append({"reason": "invalid_label", "item": item})
            continue
        result.skills.append({"text": text, "label": label})
    return result
