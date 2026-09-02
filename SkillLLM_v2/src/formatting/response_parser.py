from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.data.dataset_utils import normalize_label
from src.runtime_utils import strip_markdown_fences

try:
    import orjson
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    orjson = None


@dataclass
class ParseResult:
    skills: list[dict[str, Any]]
    rejected_items: list[dict[str, Any]] = field(default_factory=list)
    repair_actions: list[str] = field(default_factory=list)
    raw_payload: Any = None


def _extract_json_candidate(text: str) -> str:
    stripped = strip_markdown_fences(text)
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    object_match = re.search(r"(\{.*\})", stripped, flags=re.DOTALL)
    if object_match:
        return object_match.group(1)

    list_match = re.search(r"(\[.*\])", stripped, flags=re.DOTALL)
    if list_match:
        return list_match.group(1)

    return stripped


def _repair_common_json_issues(text: str) -> tuple[str, list[str]]:
    repaired = text.strip()
    actions: list[str] = []

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

    trailing_comma_fixed = re.sub(r",(\s*[}\]])", r"\1", repaired)
    if trailing_comma_fixed != repaired:
        repaired = trailing_comma_fixed
        actions.append("removed_trailing_commas")

    if repaired.startswith("json"):
        repaired = repaired[4:].strip()
        actions.append("removed_json_prefix")

    return repaired, actions


def _json_loads(text: str) -> Any:
    if orjson is not None:
        return orjson.loads(text)
    return json.loads(text)


def _load_payload(text: str, repair_actions: list[str]) -> Any:
    candidate = _extract_json_candidate(text)
    candidate, actions = _repair_common_json_issues(candidate)
    repair_actions.extend(actions)

    try:
        return _json_loads(candidate)
    except Exception:
        pass

    try:
        payload = ast.literal_eval(candidate)
        repair_actions.append("literal_eval_fallback")
        return payload
    except Exception as exc:
        raise ValueError(f"Could not parse model output as JSON: {exc}") from exc


def _flatten_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("skills"), list):
            return list(payload["skills"])

        flattened: list[dict[str, Any]] = []
        for key, values in payload.items():
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                flattened.append(
                    {
                        "text": item.get("text") or item.get("skill_span"),
                        "label": normalize_label(str(key)),
                        "start": item.get("start"),
                        "end": item.get("end"),
                    }
                )
        if flattened:
            return flattened
    raise ValueError("Parsed output does not match the expected schema.")


def _validate_item(item: Any, source_text: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(item, dict):
        return None, {"reason": "item_not_object", "item": item}

    text = item.get("text")
    label = normalize_label(item.get("label"))
    if not isinstance(text, str) or not text.strip():
        return None, {"reason": "missing_text", "item": item}
    if label is None:
        return None, {"reason": "invalid_label", "item": item}

    validated = {"text": text, "label": label}
    has_start = "start" in item and item.get("start") is not None
    has_end = "end" in item and item.get("end") is not None

    if has_start != has_end:
        return None, {"reason": "partial_offset", "item": item}

    if has_start and has_end:
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (TypeError, ValueError):
            return None, {"reason": "non_integer_offsets", "item": item}

        if source_text is not None:
            if start < 0 or end > len(source_text) or start >= end:
                return None, {"reason": "invalid_bounds", "item": item}
            if source_text[start:end] != text:
                return None, {
                    "reason": "offset_text_mismatch",
                    "item": item,
                    "source_substring": source_text[start:end],
                }

        validated["start"] = start
        validated["end"] = end

    return validated, None


def parse_model_response(raw_text: str, source_text: str | None = None) -> ParseResult:
    repair_actions: list[str] = []
    payload = _load_payload(raw_text, repair_actions)
    items = _flatten_payload(payload)

    parsed = ParseResult(skills=[], rejected_items=[], repair_actions=repair_actions, raw_payload=payload)
    for item in items:
        validated, rejected = _validate_item(item, source_text)
        if validated is not None:
            parsed.skills.append(validated)
        else:
            parsed.rejected_items.append(rejected)

    return parsed
