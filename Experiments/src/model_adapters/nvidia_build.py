import ast
import json
import re
from typing import Dict, List

from model_adapters.gpt5 import normalize_gpt5_response


THINK_TAG_PATTERN = re.compile(r"<\|channel\|>thought\s*.*?<channel\|>", re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def clean_generation_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = THINK_TAG_PATTERN.sub("", cleaned).strip()
    cleaned = CODE_FENCE_PATTERN.sub("", cleaned).strip()
    return cleaned


def extract_json_payload(raw_text: str) -> Dict:
    cleaned = clean_generation_text(raw_text)
    for loader in (json.loads, ast.literal_eval):
        try:
            payload = loader(cleaned)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        for loader in (json.loads, ast.literal_eval):
            try:
                payload = loader(candidate)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
    skills_key = re.search(r'"skills"\s*:\s*\[', cleaned, re.IGNORECASE)
    if skills_key:
        list_fragment = cleaned[skills_key.end() :]
        values = []
        for match in re.finditer(r'"((?:[^"\\]|\\.)*)"', list_fragment):
            try:
                values.append(json.loads(f'"{match.group(1)}"'))
            except json.JSONDecodeError:
                values.append(match.group(1))
        if values:
            return {"skills": values}
    raise ValueError("Could not parse NVIDIA Build response as JSON.")


def _extract_skill_strings(payload: Dict) -> List[str]:
    skills = payload.get("skills", [])
    if isinstance(skills, list):
        values = []
        for item in skills:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("skill") or "").strip()
            else:
                text = ""
            if text:
                values.append(text)
        return values
    return []


def _find_skill_span(text: str, surface_form: str, occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    patterns = [
        re.compile(rf"\b{re.escape(surface_form)}\b", re.IGNORECASE),
        re.compile(re.escape(surface_form), re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            candidate = (match.start(), match.end() - 1)
            if candidate in occupied:
                continue
            occupied.add(candidate)
            return candidate
    return None


def normalize_nvidia_build_response(record_text: str, raw_text: str) -> List[Dict]:
    payload = extract_json_payload(raw_text)
    if payload.get("skills") and payload["skills"] and not isinstance(payload["skills"][0], dict):
        occupied = set()
        spans = []
        for skill_text in _extract_skill_strings(payload):
            location = _find_skill_span(record_text, skill_text, occupied)
            if location is None:
                continue
            start, end = location
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": "SKILL",
                    "text": record_text[start : end + 1],
                }
            )
        spans.sort(key=lambda item: (item["start"], item["end"]))
        return spans
    return normalize_gpt5_response(record_text, payload)
