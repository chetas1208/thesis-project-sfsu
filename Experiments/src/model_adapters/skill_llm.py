import ast
import json
import re
from pathlib import Path
from typing import Dict, List

from common import ensure_dir, write_json, write_jsonl


INSTRUCTION = "Extract skill entities and knowledge entities from the given sentence."


def _to_instruction_record(record: Dict) -> Dict:
    skills = []
    for span in record["spans"]:
        if span["text"] not in skills:
            skills.append(span["text"])
    output = {"skills": skills, "knowledge": []}
    return {
        "id": record["id"],
        "resume_id": record.get("resume_id"),
        "section_name": record.get("section_name"),
        "instruction": INSTRUCTION,
        "input": record["text"],
        "output": json.dumps(output, ensure_ascii=False),
    }


def export_skill_llm_dataset(splits: Dict[str, List[Dict]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    for split_name, records in splits.items():
        canonical_name = "validation" if split_name == "dev" else split_name
        write_jsonl(output_dir / f"{canonical_name}.jsonl", records)
        write_json(output_dir / f"{canonical_name}.json", [_to_instruction_record(record) for record in records])

    dataset_info = {
        "skillspan_train": {
            "file_name": "train.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "skillspan_dev": {
            "file_name": "validation.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "skillspan_test": {
            "file_name": "test.json",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }
    write_json(output_dir / "dataset_info.json", dataset_info)


def _parse_generation_payload(payload) -> Dict:
    if isinstance(payload, dict):
        return payload
    if payload in (None, ""):
        return {}
    if not isinstance(payload, str):
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(payload)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            continue
    return {}


def _extract_skill_strings(payload: Dict) -> List[str]:
    if "skills" in payload and isinstance(payload["skills"], list):
        return [str(item).strip() for item in payload["skills"] if str(item).strip()]

    skills = []
    for key in ("SKILL", "skill"):
        for item in payload.get(key, []):
            if isinstance(item, dict):
                value = item.get("skill_span") or item.get("text")
            else:
                value = item
            value = str(value).strip() if value is not None else ""
            if value:
                skills.append(value)
    return skills


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


def generated_skill_llm_predictions_to_unified(prediction_rows: List[Dict], canonical_records: List[Dict]) -> List[Dict]:
    if len(prediction_rows) != len(canonical_records):
        raise ValueError(
            f"Prediction row count mismatch: {len(prediction_rows)} Skill-LLM rows vs "
            f"{len(canonical_records)} canonical records."
        )
    unified = []
    for raw_prediction, record in zip(prediction_rows, canonical_records):
        payload = _parse_generation_payload(
            raw_prediction.get("predict")
            or raw_prediction.get("response")
            or raw_prediction.get("output")
        )
        occupied = set()
        spans = []
        for skill_text in _extract_skill_strings(payload):
            location = _find_skill_span(record["text"], skill_text, occupied)
            if location is None:
                continue
            start, end = location
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": "SKILL",
                    "text": record["text"][start : end + 1],
                }
            )
        spans.sort(key=lambda item: (item["start"], item["end"]))
        unified.append({"id": record["id"], "predicted_spans": spans})
    return unified


def processed_skill_llm_to_unified(records: List[Dict]) -> List[Dict]:
    unified = []
    for record in records:
        spans = []
        for span in record.get("spans", []):
            spans.append(
                {
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "label": "SKILL",
                    "text": record["text"][int(span["start"]) : int(span["end"]) + 1],
                }
            )
        unified.append({"id": record.get("id"), "predicted_spans": spans, "text": record.get("text")})
    return unified
