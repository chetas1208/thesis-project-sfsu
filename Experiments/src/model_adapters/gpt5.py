from pathlib import Path
from typing import Dict, List

from common import ensure_dir, write_jsonl


def build_prompt(prompt_template: str, record: Dict) -> str:
    metadata = []
    if record.get("resume_id"):
        metadata.append(f"Resume ID: {record['resume_id']}")
    if record.get("section_name"):
        metadata.append(f"Section: {record['section_name']}")
    if record.get("role_hint"):
        metadata.append(f"Role hint: {record['role_hint']}")

    prompt_parts = [prompt_template.strip()]
    if metadata:
        prompt_parts.append("\n".join(metadata))
    prompt_parts.append(f"Input sentence:\n{record['text']}")
    return "\n\n".join(prompt_parts)


def export_gpt5_prompts(splits: Dict[str, List[Dict]], output_dir: Path, prompt_template: str) -> None:
    ensure_dir(output_dir)
    for split_name, records in splits.items():
        prompts = []
        for record in records:
            prompts.append(
                {
                    "id": record["id"],
                    "resume_id": record.get("resume_id"),
                    "section_name": record.get("section_name"),
                    "text": record["text"],
                    "prompt": build_prompt(prompt_template, record),
                }
            )
        write_jsonl(output_dir / f"{split_name}_prompts.jsonl", prompts)


def normalize_gpt5_response(record_text: str, response_payload: Dict) -> List[Dict]:
    spans = []
    for item in response_payload.get("skills", []):
        start = int(item.get("start_char", item.get("start")))
        end = int(item.get("end_char", item.get("end")))
        if start < 0 or end < start or end >= len(record_text):
            continue
        text = record_text[start : end + 1]
        if item.get("text") and text != item["text"]:
            continue
        spans.append(
            {
                "start": start,
                "end": end,
                "label": "SKILL",
                "text": text,
                "score": item.get("confidence", item.get("score")),
                "canonical_name": item.get("canonical_name"),
            }
        )
    spans.sort(key=lambda span: (span["start"], span["end"]))
    return spans
