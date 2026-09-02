from pathlib import Path
from typing import Dict, List

from common import ensure_dir, write_json


def export_gliner_dataset(splits: Dict[str, List[Dict]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    mapping = {"train": "train.json", "dev": "dev.json", "test": "test.json"}
    for split_name, filename in mapping.items():
        converted = []
        for record in splits[split_name]:
            spans = []
            for span in record["spans"]:
                covered = [
                    idx
                    for idx, token in enumerate(record["tokens"])
                    if int(token["start"]) >= int(span["start"]) and int(token["end"]) <= int(span["end"])
                ]
                if not covered:
                    continue
                spans.append([covered[0], covered[-1], "skill"])
            converted.append(
                {
                    "id": record["id"],
                    "resume_id": record.get("resume_id"),
                    "section_name": record.get("section_name"),
                    "tokenized_text": [token["text"] for token in record["tokens"]],
                    "ner": spans,
                }
            )
        write_json(output_dir / filename, converted)


def normalize_gliner_entities(text: str, entities: List[Dict]) -> List[Dict]:
    spans = []
    for entity in entities:
        if "start" in entity and "end" in entity:
            start = int(entity["start"])
            end = int(entity["end"])
            if end > 0 and text[start:end] == entity.get("text", text[start:end]):
                end -= 1
        else:
            span_text = entity["text"]
            start = text.find(span_text)
            if start < 0:
                continue
            end = start + len(span_text) - 1
        spans.append(
            {
                "start": start,
                "end": end,
                "label": "SKILL",
                "text": text[start : end + 1],
                "score": entity.get("score"),
            }
        )
    spans.sort(key=lambda item: (item["start"], item["end"]))
    return spans
