from pathlib import Path
from typing import Dict, Iterable, List

from common import bio_tags_from_spans, ensure_dir, labeled_bio_tags_from_spans, write_jsonl


def export_token_classification_split(records: Iterable[Dict], output_path: Path) -> None:
    rows: List[Dict] = []
    for record in records:
        rows.append(
            {
                "idx": record["id"],
                "tokens": [token["text"] for token in record["tokens"]],
                "tags_skill": bio_tags_from_spans(record["tokens"], record["spans"]),
                "ner_tags": labeled_bio_tags_from_spans(record["tokens"], record["spans"]),
            }
        )
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(__import__("json").dumps(row, ensure_ascii=False))
            handle.write("\n")


def export_token_classification_dataset(splits: Dict[str, List[Dict]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    mapping = {"train": "train.json", "dev": "dev.json", "test": "test.json"}
    for split_name, filename in mapping.items():
        export_token_classification_split(splits[split_name], output_dir / filename)
