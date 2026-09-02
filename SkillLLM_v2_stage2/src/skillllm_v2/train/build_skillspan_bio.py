from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from skillllm_v2.io_utils import write_jsonl
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

LABELS = ["O", "B-HARD_SKILL", "I-HARD_SKILL", "B-SOFT_SKILL", "I-SOFT_SKILL"]


def build_bio_dataset(dataset_root: str | Path, output_dir: str | Path, limit: int | None = None) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ["train", "dev", "test"]:
        docs = load_skillspan_split(Path(dataset_root), split, limit=limit)
        rows = [_doc_to_bio(doc) for doc in docs]
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = str(path)
    return paths


def _doc_to_bio(doc: dict[str, Any]) -> dict[str, Any]:
    labels = ["O"] * len(doc["tokens"])
    token_offsets = _token_offsets(doc["text"], doc["tokens"])
    for span in doc["gold_spans"]:
        covered = [i for i, (start, end) in enumerate(token_offsets) if max(start, span["start"]) < min(end, span["end"])]
        if not covered:
            continue
        labels[covered[0]] = f"B-{span['label']}"
        for index in covered[1:]:
            labels[index] = f"I-{span['label']}"
    return {"id": doc["id"], "tokens": doc["tokens"], "labels": labels, "text": doc["text"], "spans": doc["gold_spans"]}


def _token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    offsets = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/source/skillspan_hf")
    parser.add_argument("--output-dir", default="data/skillllm_v2/bio")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build_bio_dataset(args.dataset_path, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
