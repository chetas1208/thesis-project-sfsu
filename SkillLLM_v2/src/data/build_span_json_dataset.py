from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data.dataset_utils import build_example_from_row, load_skillspan_dataset
from src.runtime_utils import ensure_dir, write_json, write_jsonl


SPLIT_NAME_MAP = {
    "train": "train",
    "validation": "val",
    "dev": "val",
    "test": "test",
}


def build_split(dataset_split, split_name: str) -> list[dict[str, object]]:
    examples = []
    for row_index, row in enumerate(dataset_split):
        examples.append(build_example_from_row(row, split_name=split_name, row_index=row_index))
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact-span JSONL files from SkillSpan.")
    parser.add_argument("--dataset", default="jjzha/skillspan", help="Hugging Face dataset id.")
    parser.add_argument("--output-dir", default="data/v2", help="Output directory for v2 jsonl files.")
    args = parser.parse_args()

    dataset = load_skillspan_dataset(args.dataset)
    output_dir = ensure_dir(args.output_dir)
    summary: dict[str, object] = {"dataset": args.dataset, "splits": {}}

    for source_split_name, target_split_name in SPLIT_NAME_MAP.items():
        if source_split_name not in dataset:
            continue
        examples = build_split(dataset[source_split_name], split_name=target_split_name)
        output_path = output_dir / f"{target_split_name}.jsonl"
        write_jsonl(output_path, examples)

        label_counter = Counter()
        for example in examples:
            for skill in example["skills"]:
                label_counter.update([skill["label"]])

        summary["splits"][target_split_name] = {
            "examples": len(examples),
            "skills": dict(label_counter),
            "output_file": str(output_path),
        }
        print(f"Wrote {len(examples)} examples to {output_path}")

    write_json(Path(output_dir) / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
