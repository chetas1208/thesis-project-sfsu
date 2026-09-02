#!/usr/bin/env python3
"""Convert raw SkillSpan JSON into the canonical JSONL format.

Reads raw downloads, maps generic B/I/O → B-SKILL/I-SKILL/O,
builds token spans, detokenises text, and writes canonical splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canonical_schema import (
    CanonicalRecord,
    bio_to_spans,
    detokenize,
    write_canonical_split,
)


def _map_bio_tags(raw_tags: list[str]) -> list[str]:
    """Map generic B/I/O tags to B-SKILL/I-SKILL/O."""
    mapped = []
    for tag in raw_tags:
        tag = tag.strip()
        if tag == "B":
            mapped.append("B-SKILL")
        elif tag == "I":
            mapped.append("I-SKILL")
        elif tag.startswith("B-"):
            mapped.append("B-SKILL")
        elif tag.startswith("I-"):
            mapped.append("I-SKILL")
        else:
            mapped.append("O")
    return mapped


def process_split(
    raw_records: list[dict],
    split_name: str,
    target_tag_key: str = "tags_skill",
) -> list[CanonicalRecord]:
    """Process one split of raw SkillSpan data into canonical records."""
    canonical: list[CanonicalRecord] = []
    # Track row number within each document idx for sentence_id generation
    doc_row_counts: Counter = Counter()

    for record in raw_records:
        tokens = record["tokens"]
        raw_tags = record.get(target_tag_key, [])
        if not raw_tags:
            raw_tags = ["O"] * len(tokens)

        doc_id = str(record["idx"])
        doc_row_counts[doc_id] += 1
        row_num = doc_row_counts[doc_id]
        sentence_id = f"{doc_id}_{row_num}"

        bio_tags = _map_bio_tags(raw_tags)
        spans = bio_to_spans(tokens, bio_tags, label="SKILL")
        text = detokenize(tokens)
        source = record.get("source", "")

        canonical.append(CanonicalRecord(
            split=split_name,
            doc_id=doc_id,
            sentence_id=sentence_id,
            tokens=tokens,
            bio_tags=bio_tags,
            spans=spans,
            text=text,
            source=source,
        ))

    return canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical JSONL from raw SkillSpan.")
    parser.add_argument("--input", default="data/raw/skillspan", help="Raw data directory")
    parser.add_argument("--out", default="data/exports/canonical", help="Canonical output directory")
    parser.add_argument("--target", default="tags_skill", help="Tag column to use")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}
    total_spans = 0
    total_records = 0

    for split_name in ("train", "dev", "test"):
        raw_path = input_dir / f"{split_name}.json"
        if not raw_path.exists():
            print(f"  WARNING: {raw_path} not found, skipping")
            continue

        raw_records = json.loads(raw_path.read_text(encoding="utf-8"))
        canonical = process_split(raw_records, split_name, target_tag_key=args.target)

        write_canonical_split(out_dir / f"{split_name}.jsonl", canonical)

        n_spans = sum(len(r.spans) for r in canonical)
        n_docs = len(set(r.doc_id for r in canonical))
        total_spans += n_spans
        total_records += len(canonical)

        print(f"  {split_name}: {len(canonical)} sentences, {n_docs} documents, {n_spans} skill spans → {out_dir / f'{split_name}.jsonl'}")

    # Write label map
    label_map_path = out_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2) + "\n", encoding="utf-8")

    print(f"\nTotal: {total_records} sentences, {total_spans} skill spans")
    print(f"Label map → {label_map_path}")
    print("Done.")


if __name__ == "__main__":
    main()
