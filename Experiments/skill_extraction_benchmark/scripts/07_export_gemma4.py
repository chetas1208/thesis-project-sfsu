#!/usr/bin/env python3
"""Export canonical data to Gemma 4 instruction-tuning format (strict JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canonical_schema import load_canonical_split


_GEMMA4_INSTRUCTION = "Extract skill entities from the given sentence and return strict JSON."


def canonical_to_gemma4(record) -> dict:
    """Convert one canonical record to Gemma 4 instruction format."""
    unique_skills = []
    seen = set()
    for span in record.spans:
        if span.label == "SKILL":
            text = span.text.strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                unique_skills.append({"text": text})

    output_dict = {"skills": unique_skills}
    output_json = json.dumps(output_dict, separators=(",", ":"), ensure_ascii=False)

    return {
        "instruction": _GEMMA4_INSTRUCTION,
        "input": record.text,
        "output": output_json,
        "id": record.sentence_id,
        "doc_id": record.doc_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical → Gemma 4 format.")
    parser.add_argument("--input", default="data/exports/canonical")
    parser.add_argument("--out", default="data/exports/gemma4")
    args = parser.parse_args()

    input_dir = Path(args.input) if Path(args.input).is_absolute() else PROJECT_ROOT / args.input
    out_dir = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        records = load_canonical_split(input_dir / f"{split}.jsonl")
        exported = [canonical_to_gemma4(r) for r in records]

        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in exported:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  {split}: {len(exported)} records → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
