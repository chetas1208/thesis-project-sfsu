#!/usr/bin/env python3
"""Export canonical data to Skill-LLM instruction-tuning format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canonical_schema import load_canonical_split
from src.data.converters import build_skill_llm_dataset_info, canonical_to_skill_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical → Skill-LLM format.")
    parser.add_argument("--input", default="data/exports/canonical")
    parser.add_argument("--out", default="data/exports/skill_llm")
    args = parser.parse_args()

    input_dir = Path(args.input) if Path(args.input).is_absolute() else PROJECT_ROOT / args.input
    out_dir = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        records = load_canonical_split(input_dir / f"{split}.jsonl")
        exported = [canonical_to_skill_llm(r) for r in records]

        out_path = out_dir / f"{split}.json"
        out_path.write_text(
            json.dumps(exported, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  {split}: {len(exported)} records → {out_path}")

    # Write dataset_info.json for LLaMA-Factory
    dataset_info = build_skill_llm_dataset_info(out_dir)
    info_path = out_dir / "dataset_info.json"
    info_path.write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  dataset_info.json → {info_path}")
    print("Done.")


if __name__ == "__main__":
    main()
