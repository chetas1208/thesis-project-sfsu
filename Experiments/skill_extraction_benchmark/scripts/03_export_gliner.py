#!/usr/bin/env python3
"""Export canonical data to GLiNER format (JSON arrays)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canonical_schema import load_canonical_split
from src.data.converters import canonical_to_gliner


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical → GLiNER format.")
    parser.add_argument("--input", default="data/exports/canonical")
    parser.add_argument("--out", default="data/exports/gliner")
    args = parser.parse_args()

    input_dir = Path(args.input) if Path(args.input).is_absolute() else PROJECT_ROOT / args.input
    out_dir = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        records = load_canonical_split(input_dir / f"{split}.jsonl")
        exported = [canonical_to_gliner(r) for r in records]

        # GLiNER expects JSON arrays, not JSONL
        out_path = out_dir / f"{split}.json"
        out_path.write_text(
            json.dumps(exported, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        n_positive = sum(1 for r in exported if r["ner"])
        print(f"  {split}: {len(exported)} records ({n_positive} with spans) → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
