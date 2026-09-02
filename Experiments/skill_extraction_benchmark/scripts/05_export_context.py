#!/usr/bin/env python3
"""Export canonical data to ConTeXT sentence–skill pair format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.canonical_schema import load_canonical_split
from src.data.converters import _collect_all_skill_texts, canonical_to_context_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical → ConTeXT pair format.")
    parser.add_argument("--input", default="data/exports/canonical")
    parser.add_argument("--out", default="data/exports/context")
    parser.add_argument("--negatives_per_positive", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input) if Path(args.input).is_absolute() else PROJECT_ROOT / args.input
    out_dir = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all splits to build the full skill vocabulary
    all_records = {}
    for split in ("train", "dev", "test"):
        all_records[split] = load_canonical_split(input_dir / f"{split}.jsonl")

    # Build candidate skill inventory from TRAIN split only
    train_skill_texts = _collect_all_skill_texts(all_records["train"])
    print(f"  Candidate skill inventory: {len(train_skill_texts)} unique skills from train split")

    candidate_path = out_dir / "candidate_skills.json"
    candidate_path.write_text(
        json.dumps(train_skill_texts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Candidate skills → {candidate_path}")

    # For negative mining, use full skill inventory from the respective split's gold + train
    rng = random.Random(42)

    for split in ("train", "dev", "test"):
        records = all_records[split]
        # Use train skill texts as the pool for all splits (avoids data leakage)
        all_pairs = []
        for rec in records:
            pairs = canonical_to_context_pairs(
                rec,
                all_skill_texts=train_skill_texts,
                negatives_per_positive=args.negatives_per_positive,
                rng=rng,
            )
            all_pairs.extend(pairs)

        out_path = out_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        n_pos = sum(1 for p in all_pairs if p["label"] == 1)
        n_neg = sum(1 for p in all_pairs if p["label"] == 0)
        print(f"  {split}: {len(all_pairs)} pairs ({n_pos} pos, {n_neg} neg) → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
