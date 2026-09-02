#!/usr/bin/env python3
"""Download the jjzha/skillspan dataset from Hugging Face and save raw splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SkillSpan from Hugging Face.")
    parser.add_argument("--dataset", default="jjzha/skillspan", help="HF dataset identifier")
    parser.add_argument("--out", default="data/raw/skillspan", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load .env for HF_TOKEN if present
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    import urllib.request

    print(f"Downloading raw JSONL from Hugging Face: {args.dataset} ...")
    base_url = f"https://huggingface.co/datasets/{args.dataset}/raw/main"
    split_map = {"train": "train.json", "validation": "dev.json", "test": "test.json"}

    for hf_split, local_name in split_map.items():
        url = f"{base_url}/{local_name}"
        out_path = out_dir / local_name
        
        try:
            req = urllib.request.Request(url)
            # HF sometimes needs a User-Agent or it might 403
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
        except Exception as e:
            print(f"  WARNING: Failed to download source {local_name} from {url} ({e})")
            continue

        # Parse JSONL to a Python list
        records = []
        for line in content.strip().split("\n"):
            if line.strip():
                records.append(json.loads(line))

        # Write as standard JSON array (which 01_prepare_canonical expects)
        out_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  {local_name}: {len(records)} records → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
