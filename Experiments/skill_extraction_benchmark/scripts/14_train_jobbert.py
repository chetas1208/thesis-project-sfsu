#!/usr/bin/env python3
"""Train the JobBERT token-classification baseline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_jobbert")


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JobBERT token classification.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/jobbert.yaml")
    args = parser.parse_args()

    gcfg_path = Path(args.global_config) if Path(args.global_config).is_absolute() else PROJECT_ROOT / args.global_config
    mcfg_path = Path(args.model_config) if Path(args.model_config).is_absolute() else PROJECT_ROOT / args.model_config

    gcfg = yaml.safe_load(gcfg_path.read_text())
    mcfg = yaml.safe_load(mcfg_path.read_text())

    data_dir = PROJECT_ROOT / gcfg["paths"]["exports"] / "jobbert"
    output_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "jobbert"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(data_dir / "train.jsonl")
    dev_records = load_records(data_dir / "dev.jsonl")

    logger.info("Training JobBERT (%s): %d train, %d dev", mcfg["base_model"], len(train_records), len(dev_records))

    from src.trainers.jobbert_trainer import train_jobbert

    checkpoint = train_jobbert(train_records, dev_records, mcfg, output_dir, seed=gcfg["seed"])
    logger.info("Done. Checkpoint: %s", checkpoint)


if __name__ == "__main__":
    main()
