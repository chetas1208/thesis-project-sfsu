#!/usr/bin/env python3
"""Train the NNOSE model: fine-tune encoder + build kNN datastore."""

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
logger = logging.getLogger("train_nnose")


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NNOSE (encoder + datastore).")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/nnose.yaml")
    parser.add_argument("--skip-encoder", action="store_true", help="Skip encoder training, only build datastore")
    args = parser.parse_args()

    gcfg_path = Path(args.global_config) if Path(args.global_config).is_absolute() else PROJECT_ROOT / args.global_config
    mcfg_path = Path(args.model_config) if Path(args.model_config).is_absolute() else PROJECT_ROOT / args.model_config

    gcfg = yaml.safe_load(gcfg_path.read_text())
    mcfg = yaml.safe_load(mcfg_path.read_text())

    data_dir = PROJECT_ROOT / gcfg["paths"]["exports"] / "nnose"
    output_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "nnose"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_records(data_dir / "train.jsonl")
    dev_records = load_records(data_dir / "dev.jsonl")

    logger.info("Train: %d records, Dev: %d records", len(train_records), len(dev_records))

    from src.trainers.nnose_trainer import build_datastore, train_encoder

    if not args.skip_encoder:
        logger.info("Phase 1: Training encoder (%s) ...", mcfg["backbone"])
        encoder_path = train_encoder(train_records, dev_records, mcfg, output_dir, seed=gcfg["seed"])
    else:
        encoder_path = output_dir / "encoder" / "best"
        if not encoder_path.exists():
            raise FileNotFoundError(f"No encoder checkpoint at {encoder_path}. Run without --skip-encoder first.")
        logger.info("Skipping encoder training, using %s", encoder_path)

    logger.info("Phase 2: Building kNN datastore ...")
    datastore_path = build_datastore(encoder_path, train_records, mcfg, output_dir, seed=gcfg["seed"])

    logger.info("Done. Encoder: %s, Datastore: %s", encoder_path, datastore_path)


if __name__ == "__main__":
    main()
