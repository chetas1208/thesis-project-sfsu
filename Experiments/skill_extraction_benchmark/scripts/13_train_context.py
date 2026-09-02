#!/usr/bin/env python3
"""Train the ConTeXT sentence-transformer model."""

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
logger = logging.getLogger("train_context")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ConTeXT sentence-transformer.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/context.yaml")
    args = parser.parse_args()

    gcfg_path = Path(args.global_config) if Path(args.global_config).is_absolute() else PROJECT_ROOT / args.global_config
    mcfg_path = Path(args.model_config) if Path(args.model_config).is_absolute() else PROJECT_ROOT / args.model_config

    gcfg = yaml.safe_load(gcfg_path.read_text())
    mcfg = yaml.safe_load(mcfg_path.read_text())

    data_dir = PROJECT_ROOT / gcfg["paths"]["exports"] / "context"
    output_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "context"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.trainers.context_trainer import sweep_threshold, train_context

    logger.info("Training ConTeXT (%s) ...", mcfg["base_model"])
    checkpoint = train_context(mcfg, data_dir, output_dir)

    logger.info("Sweeping threshold on dev set ...")
    threshold, dev_metrics = sweep_threshold(
        checkpoint,
        data_dir / "dev.jsonl",
        encode_batch_size=mcfg["inference"]["encode_batch_size"],
    )

    # Save threshold for inference
    threshold_path = output_dir / "best_threshold.json"
    threshold_path.write_text(
        json.dumps({"threshold": threshold, "dev_metrics": dev_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("Done. Checkpoint: %s, Threshold: %.4f (dev F1=%.4f)", checkpoint, threshold, dev_metrics.get("f1", 0.0))


if __name__ == "__main__":
    main()
