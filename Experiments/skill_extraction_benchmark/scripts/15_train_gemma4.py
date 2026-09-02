#!/usr/bin/env python3
"""Train Gemma 4 (QLoRA SFT)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_gemma4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gemma 4 via QLoRA.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/gemma4.yaml")
    parser.add_argument("--data", default=None, help="Override data dir")
    parser.add_argument("--output", default=None, help="Override output dir")
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    mcfg = yaml.safe_load((PROJECT_ROOT / args.model_config).read_text())

    data_dir = Path(args.data) if args.data else PROJECT_ROOT / gcfg["paths"]["exports"] / "gemma4"
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / gcfg["paths"]["models"] / "gemma4"
    
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    from src.trainers.gemma4_trainer import train_gemma4

    logger.info("Training Gemma 4 (%s) with QLoRA ...", mcfg["model_name"])
    adapter_path = train_gemma4(mcfg, data_dir, output_dir)
    logger.info("Done. Checkpoint: %s", adapter_path)


if __name__ == "__main__":
    main()
