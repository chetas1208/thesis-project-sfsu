#!/usr/bin/env python3
"""Train Skill-LLM via QLoRA SFT using LLaMA-Factory."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("train_skillllm")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Skill-LLM (QLoRA SFT on Llama 3).")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/skill_llm.yaml")
    args = parser.parse_args()

    gcfg_path = Path(args.global_config) if Path(args.global_config).is_absolute() else PROJECT_ROOT / args.global_config
    mcfg_path = Path(args.model_config) if Path(args.model_config).is_absolute() else PROJECT_ROOT / args.model_config

    gcfg = yaml.safe_load(gcfg_path.read_text())
    mcfg = yaml.safe_load(mcfg_path.read_text())

    data_dir = PROJECT_ROOT / gcfg["paths"]["exports"] / "skill_llm"
    output_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "skill_llm"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.trainers.skill_llm_trainer import train_skill_llm

    logger.info("Training Skill-LLM (%s) with QLoRA ...", mcfg["base_model"])
    adapter_path = train_skill_llm(mcfg, data_dir, output_dir)
    logger.info("Done. Adapter: %s", adapter_path)


if __name__ == "__main__":
    main()
