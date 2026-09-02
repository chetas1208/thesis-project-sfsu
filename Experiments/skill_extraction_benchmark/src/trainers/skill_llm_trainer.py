"""Skill-LLM trainer: QLoRA SFT via LLaMA-Factory."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)


def build_train_yaml(config: Dict[str, Any], data_dir: Path, output_dir: Path) -> str:
    """Build a LLaMA-Factory training YAML config."""
    base = config["base_model"]
    template = config["template"]
    lora = config["lora"]
    train = config["training"]

    lines = [
        f"model_name_or_path: {base}",
        "stage: sft",
        "do_train: true",
        "finetuning_type: lora",
        f"lora_target: {lora['target_modules']}",
        f"lora_rank: {lora['rank']}",
        f"lora_alpha: {lora['alpha']}",
        f"lora_dropout: {lora['dropout']}",
        "",
        f"dataset_dir: {data_dir}",
        "dataset: skillspan_train",
        f"template: {template}",
        f"cutoff_len: {train['cutoff_len']}",
        "max_samples: 100000",
        "overwrite_cache: true",
        "preprocessing_num_workers: 4",
        "",
        f"output_dir: {output_dir}",
        f"logging_steps: {train['logging_steps']}",
        f"save_steps: {train['save_steps']}",
        "plot_loss: false",
        "report_to: none",
        "overwrite_output_dir: true",
        "",
        f"per_device_train_batch_size: {train['per_device_train_batch_size']}",
        f"gradient_accumulation_steps: {train['gradient_accumulation_steps']}",
        f"learning_rate: {train['learning_rate']}",
        f"num_train_epochs: {train['num_train_epochs']}",
        f"lr_scheduler_type: {train['lr_scheduler_type']}",
        f"warmup_ratio: {train['warmup_ratio']}",
        f"max_grad_norm: {train['max_grad_norm']}",
        f"bf16: {str(train['bf16']).lower()}",
        f"gradient_checkpointing: {str(train['gradient_checkpointing']).lower()}",
        "ddp_timeout: 180000000",
        "flash_attn: auto",
        f"optim: {train['optim']}",
        f"weight_decay: {train['weight_decay']}",
        "include_num_input_tokens_seen: true",
        "",
        "# QLoRA 4-bit quantization",
        "quantization_bit: 4",
        "quantization_method: bitsandbytes",
    ]
    return "\n".join(lines) + "\n"


def build_eval_yaml(config: Dict[str, Any], data_dir: Path, adapter_dir: Path, output_dir: Path) -> str:
    """Build a LLaMA-Factory evaluation YAML config."""
    base = config["base_model"]
    template = config["template"]
    gen = config["generation"]

    lines = [
        f"model_name_or_path: {base}",
        "stage: sft",
        "do_predict: true",
        "finetuning_type: lora",
        "",
        f"dataset_dir: {data_dir}",
        "eval_dataset: skillspan_test",
        f"template: {template}",
        "cutoff_len: 2048",
        "max_samples: 100000",
        "overwrite_cache: true",
        "preprocessing_num_workers: 4",
        "",
        f"adapter_name_or_path: {adapter_dir}",
        f"output_dir: {output_dir}",
        "report_to: none",
        "overwrite_output_dir: true",
        "bf16: true",
        "per_device_eval_batch_size: 1",
        "ddp_timeout: 180000000",
        "flash_attn: auto",
        "",
        "predict_with_generate: true",
        f"max_new_tokens: {gen['max_new_tokens']}",
        f"temperature: {gen['temperature']}",
        f"top_p: {gen['top_p']}",
        "",
        "quantization_bit: 4",
        "quantization_method: bitsandbytes",
    ]
    return "\n".join(lines) + "\n"


def train_skill_llm(
    config: Dict[str, Any],
    data_dir: Path,
    output_dir: Path,
) -> Path:
    """Run Skill-LLM QLoRA SFT training via LLaMA-Factory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    sft_dir = output_dir / "sft"

    train_yaml_str = build_train_yaml(config, data_dir, sft_dir)
    train_yaml_path = runtime_dir / "train.yaml"
    train_yaml_path.write_text(train_yaml_str, encoding="utf-8")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node", "2",
        "--module", "llamafactory.launcher",
        str(train_yaml_path),
    ]
    logger.info("Running Skill-LLM training: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    logger.info("Skill-LLM adapter saved to %s", sft_dir)
    return sft_dir


def predict_skill_llm(
    config: Dict[str, Any],
    data_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> Path:
    """Run Skill-LLM inference via LLaMA-Factory predict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    eval_yaml_str = build_eval_yaml(config, data_dir, adapter_dir, raw_dir)
    eval_yaml_path = runtime_dir / "eval.yaml"
    eval_yaml_path.write_text(eval_yaml_str, encoding="utf-8")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node", "2",
        "--module", "llamafactory.launcher",
        str(eval_yaml_path),
    ]
    logger.info("Running Skill-LLM inference: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    return raw_dir / "generated_predictions.jsonl"


def parse_skill_llm_output(generated_text: str) -> List[str]:
    """Parse a Skill-LLM generated JSON output to extract skill texts.

    Expected format: {"SKILL": [{"skill_span": "...", "context": "..."}]}
    Falls back to regex extraction if JSON parsing fails.
    """
    import re

    generated_text = generated_text.strip()
    skills: List[str] = []

    # Try JSON parsing first
    try:
        parsed = json.loads(generated_text)
        if isinstance(parsed, dict):
            for key in ("SKILL", "skill", "Skill", "skills", "SKILLS"):
                if key in parsed:
                    entries = parsed[key]
                    if isinstance(entries, list):
                        for entry in entries:
                            if isinstance(entry, dict):
                                text = entry.get("skill_span", entry.get("text", ""))
                            elif isinstance(entry, str):
                                text = entry
                            else:
                                continue
                            if text.strip():
                                skills.append(text.strip())
                    break
        if skills:
            return skills
    except (json.JSONDecodeError, TypeError, KeyError):
        pass

    # Fallback: extract from skill_span fields via regex
    for match in re.finditer(r'"skill_span"\s*:\s*"([^"]+)"', generated_text):
        skills.append(match.group(1).strip())

    return skills
