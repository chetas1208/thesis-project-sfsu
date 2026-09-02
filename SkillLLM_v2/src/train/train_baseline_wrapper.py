from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

from src.data.dataset_utils import normalize_label
from src.runtime_utils import (
    apply_overrides,
    ensure_dir,
    load_yaml_config,
    read_jsonl,
    save_yaml_config,
    write_jsonl,
)


ROOT_DIR = Path(__file__).resolve().parents[2]


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def build_train_yaml(config: dict[str, object]) -> dict[str, object]:
    return {
        "model_name_or_path": config["local_model_path"],
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": ",".join(config["target_modules"]),
        "lora_rank": int(config["lora_r"]),
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "dataset_dir": "data",
        "dataset": "skillspan_train",
        "template": "llama3",
        "cutoff_len": int(config["max_seq_length"]),
        "max_samples": 100000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "output_dir": str(Path(config["output_dir"]) / "baseline_lora" / "sft"),
        "logging_steps": 10,
        "save_steps": 200,
        "plot_loss": False,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": int(config["batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["epochs"]),
        "lr_scheduler_type": "cosine",
        "warmup_ratio": float(config.get("warmup_ratio", 0.1)),
        "max_grad_norm": 0.3,
        "bf16": False,
        "fp16": True,
        "ddp_timeout": 180000000,
        "flash_attn": "auto" if config.get("use_flash_attention_if_available", True) else "disabled",
        "optim": "adamw_torch",
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "include_num_input_tokens_seen": True,
    }


def build_eval_yaml(config: dict[str, object], adapter_dir: Path) -> dict[str, object]:
    return {
        "model_name_or_path": config["local_model_path"],
        "stage": "sft",
        "do_predict": True,
        "finetuning_type": "lora",
        "dataset_dir": "data",
        "eval_dataset": "skillspan_test",
        "template": "llama3",
        "cutoff_len": int(config["max_seq_length"]),
        "max_samples": 100000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "adapter_name_or_path": str(adapter_dir),
        "output_dir": str(adapter_dir.parent / "results"),
        "overwrite_output_dir": True,
        "bf16": False,
        "fp16": True,
        "per_device_eval_batch_size": int(config.get("eval_batch_size", config["batch_size"])),
        "ddp_timeout": 180000000,
        "flash_attn": "auto" if config.get("use_flash_attention_if_available", True) else "disabled",
        "predict_with_generate": True,
        "max_new_tokens": int(config.get("max_new_tokens", 512)),
        "top_p": 1.0,
        "temperature": 0.0,
    }


def convert_processed_predictions(processed_file: Path, gold_file: Path, output_file: Path) -> None:
    gold_records = read_jsonl(gold_file)
    processed_records = read_jsonl(processed_file)

    gold_by_text: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    for record in gold_records:
        gold_by_text[record["text"]].append(record)

    converted = []
    for index, record in enumerate(processed_records):
        gold_record = gold_by_text[record["text"]].popleft() if gold_by_text[record["text"]] else None
        predicted_skills = []
        for span in record.get("spans", []):
            label = normalize_label(span.get("label"))
            if label is None:
                continue
            start = int(span["start"])
            end = int(span["end"]) + 1
            predicted_skills.append(
                {
                    "text": record["text"][start:end],
                    "label": label,
                    "start": start,
                    "end": end,
                    "recovery_type": "exact",
                }
            )

        converted.append(
            {
                "id": gold_record["id"] if gold_record else f"baseline_test_{index}",
                "text": record["text"],
                "predicted_skills": predicted_skills,
            }
        )

    write_jsonl(output_file, converted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the upstream Skill-LLM baseline with local models.")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--override", action="append", default=[], help="Config override in key=value format.")
    parser.add_argument("--skip-data", action="store_true", help="Skip upstream data preparation.")
    args = parser.parse_args()

    config = apply_overrides(load_yaml_config(args.config), args.override)
    output_dir = ensure_dir(config["output_dir"])
    prediction_dir = ensure_dir(output_dir / "predictions")
    run_config_dir = ensure_dir(output_dir / "llamafactory_configs")

    if not args.skip_data:
        run_command(["bash", "src/data_preparation.sh"])

    train_yaml = run_config_dir / "baseline_train.yaml"
    eval_yaml = run_config_dir / "baseline_eval.yaml"
    train_payload = build_train_yaml(config)
    eval_payload = build_eval_yaml(config, Path(train_payload["output_dir"]))
    save_yaml_config(train_yaml, train_payload)
    save_yaml_config(eval_yaml, eval_payload)

    run_command([sys.executable, "-c", "from llamafactory.train.tuner import run_exp; run_exp()", str(train_yaml)])
    run_command([sys.executable, "-c", "from llamafactory.train.tuner import run_exp; run_exp()", str(eval_yaml)])

    generated_predictions = Path(eval_payload["output_dir"]) / "generated_predictions.jsonl"
    processed_predictions = Path(eval_payload["output_dir"]) / "processed_predictions.jsonl"
    run_command(
        [
            sys.executable,
            "src/postprocessing/combine_prediction_results.py",
            "--base_file",
            "data/test.jsonl",
            "--input_file",
            "data/test.json",
            "--prediction_file",
            str(generated_predictions),
            "--output_file",
            str(processed_predictions),
            "--tokenizer",
            "white_space",
        ]
    )

    convert_processed_predictions(
        processed_file=processed_predictions,
        gold_file=ROOT_DIR / config["test_file"],
        output_file=prediction_dir / "test_predictions.jsonl",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
