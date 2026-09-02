from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from skillllm_v2.io_utils import write_json


def train_or_verify_mistral_lora(
    model_path: str = "models/mistral-small-3.1-24b-instruct-2503",
    adapter_path: str = "artifacts/mistral-small-3.1-24b-skillspan-lora",
    output_dir: str = "artifacts/skillllm_v2/mistral_lora/model",
    train_mistral_lora: bool = False,
    train_file: str = "data/skillllm_v2/mistral_sft/train.jsonl",
    dev_file: str = "data/skillllm_v2/mistral_sft/dev.jsonl",
    num_train_epochs: int | float = 3,
    num_processes: int = 2,
    cuda_visible_devices: str = "0,1",
    launch_training: bool = True,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_exists = Path(model_path).exists()
    adapter_exists = Path(adapter_path).exists()
    if train_mistral_lora:
        require_paths = [model_path, train_file, dev_file]
        missing = [path for path in require_paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Cannot train Mistral LoRA; missing required paths: {missing}")
        train_config = _build_training_config(
            model_path=model_path,
            output_dir=str(output),
            train_file=train_file,
            dev_file=dev_file,
            num_train_epochs=num_train_epochs,
            extra_config=extra_config or {},
        )
        train_config_path = output / "mistral_lora_train_config.yaml"
        _write_yaml(train_config_path, train_config)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(num_processes),
            "-m",
            "skillllm_v2.train.mistral_lora_trainer",
            "--config",
            str(train_config_path),
        ]
        status = {
            "training_mode": "launching_lora_training" if launch_training else "training_command_prepared",
            "model_path": model_path,
            "output_dir": str(output),
            "train_file": train_file,
            "dev_file": dev_file,
            "num_train_epochs": num_train_epochs,
            "num_processes": num_processes,
            "cuda_visible_devices": cuda_visible_devices,
            "command": command,
        }
        write_json(output / "adapter_status.json", status)
        if launch_training:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            env.setdefault("PYTHONPATH", "src")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            completed = subprocess.run(command, env=env, text=True)
            if completed.returncode != 0:
                status["training_mode"] = "failed_lora_training"
                status["returncode"] = completed.returncode
                write_json(output / "adapter_status.json", status)
                raise RuntimeError(f"Mistral LoRA training failed with return code {completed.returncode}")
            summary_path = output / "training_summary.json"
            status["training_mode"] = "trained_lora"
            status["training_summary_path"] = str(summary_path)
    elif adapter_exists:
        status = {"training_mode": "reused_existing_adapter", "model_path": model_path, "adapter_path": adapter_path, "model_exists": model_exists}
    else:
        status = {"training_mode": "adapter_missing_smoke_fallback", "model_path": model_path, "adapter_path": adapter_path, "model_exists": model_exists}
    write_json(output / "adapter_status.json", status)
    return status


def _build_training_config(
    model_path: str,
    output_dir: str,
    train_file: str,
    dev_file: str,
    num_train_epochs: int | float,
    extra_config: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "model_path": model_path,
        "output_dir": output_dir,
        "train_file": train_file,
        "dev_file": dev_file,
        "num_train_epochs": num_train_epochs,
        "max_seq_length": 2048,
        "learning_rate": 0.00012,
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        "use_4bit": True,
        "gradient_checkpointing": True,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "seed": 42,
        "save_total_limit": 2,
        "attn_implementation": "sdpa",
    }
    config.update(extra_config)
    return config


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/mistral-small-3.1-24b-instruct-2503")
    parser.add_argument("--adapter-path", default="artifacts/mistral-small-3.1-24b-skillspan-lora")
    parser.add_argument("--output-dir", default="artifacts/skillllm_v2/mistral_lora/model")
    parser.add_argument("--train-mistral-lora", action="store_true")
    parser.add_argument("--train-file", default="data/skillllm_v2/mistral_sft/train.jsonl")
    parser.add_argument("--dev-file", default="data/skillllm_v2/mistral_sft/dev.jsonl")
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    train_or_verify_mistral_lora(
        args.model_path,
        args.adapter_path,
        args.output_dir,
        args.train_mistral_lora,
        args.train_file,
        args.dev_file,
        args.num_train_epochs,
        args.num_processes,
        args.cuda_visible_devices,
        not args.prepare_only,
    )


if __name__ == "__main__":
    main()
