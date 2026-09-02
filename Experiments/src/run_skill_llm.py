import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import (
    ensure_benchmark_env,
    ensure_dir,
    load_config,
    load_env_file,
    read_json_records,
    resolve_path,
    write_jsonl,
    write_run_metadata,
)
from model_adapters.skill_llm import generated_skill_llm_predictions_to_unified


def infer_template(base_model: str) -> str:
    model_name = base_model.lower()
    if "qwen" in model_name:
        return "qwen"
    if "gemma" in model_name:
        return "gemma"
    if "phi" in model_name:
        return "phi"
    return "llama3"


def build_train_yaml(base_model: str, input_dir: Path, output_dir: Path, template: str) -> str:
    return "\n".join(
        [
            f"model_name_or_path: {base_model}",
            "stage: sft",
            "do_train: true",
            "finetuning_type: lora",
            "lora_target: q_proj,v_proj",
            "lora_rank: 64",
            "lora_alpha: 128",
            "lora_dropout: 0.1",
            f"dataset_dir: {input_dir}",
            "dataset: skillspan_train",
            f"template: {template}",
            "cutoff_len: 2000",
            "max_samples: 100000",
            "overwrite_cache: true",
            "preprocessing_num_workers: 4",
            f"output_dir: {output_dir}",
            "logging_steps: 10",
            "save_steps: 200",
            "plot_loss: false",
            "report_to: none",
            "overwrite_output_dir: true",
            "per_device_train_batch_size: 2",
            "gradient_accumulation_steps: 2",
            "learning_rate: 2.0e-4",
            "num_train_epochs: 2.0",
            "lr_scheduler_type: cosine",
            "warmup_ratio: 0.1",
            "max_grad_norm: 0.3",
            "bf16: true",
            "gradient_checkpointing: true",
            "ddp_timeout: 180000000",
            "flash_attn: auto",
            "optim: adamw_torch",
            "weight_decay: 0.0",
            "include_num_input_tokens_seen: true",
        ]
    )


def build_eval_yaml(base_model: str, input_dir: Path, adapter_path: Path, output_dir: Path, template: str) -> str:
    return "\n".join(
        [
            f"model_name_or_path: {base_model}",
            "stage: sft",
            "do_predict: true",
            "finetuning_type: lora",
            f"dataset_dir: {input_dir}",
            "eval_dataset: skillspan_test",
            f"template: {template}",
            "cutoff_len: 2000",
            "max_samples: 100000",
            "overwrite_cache: true",
            "preprocessing_num_workers: 4",
            f"adapter_name_or_path: {adapter_path}",
            f"output_dir: {output_dir}",
            "report_to: none",
            "overwrite_output_dir: true",
            "bf16: true",
            "per_device_eval_batch_size: 2",
            "ddp_timeout: 180000000",
            "flash_attn: auto",
            "predict_with_generate: true",
            "max_new_tokens: 1000",
            "top_p: 0.95",
            "temperature: 0.01",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run Skill-LLM with benchmark data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    model_cfg = config["models"]["skill_llm"]
    base_model = os.environ.get("SKILL_LLM_BASE_MODEL_OVERRIDE", model_cfg["base_model"])
    template = os.environ.get("SKILL_LLM_TEMPLATE_OVERRIDE", infer_template(base_model))
    repo_dir = resolve_path(model_cfg["repo_dir"])
    input_dir = resolve_path(model_cfg["input_dir"])
    results_dir = ensure_dir(resolve_path(model_cfg["results_dir"]))
    runtime_dir = ensure_dir(results_dir / "runtime")
    raw_dir = ensure_dir(results_dir / "raw")
    sft_dir = results_dir / "sft"

    train_yaml = runtime_dir / "skill_llm_train.yaml"
    eval_yaml = runtime_dir / "skill_llm_eval.yaml"
    train_yaml.write_text(build_train_yaml(base_model, input_dir, sft_dir, template), encoding="utf-8")
    eval_yaml.write_text(build_eval_yaml(base_model, input_dir, sft_dir, raw_dir, template), encoding="utf-8")

    train_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--module",
        "llamafactory.launcher",
        str(train_yaml),
    ]
    predict_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--module",
        "llamafactory.launcher",
        str(eval_yaml),
    ]

    if args.dry_run:
        if args.mode in {"train", "full"}:
            print(" ".join(train_command))
        if args.mode in {"predict", "full"}:
            print(" ".join(predict_command))
        return

    if args.mode in {"train", "full"}:
        subprocess.run(train_command, check=True, cwd=repo_dir)
    if args.mode in {"predict", "full"}:
        subprocess.run(predict_command, check=True, cwd=repo_dir)
        generated_predictions = read_json_records(raw_dir / "generated_predictions.jsonl")
        canonical_test_records = read_json_records(resolve_path(config["paths"]["canonical_dir"]) / "test.jsonl")
        unified = generated_skill_llm_predictions_to_unified(generated_predictions, canonical_test_records)
        write_jsonl(raw_dir / "processed_predictions.jsonl", unified)
        write_jsonl(results_dir / "predictions.jsonl", unified)
        write_run_metadata(
            results_dir / "run_metadata.json",
            {
                "model": "skill_llm",
                "base_model": base_model,
                "checkpoint": str(sft_dir),
                "raw_output_file": str(raw_dir / "generated_predictions.jsonl"),
                "postprocessor": "local_generated_skill_llm_predictions_to_unified",
                "template": template,
            },
        )
        print(f"Unified predictions written to {results_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
