import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from common import (
    ensure_benchmark_env,
    ensure_dir,
    load_config,
    load_env_file,
    parse_conll_prediction_file,
    read_json_records,
    resolve_path,
    write_jsonl,
    write_run_metadata,
)


def command_to_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the JobBERT token-classification baseline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    model_cfg = config["models"]["jobbert"]
    defaults = config["training_defaults"]["token_classification"]
    input_dir = resolve_path(model_cfg["input_dir"])
    results_dir = ensure_dir(resolve_path(model_cfg["results_dir"]))
    checkpoint_dir = ensure_dir(results_dir / "checkpoints")
    raw_dir = ensure_dir(results_dir / "raw")
    nnose_repo = resolve_path(config["models"]["nnose"]["repo_dir"])

    train_file = input_dir / "train.json"
    dev_file = input_dir / "dev.json"
    test_file = input_dir / "test.json"
    canonical_test = resolve_path(config["paths"]["canonical_dir"]) / "test.jsonl"

    train_command = [
        sys.executable,
        str(nnose_repo / "src" / "run_ner_no_trainer.py"),
        "--model_name_or_path",
        model_cfg["base_model"],
        "--train_file",
        str(train_file),
        "--validation_file",
        str(dev_file),
        "--text_column_name",
        "tokens",
        "--label_column_name",
        "tags_skill",
        "--output_dir",
        str(checkpoint_dir),
        "--per_device_train_batch_size",
        str(defaults["train_batch_size"]),
        "--per_device_eval_batch_size",
        str(defaults["eval_batch_size"]),
        "--learning_rate",
        str(defaults["learning_rate"]),
        "--checkpointing_steps",
        "epoch",
        "--num_train_epochs",
        str(defaults["num_train_epochs"]),
        "--patience",
        str(defaults["patience"]),
        "--seed",
        str(config["seed"]),
    ]
    inference_model = checkpoint_dir if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) else model_cfg["base_model"]
    predict_command = [
        sys.executable,
        str(nnose_repo / "src" / "run_inference.py"),
        "--model_name_or_path",
        str(inference_model),
        "--train_file",
        str(train_file),
        "--validation_file",
        str(test_file),
        "--text_column_name",
        "tokens",
        "--label_column_name",
        "tags_skill",
        "--seed",
        str(config["seed"]),
        "--write_output",
        str(raw_dir) + "/",
    ]

    if args.dry_run:
        if args.mode in {"train", "full"}:
            print(command_to_text(train_command))
        if args.mode in {"predict", "full"}:
            print(command_to_text(predict_command))
        return

    if args.mode in {"train", "full"}:
        subprocess.run(train_command, check=True, cwd=results_dir)
    if args.mode in {"predict", "full"}:
        subprocess.run(predict_command, check=True, cwd=results_dir)

    output_files = sorted(raw_dir.glob("*.out"), key=lambda path: path.stat().st_mtime)
    if not output_files:
        raise FileNotFoundError(f"No prediction .out files found in {raw_dir}")
    predictions = parse_conll_prediction_file(output_files[-1], read_json_records(canonical_test))
    write_jsonl(results_dir / "predictions.jsonl", predictions)
    write_run_metadata(
        results_dir / "run_metadata.json",
        {
            "model": "jobbert",
            "base_model": model_cfg["base_model"],
            "checkpoint": str(inference_model),
            "raw_output_file": str(output_files[-1]),
        },
    )
    print(f"Unified predictions written to {results_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()

