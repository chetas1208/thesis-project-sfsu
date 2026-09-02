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
    parser = argparse.ArgumentParser(description="Train and run the NNOSE benchmark pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "prepare-datastore", "predict", "full"], default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    model_cfg = config["models"]["nnose"]
    defaults = config["training_defaults"]["token_classification"]
    nnose_defaults = config["training_defaults"]["nnose"]

    repo_dir = resolve_path(model_cfg["repo_dir"])
    input_dir = resolve_path(model_cfg["input_dir"])
    results_dir = ensure_dir(resolve_path(model_cfg["results_dir"]))
    checkpoint_dir = ensure_dir(results_dir / "checkpoints")
    datastore_dir = ensure_dir(results_dir / "datastore")
    representation_dir = ensure_dir(datastore_dir / "saved_embedding_jobbert")
    index_dir = ensure_dir(datastore_dir / "index_jobbert")
    raw_dir = ensure_dir(results_dir / "raw")

    train_file = input_dir / "train.json"
    dev_file = input_dir / "dev.json"
    test_file = input_dir / "test.json"
    canonical_test = resolve_path(config["paths"]["canonical_dir"]) / "test.jsonl"

    train_command = [
        sys.executable,
        str(repo_dir / "src" / "run_ner_no_trainer.py"),
        "--model_name_or_path",
        model_cfg["base_encoder"],
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
    representation_command = [
        sys.executable,
        str(repo_dir / "src" / "get_representations.py"),
        "--model_name_or_path",
        str(checkpoint_dir),
        "--train_file",
        str(train_file),
        "--text_column_name",
        "tokens",
        "--label_column_name",
        "tags_skill",
        "--max_length",
        str(defaults["max_length"]),
        "--seed",
        str(config["seed"]),
        "--save_path",
        str(representation_dir),
    ]
    datastore_command = [
        sys.executable,
        str(repo_dir / "src" / "create_datastore.py"),
        "--feature_dir",
        str(representation_dir),
        "--output_dir",
        str(index_dir),
        "--sample_percentage",
        "100",
        "--seed",
        str(config["seed"]),
        "--whitening",
        "--dim_reduction",
        "768",
    ]
    predict_command = [
        sys.executable,
        str(repo_dir / "src" / "run_inference.py"),
        "--model_name_or_path",
        str(checkpoint_dir),
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
        "--datastore_path",
        str(index_dir),
        "--knn",
        "--k",
        str(nnose_defaults["k"]),
        "--lambda_value",
        str(nnose_defaults["lambda_value"]),
        "--temperature",
        str(nnose_defaults["temperature"]),
        "--write_output",
        str(raw_dir) + "/",
    ]
    if nnose_defaults.get("projection", False):
        predict_command.append("--projection")

    if args.dry_run:
        if args.mode in {"train", "full"}:
            print(command_to_text(train_command))
        if args.mode in {"prepare-datastore", "full"}:
            print(command_to_text(representation_command))
            print(command_to_text(datastore_command))
        if args.mode in {"predict", "full"}:
            print(command_to_text(predict_command))
        return

    if args.mode in {"train", "full"}:
        subprocess.run(train_command, check=True, cwd=results_dir)
    if args.mode in {"prepare-datastore", "full"}:
        subprocess.run(representation_command, check=True, cwd=results_dir)
        subprocess.run(datastore_command, check=True, cwd=results_dir)
    if args.mode in {"predict", "full"}:
        subprocess.run(predict_command, check=True, cwd=results_dir)

    output_files = sorted(raw_dir.glob("*_knn.out"), key=lambda path: path.stat().st_mtime)
    if not output_files:
        raise FileNotFoundError(f"No NNOSE KNN output files found in {raw_dir}")
    predictions = parse_conll_prediction_file(output_files[-1], read_json_records(canonical_test))
    write_jsonl(results_dir / "predictions.jsonl", predictions)
    write_run_metadata(
        results_dir / "run_metadata.json",
        {
            "model": "nnose",
            "base_model": model_cfg["base_encoder"],
            "checkpoint": str(checkpoint_dir),
            "datastore": str(index_dir),
            "raw_output_file": str(output_files[-1]),
            "k": nnose_defaults["k"],
            "lambda_value": nnose_defaults["lambda_value"],
            "temperature": nnose_defaults["temperature"],
        },
    )
    print(f"Unified predictions written to {results_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()

