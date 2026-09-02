import argparse
import json
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
    write_json,
    write_jsonl,
    write_run_metadata,
)
from model_adapters.gliner import normalize_gliner_entities


def filter_positive_gliner_records(records: list[dict]) -> list[dict]:
    return [record for record in records if record.get("ner")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run GLiNER for skill extraction.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    model_cfg = config["models"]["gliner"]
    threshold = args.threshold if args.threshold is not None else config["training_defaults"]["gliner"]["threshold"]

    input_dir = resolve_path(model_cfg["input_dir"])
    results_dir = ensure_dir(resolve_path(model_cfg["results_dir"]))
    runtime_dir = ensure_dir(results_dir / "runtime")
    raw_dir = ensure_dir(results_dir / "raw")
    repo_dir = resolve_path(model_cfg["repo_dir"])

    train_data = input_dir / "train.json"
    val_data = input_dir / "dev.json"
    canonical_test_path = resolve_path(config["paths"]["canonical_dir"]) / "test.jsonl"
    test_records = read_json_records(canonical_test_path) if canonical_test_path.exists() else []

    filtered_train_data = runtime_dir / "train_positive_only.json"
    filtered_val_data = runtime_dir / "dev_positive_only.json"
    train_records = filter_positive_gliner_records(json.loads(train_data.read_text(encoding="utf-8")))
    val_records = filter_positive_gliner_records(json.loads(val_data.read_text(encoding="utf-8")))
    if not train_records:
        raise ValueError("GLiNER training split has no positive examples after filtering.")
    if not val_records:
        raise ValueError("GLiNER validation split has no positive examples after filtering.")
    write_json(filtered_train_data, train_records)
    write_json(filtered_val_data, val_records)

    runtime_config = runtime_dir / "gliner_train_config.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "model:",
                f"  model_name: {model_cfg['base_model']}",
                '  name: "skill gliner"',
                "  max_width: 12",
                "  hidden_size: 768",
                "  dropout: 0.3",
                "  fine_tune: true",
                "  subtoken_pooling: first",
                "  span_mode: markerV0",
                "  max_types: 16",
                "  max_len: 512",
                "  max_neg_type_ratio: 1",
                '  decoder_mode: "span"',
                "data:",
                f"  root_dir: {results_dir}",
                f"  train_data: '{filtered_train_data}'",
                f"  val_data_dir: '{filtered_val_data}'",
                "training:",
                f"  prev_path: {model_cfg['base_model']}",
                "  num_steps: 15000",
                "  train_batch_size: 2",
                "  eval_every: 500",
                "  warmup_ratio: 0.05",
                '  scheduler_type: "cosine"',
                "  loss_alpha: 0.75",
                "  loss_gamma: 0",
                "  loss_prob_margin: 0",
                '  loss_reduction: "sum"',
                "  negatives: 1.0",
                "  masking: none",
                "  lr_encoder: 1e-5",
                "  lr_others: 3e-5",
                "  weight_decay_encoder: 0.1",
                "  weight_decay_other: 0.01",
                "  max_grad_norm: 10.0",
                "  save_total_limit: 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    train_command = [sys.executable, str(repo_dir / "train.py"), "--config", str(runtime_config)]
    if args.dry_run and args.mode in {"train", "full"}:
        print(" ".join(train_command))

    if not args.dry_run and args.mode in {"train", "full"}:
        subprocess.run(train_command, check=True, cwd=results_dir)

    if args.mode in {"predict", "full"}:
        if args.dry_run:
            checkpoint = results_dir / "models"
            print(f"GLiNER predict using checkpoint={checkpoint if checkpoint.exists() else model_cfg['base_model']}")
            return

        if not test_records:
            raise FileNotFoundError(f"Canonical test split not found at {canonical_test_path}")

        from gliner import GLiNER  # pylint: disable=import-outside-toplevel

        checkpoint_dir = results_dir / "models"
        model_path = str(checkpoint_dir) if checkpoint_dir.exists() else model_cfg["base_model"]
        model = GLiNER.from_pretrained(model_path)
        raw_records = []
        unified_records = []
        for record in test_records:
            entities = model.predict_entities(record["text"], ["skill"], threshold=threshold)
            raw_records.append({"id": record["id"], "entities": entities})
            unified_records.append({"id": record["id"], "predicted_spans": normalize_gliner_entities(record["text"], entities)})
        write_json(raw_dir / "raw_predictions.json", raw_records)
        write_jsonl(results_dir / "predictions.jsonl", unified_records)
        write_run_metadata(
            results_dir / "run_metadata.json",
            {
                "model": "gliner",
                "base_model": model_cfg["base_model"],
                "checkpoint": model_path,
                "threshold": threshold,
                "train_records": len(train_records),
                "dev_records": len(val_records),
                "positive_only_training": True,
            },
        )
        print(f"Unified predictions written to {results_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
