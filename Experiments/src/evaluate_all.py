import argparse
import csv
from pathlib import Path

from common import (
    exact_span_metrics,
    load_config,
    load_env_file,
    read_json_records,
    read_optional_json,
    resolve_path,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate exact span metrics across all benchmark models.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    canonical_test_path = resolve_path(config["paths"]["canonical_dir"]) / "test.jsonl"
    canonical_test = read_json_records(canonical_test_path) if canonical_test_path.exists() else []
    span_summary_rows = []
    retrieval_summary_rows = []

    for model_key, model_cfg in config["models"].items():
        task_type = model_cfg.get("task_type", "span_extraction")
        results_dir = resolve_path(model_cfg["results_dir"])
        metadata_path = results_dir / "run_metadata.json"
        metadata = read_optional_json(metadata_path)
        base_model = metadata.get("base_model", model_cfg.get("base_model", model_cfg.get("base_encoder", "")))

        if task_type == "retrieval":
            pair_scores_path = results_dir / "pair_scores.jsonl"
            metrics_path = results_dir / "metrics.json"
            retrieval_metrics_path = results_dir / "retrieval_metrics.json"
            metrics = read_optional_json(metrics_path)
            retrieval_metrics = read_optional_json(retrieval_metrics_path)

            if not metrics or not retrieval_metrics:
                retrieval_summary_rows.append(
                    {
                        "model": model_key,
                        "status": "missing_metrics",
                        "pair_precision": "",
                        "pair_recall": "",
                        "pair_f1": "",
                        "pair_accuracy": "",
                        "recall_at_1": "",
                        "recall_at_3": "",
                        "recall_at_5": "",
                        "oov_gold_skill_rate": "",
                        "pair_scores_file": str(pair_scores_path),
                        "checkpoint": metadata.get("checkpoint", ""),
                        "base_model": base_model,
                        "env_name": model_cfg["env_name"],
                    }
                )
                continue

            retrieval_summary_rows.append(
                {
                    "model": model_key,
                    "status": "ok",
                    "pair_precision": f"{metrics.get('precision', 0.0):.6f}",
                    "pair_recall": f"{metrics.get('recall', 0.0):.6f}",
                    "pair_f1": f"{metrics.get('f1', 0.0):.6f}",
                    "pair_accuracy": f"{metrics.get('accuracy', 0.0):.6f}",
                    "recall_at_1": f"{retrieval_metrics.get('recall_at_1', 0.0):.6f}",
                    "recall_at_3": f"{retrieval_metrics.get('recall_at_3', 0.0):.6f}",
                    "recall_at_5": f"{retrieval_metrics.get('recall_at_5', 0.0):.6f}",
                    "oov_gold_skill_rate": f"{retrieval_metrics.get('oov_gold_skill_rate', 0.0):.6f}",
                    "pair_scores_file": str(pair_scores_path),
                    "checkpoint": metadata.get("checkpoint", ""),
                    "base_model": base_model,
                    "env_name": model_cfg["env_name"],
                }
            )
            continue

        predictions_path = results_dir / "predictions.jsonl"

        if not predictions_path.exists():
            span_summary_rows.append(
                {
                    "model": model_key,
                    "status": "missing_predictions",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "prediction_file": str(predictions_path),
                    "checkpoint": metadata.get("checkpoint", ""),
                    "base_model": base_model,
                    "env_name": model_cfg["env_name"],
                }
            )
            continue

        if not canonical_test:
            span_summary_rows.append(
                {
                    "model": model_key,
                    "status": "missing_canonical_test",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "prediction_file": str(predictions_path),
                    "checkpoint": metadata.get("checkpoint", ""),
                    "base_model": base_model,
                    "env_name": model_cfg["env_name"],
                }
            )
            continue

        predictions = read_json_records(predictions_path)
        if len(predictions) != len(canonical_test):
            span_summary_rows.append(
                {
                    "model": model_key,
                    "status": f"partial_predictions ({len(predictions)}/{len(canonical_test)})",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "tp": "",
                    "fp": "",
                    "fn": "",
                    "prediction_file": str(predictions_path),
                    "checkpoint": metadata.get("checkpoint", ""),
                    "base_model": base_model,
                    "env_name": model_cfg["env_name"],
                }
            )
            continue
        metrics = exact_span_metrics(canonical_test, predictions)
        write_json(results_dir / "metrics.json", metrics)
        span_summary_rows.append(
            {
                "model": model_key,
                "status": "ok",
                "precision": f"{metrics['precision']:.6f}",
                "recall": f"{metrics['recall']:.6f}",
                "f1": f"{metrics['f1']:.6f}",
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "prediction_file": str(predictions_path),
                "checkpoint": metadata.get("checkpoint", ""),
                "base_model": base_model,
                "env_name": model_cfg["env_name"],
            }
        )

    summary_path = resolve_path(config["paths"]["results_dir"]) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "status",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "prediction_file",
        "checkpoint",
        "base_model",
        "env_name",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(span_summary_rows)

    retrieval_summary_path = resolve_path(config["paths"]["results_dir"]) / "retrieval_summary.csv"
    retrieval_fieldnames = [
        "model",
        "status",
        "pair_precision",
        "pair_recall",
        "pair_f1",
        "pair_accuracy",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "oov_gold_skill_rate",
        "pair_scores_file",
        "checkpoint",
        "base_model",
        "env_name",
    ]
    with retrieval_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=retrieval_fieldnames)
        writer.writeheader()
        writer.writerows(retrieval_summary_rows)

    print(f"Summary written to {summary_path}")
    print(f"Retrieval summary written to {retrieval_summary_path}")


if __name__ == "__main__":
    main()
