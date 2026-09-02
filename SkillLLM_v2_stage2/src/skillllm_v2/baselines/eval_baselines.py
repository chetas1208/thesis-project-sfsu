from __future__ import annotations

from pathlib import Path
from typing import Any

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.io_utils import read_jsonl, write_json
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split


def evaluate_prediction_file(dataset_root: str | Path, split: str, prediction_path: str | Path, metrics_path: str | Path, notes: str = "") -> dict[str, Any]:
    gold_docs = load_skillspan_split(Path(dataset_root), split)
    predictions = read_jsonl(prediction_path)
    by_id = {row["id"]: row.get("predictions", []) for row in predictions}
    gold = [doc["gold_spans"] for doc in gold_docs if doc["id"] in by_id]
    pred = [by_id[doc["id"]] for doc in gold_docs if doc["id"] in by_id]
    metrics = evaluate_extraction(gold, pred)
    metrics["notes"] = notes
    write_json(metrics_path, metrics)
    return metrics
