#!/usr/bin/env python3
"""Run JobBERT token-classification inference on the test set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_jobbert")


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="JobBERT inference.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/jobbert.yaml")
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    mcfg = yaml.safe_load((PROJECT_ROOT / args.model_config).read_text())

    data_dir = PROJECT_ROOT / gcfg["paths"]["exports"] / "jobbert"
    model_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "jobbert"
    pred_dir = PROJECT_ROOT / gcfg["paths"]["predictions"]
    doc_pred_dir = PROJECT_ROOT / gcfg["paths"]["document_predictions"]
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    doc_pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_records = load_records(data_dir / "test.jsonl")

    checkpoint_dir = model_dir / "checkpoint" / "best"
    model_path = checkpoint_dir if checkpoint_dir.exists() else Path(mcfg["base_model"])

    from src.data.canonical_schema import load_canonical_split
    from src.evaluation.memory import get_peak_gpu_memory_mb
    from src.inference.aggregation import aggregate_to_documents, write_document_predictions
    from src.inference.postprocess import write_predictions
    from src.trainers.jobbert_trainer import predict_jobbert

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("Running JobBERT inference on %d test records ...", len(test_records))
    start = time.perf_counter()
    predictions = predict_jobbert(model_path, test_records, max_length=mcfg["training"]["max_length"])
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_gpu_memory_mb()

    write_predictions(pred_dir / "jobbert_test.jsonl", predictions)
    logger.info("Sentence predictions → %s", pred_dir / "jobbert_test.jsonl")

    canonical_test = load_canonical_split(PROJECT_ROOT / gcfg["paths"]["canonical"] / "test.jsonl")
    doc_preds = aggregate_to_documents(predictions, gold_records=[r.to_dict() for r in canonical_test])
    write_document_predictions(doc_pred_dir / "jobbert_doc.jsonl", doc_preds)

    efficiency = {
        "model": "jobbert",
        "total_seconds": round(elapsed, 2),
        "num_sentences": len(test_records),
        "seconds_per_sentence": round(elapsed / max(1, len(test_records)), 4),
        "peak_gpu_memory_mb": peak_mem,
    }
    (metrics_dir / "efficiency_jobbert.json").write_text(json.dumps(efficiency, indent=2) + "\n")
    logger.info("Done. %.2fs total, %.4fs/sentence, %.0f MB GPU", elapsed, efficiency["seconds_per_sentence"], peak_mem)


if __name__ == "__main__":
    main()
