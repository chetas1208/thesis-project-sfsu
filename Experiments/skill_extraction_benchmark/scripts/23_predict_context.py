#!/usr/bin/env python3
"""Run ConTeXT retrieval-based inference on the test set."""

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
logger = logging.getLogger("predict_context")


def main() -> None:
    parser = argparse.ArgumentParser(description="ConTeXT retrieval inference.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/context.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    mcfg = yaml.safe_load((PROJECT_ROOT / args.model_config).read_text())

    model_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "context"
    data_dir = PROJECT_ROOT / gcfg["paths"]["exports"] / "context"
    pred_dir = PROJECT_ROOT / gcfg["paths"]["predictions"]
    doc_pred_dir = PROJECT_ROOT / gcfg["paths"]["document_predictions"]
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    doc_pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load candidate skills
    candidate_path = data_dir / "candidate_skills.json"
    candidate_skills = json.loads(candidate_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d candidate skills", len(candidate_skills))

    # Load threshold
    threshold = args.threshold
    if threshold is None:
        threshold_path = model_dir / "best_threshold.json"
        if threshold_path.exists():
            threshold = json.loads(threshold_path.read_text())["threshold"]
        else:
            threshold = mcfg["inference"]["default_threshold"]
    top_k = mcfg["inference"]["top_k"]

    # Load test records
    from src.data.canonical_schema import load_canonical_split
    from src.evaluation.memory import get_peak_gpu_memory_mb
    from src.inference.aggregation import aggregate_to_documents, write_document_predictions
    from src.inference.postprocess import write_predictions
    from src.trainers.context_trainer import predict_context

    canonical_test = load_canonical_split(PROJECT_ROOT / gcfg["paths"]["canonical"] / "test.jsonl")
    test_dicts = [r.to_dict() for r in canonical_test]

    checkpoint_dir = model_dir / "checkpoint"
    model_path = checkpoint_dir if checkpoint_dir.exists() else Path(mcfg["base_model"])

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("Running ConTeXT inference (threshold=%.4f, top_k=%d) ...", threshold, top_k)
    start = time.perf_counter()
    predictions = predict_context(
        model_path, candidate_skills, test_dicts,
        threshold=threshold, top_k=top_k,
        encode_batch_size=mcfg["inference"]["encode_batch_size"],
    )
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_gpu_memory_mb()

    write_predictions(pred_dir / "context_test.jsonl", predictions)
    logger.info("Sentence predictions → %s", pred_dir / "context_test.jsonl")

    doc_preds = aggregate_to_documents(predictions, gold_records=test_dicts)
    write_document_predictions(doc_pred_dir / "context_doc.jsonl", doc_preds)

    efficiency = {
        "model": "context",
        "total_seconds": round(elapsed, 2),
        "num_sentences": len(test_dicts),
        "seconds_per_sentence": round(elapsed / max(1, len(test_dicts)), 4),
        "peak_gpu_memory_mb": peak_mem,
        "threshold": threshold,
        "top_k": top_k,
        "candidate_pool_size": len(candidate_skills),
    }
    (metrics_dir / "efficiency_context.json").write_text(json.dumps(efficiency, indent=2) + "\n")
    logger.info("Done. %.2fs total, %.4fs/sentence, %.0f MB GPU", elapsed, efficiency["seconds_per_sentence"], peak_mem)


if __name__ == "__main__":
    main()
