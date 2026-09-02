#!/usr/bin/env python3
"""Run GLiNER inference on the test set."""

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
logger = logging.getLogger("predict_gliner")


def main() -> None:
    parser = argparse.ArgumentParser(description="GLiNER inference for skill extraction.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/gliner.yaml")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    mcfg = yaml.safe_load((PROJECT_ROOT / args.model_config).read_text())

    model_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "gliner"
    pred_dir = PROJECT_ROOT / gcfg["paths"]["predictions"]
    doc_pred_dir = PROJECT_ROOT / gcfg["paths"]["document_predictions"]
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    doc_pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold if args.threshold is not None else mcfg["inference"]["threshold"]
    entity_types = mcfg["inference"]["entity_types"]

    # Load canonical test records (GLiNER needs text, not token classification format)
    from src.data.canonical_schema import load_canonical_split
    canonical_test = load_canonical_split(PROJECT_ROOT / gcfg["paths"]["canonical"] / "test.jsonl")
    test_dicts = [r.to_dict() for r in canonical_test]

    # Determine model path
    checkpoint_dir = model_dir / "models"
    model_path = mcfg["base_model"]
    
    if (checkpoint_dir / "gliner_config.json").exists():
        model_path = str(checkpoint_dir)
    elif checkpoint_dir.exists():
        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda d: int(d.name.split("-")[-1]))
            if (latest / "gliner_config.json").exists():
                model_path = str(latest)

    from src.evaluation.memory import get_peak_gpu_memory_mb
    from src.inference.aggregation import aggregate_to_documents, write_document_predictions
    from src.inference.postprocess import write_predictions
    from src.trainers.gliner_trainer import predict_gliner

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info("Running GLiNER inference on %d test records (threshold=%.2f) ...", len(test_dicts), threshold)
    start = time.perf_counter()
    predictions = predict_gliner(model_path, test_dicts, threshold=threshold, entity_types=entity_types)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_gpu_memory_mb()

    write_predictions(pred_dir / "gliner_test.jsonl", predictions)
    logger.info("Sentence predictions → %s", pred_dir / "gliner_test.jsonl")

    doc_preds = aggregate_to_documents(predictions, gold_records=test_dicts)
    write_document_predictions(doc_pred_dir / "gliner_doc.jsonl", doc_preds)

    efficiency = {
        "model": "gliner",
        "total_seconds": round(elapsed, 2),
        "num_sentences": len(test_dicts),
        "seconds_per_sentence": round(elapsed / max(1, len(test_dicts)), 4),
        "peak_gpu_memory_mb": peak_mem,
        "threshold": threshold,
    }
    (metrics_dir / "efficiency_gliner.json").write_text(json.dumps(efficiency, indent=2) + "\n")
    logger.info("Done. %.2fs total, %.4fs/sentence, %.0f MB GPU", elapsed, efficiency["seconds_per_sentence"], peak_mem)


if __name__ == "__main__":
    main()
