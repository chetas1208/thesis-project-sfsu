#!/usr/bin/env python3
"""Run Skill-LLM inference on the test set."""

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
logger = logging.getLogger("predict_skillllm")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill-LLM inference.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/skill_llm.yaml")
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    mcfg = yaml.safe_load((PROJECT_ROOT / args.model_config).read_text())

    data_dir = PROJECT_ROOT / gcfg["paths"]["exports"] / "skill_llm"
    model_dir = PROJECT_ROOT / gcfg["paths"]["models"] / "skill_llm"
    pred_dir = PROJECT_ROOT / gcfg["paths"]["predictions"]
    doc_pred_dir = PROJECT_ROOT / gcfg["paths"]["document_predictions"]
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    doc_pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    from src.data.canonical_schema import load_canonical_split
    from src.evaluation.memory import get_peak_gpu_memory_mb
    from src.inference.aggregation import aggregate_to_documents, write_document_predictions
    from src.inference.postprocess import write_predictions
    from src.trainers.skill_llm_trainer import parse_skill_llm_output, predict_skill_llm

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    adapter_dir = model_dir / "sft"
    raw_dir = model_dir / "raw"

    logger.info("Running Skill-LLM inference ...")
    start = time.perf_counter()
    raw_output_path = predict_skill_llm(mcfg, data_dir, adapter_dir, model_dir)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_gpu_memory_mb()

    # Post-process: parse generated JSON into unified format
    canonical_test = load_canonical_split(PROJECT_ROOT / gcfg["paths"]["canonical"] / "test.jsonl")
    raw_predictions = []
    if raw_output_path.exists():
        for line in raw_output_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                raw_predictions.append(json.loads(line))

    predictions = []
    for i, canonical_rec in enumerate(canonical_test):
        generated_text = ""
        if i < len(raw_predictions):
            generated_text = raw_predictions[i].get("predict", raw_predictions[i].get("output", ""))

        skill_texts = parse_skill_llm_output(generated_text)
        predictions.append({
            "id": canonical_rec.sentence_id,
            "doc_id": canonical_rec.doc_id,
            "tokens": canonical_rec.tokens,
            "predicted_spans": [{"text": s, "start_token": -1, "end_token": -1, "score": 1.0} for s in skill_texts],
            "predicted_skill_texts": skill_texts,
        })

    write_predictions(pred_dir / "skill_llm_test.jsonl", predictions)
    logger.info("Sentence predictions → %s", pred_dir / "skill_llm_test.jsonl")

    doc_preds = aggregate_to_documents(predictions, gold_records=[r.to_dict() for r in canonical_test])
    write_document_predictions(doc_pred_dir / "skill_llm_doc.jsonl", doc_preds)

    efficiency = {
        "model": "skill_llm",
        "total_seconds": round(elapsed, 2),
        "num_sentences": len(canonical_test),
        "seconds_per_sentence": round(elapsed / max(1, len(canonical_test)), 4),
        "peak_gpu_memory_mb": peak_mem,
    }
    (metrics_dir / "efficiency_skill_llm.json").write_text(json.dumps(efficiency, indent=2) + "\n")
    logger.info("Done. %.2fs total, %.0f MB GPU", elapsed, peak_mem)


if __name__ == "__main__":
    main()
