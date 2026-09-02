#!/usr/bin/env python3
"""Run Gemma 4 inference on the test set."""

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
logger = logging.getLogger("predict_gemma4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 inference.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--model-config", default="configs/gemma4.yaml")
    parser.add_argument("--data", default=None, help="Override input canonical test jsonl")
    parser.add_argument("--model_dir", default=None, help="Override model directory")
    parser.add_argument("--output", default=None, help="Override predictions output path")

    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    mcfg = yaml.safe_load((PROJECT_ROOT / args.model_config).read_text())

    model_dir = Path(args.model_dir) if args.model_dir else PROJECT_ROOT / gcfg["paths"]["models"] / "gemma4"
    pred_dir = PROJECT_ROOT / gcfg["paths"]["predictions"]
    doc_pred_dir = PROJECT_ROOT / gcfg["paths"]["document_predictions"]
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    doc_pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    test_source = Path(args.data) if args.data else PROJECT_ROOT / gcfg["paths"]["canonical"] / "test.jsonl"
    out_file = Path(args.output) if args.output else pred_dir / "gemma4_test.jsonl"
    doc_out_file = doc_pred_dir / "gemma4_doc.jsonl"

    from src.data.canonical_schema import load_canonical_split
    from src.evaluation.memory import get_peak_gpu_memory_mb
    from src.inference.aggregation import aggregate_to_documents, write_document_predictions
    from src.inference.postprocess import write_predictions
    from src.trainers.gemma4_trainer import generate_gemma4_batch, parse_strict_json_skills, recover_span_from_text

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    canonical_test = load_canonical_split(test_source)
    test_records = [r.to_dict() for r in canonical_test]
    
    adapter_dir = model_dir / "checkpoint" / "best"
    if not adapter_dir.exists():
        logger.warning("Adapter directory %s not found. Proceeding with zero-shot if possible.", adapter_dir)
        adapter_dir = model_dir

    logger.info("Running Gemma 4 inference ...")
    start = time.perf_counter()
    raw_predictions = generate_gemma4_batch(adapter_dir, test_records, mcfg)
    elapsed = time.perf_counter() - start
    peak_mem = get_peak_gpu_memory_mb()

    # Parse JSON and map to unified format
    predictions = []
    for i, rec in enumerate(test_records):
        generated_text = raw_predictions[i]
        skill_texts = parse_strict_json_skills(generated_text)
        
        predicted_spans = []
        for st in skill_texts:
            s_tok, e_tok = recover_span_from_text(st, rec["tokens"])
            predicted_spans.append({
                "text": st,
                "start_token": s_tok,
                "end_token": e_tok,
                "score": 1.0
            })
            
        predictions.append({
            "id": rec["sentence_id"],
            "doc_id": rec["doc_id"],
            "model": "gemma4",
            "tokens": rec["tokens"],
            "predicted_spans": predicted_spans,
            "predicted_skill_texts": skill_texts,
        })

    write_predictions(out_file, predictions)
    logger.info("Sentence predictions → %s", out_file)

    doc_preds = aggregate_to_documents(predictions, gold_records=test_records)
    write_document_predictions(doc_out_file, doc_preds)

    efficiency = {
        "model": "gemma4",
        "total_seconds": round(elapsed, 2),
        "num_sentences": len(test_records),
        "seconds_per_sentence": round(elapsed / max(1, len(test_records)), 4),
        "peak_gpu_memory_mb": peak_mem,
    }
    (metrics_dir / "efficiency_gemma4.json").write_text(json.dumps(efficiency, indent=2) + "\n")
    logger.info("Done. %.2fs total, %.0f MB GPU", elapsed, peak_mem)


if __name__ == "__main__":
    main()
