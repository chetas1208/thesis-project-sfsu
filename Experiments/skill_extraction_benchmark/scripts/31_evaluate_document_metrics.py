#!/usr/bin/env python3
"""Run document-level evaluation scripts and compute set metrics for all models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_doc")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate document-level metrics for all models.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    doc_pred_dir = PROJECT_ROOT / gcfg["paths"]["document_predictions"]
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    from src.evaluation.document_sets import document_set_metrics

    models = ["nnose", "gliner", "skill_llm", "context", "jobbert", "gemma4"]
    all_metrics = {}

    for model in models:
        doc_pred_path = doc_pred_dir / f"{model}_doc.jsonl"
        if not doc_pred_path.exists():
            logger.warning("Document predictions not found for %s (%s)", model, doc_pred_path)
            continue

        doc_preds = []
        for line in doc_pred_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                doc_preds.append(json.loads(line))

        metrics = document_set_metrics(doc_preds)
        all_metrics[model] = metrics
        logger.info("%s doc eval complete (Set F1: %.4f, Jaccard: %.4f)", model, metrics.get("doc_f1", 0), metrics.get("doc_jaccard", 0))

    out_path = metrics_dir / "document_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Document metrics written to %s", out_path)


if __name__ == "__main__":
    main()
