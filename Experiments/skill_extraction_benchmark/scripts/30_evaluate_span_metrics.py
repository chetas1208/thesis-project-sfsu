#!/usr/bin/env python3
"""Run evaluation scripts and compute span/token level metrics for all models."""

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
logger = logging.getLogger("evaluate_span")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate span-level metrics for all models.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())

    canonical_dir = PROJECT_ROOT / gcfg["paths"]["canonical"]
    pred_dir = PROJECT_ROOT / gcfg["paths"]["predictions"]
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    from src.data.canonical_schema import load_canonical_split
    from src.evaluation.spans import exact_span_match_metrics, text_match_metrics, token_level_metrics
    from src.inference.postprocess import load_predictions

    test_split_path = canonical_dir / "test.jsonl"
    if not test_split_path.exists():
        logger.error("Test split not found at %s", test_split_path)
        return

    canonical_test = load_canonical_split(test_split_path)
    test_dicts = [r.to_dict() for r in canonical_test]

    models = ["nnose", "gliner", "skill_llm", "context", "jobbert", "gemma4"]
    all_metrics = {}

    for model in models:
        pred_path = pred_dir / f"{model}_test.jsonl"
        if not pred_path.exists():
            logger.warning("Predictions not found for %s (%s)", model, pred_path)
            continue

        preds = load_predictions(pred_path)
        if len(preds) != len(test_dicts):
            logger.warning("Prediction count mismatch for %s: %d vs %d", model, len(preds), len(test_dicts))
            continue

        exact = exact_span_match_metrics(test_dicts, preds)
        text_match = text_match_metrics(test_dicts, preds)
        
        # Token metrics only apply well to sequence labeling models
        if model in {"nnose", "jobbert"}:
            token = token_level_metrics(test_dicts, preds)
        else:
            token = {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

        all_metrics[model] = {**exact, **text_match, **token}
        logger.info("%s eval complete (Span F1: %.4f, Text F1: %.4f)", model, exact.get("exact_span_f1", 0), text_match.get("text_match_f1", 0))

    out_path = metrics_dir / "span_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Span metrics written to %s", out_path)


if __name__ == "__main__":
    main()
