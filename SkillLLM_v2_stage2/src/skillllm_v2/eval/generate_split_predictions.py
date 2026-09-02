"""generate_split_predictions.py

One-shot inference pass with the existing 80-epoch Raw Mistral LoRA adapter
over a SkillSpan split (e.g. ``dev``), producing a prediction cache used to
tune wrapper components (boundary refiner, candidate merger, confidence
calibrator) against dev-set extraction F1.

This module performs INFERENCE ONLY. It loads the already-trained adapter at
``artifacts/skillllm_v2/mistral_lora/model`` (falling back to
``model.adapter_path`` from the config only if that trained adapter is
absent) and never calls a training entrypoint. Running this script does not
retrain Mistral -- it mirrors ``eval_raw_mistral_lora`` in
``skillllm_v2.pipeline.smoke_runner`` but targets an arbitrary split.

Usage:
    python -m skillllm_v2.eval.generate_split_predictions --split dev
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from skillllm_v2.config import load_yaml
from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.extract import MistralExtractor
from skillllm_v2.io_utils import write_json, write_jsonl
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

DEFAULT_CONFIG_PATH = "configs/skillllm_v2_training_80ep_2gpu.yaml"
TRAINED_ADAPTER_PATH = Path("artifacts/skillllm_v2/mistral_lora/model")


def _trained_adapter_path(config: dict[str, Any]) -> str | None:
    if (TRAINED_ADAPTER_PATH / "adapter_config.json").exists():
        return str(TRAINED_ADAPTER_PATH)
    return config.get("model", {}).get("adapter_path")


def _dataset_root(config: dict[str, Any]) -> str:
    return config.get("data", {}).get("skillspan_root", "data/source/skillspan_hf")


def build_extractor(config: dict[str, Any]) -> MistralExtractor:
    model_cfg = config.get("model", {})
    return MistralExtractor(
        model_path=model_cfg.get("model_path"),
        adapter_path=_trained_adapter_path(config),
        use_4bit=bool(model_cfg.get("use_4bit", True)),
        max_new_tokens=int(model_cfg.get("max_new_tokens", 512)),
        allow_fallback=False,
    )


def generate_split_predictions(
    config_path: str = DEFAULT_CONFIG_PATH,
    split: str = "dev",
    limit: int | None = None,
    output_path: str | None = None,
    metrics_path: str | None = None,
    progress_every: int = 50,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    docs = load_skillspan_split(Path(_dataset_root(config)), split, limit=limit)
    docs = [doc for doc in docs if doc["gold_spans"]]

    extractor = build_extractor(config)
    output_path = output_path or f"artifacts/skillllm_v2/mistral_lora/predictions_{split}.jsonl"
    metrics_path = metrics_path or f"artifacts/skillllm_v2/mistral_lora/metrics_{split}.json"

    rows: list[dict[str, Any]] = []
    gold: list[list[dict[str, Any]]] = []
    pred: list[list[dict[str, Any]]] = []
    started = time.perf_counter()
    for i, doc in enumerate(docs):
        predictions = extractor.extract(doc["text"])
        rows.append({"id": doc["id"], "text": doc["text"], "gold_spans": doc["gold_spans"], "predictions": predictions})
        gold.append(doc["gold_spans"])
        pred.append(predictions)
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"[{split}] {i + 1}/{len(docs)} docs in {elapsed:.1f}s", flush=True)

    write_jsonl(output_path, rows)
    metrics = evaluate_extraction(gold, pred)
    metrics["notes"] = (
        f"Raw Mistral LoRA inference (no retraining) on '{split}' split ({len(docs)} docs). "
        f"Adapter: {extractor.adapter_path}. Model load fallback used: {bool(extractor.load_error)}."
        + (f" Load error: {extractor.load_error[:300]}" if extractor.load_error else "")
    )
    write_json(metrics_path, metrics)
    print(f"Wrote {len(rows)} predictions to {output_path}", flush=True)
    print(f"Metrics ({split}): {metrics}", flush=True)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Raw Mistral LoRA predictions for a SkillSpan split (inference only, no retraining)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
    args = parser.parse_args()

    generate_split_predictions(args.config, args.split, args.limit, args.output, args.metrics_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
