"""Resumable held-out TEST evaluation of the QLoRA-loop best adapter.

Mirrors smoke_runner.eval_raw_mistral_lora exactly (same config, docs filter
[test split, gold-bearing docs], MistralExtractor, one-doc-at-a-time greedy
decoding, evaluate_extraction); only the adapter path differs. Predictions are
appended per doc to a jsonl so a restart skips finished docs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.extract import MistralExtractor
from skillllm_v2.io_utils import read_json, write_json
from skillllm_v2.pipeline.smoke_runner import _cfg, _dataset_root, _docs

CONFIG = "configs/skillllm_v2_training_80ep_2gpu.yaml"  # same config the anchor eval used
# Defaults = v1 run (unchanged behaviour). Override via env for other runs (e.g. v2).
ADAPTER = os.environ.get("QLORA_ADAPTER", "artifacts/skillllm_v2/qlora_loop/best")
OUT = Path(os.environ.get("QLORA_EVAL_OUT", "artifacts/skillllm_v2/qlora_loop_eval"))
REPORT_METRICS = os.environ.get("QLORA_REPORT_METRICS", "reports/skillllm_v2/qlora_loop_test_metrics.json")
RUN_LABEL = os.environ.get("QLORA_RUN_LABEL", "round 1")
PRED = OUT / "predictions.jsonl"
TIMING = OUT / "timing_segments.jsonl"


def main() -> None:
    limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
    OUT.mkdir(parents=True, exist_ok=True)
    config = _cfg(CONFIG)
    docs = _docs(config, "test", "full", limit)
    print("test docs:", len(docs), flush=True)
    done = {}
    if PRED.exists():
        for line in PRED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail from an interrupted write
                done[row["id"]] = row
    print("already done:", len(done), flush=True)
    todo = [d for d in docs if d["id"] not in done]

    seg = {"start_unix": time.time(), "docs_before": len(done)}
    peak_gb = 0.0
    if todo:
        import torch

        model_cfg = config.get("model", {})
        extractor = MistralExtractor(
            model_path=model_cfg.get("model_path"),
            adapter_path=ADAPTER,
            use_4bit=bool(model_cfg.get("use_4bit", True)),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 512)),
            allow_fallback=False,
        )
        t_load = time.perf_counter()
        extractor.load()
        seg["load_seconds"] = time.perf_counter() - t_load
        print(f"model loaded in {seg['load_seconds']:.0f}s; adapter={ADAPTER}; alloc={torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with PRED.open("a", encoding="utf-8") as fh:
            for i, doc in enumerate(todo):
                predictions = extractor.extract(doc["text"])
                row = {"id": doc["id"], "text": doc["text"], "predictions": predictions}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[doc["id"]] = row
                if i % 25 == 0:
                    el = time.perf_counter() - t0
                    print(f"[{i+1}/{len(todo)}] {el:.0f}s elapsed, {el/(i+1):.2f}s/doc", flush=True)
        seg["gen_seconds"] = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        seg["peak_alloc_gb"] = peak_gb
        seg["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9
        seg["docs_generated"] = len(todo)
        seg["end_unix"] = time.time()
        with TIMING.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(seg) + "\n")

    rows = [done[d["id"]] for d in docs]
    metrics = evaluate_extraction([d["gold_spans"] for d in docs], [r["predictions"] for r in rows])
    metrics["notes"] = (
        f"QLoRA-loop best adapter ({ADAPTER}, {RUN_LABEL}) via the Raw Mistral LoRA eval path "
        "(MistralExtractor, greedy, max_new_tokens=512, 4-bit NF4, one doc at a time). Model load fallback used: False."
    )
    if limit:
        print("LIMIT set; not writing report metrics")
        print(metrics)
        return
    write_json(OUT / "metrics.json", metrics)
    write_json(REPORT_METRICS, metrics)
    print("final metrics:", metrics, flush=True)


if __name__ == "__main__":
    main()
