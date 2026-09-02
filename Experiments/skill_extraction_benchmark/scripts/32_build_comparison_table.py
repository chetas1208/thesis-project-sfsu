#!/usr/bin/env python3
"""Build the final comparison tables from computed metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("build_comparison")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified benchmark comparison report.")
    parser.add_argument("--global-config", default="configs/global.yaml")
    args = parser.parse_args()

    gcfg = yaml.safe_load((PROJECT_ROOT / args.global_config).read_text())
    metrics_dir = PROJECT_ROOT / gcfg["paths"]["metrics"]
    reports_dir = PROJECT_ROOT / gcfg["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    span_metrics_path = metrics_dir / "span_metrics.json"
    doc_metrics_path = metrics_dir / "document_metrics.json"

    if not span_metrics_path.exists() or not doc_metrics_path.exists():
        logger.error("Metrics files missing. Run evaluation scripts first.")
        return

    span_metrics = json.loads(span_metrics_path.read_text())
    doc_metrics = json.loads(doc_metrics_path.read_text())

    models = ["nnose", "gliner", "skill_llm", "context", "jobbert", "gemma4"]
    
    rows = []
    for model in models:
        s = span_metrics.get(model, {})
        d = doc_metrics.get(model, {})
        
        eff_path = metrics_dir / f"efficiency_{model}.json"
        e = json.loads(eff_path.read_text()) if eff_path.exists() else {}

        rows.append({
            "model": model,
            "span_f1": s.get("exact_span_f1", 0.0),
            "text_f1": s.get("text_match_f1", 0.0),
            "token_f1": s.get("token_f1", 0.0),
            "doc_f1": d.get("doc_f1", 0.0),
            "doc_jaccard": d.get("doc_jaccard", 0.0),
            "sec_per_sec": e.get("seconds_per_sentence", 0.0),
            "peak_mem_mb": e.get("peak_gpu_memory_mb", 0.0),
        })

    # Write CSV
    csv_path = reports_dir / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Write Markdown
    md_path = reports_dir / "comparison.md"
    with md_path.open("w") as f:
        f.write("# Skill Extraction Benchmark Comparison\n\n")
        f.write("| Model | Span F1 | Text F1 | Token F1 | Doc Set F1 | Doc Jaccard | Time/Sent (s) | Peak Mem (MB) |\n")
        f.write("|-------|---------|---------|----------|------------|-------------|---------------|---------------|\n")
        for r in rows:
            f.write(f"| {r['model']} | {r['span_f1']:.4f} | {r['text_f1']:.4f} | {r['token_f1']:.4f} | "
                    f"{r['doc_f1']:.4f} | {r['doc_jaccard']:.4f} | {r['sec_per_sec']:.4f} | {r['peak_mem_mb']:.1f} |\n")

    logger.info("Comparison tables written to %s", reports_dir)


if __name__ == "__main__":
    main()
