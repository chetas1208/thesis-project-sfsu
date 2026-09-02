from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.runtime_utils import write_json


LEADERBOARD_COLUMNS = [
    "model_name",
    "exact_span_precision",
    "exact_span_recall",
    "exact_span_f1",
    "token_precision",
    "token_recall",
    "token_f1",
    "partial_span_f1",
    "macro_f1",
    "micro_f1",
    "parse_failure_rate",
    "unaligned_prediction_rate",
    "avg_latency_seconds",
    "total_runtime_minutes",
]


def _format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def write_leaderboard(output_dir: str | Path, model_results: dict[str, dict[str, Any]]) -> None:
    root = Path(output_dir)
    csv_path = root / "leaderboard.csv"
    md_path = root / "leaderboard.md"
    rows = []
    for model_name in sorted(model_results):
        metrics = model_results[model_name].get("metrics") or {}
        row = {"model_name": model_name}
        for column in LEADERBOARD_COLUMNS[1:]:
            row[column] = metrics.get(column, "")
        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# SkillSpan Leaderboard", "", "|" + "|".join(LEADERBOARD_COLUMNS) + "|", "|" + "|".join(["---"] * len(LEADERBOARD_COLUMNS)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(row[column] if column == "model_name" else _format_float(row[column]) for column in LEADERBOARD_COLUMNS) + "|")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(
    output_dir: str | Path,
    benchmark_config: dict[str, Any],
    dataset_metadata: dict[str, Any],
    model_results: dict[str, dict[str, Any]],
) -> None:
    root = Path(output_dir)
    lines = [
        "# SkillSpan Benchmark Summary",
        "",
        "## Dataset",
        "",
        f"- Requested path: `{dataset_metadata['requested_dataset_path']}`",
        f"- Resolved root: `{dataset_metadata['dataset_root']}`",
        f"- Detected type: `{dataset_metadata['dataset_type']}`",
        f"- Used fallback split: `{dataset_metadata['used_fallback_split']}`",
        f"- Limit: `{dataset_metadata['limit']}`",
        "",
        "## Models",
        "",
    ]

    for model_name in sorted(model_results):
        result = model_results[model_name]
        status = result.get("status", "unknown")
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(f"- Status: `{status}`")
        if result.get("failure_reason"):
            lines.append(f"- Failure: `{result['failure_reason']}`")
        if result.get("metrics"):
            metrics = result["metrics"]
            lines.append(f"- Exact span F1: `{metrics.get('exact_span_f1', 0.0):.6f}`")
            lines.append(f"- Token F1: `{metrics.get('token_f1', 0.0):.6f}`")
            lines.append(f"- Partial span F1: `{metrics.get('partial_span_f1', 0.0):.6f}`")
            lines.append(f"- Parse failure rate: `{metrics.get('parse_failure_rate', 0.0):.6f}`")
        lines.append("")

    summary_path = root / "summary_report.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "summary_report.json", {"benchmark": benchmark_config, "dataset": dataset_metadata, "models": model_results})
