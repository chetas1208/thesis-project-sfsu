from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.runtime_utils import read_json, write_json


SUFFIXES = ("exact", "relaxed", "doc_set", "classification", "latency")


def split_experiment_name(stem: str) -> tuple[str, str] | None:
    for suffix in SUFFIXES:
        token = f"_{suffix}"
        if stem.endswith(token):
            return stem[: -len(token)], suffix
    return None


def load_metric_groups(metrics_dir: Path) -> dict[str, dict[str, dict]]:
    groups: dict[str, dict[str, dict]] = {}
    for metric_file in metrics_dir.glob("*.json"):
        parsed = split_experiment_name(metric_file.stem)
        if parsed is None:
            continue
        experiment_name, suffix = parsed
        groups.setdefault(experiment_name, {})[suffix] = read_json(metric_file)
    return groups


def build_rows(metric_groups: dict[str, dict[str, dict]]) -> list[dict[str, object]]:
    rows = []
    for experiment_name, metrics in metric_groups.items():
        if experiment_name.startswith("ablation_"):
            continue
        exact = metrics.get("exact", {})
        relaxed = metrics.get("relaxed", {})
        doc_set = metrics.get("doc_set", {})
        latency = metrics.get("latency", {})
        rows.append(
            {
                "experiment_name": experiment_name,
                "exact_f1": exact.get("overall", {}).get("f1", 0.0),
                "exact_precision": exact.get("overall", {}).get("precision", 0.0),
                "exact_recall": exact.get("overall", {}).get("recall", 0.0),
                "relaxed_f1": relaxed.get("relaxed_span", {}).get("f1", 0.0),
                "text_match_f1": relaxed.get("text_match", {}).get("f1", 0.0),
                "doc_set_f1": doc_set.get("document_set", {}).get("f1", 0.0),
                "hard_skill_f1": exact.get("per_label", {}).get("hard_skill", {}).get("f1", 0.0),
                "soft_skill_f1": exact.get("per_label", {}).get("soft_skill", {}).get("f1", 0.0),
                "avg_latency_s": latency.get("average_latency_seconds", 0.0),
                "throughput_sps": latency.get("throughput_samples_per_second", 0.0),
                "gpu_memory_mb": latency.get("gpu_memory_mb", 0.0) or 0.0,
            }
        )
    return sorted(rows, key=lambda row: (-row["exact_f1"], row["experiment_name"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["experiment_name"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("experiment_name", "Experiment"),
        ("exact_f1", "Exact F1"),
        ("relaxed_f1", "Relaxed F1"),
        ("text_match_f1", "Text F1"),
        ("doc_set_f1", "DocSet F1"),
        ("hard_skill_f1", "Hard F1"),
        ("soft_skill_f1", "Soft F1"),
        ("throughput_sps", "Throughput"),
    ]
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        " & ".join(column_name for _, column_name in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_report(path: Path, rows: list[dict[str, object]], ablation_summary: dict | None) -> None:
    baseline = next((row for row in rows if "baseline" in row["experiment_name"]), None)
    best_row = rows[0] if rows else None
    improvement = None
    if baseline and best_row:
        improvement = best_row["exact_f1"] - baseline["exact_f1"]

    lines = ["# Final Model Comparison", ""]
    if best_row:
        lines.append(f"Best exact-span model: `{best_row['experiment_name']}`")
        lines.append("")
    if baseline and improvement is not None:
        lines.append(f"Exact-span F1 gain over baseline: `{improvement:.4f}`")
        lines.append("")
    if ablation_summary:
        lines.append(f"Largest ablation gain: `{ablation_summary.get('largest_gain_stage', 'n/a')}`")
        lines.append("")

    lines.append("| Experiment | Exact F1 | Relaxed F1 | Text F1 | DocSet F1 | Hard F1 | Soft F1 | Throughput |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['experiment_name']} | {row['exact_f1']:.4f} | {row['relaxed_f1']:.4f} | "
            f"{row['text_match_f1']:.4f} | {row['doc_set_f1']:.4f} | {row['hard_skill_f1']:.4f} | "
            f"{row['soft_skill_f1']:.4f} | {row['throughput_sps']:.2f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\section{Final Model Comparison}",
        r"\begin{itemize}",
    ]
    if rows:
        best = rows[0]
        lines.append(rf"\item Best exact-span system: \texttt{{{best['experiment_name']}}} with exact F1 {best['exact_f1']:.4f}.")
    lines.append(r"\item Table \ref{tab:model-leaderboard} contains the full backbone comparison.")
    lines.append(r"\end{itemize}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate metric files into paper-ready tables.")
    parser.add_argument("--metrics-dir", default="reports/metrics", help="Directory containing metric json files.")
    parser.add_argument("--output-dir", default="reports/tables", help="Directory for generated tables.")
    parser.add_argument("--report-dir", default="reports", help="Directory for the markdown and tex summaries.")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    rows = build_rows(load_metric_groups(metrics_dir))
    if not rows:
        print("No metric files found.")
        return 0

    leaderboard_csv = output_dir / "model_leaderboard.csv"
    leaderboard_tex = output_dir / "model_leaderboard.tex"
    write_csv(leaderboard_csv, rows)
    write_latex(leaderboard_tex, rows)

    ablation_summary_path = report_dir / "metrics" / "ablation_best_summary.json"
    ablation_summary = read_json(ablation_summary_path) if ablation_summary_path.exists() else None
    write_markdown_report(report_dir / "final_model_comparison.md", rows, ablation_summary)
    write_tex_report(report_dir / "final_model_comparison.tex", rows)

    write_json(report_dir / "metrics" / "model_leaderboard_summary.json", {"rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
