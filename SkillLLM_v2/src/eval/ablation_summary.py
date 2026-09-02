from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.runtime_utils import read_json, write_json


def load_exact_metrics(metrics_dir: Path) -> dict[str, dict]:
    metrics = {}
    for metric_file in metrics_dir.glob("ablation_*_exact.json"):
        metrics[metric_file.stem[: -len("_exact")]] = read_json(metric_file)
    return metrics


def stage_label(experiment_name: str) -> str:
    return experiment_name.split("_")[1].upper() if "_" in experiment_name else experiment_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ablation metrics.")
    parser.add_argument("--metrics-dir", default="reports/metrics", help="Directory containing metric json files.")
    parser.add_argument("--output-dir", default="reports/tables", help="Directory for generated tables.")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_metrics = load_exact_metrics(metrics_dir)
    rows = []
    for experiment_name, payload in exact_metrics.items():
        rows.append(
            {
                "experiment_name": experiment_name,
                "stage": stage_label(experiment_name),
                "exact_f1": payload.get("overall", {}).get("f1", 0.0),
                "hard_skill_f1": payload.get("per_label", {}).get("hard_skill", {}).get("f1", 0.0),
                "soft_skill_f1": payload.get("per_label", {}).get("soft_skill", {}).get("f1", 0.0),
            }
        )

    rows.sort(key=lambda row: row["stage"])

    csv_path = output_dir / "ablation_table.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["experiment_name"])
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Stage & Exact F1 & Hard F1 & Soft F1 \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['stage']} & {row['exact_f1']:.4f} & {row['hard_skill_f1']:.4f} & {row['soft_skill_f1']:.4f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "ablation_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    best_row = max(rows, key=lambda row: row["exact_f1"], default=None)
    largest_gain_stage = None
    if len(rows) >= 2:
        deltas = []
        for left, right in zip(rows, rows[1:]):
            deltas.append((right["stage"], right["exact_f1"] - left["exact_f1"]))
        if deltas:
            largest_gain_stage = max(deltas, key=lambda item: item[1])[0]

    summary = {
        "rows": rows,
        "best_experiment": best_row,
        "largest_gain_stage": largest_gain_stage,
    }
    write_json(metrics_dir / "ablation_best_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
