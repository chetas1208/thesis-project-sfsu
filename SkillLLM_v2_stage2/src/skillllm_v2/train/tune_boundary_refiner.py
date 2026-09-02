"""tune_boundary_refiner.py

Dev-set grid search over :class:`skillllm_v2.verify.boundary_selector.BoundarySelector`
configuration. Selects the configuration that maximizes Exact Span F1 on the
914-doc dev split (Raw Mistral LoRA predictions, inference cached by
``skillllm_v2.eval.generate_split_predictions``) subject to the guardrail that
Soft Skill F1 must not drop by more than ``SOFT_F1_MAX_DROP`` relative to the
unrefined (no-op) Raw Mistral dev predictions.

If no grid configuration beats the no-op baseline on Exact Span F1, the no-op
configuration (``allowed_edit_types: []``, i.e. the boundary refiner always
preserves the original Raw Mistral span) is written as the tuned config. This
guarantees the boundary refiner stage of the wrapper can never regress Exact
Span F1 relative to Raw Mistral.

This module performs NO model inference and NO retraining -- it only replays
cached dev predictions through the (pure-Python) boundary refiner/selector.

Usage:
    python -m skillllm_v2.train.tune_boundary_refiner
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.io_utils import read_jsonl, write_text
from skillllm_v2.verify.boundary_refiner import BoundaryRefiner
from skillllm_v2.verify.boundary_selector import DEFAULT_CONFIG, BoundarySelector

DEV_PREDICTIONS_PATH = "artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl"
TUNED_CONFIG_PATH = "artifacts/skillllm_v2/boundary_refiner/tuned_config.yaml"
RESULTS_CSV_PATH = "artifacts/skillllm_v2/boundary_refiner/dev_tuning_results.csv"
REPORT_PATH = "reports/skillllm_v2/boundary_refiner_tuning.md"

ALLOWED_EDIT_TYPE_GRID: list[list[str]] = [
    [],
    ["trim_leading_filler"],
    ["taxonomy_normalized"],
    ["trim_leading_filler", "taxonomy_normalized"],
]
HARD_MARGIN_GRID = [0.06, 0.08, 0.10, 0.12]
SOFT_MARGIN_GRID = [0.12, 0.15, 0.18, 0.22]
SOFT_F1_MAX_DROP = 0.03

METRIC_COLUMNS = [
    "exact_span_f1",
    "exact_span_precision",
    "exact_span_recall",
    "soft_skill_f1",
    "hard_skill_f1",
    "token_f1",
    "partial_span_f1",
]


def _copy_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in config.items()}


def _config_variant(allowed_edit_types: list[str], hard_margin: float | None, soft_margin: float | None) -> dict[str, Any]:
    config = _copy_config(DEFAULT_CONFIG)
    config["allowed_edit_types"] = list(allowed_edit_types)
    if hard_margin is not None:
        config["label_thresholds"]["HARD_SKILL"] = {"min_margin_to_replace": hard_margin}
    if soft_margin is not None:
        config["label_thresholds"]["SOFT_SKILL"] = {"min_margin_to_replace": soft_margin}
    return config


def _grid() -> list[tuple[list[str], float | None, float | None]]:
    combos: list[tuple[list[str], float | None, float | None]] = [([], None, None)]
    for allowed in ALLOWED_EDIT_TYPE_GRID:
        if not allowed:
            continue
        for hard_margin, soft_margin in product(HARD_MARGIN_GRID, SOFT_MARGIN_GRID):
            combos.append((allowed, hard_margin, soft_margin))
    return combos


def _load_dev_rows(path: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(
            f"No dev predictions found at {path}. Run "
            "`python -m skillllm_v2.eval.generate_split_predictions --split dev` first "
            "(inference only, reuses the existing 80-epoch adapter)."
        )
    return rows


def _evaluate_config(rows: list[dict[str, Any]], refiner: BoundaryRefiner, config: dict[str, Any]) -> dict[str, float]:
    selector = BoundarySelector(config=config, tuned_config_path=None)
    gold: list[list[dict[str, Any]]] = []
    pred: list[list[dict[str, Any]]] = []
    for row in rows:
        refined = refiner.refine(row["text"], row["predictions"], selector=selector)
        gold.append(row["gold_spans"])
        pred.append(refined)
    return evaluate_extraction(gold, pred)


def _label(allowed_edit_types: list[str], hard_margin: float | None, soft_margin: float | None) -> str:
    if not allowed_edit_types:
        return "no_op"
    return f"{'+'.join(allowed_edit_types)} (hard>={hard_margin}, soft>={soft_margin})"


def tune(
    dev_predictions_path: str = DEV_PREDICTIONS_PATH,
    tuned_config_path: str = TUNED_CONFIG_PATH,
    results_csv_path: str = RESULTS_CSV_PATH,
    report_path: str = REPORT_PATH,
) -> dict[str, Any]:
    rows = _load_dev_rows(dev_predictions_path)
    refiner = BoundaryRefiner()

    no_op_config = _config_variant([], None, None)
    no_op_metrics = _evaluate_config(rows, refiner, no_op_config)
    soft_floor = no_op_metrics["soft_skill_f1"] - SOFT_F1_MAX_DROP

    results: list[dict[str, Any]] = []
    for allowed_edit_types, hard_margin, soft_margin in _grid():
        config = _config_variant(allowed_edit_types, hard_margin, soft_margin)
        metrics = _evaluate_config(rows, refiner, config)
        result = {
            "allowed_edit_types": "+".join(allowed_edit_types) if allowed_edit_types else "(none)",
            "hard_skill_min_margin": hard_margin if hard_margin is not None else "",
            "soft_skill_min_margin": soft_margin if soft_margin is not None else "",
            **{col: metrics[col] for col in METRIC_COLUMNS},
            "delta_exact_vs_no_op": metrics["exact_span_f1"] - no_op_metrics["exact_span_f1"],
            "delta_soft_vs_no_op": metrics["soft_skill_f1"] - no_op_metrics["soft_skill_f1"],
            "passes_soft_guardrail": metrics["soft_skill_f1"] >= soft_floor,
            "config": config,
            "label": _label(allowed_edit_types, hard_margin, soft_margin),
        }
        results.append(result)

    eligible = [r for r in results if r["passes_soft_guardrail"]]
    best = max(eligible, key=lambda r: r["exact_span_f1"])
    no_op_result = next(r for r in results if r["allowed_edit_types"] == "(none)")

    if best["exact_span_f1"] > no_op_result["exact_span_f1"]:
        selected = best
        decision = "tuned"
    else:
        selected = no_op_result
        decision = "no_op_fallback"

    selected_config = selected["config"]

    _write_results_csv(results_csv_path, results)
    _write_tuned_config(tuned_config_path, selected_config, selected, no_op_result, decision, len(rows), soft_floor)
    _write_report(report_path, results, no_op_result, selected, decision, len(rows), soft_floor)

    return {
        "decision": decision,
        "no_op_metrics": no_op_metrics,
        "selected_metrics": {col: selected[col] for col in METRIC_COLUMNS},
        "selected_label": selected["label"],
        "tuned_config_path": tuned_config_path,
        "results_csv_path": results_csv_path,
        "report_path": report_path,
        "dev_doc_count": len(rows),
    }


def _write_results_csv(path: str, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "allowed_edit_types",
        "hard_skill_min_margin",
        "soft_skill_min_margin",
        *METRIC_COLUMNS,
        "delta_exact_vs_no_op",
        "delta_soft_vs_no_op",
        "passes_soft_guardrail",
    ]
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})


def _write_tuned_config(
    path: str,
    selected_config: dict[str, Any],
    selected: dict[str, Any],
    no_op_result: dict[str, Any],
    decision: str,
    dev_doc_count: int,
    soft_floor: float,
) -> None:
    payload = {
        "boundary_selector": selected_config,
        "tuning_meta": {
            "decision": decision,
            "selected_label": selected["label"],
            "dev_doc_count": dev_doc_count,
            "soft_skill_f1_floor": soft_floor,
            "no_op_dev_metrics": {col: no_op_result[col] for col in METRIC_COLUMNS},
            "selected_dev_metrics": {col: selected[col] for col in METRIC_COLUMNS},
            "note": (
                "decision == 'no_op_fallback' means no boundary-edit configuration beat "
                "the unrefined Raw Mistral predictions on dev Exact Span F1 within the "
                "Soft Skill F1 guardrail, so the boundary refiner stage is configured to "
                "always preserve the original Raw Mistral span (non-regressive passthrough)."
            ),
        },
    }
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_report(
    path: str,
    results: list[dict[str, Any]],
    no_op_result: dict[str, Any],
    selected: dict[str, Any],
    decision: str,
    dev_doc_count: int,
    soft_floor: float,
) -> None:
    top5 = sorted(
        (r for r in results if r["passes_soft_guardrail"]),
        key=lambda r: r["exact_span_f1"],
        reverse=True,
    )[:5]

    lines = [
        "# Boundary Refiner Dev-Set Tuning",
        "",
        f"Dev split: {dev_doc_count} documents (SkillSpan dev split, Raw Mistral LoRA "
        "predictions from `skillllm_v2.eval.generate_split_predictions`, inference only -- "
        "no retraining).",
        "",
        f"Soft Skill F1 guardrail: selected config must score >= {soft_floor:.6f} "
        f"(no-op Soft Skill F1 {no_op_result['soft_skill_f1']:.6f} minus {SOFT_F1_MAX_DROP}).",
        "",
        "## No-op baseline (boundary refiner disabled, Raw Mistral passthrough)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for col in METRIC_COLUMNS:
        lines.append(f"| {col} | {no_op_result[col]:.6f} |")

    lines += [
        "",
        "## Top configurations (within Soft Skill F1 guardrail, ranked by Exact Span F1)",
        "",
        "| Rank | Config | Exact F1 | Soft F1 | Hard F1 | Token F1 | Delta Exact vs no-op |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(top5, start=1):
        lines.append(
            f"| {i} | {r['label']} | {r['exact_span_f1']:.6f} | {r['soft_skill_f1']:.6f} | "
            f"{r['hard_skill_f1']:.6f} | {r['token_f1']:.6f} | {r['delta_exact_vs_no_op']:+.6f} |"
        )

    lines += [
        "",
        "## Decision",
        "",
    ]
    if decision == "tuned":
        lines += [
            f"Selected configuration **{selected['label']}**, which improves dev Exact Span F1 "
            f"by {selected['delta_exact_vs_no_op']:+.6f} over the no-op baseline while staying "
            f"within the Soft Skill F1 guardrail "
            f"({selected['soft_skill_f1']:.6f} >= {soft_floor:.6f}).",
            "",
            f"Written to `{TUNED_CONFIG_PATH}` and consumed by "
            "`skillllm_v2.verify.boundary_selector.BoundarySelector` via "
            "`load_selector_config()`.",
        ]
    else:
        lines += [
            "No grid configuration improved dev Exact Span F1 over the no-op baseline within "
            "the Soft Skill F1 guardrail. The boundary refiner stage falls back to "
            "**no-op** (`allowed_edit_types: []`): every Raw Mistral span is preserved "
            "unchanged, so this stage cannot regress Exact Span F1 relative to Raw Mistral.",
            "",
            f"Written to `{TUNED_CONFIG_PATH}` and consumed by "
            "`skillllm_v2.verify.boundary_selector.BoundarySelector` via "
            "`load_selector_config()`.",
        ]

    write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune the boundary refiner selector on dev-set Raw Mistral predictions")
    parser.add_argument("--dev-predictions", default=DEV_PREDICTIONS_PATH)
    parser.add_argument("--tuned-config", default=TUNED_CONFIG_PATH)
    parser.add_argument("--results-csv", default=RESULTS_CSV_PATH)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()

    summary = tune(args.dev_predictions, args.tuned_config, args.results_csv, args.report)
    print(f"Decision: {summary['decision']} ({summary['selected_label']})")
    print(f"No-op dev metrics: {summary['no_op_metrics']}")
    print(f"Selected dev metrics: {summary['selected_metrics']}")
    print(f"Tuned config: {summary['tuned_config_path']}")
    print(f"Results CSV: {summary['results_csv_path']}")
    print(f"Report: {summary['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
