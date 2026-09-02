"""tune_confidence_calibrator.py

Dev-set grid search over :class:`skillllm_v2.verify.confidence_calibrator.ConfidenceCalibrator`
thresholds (``hard_skill_threshold``, ``soft_skill_threshold``,
``dictionary_only_threshold``, ``multi_source_threshold``).

The calibrator never rejects a candidate whose ``metadata.sources``
intersects ``protect_anchor_sources`` (default ``{"mistral"}``) -- so, like
``tune_candidate_merger.py``, this tuning only controls how aggressively
brand-new GLiNER/dictionary-only additions are filtered AFTER they have
already passed ``CandidateMerger``'s add-only confidence/alias-evidence gate
and ``SpanVerifier``'s source-aware quality checks.

The full dev pipeline replayed here is:
    (mistral + gliner + dictionary candidates from the dev candidate cache)
    -> CandidateMerger().merge()    [tuned config from tune_candidate_merger, if present]
    -> BoundaryRefiner().refine()   [tuned config from tune_boundary_refiner, if present]
    -> SpanVerifier().filter()      [source-aware, not tuned]
    -> ConfidenceCalibrator(config=...).calibrate()   [grid-searched here]

Selection mirrors the other tuning scripts: pick the grid configuration that
maximizes dev Exact Span F1 subject to the guardrail that Soft Skill F1 must
not drop by more than ``SOFT_F1_MAX_DROP`` relative to the no-op (calibrator
never rejects anything, i.e. all four thresholds set to
``ALWAYS_ACCEPT``). If nothing beats the no-op, the no-op configuration is
written as the tuned config -- this guarantees the calibrator stage of the
wrapper can never regress Exact Span F1 relative to the rest of the pipeline.

This module performs NO Mistral inference and reuses the dev candidate cache
built by ``tune_candidate_merger.py``
(``artifacts/skillllm_v2/candidate_merger/dev_candidates_cache.jsonl``),
rebuilding it (CPU GLiNER + dictionary only) if missing.

Usage:
    python -m skillllm_v2.train.tune_confidence_calibrator
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.extract import CandidateMerger
from skillllm_v2.io_utils import write_text
from skillllm_v2.train.tune_candidate_merger import DEV_CANDIDATES_CACHE_PATH, DEV_PREDICTIONS_PATH, _load_dev_candidate_cache
from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier
from skillllm_v2.verify.confidence_calibrator import ALWAYS_ACCEPT, DEFAULT_CONFIG

TUNED_CONFIG_PATH = "artifacts/skillllm_v2/confidence_calibrator/tuned_config.yaml"
RESULTS_CSV_PATH = "artifacts/skillllm_v2/confidence_calibrator/dev_tuning_results.csv"
REPORT_PATH = "reports/skillllm_v2/confidence_calibrator_tuning.md"

HARD_SKILL_THRESHOLD_GRID = [ALWAYS_ACCEPT, 0.35, 0.45]
SOFT_SKILL_THRESHOLD_GRID = [ALWAYS_ACCEPT, 0.30, 0.42]
DICTIONARY_ONLY_THRESHOLD_GRID = [ALWAYS_ACCEPT, 0.55, 0.70]
MULTI_SOURCE_THRESHOLD_GRID = [ALWAYS_ACCEPT, 0.30, 0.40]
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
    return {k: (list(v) if isinstance(v, list) else v) for k, v in config.items()}


def _config_variant(hard: float, soft: float, dictionary_only: float, multi_source: float) -> dict[str, Any]:
    config = _copy_config(DEFAULT_CONFIG)
    config["hard_skill_threshold"] = hard
    config["soft_skill_threshold"] = soft
    config["dictionary_only_threshold"] = dictionary_only
    config["multi_source_threshold"] = multi_source
    return config


def _no_op_config() -> dict[str, Any]:
    return _config_variant(ALWAYS_ACCEPT, ALWAYS_ACCEPT, ALWAYS_ACCEPT, ALWAYS_ACCEPT)


def _grid() -> list[tuple[float, float, float, float]]:
    return list(product(HARD_SKILL_THRESHOLD_GRID, SOFT_SKILL_THRESHOLD_GRID, DICTIONARY_ONLY_THRESHOLD_GRID, MULTI_SOURCE_THRESHOLD_GRID))


def _label(hard: float, soft: float, dictionary_only: float, multi_source: float) -> str:
    if hard == soft == dictionary_only == multi_source == ALWAYS_ACCEPT:
        return "no_op (calibrator passthrough)"

    def fmt(value: float) -> str:
        return "always" if value == ALWAYS_ACCEPT else f"{value:.2f}"

    return f"hard>={fmt(hard)}, soft>={fmt(soft)}, dict_only>={fmt(dictionary_only)}, multi>={fmt(multi_source)}"


def _build_pred(rows: list[dict[str, Any]], merger: CandidateMerger, refiner: BoundaryRefiner, verifier: SpanVerifier) -> list[list[dict[str, Any]]]:
    """Replay merge -> refine -> verify once; calibration is the only varying stage."""
    verified: list[list[dict[str, Any]]] = []
    for row in rows:
        candidates = list(row["mistral_predictions"]) + list(row["gliner_candidates"]) + list(row["dictionary_candidates"])
        merged = merger.merge(candidates)
        refined = refiner.refine(row["text"], merged)
        verified.append(verifier.filter(refined))
    return verified


def _evaluate_config(rows: list[dict[str, Any]], verified: list[list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, float]:
    calibrator = ConfidenceCalibrator(config=config, tuned_config_path=None)
    gold: list[list[dict[str, Any]]] = []
    pred: list[list[dict[str, Any]]] = []
    for row, candidates in zip(rows, verified):
        gold.append(row["gold_spans"])
        pred.append(calibrator.calibrate(candidates))
    return evaluate_extraction(gold, pred)


def tune(
    dev_predictions_path: str = DEV_PREDICTIONS_PATH,
    cache_path: str = DEV_CANDIDATES_CACHE_PATH,
    tuned_config_path: str = TUNED_CONFIG_PATH,
    results_csv_path: str = RESULTS_CSV_PATH,
    report_path: str = REPORT_PATH,
    rebuild_cache: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    rows = _load_dev_candidate_cache(dev_predictions_path, cache_path, rebuild=rebuild_cache, limit=limit)

    merger = CandidateMerger()
    refiner = BoundaryRefiner()
    verifier = SpanVerifier()
    verified = _build_pred(rows, merger, refiner, verifier)

    no_op_config = _no_op_config()
    no_op_metrics = _evaluate_config(rows, verified, no_op_config)
    soft_floor = no_op_metrics["soft_skill_f1"] - SOFT_F1_MAX_DROP

    results: list[dict[str, Any]] = []
    for hard, soft, dictionary_only, multi_source in _grid():
        config = _config_variant(hard, soft, dictionary_only, multi_source)
        metrics = _evaluate_config(rows, verified, config)
        result = {
            "hard_skill_threshold": hard,
            "soft_skill_threshold": soft,
            "dictionary_only_threshold": dictionary_only,
            "multi_source_threshold": multi_source,
            **{col: metrics[col] for col in METRIC_COLUMNS},
            "delta_exact_vs_no_op": metrics["exact_span_f1"] - no_op_metrics["exact_span_f1"],
            "delta_soft_vs_no_op": metrics["soft_skill_f1"] - no_op_metrics["soft_skill_f1"],
            "passes_soft_guardrail": metrics["soft_skill_f1"] >= soft_floor,
            "config": config,
            "label": _label(hard, soft, dictionary_only, multi_source),
        }
        results.append(result)

    eligible = [r for r in results if r["passes_soft_guardrail"]]
    best = max(eligible, key=lambda r: r["exact_span_f1"])
    no_op_result = next(r for r in results if r["label"].startswith("no_op"))

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
        "cache_path": cache_path,
        "dev_doc_count": len(rows),
    }


def _write_results_csv(path: str, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "hard_skill_threshold",
        "soft_skill_threshold",
        "dictionary_only_threshold",
        "multi_source_threshold",
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
        "confidence_calibrator": selected_config,
        "tuning_meta": {
            "decision": decision,
            "selected_label": selected["label"],
            "dev_doc_count": dev_doc_count,
            "soft_skill_f1_floor": soft_floor,
            "no_op_dev_metrics": {col: no_op_result[col] for col in METRIC_COLUMNS},
            "selected_dev_metrics": {col: selected[col] for col in METRIC_COLUMNS},
            "note": (
                "decision == 'no_op_fallback' means no calibration-threshold "
                "configuration beat the calibrator passthrough (ALWAYS_ACCEPT for all "
                "four thresholds) on dev Exact Span F1 within the Soft Skill F1 "
                "guardrail, so the confidence-calibrator stage is configured to never "
                "reject a candidate, making it a non-regressive passthrough. "
                "Candidates whose metadata.sources intersects protect_anchor_sources "
                "(default ['mistral']) are always accepted regardless of this "
                "configuration."
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
        "# Confidence Calibrator Dev-Set Tuning",
        "",
        f"Dev split: {dev_doc_count} documents (SkillSpan dev split, Raw Mistral LoRA "
        "predictions plus freshly computed GLiNER and dictionary candidates, replayed "
        "through CandidateMerger -> BoundaryRefiner -> SpanVerifier with their own "
        "tuned/default configs -- inference only, no retraining).",
        "",
        "The calibrator never rejects a candidate whose `metadata.sources` intersects "
        "`protect_anchor_sources` (default `{\"mistral\"}`). This tuning only controls "
        "how aggressively brand-new GLiNER/dictionary-only additions are filtered.",
        "",
        f"Soft Skill F1 guardrail: selected config must score >= {soft_floor:.6f} "
        f"(no-op Soft Skill F1 {no_op_result['soft_skill_f1']:.6f} minus {SOFT_F1_MAX_DROP}).",
        "",
        "## No-op baseline (calibrator passthrough, ALWAYS_ACCEPT for all thresholds)",
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
            f"by {selected['delta_exact_vs_no_op']:+.6f} over the calibrator-passthrough no-op "
            f"while staying within the Soft Skill F1 guardrail "
            f"({selected['soft_skill_f1']:.6f} >= {soft_floor:.6f}).",
            "",
            f"Written to `{TUNED_CONFIG_PATH}` and consumed by "
            "`skillllm_v2.verify.confidence_calibrator.ConfidenceCalibrator` via "
            "`load_calibrator_config()`.",
        ]
    else:
        lines += [
            "No grid configuration improved dev Exact Span F1 over the calibrator "
            "passthrough within the Soft Skill F1 guardrail. The confidence-calibrator "
            "stage falls back to **no-op**: all four thresholds are set to "
            "`ALWAYS_ACCEPT`, so no candidate is ever rejected on confidence grounds "
            "(this stage becomes a non-regressive passthrough).",
            "",
            f"Written to `{TUNED_CONFIG_PATH}` and consumed by "
            "`skillllm_v2.verify.confidence_calibrator.ConfidenceCalibrator` via "
            "`load_calibrator_config()`.",
        ]

    write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune the confidence calibrator's non-anchor thresholds on dev-set Raw Mistral predictions")
    parser.add_argument("--dev-predictions", default=DEV_PREDICTIONS_PATH)
    parser.add_argument("--cache", default=DEV_CANDIDATES_CACHE_PATH)
    parser.add_argument("--tuned-config", default=TUNED_CONFIG_PATH)
    parser.add_argument("--results-csv", default=RESULTS_CSV_PATH)
    parser.add_argument("--report", default=REPORT_PATH)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    summary = tune(args.dev_predictions, args.cache, args.tuned_config, args.results_csv, args.report, args.rebuild_cache, args.limit)
    print(f"Decision: {summary['decision']} ({summary['selected_label']})")
    print(f"No-op dev metrics: {summary['no_op_metrics']}")
    print(f"Selected dev metrics: {summary['selected_metrics']}")
    print(f"Dev candidate cache: {summary['cache_path']}")
    print(f"Tuned config: {summary['tuned_config_path']}")
    print(f"Results CSV: {summary['results_csv_path']}")
    print(f"Report: {summary['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
