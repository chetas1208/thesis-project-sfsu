"""tune_candidate_merger.py

Dev-set grid search over the "add-only" policy in
:class:`skillllm_v2.extract.candidate_merger.CandidateMerger`
(``min_confidence_new_span`` per source combination and
``require_alias_evidence_for_new_span``).

The merger never removes a candidate whose ``metadata.sources`` includes
``"mistral"`` -- so the only thing this tuning script can do is decide how
permissive the merger is about ADDING brand-new GLiNER/dictionary-only spans
on top of the cached Raw Mistral LoRA dev predictions
(``artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl``, produced by
``skillllm_v2.eval.generate_split_predictions``, inference only -- no
retraining).

Selection mirrors ``tune_boundary_refiner.py``: pick the grid configuration
that maximizes dev Exact Span F1 subject to the guardrail that Soft Skill F1
must not drop by more than ``SOFT_F1_MAX_DROP`` relative to the
"mistral-only" no-op (every non-Mistral candidate dropped, so the merger is a
pure passthrough over Raw Mistral). If nothing beats the no-op on Exact Span
F1, the no-op configuration is written as the tuned config -- this guarantees
the candidate-merger stage of the wrapper can never regress Exact Span F1
relative to Raw Mistral.

This module performs NO Mistral inference. It only runs the (CPU, fast)
GLiNER and dictionary extractors to build a reusable dev candidate cache
(``artifacts/skillllm_v2/candidate_merger/dev_candidates_cache.jsonl``), then
replays that cache through the (pure-Python) candidate merger.

Usage:
    python -m skillllm_v2.train.tune_candidate_merger
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.extract import CandidateMerger, DictionaryExtractor, GLiNERExtractor
from skillllm_v2.extract.candidate_merger import DEFAULT_CONFIG
from skillllm_v2.io_utils import read_jsonl, write_jsonl, write_text

DEV_PREDICTIONS_PATH = "artifacts/skillllm_v2/mistral_lora/predictions_dev.jsonl"
DEV_CANDIDATES_CACHE_PATH = "artifacts/skillllm_v2/candidate_merger/dev_candidates_cache.jsonl"
TUNED_CONFIG_PATH = "artifacts/skillllm_v2/candidate_merger/tuned_config.yaml"
RESULTS_CSV_PATH = "artifacts/skillllm_v2/candidate_merger/dev_tuning_results.csv"
REPORT_PATH = "reports/skillllm_v2/candidate_merger_tuning.md"

# 1.01 is an unreachable confidence (confidences are <= 1.0), so it means
# "never accept a new span from this source combination".
NEVER = 1.01

GLINER_THRESHOLD_GRID = [NEVER, 0.85, 0.75, 0.65, 0.55, 0.45]
DICTIONARY_THRESHOLD_GRID = [NEVER, 0.90, 0.82, 0.75, 0.65]
ALIAS_EVIDENCE_GRID = [True, False]
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
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}


def _config_variant(gliner_threshold: float, dictionary_threshold: float, alias_evidence_required: bool) -> dict[str, Any]:
    config = _copy_config(DEFAULT_CONFIG)
    combined = max(0.0, min(gliner_threshold, dictionary_threshold) - 0.10)
    config["min_confidence_new_span"] = {
        "gliner": gliner_threshold,
        "dictionary": dictionary_threshold,
        # key order matches sorted({"dictionary", "gliner"}) -> "dictionary+gliner"
        "dictionary+gliner": combined,
    }
    config["default_min_confidence_new_span"] = max(gliner_threshold, dictionary_threshold)
    config["require_alias_evidence_for_new_span"] = alias_evidence_required
    return config


def _no_op_config() -> dict[str, Any]:
    return _config_variant(NEVER, NEVER, True)


def _grid() -> list[tuple[float, float, bool]]:
    combos: list[tuple[float, float, bool]] = [(NEVER, NEVER, True)]
    for gliner_threshold, dictionary_threshold, alias_evidence_required in product(GLINER_THRESHOLD_GRID, DICTIONARY_THRESHOLD_GRID, ALIAS_EVIDENCE_GRID):
        if gliner_threshold == NEVER and dictionary_threshold == NEVER:
            continue
        combos.append((gliner_threshold, dictionary_threshold, alias_evidence_required))
    return combos


def _label(gliner_threshold: float, dictionary_threshold: float, alias_evidence_required: bool) -> str:
    if gliner_threshold == NEVER and dictionary_threshold == NEVER:
        return "no_op (mistral-only passthrough)"
    parts = []
    parts.append("gliner>=" + ("never" if gliner_threshold == NEVER else f"{gliner_threshold:.2f}"))
    parts.append("dictionary>=" + ("never" if dictionary_threshold == NEVER else f"{dictionary_threshold:.2f}"))
    parts.append("alias_evidence" if alias_evidence_required else "no_alias_evidence")
    return ", ".join(parts)


def _build_dev_candidate_cache(dev_predictions_path: str, cache_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(dev_predictions_path)
    if not rows:
        raise FileNotFoundError(
            f"No dev predictions found at {dev_predictions_path}. Run "
            "`python -m skillllm_v2.eval.generate_split_predictions --split dev` first "
            "(inference only, reuses the existing 80-epoch adapter)."
        )
    if limit:
        rows = rows[:limit]

    gliner = GLiNERExtractor()
    dictionary = DictionaryExtractor()
    cache_rows: list[dict[str, Any]] = []
    for row in rows:
        cache_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "gold_spans": row["gold_spans"],
                "mistral_predictions": row["predictions"],
                "gliner_candidates": gliner.extract(row["text"]),
                "dictionary_candidates": dictionary.extract(row["text"]),
            }
        )
    write_jsonl(cache_path, cache_rows)
    return cache_rows


def _load_dev_candidate_cache(dev_predictions_path: str, cache_path: str, rebuild: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    if not rebuild:
        cached = read_jsonl(cache_path)
        if cached:
            return cached[:limit] if limit else cached
    return _build_dev_candidate_cache(dev_predictions_path, cache_path, limit=limit)


def _evaluate_config(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    merger = CandidateMerger(config=config, tuned_config_path=None)
    gold: list[list[dict[str, Any]]] = []
    pred: list[list[dict[str, Any]]] = []
    for row in rows:
        candidates = list(row["mistral_predictions"]) + list(row["gliner_candidates"]) + list(row["dictionary_candidates"])
        merged = merger.merge(candidates)
        gold.append(row["gold_spans"])
        pred.append(merged)
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

    no_op_config = _no_op_config()
    no_op_metrics = _evaluate_config(rows, no_op_config)
    soft_floor = no_op_metrics["soft_skill_f1"] - SOFT_F1_MAX_DROP

    results: list[dict[str, Any]] = []
    for gliner_threshold, dictionary_threshold, alias_evidence_required in _grid():
        config = _config_variant(gliner_threshold, dictionary_threshold, alias_evidence_required)
        metrics = _evaluate_config(rows, config)
        result = {
            "gliner_threshold": "never" if gliner_threshold == NEVER else gliner_threshold,
            "dictionary_threshold": "never" if dictionary_threshold == NEVER else dictionary_threshold,
            "require_alias_evidence_for_new_span": alias_evidence_required,
            **{col: metrics[col] for col in METRIC_COLUMNS},
            "delta_exact_vs_no_op": metrics["exact_span_f1"] - no_op_metrics["exact_span_f1"],
            "delta_soft_vs_no_op": metrics["soft_skill_f1"] - no_op_metrics["soft_skill_f1"],
            "passes_soft_guardrail": metrics["soft_skill_f1"] >= soft_floor,
            "config": config,
            "label": _label(gliner_threshold, dictionary_threshold, alias_evidence_required),
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
        "gliner_threshold",
        "dictionary_threshold",
        "require_alias_evidence_for_new_span",
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
        "candidate_merger": selected_config,
        "tuning_meta": {
            "decision": decision,
            "selected_label": selected["label"],
            "dev_doc_count": dev_doc_count,
            "soft_skill_f1_floor": soft_floor,
            "no_op_dev_metrics": {col: no_op_result[col] for col in METRIC_COLUMNS},
            "selected_dev_metrics": {col: selected[col] for col in METRIC_COLUMNS},
            "note": (
                "decision == 'no_op_fallback' means no add-only configuration beat the "
                "mistral-only passthrough on dev Exact Span F1 within the Soft Skill F1 "
                "guardrail, so the candidate-merger stage is configured to never accept "
                "a new GLiNER/dictionary-only span (min_confidence_new_span thresholds "
                "set to an unreachable 1.01), making it a non-regressive passthrough over "
                "Raw Mistral. Candidates whose metadata.sources includes 'mistral' are "
                "always kept regardless of this configuration."
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
        "# Candidate Merger Dev-Set Tuning",
        "",
        f"Dev split: {dev_doc_count} documents (SkillSpan dev split, Raw Mistral LoRA "
        "predictions from `skillllm_v2.eval.generate_split_predictions`, plus freshly "
        "computed GLiNER and dictionary candidates -- inference only, no retraining).",
        "",
        "The candidate merger never removes a candidate whose `metadata.sources` "
        "includes `\"mistral\"`. This tuning only controls whether brand-new "
        "GLiNER/dictionary-only spans are ADDED on top of Raw Mistral.",
        "",
        f"Soft Skill F1 guardrail: selected config must score >= {soft_floor:.6f} "
        f"(mistral-only no-op Soft Skill F1 {no_op_result['soft_skill_f1']:.6f} minus {SOFT_F1_MAX_DROP}).",
        "",
        "## No-op baseline (mistral-only passthrough, no GLiNER/dictionary additions)",
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
            f"by {selected['delta_exact_vs_no_op']:+.6f} over the mistral-only no-op while staying "
            f"within the Soft Skill F1 guardrail "
            f"({selected['soft_skill_f1']:.6f} >= {soft_floor:.6f}).",
            "",
            f"Written to `{TUNED_CONFIG_PATH}` and consumed by "
            "`skillllm_v2.extract.candidate_merger.CandidateMerger` via `load_merger_config()`.",
        ]
    else:
        lines += [
            "No grid configuration improved dev Exact Span F1 over the mistral-only "
            "no-op within the Soft Skill F1 guardrail. The candidate-merger stage falls "
            "back to **no-op**: `min_confidence_new_span` thresholds are set to an "
            "unreachable 1.01, so no GLiNER/dictionary-only span is ever added and the "
            "merger is a non-regressive passthrough over Raw Mistral (candidates with "
            "`\"mistral\"` in `metadata.sources` are always kept).",
            "",
            f"Written to `{TUNED_CONFIG_PATH}` and consumed by "
            "`skillllm_v2.extract.candidate_merger.CandidateMerger` via `load_merger_config()`.",
        ]

    write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune the candidate merger's add-only policy on dev-set Raw Mistral predictions")
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
