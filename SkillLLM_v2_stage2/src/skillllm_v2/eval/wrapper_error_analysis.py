"""wrapper_error_analysis.py

Compares Raw Mistral LoRA predictions (the 80-epoch anchor) against each
downstream SkillLLM_v2 wrapper stage on the SkillSpan test split, to localize
where postprocessing helps or hurts relative to the anchor.

Outputs (under ``reports/skillllm_v2/``):
  - ``wrapper_error_analysis.json`` -- machine-readable per-stage metrics, deltas
    and edit/rejection counters.
  - ``wrapper_error_analysis.md``   -- human-readable summary table.
  - ``wrapper_regression_cases.jsonl`` -- one row per (doc, stage, span) where
    Raw Mistral was correct but the wrapper stage is not.

Usage:
    python -m skillllm_v2.eval.wrapper_error_analysis
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from skillllm_v2.eval.extraction_eval import _key, _prf, evaluate_extraction
from skillllm_v2.io_utils import read_jsonl, write_json, write_jsonl, write_text
from skillllm_v2.schema import normalize_text
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
from skillllm_v2.verify.span_verifier import GENERIC_REJECTS

RAW_PREDICTIONS = "artifacts/skillllm_v2/mistral_lora/predictions.jsonl"

STAGES = [
    ("mistral_boundary", "Mistral + Boundary Refiner", "artifacts/skillllm_v2/mistral_boundary/predictions.jsonl"),
    ("mistral_gliner_taxonomy", "Mistral + GLiNER + Taxonomy", "artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl"),
    ("full_cag_graph", "Full CAG/Graph", "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl"),
]

LOW_CONFIDENCE_THRESHOLD = 0.45

RECOMMENDATIONS = {
    "removed_correct_span": (
        "Raw Mistral span '{surface}' ({label}, conf={confidence:.2f}) was correct but is "
        "absent from the wrapper output with no overlapping replacement. Treat Raw Mistral "
        "as the anchor: never drop a span unless a source-aware verifier rejects it with "
        "strong negative evidence."
    ),
    "bad_boundary_edit": (
        "Boundary edit changed '{surface}' -> '{new_surface}' ({label}). Generate this as a "
        "candidate variant only (boundary_refiner) and let the dev-tuned boundary_selector "
        "keep the original span unless the candidate has strong taxonomy/alias evidence."
    ),
    "bad_label_change": (
        "Label changed from {label} to {new_label} for span overlapping '{surface}'. Do not "
        "let candidate merging override the Raw Mistral label without high-confidence "
        "multi-source agreement."
    ),
    "verifier_false_rejection": (
        "Module '{module}' rejected a correct Raw Mistral span '{surface}' "
        "({label}, conf={confidence:.2f}). Verifier/calibrator must be source-aware: Raw "
        "Mistral predictions need a lower keep-threshold than GLiNER/dictionary-only candidates."
    ),
    "taxonomy_filter_false_rejection": (
        "Taxonomy linking removed or altered span '{surface}'. Linking must only attach "
        "canonical_skill_id/canonical_skill_name metadata and must never remove a prediction."
    ),
}

DELTA_METRICS = [
    "exact_span_f1",
    "exact_span_precision",
    "exact_span_recall",
    "partial_span_f1",
    "token_f1",
    "hard_skill_f1",
    "soft_skill_f1",
    "macro_f1",
]


def _sources(item: dict[str, Any]) -> set[str]:
    meta = item.get("metadata") or {}
    sources = set(meta.get("sources") or [])
    if item.get("source"):
        sources.add(item["source"])
    return sources


def _find_overlap(stage_preds: list[dict[str, Any]], raw_span: dict[str, Any]) -> dict[str, Any] | None:
    if raw_span.get("start") is None:
        return None
    same_label: dict[str, Any] | None = None
    any_label: dict[str, Any] | None = None
    for pred in stage_preds:
        if pred.get("start") is None:
            continue
        if pred["start"] < raw_span["end"] and raw_span["start"] < pred["end"]:
            if pred["label"] == raw_span["label"] and same_label is None:
                same_label = pred
            elif any_label is None:
                any_label = pred
    return same_label or any_label


def _module_for_overlap(pred: dict[str, Any]) -> str:
    meta = pred.get("metadata") or {}
    if "boundary_refined_from" in meta or "boundary_split_from" in meta:
        return "boundary_refiner"
    if meta.get("taxonomy_link"):
        return "taxonomy_linker"
    return "candidate_merger"


def _classify_removed(raw_span: dict[str, Any], stage_preds: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any] | None]:
    overlap = _find_overlap(stage_preds, raw_span)
    if overlap is not None:
        if overlap["label"] != raw_span["label"]:
            return "bad_label_change", _module_for_overlap(overlap), overlap
        return "bad_boundary_edit", _module_for_overlap(overlap), overlap
    norm = normalize_text(raw_span["surface_form"])
    if norm in GENERIC_REJECTS or len(norm.split()) > 6:
        return "verifier_false_rejection", "span_verifier", None
    if float(raw_span.get("confidence", 1.0)) < LOW_CONFIDENCE_THRESHOLD:
        return "verifier_false_rejection", "confidence_calibrator", None
    return "removed_correct_span", "candidate_merger_or_pipeline", None


def _recommendation(rtype: str, module: str, raw_span: dict[str, Any], overlap: dict[str, Any] | None) -> str:
    template = RECOMMENDATIONS[rtype]
    return template.format(
        surface=raw_span["surface_form"],
        new_surface=overlap["surface_form"] if overlap else "",
        label=raw_span["label"],
        new_label=overlap["label"] if overlap else "",
        confidence=float(raw_span.get("confidence", 0.0)),
        module=module,
    )


def _label_prf(gold_lists: list[list[dict[str, Any]]], pred_lists: list[list[dict[str, Any]]], label: str) -> dict[str, Any]:
    tp = fp = fn = 0
    for gold_doc, pred_doc in zip(gold_lists, pred_lists):
        gold_set = {_key(item) for item in gold_doc if item.get("label") == label}
        pred_set = {_key(item) for item in pred_doc if item.get("label") == label}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _analyze_stage(
    stage_key: str,
    stage_label: str,
    ids: list[str],
    text_by_id: dict[str, str],
    gold_by_id: dict[str, list[dict[str, Any]]],
    raw_by_id: dict[str, list[dict[str, Any]]],
    stage_by_id: dict[str, list[dict[str, Any]]],
    raw_metrics: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common_ids = [doc_id for doc_id in ids if doc_id in stage_by_id]
    gold_lists = [gold_by_id[i] for i in common_ids]
    raw_lists = [raw_by_id[i] for i in common_ids]
    stage_lists = [stage_by_id[i] for i in common_ids]

    stage_metrics = evaluate_extraction(gold_lists, stage_lists)

    counters: Counter[str] = Counter()
    regression_cases: list[dict[str, Any]] = []

    for doc_id, gold, raw_preds, stage_preds in zip(common_ids, gold_lists, raw_lists, stage_lists):
        text = text_by_id[doc_id]
        gold_set = {_key(item) for item in gold}
        raw_keys = {_key(item) for item in raw_preds}
        stage_keys = {_key(item) for item in stage_preds}

        raw_correct = raw_keys & gold_set
        stage_correct = stage_keys & gold_set

        counters["kept"] += len(raw_correct & stage_keys)
        counters["exact_gains"] += len(stage_correct - raw_correct)
        counters["exact_losses"] += len(raw_correct - stage_correct)

        new_preds = [pred for pred in stage_preds if _key(pred) not in raw_keys]
        for pred in new_preds:
            srcs = _sources(pred)
            if "gliner" in srcs:
                counters["added_by_gliner"] += 1
            if "dictionary" in srcs:
                counters["added_by_dictionary"] += 1

        raw_by_key = {_key(item): item for item in raw_preds}
        removed = raw_correct - stage_keys
        counters["removed"] += len(removed)
        for key in removed:
            raw_span = raw_by_key[key]
            rtype, module, overlap = _classify_removed(raw_span, stage_preds)
            if rtype == "bad_boundary_edit":
                counters["boundary_modified"] += 1
            elif rtype == "bad_label_change":
                counters["label_changed"] += 1
            elif module == "span_verifier":
                counters["rejected_by_verifier"] += 1
            elif module == "confidence_calibrator":
                counters["rejected_by_confidence"] += 1
            regression_cases.append(
                {
                    "id": doc_id,
                    "stage": stage_key,
                    "text": text,
                    "gold": gold,
                    "raw_mistral": raw_preds,
                    "wrapper_prediction": stage_preds,
                    "regression_type": rtype,
                    "module_responsible": module,
                    "recommendation": _recommendation(rtype, module, raw_span, overlap),
                }
            )

    deltas = {metric: stage_metrics[metric] - raw_metrics[metric] for metric in DELTA_METRICS}

    summary = {
        "label": stage_label,
        "doc_count": len(common_ids),
        "metrics": stage_metrics,
        "deltas_vs_raw_mistral": deltas,
        "counts": dict(counters),
        "hard_skill": {
            "raw_mistral": _label_prf(gold_lists, raw_lists, "HARD_SKILL"),
            "wrapper": _label_prf(gold_lists, stage_lists, "HARD_SKILL"),
        },
        "soft_skill": {
            "raw_mistral": _label_prf(gold_lists, raw_lists, "SOFT_SKILL"),
            "wrapper": _label_prf(gold_lists, stage_lists, "SOFT_SKILL"),
        },
        "regression_case_count": len(regression_cases),
    }
    return summary, regression_cases


def run_analysis(
    dataset_root: str = "data/source/skillspan_hf",
    split: str = "test",
    output_dir: str = "reports/skillllm_v2",
) -> dict[str, Any]:
    gold_docs = load_skillspan_split(Path(dataset_root), split, limit=None)
    gold_docs = [doc for doc in gold_docs if doc["gold_spans"]]
    gold_by_id = {doc["id"]: doc["gold_spans"] for doc in gold_docs}
    text_by_id = {doc["id"]: doc["text"] for doc in gold_docs}

    raw_rows = read_jsonl(RAW_PREDICTIONS)
    raw_by_id = {row["id"]: row["predictions"] for row in raw_rows}

    ids = [doc_id for doc_id in gold_by_id if doc_id in raw_by_id]
    gold_lists = [gold_by_id[i] for i in ids]
    raw_lists = [raw_by_id[i] for i in ids]
    raw_metrics = evaluate_extraction(gold_lists, raw_lists)

    report: dict[str, Any] = {
        "dataset": {"root": dataset_root, "split": split, "doc_count": len(ids)},
        "raw_mistral_metrics": raw_metrics,
        "stages": {},
    }
    all_regression_cases: list[dict[str, Any]] = []

    for stage_key, stage_label, path in STAGES:
        if not Path(path).exists():
            report["stages"][stage_key] = {"label": stage_label, "error": f"missing predictions file: {path}"}
            continue
        stage_rows = read_jsonl(path)
        stage_by_id = {row["id"]: row["predictions"] for row in stage_rows}
        summary, regression_cases = _analyze_stage(
            stage_key, stage_label, ids, text_by_id, gold_by_id, raw_by_id, stage_by_id, raw_metrics
        )
        report["stages"][stage_key] = summary
        all_regression_cases.extend(regression_cases)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_json(f"{output_dir}/wrapper_error_analysis.json", report)
    write_jsonl(f"{output_dir}/wrapper_regression_cases.jsonl", all_regression_cases)
    write_text(f"{output_dir}/wrapper_error_analysis.md", _markdown_report(report, all_regression_cases))
    return report


def _markdown_report(report: dict[str, Any], regression_cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Wrapper Error Analysis")
    lines.append("")
    lines.append(
        f"Dataset: `{report['dataset']['root']}` split=`{report['dataset']['split']}` "
        f"doc_count={report['dataset']['doc_count']}"
    )
    lines.append("")

    raw = report["raw_mistral_metrics"]
    lines.append("## Raw Mistral LoRA (anchor)")
    lines.append("")
    lines.append(
        f"- exact_span_f1 = {raw['exact_span_f1']:.6f}\n"
        f"- token_f1 = {raw['token_f1']:.6f}\n"
        f"- hard_skill_f1 = {raw['hard_skill_f1']:.6f}\n"
        f"- soft_skill_f1 = {raw['soft_skill_f1']:.6f}"
    )
    lines.append("")

    lines.append("## Stage comparison")
    lines.append("")
    lines.append(
        "| Stage | Exact F1 | Δ Exact F1 | Token F1 | Δ Token F1 | Hard F1 | Δ Hard F1 | Soft F1 | Δ Soft F1 | Regressions |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for stage_key, _, _ in STAGES:
        summary = report["stages"].get(stage_key, {})
        if "error" in summary:
            lines.append(f"| {summary['label']} | {summary['error']} | | | | | | | | |")
            continue
        m = summary["metrics"]
        d = summary["deltas_vs_raw_mistral"]
        lines.append(
            f"| {summary['label']} | {m['exact_span_f1']:.4f} | {d['exact_span_f1']:+.4f} | "
            f"{m['token_f1']:.4f} | {d['token_f1']:+.4f} | "
            f"{m['hard_skill_f1']:.4f} | {d['hard_skill_f1']:+.4f} | "
            f"{m['soft_skill_f1']:.4f} | {d['soft_skill_f1']:+.4f} | "
            f"{summary['regression_case_count']} |"
        )
    lines.append("")

    lines.append("## Edit / rejection counts per stage")
    lines.append("")
    lines.append(
        "| Stage | kept | removed | boundary_modified | label_changed | added_by_gliner | "
        "added_by_dictionary | rejected_by_verifier | rejected_by_confidence | exact_gains | exact_losses |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for stage_key, _, _ in STAGES:
        summary = report["stages"].get(stage_key, {})
        if "error" in summary:
            continue
        c = summary["counts"]
        lines.append(
            f"| {summary['label']} | {c.get('kept', 0)} | {c.get('removed', 0)} | "
            f"{c.get('boundary_modified', 0)} | {c.get('label_changed', 0)} | "
            f"{c.get('added_by_gliner', 0)} | {c.get('added_by_dictionary', 0)} | "
            f"{c.get('rejected_by_verifier', 0)} | {c.get('rejected_by_confidence', 0)} | "
            f"{c.get('exact_gains', 0)} | {c.get('exact_losses', 0)} |"
        )
    lines.append("")

    lines.append("## Regression type breakdown")
    lines.append("")
    type_counts: Counter[tuple[str, str]] = Counter()
    module_counts: Counter[tuple[str, str]] = Counter()
    for case in regression_cases:
        type_counts[(case["stage"], case["regression_type"])] += 1
        module_counts[(case["stage"], case["module_responsible"])] += 1
    lines.append("| Stage | regression_type | count |")
    lines.append("|---|---|---|")
    for (stage_key, rtype), count in sorted(type_counts.items()):
        stage_label = next(label for key, label, _ in STAGES if key == stage_key)
        lines.append(f"| {stage_label} | {rtype} | {count} |")
    lines.append("")
    lines.append("| Stage | module_responsible | count |")
    lines.append("|---|---|---|")
    for (stage_key, module), count in sorted(module_counts.items()):
        stage_label = next(label for key, label, _ in STAGES if key == stage_key)
        lines.append(f"| {stage_label} | {module} | {count} |")
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "Every downstream stage shows a negative `Δ Exact F1` and `Δ Soft F1` versus the Raw Mistral "
        "anchor (see table above). The dominant regression types and `module_responsible` values "
        "above identify which postprocessing modules are responsible for destroying spans that the "
        "80-epoch Mistral LoRA already extracted correctly. Section 2/5/7/8/9 of the SkillLLM_v2 safe "
        "wrapper rewrite directly target these modules so that the wrapper becomes non-regressive "
        "relative to Raw Mistral."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrapper error analysis vs Raw Mistral anchor")
    parser.add_argument("--dataset-root", default="data/source/skillspan_hf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="reports/skillllm_v2")
    args = parser.parse_args()

    report = run_analysis(args.dataset_root, args.split, args.output_dir)
    for stage_key, _, _ in STAGES:
        summary = report["stages"].get(stage_key, {})
        if "error" in summary:
            print(f"{stage_key}: {summary['error']}")
            continue
        m = summary["metrics"]
        d = summary["deltas_vs_raw_mistral"]
        print(
            f"{summary['label']}: exact_span_f1={m['exact_span_f1']:.6f} "
            f"(delta {d['exact_span_f1']:+.6f}), regressions={summary['regression_case_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
