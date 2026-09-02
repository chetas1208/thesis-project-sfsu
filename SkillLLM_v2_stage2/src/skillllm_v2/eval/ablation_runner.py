from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.eval.non_regression_gate import evaluate_gate, load_guardrails
from skillllm_v2.pipeline import SkillLLMv2Pipeline
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier

GATED_VARIANT_NAME = "G_safe_wrapper"


def _linking_coverage(predicted: list[list[dict[str, Any]]]) -> float | None:
    total = sum(len(doc) for doc in predicted)
    if not total:
        return None
    linked = sum(1 for doc in predicted for item in doc if item.get("canonical_skill_name"))
    return linked / total


def run_ablation(dataset_root: str, output_dir: str, limit: int = 20) -> list[dict[str, Any]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_docs = load_skillspan_split(Path(dataset_root), "dev", limit=None)
    docs = [doc for doc in all_docs if doc["gold_spans"]][:limit]
    gold = [doc["gold_spans"] for doc in docs]
    pipeline = SkillLLMv2Pipeline()
    rows = []
    variants = [
        ("A_mistral_only", {"gliner": False, "dictionary": False, "refine": False, "verify": False, "normalize": False}),
        ("B_mistral_boundary", {"gliner": False, "dictionary": False, "refine": True, "verify": False, "normalize": False}),
        ("C_mistral_gliner", {"gliner": True, "dictionary": False, "refine": True, "verify": False, "normalize": False}),
        ("D_mistral_gliner_dictionary", {"gliner": True, "dictionary": True, "refine": True, "verify": False, "normalize": False}),
        ("E_full_wrapper", {"gliner": True, "dictionary": True, "refine": True, "verify": True, "normalize": False}),
        ("F_full_wrapper_taxonomy", {"gliner": True, "dictionary": True, "refine": True, "verify": True, "normalize": True}),
    ]
    predicted_by_variant: dict[str, list[list[dict[str, Any]]]] = {}
    for name, flags in variants:
        predicted = []
        for doc in docs:
            candidates = list(pipeline.mistral.extract(doc["text"]))
            if flags["gliner"]:
                candidates.extend(pipeline.gliner.extract(doc["text"]))
            if flags["dictionary"]:
                candidates.extend(pipeline.dictionary.extract(doc["text"]))
            candidates = pipeline.merger.merge(candidates)
            if flags["refine"]:
                candidates = BoundaryRefiner().refine(doc["text"], candidates)
            if flags["verify"]:
                candidates = ConfidenceCalibrator().calibrate(SpanVerifier().filter(candidates))
            if flags["normalize"]:
                candidates = pipeline.linker.link_candidates(candidates)
            predicted.append(candidates)
        metrics = evaluate_extraction(gold, predicted)
        metrics["variant"] = name
        metrics["linking_coverage"] = _linking_coverage(predicted) if flags["normalize"] else None
        metrics["gate_decision"] = None
        rows.append(metrics)
        predicted_by_variant[name] = predicted

    # Gated "safe wrapper" row: gate the full wrapper+taxonomy variant (F)
    # against the mistral-only variant (A) as this ablation's own anchor --
    # mirrors skillllm_v2.eval.run_safe_wrapper's non-regression gate, but on
    # this dev smoke subset (both variants share the same Mistral extractor
    # instance, so the comparison is apples-to-apples for THIS subset even
    # when that extractor is the smoke fallback).
    guardrails = load_guardrails()
    anchor_metrics = rows[0]
    candidate_metrics = rows[-1]
    gate_result = evaluate_gate(GATED_VARIANT_NAME, candidate_metrics, anchor_metrics, guardrails)
    if gate_result.decision == "accept":
        selected_metrics = dict(candidate_metrics)
    else:
        selected_metrics = dict(anchor_metrics)
        # Taxonomy/job/course outputs are still produced from the wrapper's
        # enriched (F) candidates even when extraction falls back to the anchor.
        selected_metrics["linking_coverage"] = candidate_metrics["linking_coverage"]
    selected_metrics["variant"] = GATED_VARIANT_NAME
    selected_metrics["gate_decision"] = gate_result.decision
    rows.append(selected_metrics)
    (out / "ablation_gate_result.json").write_text(json.dumps(gate_result.to_dict(), indent=2), encoding="utf-8")

    csv_path = out / "extraction_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path = out / "extraction_ablation.md"
    md_path.write_text(_markdown_table(rows), encoding="utf-8")
    (out / "ablation_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = ["variant", "exact_span_f1", "partial_span_f1", "token_f1", "hard_skill_f1", "soft_skill_f1", "linking_coverage", "gate_decision"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                cells.append(str(round(value, 4)))
            elif value is None:
                cells.append("NA")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
