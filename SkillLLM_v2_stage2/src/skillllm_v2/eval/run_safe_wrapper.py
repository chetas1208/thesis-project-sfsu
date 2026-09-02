"""run_safe_wrapper.py

The SkillLLM_v2_stage2 "safe wrapper" entry point.

This script does NOT retrain or re-run inference for Mistral. It loads the
cached 80-epoch Raw Mistral LoRA predictions
(``artifacts/skillllm_v2/mistral_lora/predictions.jsonl``, 974 test docs) as
the non-regression anchor and builds three wrapper extraction variants on top
of them, using only CPU-side components (GLiNER, the dictionary extractor,
``CandidateMerger``, ``BoundaryRefiner``, ``SpanVerifier``,
``ConfidenceCalibrator``, ``TaxonomyLinker``):

  - ``boundary_refiner``: anchor predictions -> BoundaryRefiner -> SpanVerifier
    -> ConfidenceCalibrator. Tests the non-destructive boundary-refiner fix
    (section 2/3) in isolation.
  - ``mistral_gliner_taxonomy``: anchor + GLiNER + dictionary candidates ->
    CandidateMerger (anchor-protected, add-only, anchor-overlap-safe) ->
    TaxonomyLinker. Tests the merger + taxonomy-linker fixes (sections 6/7)
    in isolation.
  - ``full_cag_graph``: anchor + GLiNER + dictionary candidates ->
    CandidateMerger -> BoundaryRefiner -> SpanVerifier -> ConfidenceCalibrator
    -> TaxonomyLinker. The full safe wrapper (sections 2-9 combined).

For each variant, ``skillllm_v2.eval.non_regression_gate`` compares the
variant's extraction metrics against the anchor's metrics on the SAME 974
docs:

  - ``accept``: the variant's own extraction predictions are written as
    ``predictions.jsonl`` for that variant.
  - ``fallback_to_anchor``: the anchor's (Raw Mistral) predictions are written
    as ``predictions.jsonl`` instead, and the gate's reason explains why.

This module never fabricates an improvement -- if a variant cannot beat (or
at least not meaningfully regress from) the anchor, it is reported as
``fallback_to_anchor`` and the honest extraction-F1 numbers for both the
anchor and the candidate are kept in ``metrics.json``.

Regardless of the gate decision, the ``mistral_gliner_taxonomy`` and
``full_cag_graph`` variants ALSO write ``linked_candidates.jsonl`` --
taxonomy-linked candidates built from the wrapper's enriched (merged /
refined / verified / calibrated) candidate set -- for downstream job/course
recommendation. This is the basis for the thesis claim that downstream
CAG/Graph improves product utility (taxonomy linking, recommendations) even
when its raw span-extraction F1 does not beat the Raw Mistral anchor.

Usage:
    python -m skillllm_v2.eval.run_safe_wrapper \\
        --config configs/skillllm_v2_training_80ep_2gpu.yaml [--limit 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from skillllm_v2.config import load_yaml
from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.eval.non_regression_gate import (
    DEFAULT_GUARDRAILS_PATH,
    GateResult,
    evaluate_gate,
    format_gate_markdown,
    load_guardrails,
    write_gate_report,
)
from skillllm_v2.extract import CandidateMerger, DictionaryExtractor, GLiNERExtractor
from skillllm_v2.io_utils import read_json, read_jsonl, write_json, write_jsonl
from skillllm_v2.normalize import TaxonomyLinker
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier

DEFAULT_CONFIG_PATH = "configs/skillllm_v2_training_80ep_2gpu.yaml"
ANCHOR_PREDICTIONS_PATH = "artifacts/skillllm_v2/mistral_lora/predictions.jsonl"
ANCHOR_METRICS_PATH = "artifacts/skillllm_v2/mistral_lora/metrics.json"
OUTPUT_DIR = "artifacts/skillllm_v2/safe_wrapper"
GATE_REPORT_PATH = "reports/skillllm_v2/safe_wrapper_gate.md"
SUMMARY_PATH = "reports/skillllm_v2/safe_wrapper_summary.json"

VARIANT_NAMES = ["boundary_refiner", "mistral_gliner_taxonomy", "full_cag_graph"]


def _dataset_root(config: dict[str, Any]) -> str:
    return config.get("data", {}).get("skillspan_root", "data/source/skillspan_hf")


def _load_test_docs(dataset_root: str, limit: int | None = None) -> list[dict[str, Any]]:
    docs = load_skillspan_split(Path(dataset_root), "test", limit=None)
    docs = [doc for doc in docs if doc["gold_spans"]]
    return docs[:limit] if limit else docs


def _load_anchor(predictions_path: str, metrics_path: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    by_id = {row["id"]: row["predictions"] for row in read_jsonl(predictions_path)}
    metrics = read_json(metrics_path, {})
    return by_id, metrics


def _linking_coverage(linked_docs: list[list[dict[str, Any]]]) -> float | None:
    total = sum(len(doc) for doc in linked_docs)
    if not total:
        return None
    linked = sum(1 for doc in linked_docs for item in doc if item.get("canonical_skill_name"))
    return linked / total


def run_safe_wrapper(
    config_path: str = DEFAULT_CONFIG_PATH,
    limit: int | None = None,
    use_gliner: bool = True,
    anchor_predictions_path: str = ANCHOR_PREDICTIONS_PATH,
    anchor_metrics_path: str = ANCHOR_METRICS_PATH,
    output_dir: str = OUTPUT_DIR,
    gate_report_path: str = GATE_REPORT_PATH,
    summary_path: str = SUMMARY_PATH,
    guardrails_path: str | None = DEFAULT_GUARDRAILS_PATH,
) -> dict[str, Any]:
    config = load_yaml(config_path) if config_path and Path(config_path).exists() else {}
    docs = _load_test_docs(_dataset_root(config), limit)
    anchor_by_id, anchor_metrics_cached = _load_anchor(anchor_predictions_path, anchor_metrics_path)

    docs = [doc for doc in docs if doc["id"] in anchor_by_id]
    if not docs:
        raise RuntimeError(
            f"No test docs with cached Raw Mistral anchor predictions found "
            f"(checked {anchor_predictions_path}). Run eval_raw_mistral_lora first."
        )

    gold = [doc["gold_spans"] for doc in docs]
    anchor_preds = [anchor_by_id[doc["id"]] for doc in docs]
    anchor_metrics = evaluate_extraction(gold, anchor_preds)

    gliner = GLiNERExtractor() if use_gliner else None
    dictionary = DictionaryExtractor()
    merger = CandidateMerger()
    refiner = BoundaryRefiner()
    verifier = SpanVerifier()
    calibrator = ConfidenceCalibrator()
    linker = TaxonomyLinker()

    boundary_preds: list[list[dict[str, Any]]] = []
    gt_preds: list[list[dict[str, Any]]] = []
    cag_preds: list[list[dict[str, Any]]] = []
    cag_linked: list[list[dict[str, Any]]] = []

    for doc, anchor_pred in zip(docs, anchor_preds):
        text = doc["text"]
        gliner_candidates = gliner.extract(text) if gliner is not None else []
        dictionary_candidates = dictionary.extract(text)

        # boundary_refiner: anchor only, non-destructive boundary edits +
        # source-aware verify/calibrate (both no-ops for anchor candidates).
        refined = refiner.refine(text, anchor_pred)
        verified = verifier.filter(refined)
        boundary_preds.append(calibrator.calibrate(verified))

        # mistral_gliner_taxonomy: anchor + add-only GLiNER/dictionary
        # additions -> taxonomy linking only (no boundary/verify/calibrate).
        merged = merger.merge(list(anchor_pred) + list(gliner_candidates) + list(dictionary_candidates))
        gt_preds.append(linker.link_candidates(merged))

        # full_cag_graph: same merged candidates, plus boundary refinement,
        # source-aware verification/calibration, then taxonomy linking.
        refined_cag = refiner.refine(text, merged)
        verified_cag = verifier.filter(refined_cag)
        calibrated_cag = calibrator.calibrate(verified_cag)
        linked_cag = linker.link_candidates(calibrated_cag)
        cag_preds.append(linked_cag)
        cag_linked.append(linked_cag)

    candidate_metrics = {
        "boundary_refiner": evaluate_extraction(gold, boundary_preds),
        "mistral_gliner_taxonomy": evaluate_extraction(gold, gt_preds),
        "full_cag_graph": evaluate_extraction(gold, cag_preds),
    }
    candidate_predictions = {
        "boundary_refiner": boundary_preds,
        "mistral_gliner_taxonomy": gt_preds,
        "full_cag_graph": cag_preds,
    }
    linked_candidates = {
        "mistral_gliner_taxonomy": gt_preds,
        "full_cag_graph": cag_linked,
    }
    linking_coverage = {
        "mistral_gliner_taxonomy": _linking_coverage(gt_preds),
        "full_cag_graph": _linking_coverage(cag_linked),
    }

    guardrails = load_guardrails(guardrails_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gate_results: list[GateResult] = []
    variant_summaries: dict[str, Any] = {}
    for variant in VARIANT_NAMES:
        result = evaluate_gate(variant, candidate_metrics[variant], anchor_metrics, guardrails)
        gate_results.append(result)
        selected_preds = candidate_predictions[variant] if result.decision == "accept" else anchor_preds
        selected_metrics = candidate_metrics[variant] if result.decision == "accept" else anchor_metrics

        variant_dir = out / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(variant_dir / "predictions.jsonl", (
            {"id": doc["id"], "text": doc["text"], "predictions": preds}
            for doc, preds in zip(docs, selected_preds)
        ))
        write_jsonl(variant_dir / "candidate_predictions.jsonl", (
            {"id": doc["id"], "text": doc["text"], "predictions": preds}
            for doc, preds in zip(docs, candidate_predictions[variant])
        ))
        if variant in linked_candidates:
            write_jsonl(variant_dir / "linked_candidates.jsonl", (
                {"id": doc["id"], "text": doc["text"], "linked_skills": preds}
                for doc, preds in zip(docs, linked_candidates[variant])
            ))

        variant_metrics: dict[str, Any] = {
            "anchor_metrics": anchor_metrics,
            "candidate_metrics": candidate_metrics[variant],
            "selected_metrics": selected_metrics,
            "gate": result.to_dict(),
        }
        if variant in linking_coverage:
            variant_metrics["linking_coverage"] = linking_coverage[variant]
        write_json(variant_dir / "metrics.json", variant_metrics)

        variant_summaries[variant] = {
            "decision": result.decision,
            "improved_over_anchor": result.improved_over_anchor,
            "candidate_exact_span_f1": candidate_metrics[variant]["exact_span_f1"],
            "candidate_soft_skill_f1": candidate_metrics[variant]["soft_skill_f1"],
            "selected_exact_span_f1": selected_metrics["exact_span_f1"],
            "selected_soft_skill_f1": selected_metrics["soft_skill_f1"],
            "linking_coverage": linking_coverage.get(variant),
        }

    write_json(out / "gate_results.json", [result.to_dict() for result in gate_results])
    write_gate_report(gate_report_path, gate_results)

    # Aggregate "safe wrapper" artifacts for thesis/report consumers that need
    # one canonical row. The full CAG/Graph safe variant is the final wrapper;
    # when it fails the gate, these predictions are the Raw Mistral anchor.
    full_summary = variant_summaries["full_cag_graph"]
    full_selected = (
        candidate_predictions["full_cag_graph"]
        if full_summary["decision"] == "accept"
        else anchor_preds
    )
    write_jsonl(out / "predictions.jsonl", (
        {"id": doc["id"], "text": doc["text"], "predictions": preds}
        for doc, preds in zip(docs, full_selected)
    ))
    write_jsonl(out / "linked_skills.jsonl", (
        {"id": doc["id"], "text": doc["text"], "linked_skills": preds}
        for doc, preds in zip(docs, cag_linked)
    ))
    non_regression_status = {
        "anchor_metrics": anchor_metrics,
        "final_variant": "full_cag_graph",
        "final_decision": full_summary["decision"],
        "non_regression_passed": full_summary["selected_exact_span_f1"] >= anchor_metrics["exact_span_f1"],
        "extraction_output_policy": (
            "candidate_wrapper_predictions"
            if full_summary["decision"] == "accept"
            else "fallback_to_raw_mistral_anchor"
        ),
        "gate_results": [result.to_dict() for result in gate_results],
    }
    write_json(out / "non_regression_status.json", non_regression_status)

    summary = {
        "doc_count": len(docs),
        "anchor_predictions_path": anchor_predictions_path,
        "anchor_metrics": anchor_metrics,
        "anchor_metrics_cached": anchor_metrics_cached,
        "use_gliner": use_gliner,
        "variants": variant_summaries,
        "output_dir": output_dir,
        "gate_report_path": gate_report_path,
        "non_regression_status": non_regression_status,
    }
    write_json(out / "metrics.json", summary)
    write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SkillLLM_v2_stage2 safe wrapper (non-regression-gated extraction variants)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Limit to the first N test docs (for smoke runs)")
    parser.add_argument("--anchor-predictions", default=ANCHOR_PREDICTIONS_PATH)
    parser.add_argument("--anchor-metrics", default=ANCHOR_METRICS_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--gate-report", default=GATE_REPORT_PATH)
    parser.add_argument("--summary", default=SUMMARY_PATH)
    parser.add_argument("--guardrails", default=DEFAULT_GUARDRAILS_PATH)
    parser.add_argument("--disable-gliner", action="store_true", help="Skip live GLiNER extraction and use Mistral+dictionary only. Useful for deterministic research-refresh runs.")
    args = parser.parse_args()

    summary = run_safe_wrapper(
        config_path=args.config,
        limit=args.limit,
        use_gliner=not args.disable_gliner,
        anchor_predictions_path=args.anchor_predictions,
        anchor_metrics_path=args.anchor_metrics,
        output_dir=args.output_dir,
        gate_report_path=args.gate_report,
        summary_path=args.summary,
        guardrails_path=args.guardrails,
    )
    print(f"Docs evaluated: {summary['doc_count']}")
    print(f"Anchor exact_span_f1: {summary['anchor_metrics']['exact_span_f1']:.6f}")
    for variant, info in summary["variants"].items():
        print(
            f"  {variant}: decision={info['decision']} improved={info['improved_over_anchor']} "
            f"candidate_exact_f1={info['candidate_exact_span_f1']:.6f} "
            f"selected_exact_f1={info['selected_exact_span_f1']:.6f} "
            f"linking_coverage={info['linking_coverage']}"
        )
    print(f"Gate report: {summary['gate_report_path']}")
    print(f"Summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
