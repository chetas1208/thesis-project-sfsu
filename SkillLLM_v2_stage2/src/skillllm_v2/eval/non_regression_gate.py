"""non_regression_gate.py

Decides, per wrapper extraction variant (e.g. boundary-refined Mistral,
Mistral+GLiNER+Taxonomy, Full CAG/Graph), whether that variant's extraction
output is "accept"-ed for downstream use or whether the pipeline must fall
back to the unmodified Raw Mistral LoRA anchor predictions for *extraction*.

Per the SkillLLM_v2_stage2 wrapper-fix spec, the 80-epoch Raw Mistral LoRA
extractor (test split, 974 docs) is the non-regression anchor:
    exact_span_f1 = 0.675034, token_f1 = 0.797798,
    soft_skill_f1 = 0.609792, hard_skill_f1 = 0.731138
(see reports/skillllm_v2/wrapper_error_analysis*.md and
configs/skillllm_v2_wrapper_guardrails.yaml).

A variant is "accept"-ed only if BOTH guardrails hold relative to the anchor:
  - exact_span_f1 >= anchor exact_span_f1 - exact_span_f1.max_drop
    ("no worse than 0.665"); clearing exact_span_f1.target (0.675034) is
    additionally reported as "improved_over_anchor" (informational only).
  - soft_skill_f1 >= anchor soft_skill_f1 - soft_skill_f1.max_drop
    (soft skill F1 must not drop by more than 0.03).

If either guardrail fails, the decision is "fallback_to_anchor": the
pipeline must use the Raw Mistral anchor's extraction predictions for that
variant, while taxonomy/job/course recommendation outputs are still produced
from the wrapper's enriched candidates separately. This module never
fabricates an "improvement" -- a variant that cannot beat (or at least not
meaningfully regress from) the anchor is reported as such.

Usage (single variant, metrics already computed via
``skillllm_v2.eval.extraction_eval.evaluate_extraction`` and saved as JSON):

    python -m skillllm_v2.eval.non_regression_gate \\
        --variant-name boundary_refiner \\
        --candidate-metrics artifacts/.../boundary_refiner_metrics.json \\
        --output artifacts/.../boundary_refiner_gate.json

Programmatic use (preferred from ``run_safe_wrapper.py`` /
``ablation_runner.py``, which compute both anchor and candidate metrics on
the same split/sample so the comparison is apples-to-apples):

    result = evaluate_gate("boundary_refiner", candidate_metrics, anchor_metrics)
    selected, result = select_extraction("boundary_refiner", anchor_preds, candidate_preds, candidate_metrics, anchor_metrics)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillllm_v2.config import load_yaml
from skillllm_v2.io_utils import read_json, write_json, write_text

DEFAULT_GUARDRAILS_PATH = "configs/skillllm_v2_wrapper_guardrails.yaml"

DEFAULT_GUARDRAILS: dict[str, Any] = {
    "anchor_metrics": {
        "exact_span_f1": 0.675034,
        "token_f1": 0.797798,
        "soft_skill_f1": 0.609792,
        "hard_skill_f1": 0.731138,
    },
    "exact_span_f1": {"target": 0.675034, "max_drop": 0.010034},
    "soft_skill_f1": {"max_drop": 0.03},
}


def load_guardrails(path: str | None = DEFAULT_GUARDRAILS_PATH) -> dict[str, Any]:
    if path and Path(path).exists():
        return load_yaml(path)
    return DEFAULT_GUARDRAILS


@dataclass
class GateCheck:
    metric: str
    anchor: float
    candidate: float
    floor: float
    passed: bool

    @property
    def delta(self) -> float:
        return self.candidate - self.anchor

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "anchor": self.anchor,
            "candidate": self.candidate,
            "floor": self.floor,
            "delta": self.delta,
            "passed": self.passed,
        }


@dataclass
class GateResult:
    variant: str
    decision: str
    improved_over_anchor: bool
    reason: str
    checks: list[GateCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "decision": self.decision,
            "improved_over_anchor": self.improved_over_anchor,
            "reason": self.reason,
            "checks": [check.to_dict() for check in self.checks],
        }


def evaluate_gate(
    variant_name: str,
    candidate_metrics: dict[str, float],
    anchor_metrics: dict[str, float] | None = None,
    guardrails: dict[str, Any] | None = None,
) -> GateResult:
    """Compare ``candidate_metrics`` against ``anchor_metrics`` for ``variant_name``.

    ``anchor_metrics`` should be the Raw Mistral anchor's metrics computed on
    the SAME split/sample as ``candidate_metrics`` -- callers comparing
    ablation/dev subsets should pass that subset's own Raw Mistral row, not
    the fixed test-split numbers in ``guardrails["anchor_metrics"]`` (those
    are only used as the default for whole-test-split CLI invocations).
    """
    guardrails = guardrails or DEFAULT_GUARDRAILS
    anchor_metrics = anchor_metrics or guardrails.get("anchor_metrics", DEFAULT_GUARDRAILS["anchor_metrics"])

    exact_cfg = guardrails.get("exact_span_f1", DEFAULT_GUARDRAILS["exact_span_f1"])
    soft_cfg = guardrails.get("soft_skill_f1", DEFAULT_GUARDRAILS["soft_skill_f1"])

    exact_floor = anchor_metrics["exact_span_f1"] - exact_cfg["max_drop"]
    soft_floor = anchor_metrics["soft_skill_f1"] - soft_cfg["max_drop"]

    exact_check = GateCheck(
        metric="exact_span_f1",
        anchor=anchor_metrics["exact_span_f1"],
        candidate=candidate_metrics["exact_span_f1"],
        floor=exact_floor,
        passed=candidate_metrics["exact_span_f1"] >= exact_floor,
    )
    soft_check = GateCheck(
        metric="soft_skill_f1",
        anchor=anchor_metrics["soft_skill_f1"],
        candidate=candidate_metrics["soft_skill_f1"],
        floor=soft_floor,
        passed=candidate_metrics["soft_skill_f1"] >= soft_floor,
    )
    checks = [exact_check, soft_check]
    decision = "accept" if all(check.passed for check in checks) else "fallback_to_anchor"
    target = exact_cfg.get("target", anchor_metrics["exact_span_f1"])
    improved = candidate_metrics["exact_span_f1"] >= target

    if decision == "accept" and improved:
        reason = (
            f"{variant_name}: exact_span_f1 {exact_check.candidate:.6f} meets the "
            f"improvement target ({target:.6f}) and soft_skill_f1 "
            f"{soft_check.candidate:.6f} stays within the regression guardrail "
            f"(>= {soft_floor:.6f}) -- accepted."
        )
    elif decision == "accept":
        reason = (
            f"{variant_name}: exact_span_f1 {exact_check.candidate:.6f} is below the "
            f"improvement target ({target:.6f}) but stays within the "
            f"non-regression floor (>= {exact_floor:.6f}); soft_skill_f1 "
            f"{soft_check.candidate:.6f} stays within its guardrail "
            f"(>= {soft_floor:.6f}) -- accepted as non-regressive."
        )
    else:
        failed = [check.metric for check in checks if not check.passed]
        reason = (
            f"{variant_name}: failed non-regression guardrail(s) on "
            f"{', '.join(failed)} -- falling back to Raw Mistral anchor "
            "extraction for this variant. Downstream taxonomy/job/course "
            "outputs are still produced from the wrapper's enriched "
            "candidates."
        )

    return GateResult(variant=variant_name, decision=decision, improved_over_anchor=improved, reason=reason, checks=checks)


def select_extraction(
    variant_name: str,
    anchor_predictions: list[list[dict[str, Any]]],
    candidate_predictions: list[list[dict[str, Any]]],
    candidate_metrics: dict[str, float],
    anchor_metrics: dict[str, float] | None = None,
    guardrails: dict[str, Any] | None = None,
) -> tuple[list[list[dict[str, Any]]], GateResult]:
    """Return ``(selected_predictions, gate_result)`` for ``variant_name``.

    ``selected_predictions`` is ``candidate_predictions`` if the gate
    decision is ``"accept"``, otherwise ``anchor_predictions`` (the Raw
    Mistral fallback).
    """
    result = evaluate_gate(variant_name, candidate_metrics, anchor_metrics, guardrails)
    selected = candidate_predictions if result.decision == "accept" else anchor_predictions
    return selected, result


def format_gate_markdown(results: list[GateResult]) -> str:
    lines = [
        "# SkillLLM_v2_stage2 Wrapper Non-Regression Gate",
        "",
        "Each variant's extraction metrics are compared against the Raw "
        "Mistral LoRA anchor on the same split/sample. `accept` means the "
        "variant's extraction output is used downstream; `fallback_to_anchor` "
        "means the wrapper falls back to the unmodified Raw Mistral "
        "predictions for extraction (taxonomy/job/course outputs are still "
        "produced from the wrapper's enriched candidates).",
        "",
        "| Variant | Decision | Improved | Exact F1 (anchor -> candidate) | Exact floor | Soft F1 (anchor -> candidate) | Soft floor |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        exact = next(check for check in result.checks if check.metric == "exact_span_f1")
        soft = next(check for check in result.checks if check.metric == "soft_skill_f1")
        lines.append(
            f"| {result.variant} | {result.decision} | {result.improved_over_anchor} | "
            f"{exact.anchor:.6f} -> {exact.candidate:.6f} | {exact.floor:.6f} | "
            f"{soft.anchor:.6f} -> {soft.candidate:.6f} | {soft.floor:.6f} |"
        )
    lines.append("")
    lines.append("## Reasons")
    lines.append("")
    for result in results:
        lines.append(f"- **{result.variant}**: {result.reason}")
    return "\n".join(lines) + "\n"


def write_gate_report(path: str, results: list[GateResult]) -> None:
    write_text(path, format_gate_markdown(results))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the SkillLLM_v2_stage2 wrapper non-regression gate for one variant")
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--candidate-metrics", required=True, help="Path to a JSON file with extraction metrics (output of evaluate_extraction)")
    parser.add_argument("--anchor-metrics", default=None, help="Path to a JSON file with the anchor's extraction metrics; defaults to the guardrails config's anchor_metrics (test-split Raw Mistral)")
    parser.add_argument("--guardrails", default=DEFAULT_GUARDRAILS_PATH)
    parser.add_argument("--output", default=None, help="Path to write the gate decision as JSON")
    args = parser.parse_args()

    guardrails = load_guardrails(args.guardrails)
    candidate_metrics = read_json(args.candidate_metrics)
    anchor_metrics = read_json(args.anchor_metrics) if args.anchor_metrics else None

    result = evaluate_gate(args.variant_name, candidate_metrics, anchor_metrics, guardrails)
    print(f"Decision: {result.decision}")
    print(f"Improved over anchor: {result.improved_over_anchor}")
    print(f"Reason: {result.reason}")
    for check in result.checks:
        print(f"  {check.metric}: anchor={check.anchor:.6f} candidate={check.candidate:.6f} floor={check.floor:.6f} passed={check.passed}")

    if args.output:
        write_json(args.output, result.to_dict())
        print(f"Gate decision written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
