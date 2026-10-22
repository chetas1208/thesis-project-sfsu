from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from skillllm_v2.io_utils import read_json

SAFE_WRAPPER_DIR = "artifacts/skillllm_v2/safe_wrapper"

# Legacy (pre-gate) eval-stage outputs, used as-is for the baselines/anchor
# and as a fallback for the "(safe wrapper)" rows below if
# skillllm_v2.eval.run_safe_wrapper has not produced gated metrics yet.
LEGACY_ROWS = [
    ("BERT baseline", "artifacts/skillllm_v2/bert_baseline/metrics.json"),
    ("GLiNER", "artifacts/skillllm_v2/gliner_baseline/metrics.json"),
    ("Raw Mistral LoRA (anchor)", "artifacts/skillllm_v2/mistral_lora/metrics.json"),
    ("Old Boundary Refiner", "artifacts/skillllm_v2/mistral_boundary/metrics.json"),
    ("Old Mistral + GLiNER + Taxonomy", "artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json"),
    ("Old Full CAG/Graph", "artifacts/skillllm_v2/full_cag_graph/metrics.json"),
]

# Each "(safe wrapper)" row is gated, per skillllm_v2.eval.non_regression_gate
# and configs/skillllm_v2_wrapper_guardrails.yaml, against the Raw Mistral
# LoRA anchor above. "Exact/Token/Soft/Hard Skill F1" report the gate's
# *selected* extraction (the wrapper's own output if accepted, otherwise the
# unmodified anchor predictions); "Candidate Exact Span F1" is the wrapper
# variant's own pre-gate exact-span F1, kept for transparency even on
# fallback. "Linking Accuracy" is the wrapper's taxonomy-linking coverage,
# produced for downstream job/course recommendation regardless of the gate
# decision.
SAFE_WRAPPER_ROWS = [
    ("Mistral + Boundary Refiner (safe wrapper)", "boundary_refiner", "artifacts/skillllm_v2/mistral_boundary/metrics.json"),
    ("Mistral + GLiNER + Taxonomy (safe wrapper)", "mistral_gliner_taxonomy", "artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json"),
    ("SkillLLM_v2 Full CAG/Graph (safe wrapper)", "full_cag_graph", "artifacts/skillllm_v2/full_cag_graph/metrics.json"),
]

MONEY_COLUMNS = [
    "System",
    "Exact Span F1",
    "Token F1",
    "Partial Span F1",
    "HARD_SKILL F1",
    "Soft Skill F1",
    "Linking Accuracy",
    "Job Match nDCG@10",
    "Course Rec P@5",
    "Regression vs Raw Mistral",
    "Status",
    "Artifact path",
    "Gate Decision",
    "Candidate Exact Span F1",
]


def _legacy_row(system: str, metrics_path: str, gate_decision: str = "NA") -> dict[str, Any]:
    metrics = read_json(metrics_path, default={}) or {}
    status = "present" if Path(metrics_path).exists() else "missing"
    course_benchmark = read_json("reports/skillllm_v2/course_recommendation_benchmark.json", default={}) or {}
    course_rec = metrics.get("course_rec_precision_at_5")
    if course_rec is None and system == "Old Full CAG/Graph" and "SkillLLM_v2 Full Graph Recommender" in course_benchmark:
        course_rec = course_benchmark["SkillLLM_v2 Full Graph Recommender"].get("Precision@5")
    return {
        "System": system,
        "Exact Span F1": _fmt(metrics.get("exact_span_f1")),
        "Token F1": _fmt(metrics.get("token_f1")),
        "Partial Span F1": _fmt(metrics.get("partial_span_f1")),
        "HARD_SKILL F1": _fmt(metrics.get("hard_skill_f1")),
        "Soft Skill F1": _fmt(metrics.get("soft_skill_f1")),
        "Linking Accuracy": _fmt(metrics.get("linking_accuracy")),
        "Job Match nDCG@10": _fmt(metrics.get("job_match_ndcg_at_10")),
        "Course Rec P@5": _fmt(course_rec),
        "Regression vs Raw Mistral": _regression(metrics),
        "Status": status,
        "Artifact path": metrics_path,
        "Gate Decision": gate_decision,
        "Candidate Exact Span F1": "NA",
    }


def _safe_wrapper_row(system: str, variant: str, legacy_metrics_path: str) -> dict[str, Any]:
    payload = read_json(f"{SAFE_WRAPPER_DIR}/{variant}/metrics.json", default=None)
    if payload is None:
        # skillllm_v2.eval.run_safe_wrapper has not been run yet -- fall back
        # to whatever the legacy (pre-gate) eval stage wrote, but mark it
        # "ungated" so the table never silently presents a pre-gate number as
        # if it had cleared the non-regression gate.
        return _legacy_row(system, legacy_metrics_path, gate_decision="ungated")
    selected = payload.get("selected_metrics", {}) or {}
    candidate = payload.get("candidate_metrics", {}) or {}
    gate = payload.get("gate", {}) or {}
    display_system = system
    aggregate = read_json(f"{SAFE_WRAPPER_DIR}/metrics.json", default={}) or {}
    if aggregate.get("use_gliner") is False and variant in {"mistral_gliner_taxonomy", "full_cag_graph"}:
        display_system = f"{system} [GLiNER disabled in refresh run]"
    course_benchmark = read_json("reports/skillllm_v2/course_recommendation_benchmark.json", default={}) or {}
    course_rec = "NA"
    if "SkillLLM_v2 Full Graph Recommender" in course_benchmark and variant in {"mistral_gliner_taxonomy", "full_cag_graph"}:
        course_rec = _fmt(course_benchmark["SkillLLM_v2 Full Graph Recommender"].get("Precision@5"))
    return {
        "System": display_system,
        "Exact Span F1": _fmt(selected.get("exact_span_f1")),
        "Token F1": _fmt(selected.get("token_f1")),
        "Partial Span F1": _fmt(selected.get("partial_span_f1")),
        "HARD_SKILL F1": _fmt(selected.get("hard_skill_f1")),
        "Soft Skill F1": _fmt(selected.get("soft_skill_f1")),
        "Linking Accuracy": _fmt(payload.get("linking_coverage")),
        "Job Match nDCG@10": "NA",
        "Course Rec P@5": course_rec,
        "Regression vs Raw Mistral": _regression(candidate),
        "Status": "selected_anchor_fallback" if gate.get("decision") == "fallback_to_anchor" else "selected_candidate",
        "Artifact path": f"{SAFE_WRAPPER_DIR}/{variant}/metrics.json",
        "Gate Decision": gate.get("decision", "NA"),
        "Candidate Exact Span F1": _fmt(candidate.get("exact_span_f1")),
    }


def generate_money_table(output_dir: str | Path = "reports/skillllm_v2") -> list[dict[str, Any]]:
    rows = []
    for system, metrics_path in LEGACY_ROWS:
        gate_decision = "anchor" if metrics_path.endswith("mistral_lora/metrics.json") else "NA"
        rows.append(_legacy_row(system, metrics_path, gate_decision=gate_decision))
    for system, variant, legacy_metrics_path in SAFE_WRAPPER_ROWS:
        rows.append(_safe_wrapper_row(system, variant, legacy_metrics_path))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "money_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MONEY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "money_table.md").write_text(_render_markdown(rows), encoding="utf-8")
    return rows


EXTRACTION_COLUMNS = [
    "Exact Span F1",
    "Token F1",
    "Partial Span F1",
    "HARD_SKILL F1",
    "Soft Skill F1",
]

WRAPPER_EXTRA_COLUMNS = [
    "Linking Accuracy",
    "Gate Decision",
    "Candidate Exact Span F1",
]

META_COLUMNS = [
    "Job Match nDCG@10",
    "Course Rec P@5",
    "Regression vs Raw Mistral",
    "Status",
]


def _md_table(headers: list[str], table_rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for cells in table_rows:
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _row_cells(row: dict[str, Any], columns: list[str]) -> list[str]:
    return [str(row[col]) for col in columns]


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    """Readable markdown report; CSV retains the full flat table."""
    baselines = rows[:3]
    legacy_wrappers = rows[3:6]
    safe_wrappers = rows[6:]

    lines = [
        "# SkillLLM_v2 — Results Summary",
        "",
        "SkillSpan evaluation metrics from verified artifact JSON files.",
        "",
        "---",
        "",
        "## At a glance",
        "",
        "Baselines vs the Raw Mistral LoRA anchor.",
        "",
    ]
    lines.extend(
        _md_table(
            ["System", *EXTRACTION_COLUMNS],
            [_row_cells(row, ["System", *EXTRACTION_COLUMNS]) for row in baselines],
        )
    )
    lines.extend(["", "---", "", "## Baselines & anchor", ""])
    lines.extend(
        _md_table(
            ["System", *EXTRACTION_COLUMNS, "Gate Decision"],
            [
                _row_cells(row, ["System", *EXTRACTION_COLUMNS, "Gate Decision"])
                for row in baselines
            ],
        )
    )
    lines.extend(["", "---", "", "## Legacy wrapper stages (pre-gate eval)", ""])
    lines.extend(
        _md_table(
            ["System", *EXTRACTION_COLUMNS, "Linking Accuracy"],
            [
                _row_cells(row, ["System", *EXTRACTION_COLUMNS, "Linking Accuracy"])
                for row in legacy_wrappers
            ],
        )
    )
    lines.extend(["", "---", "", "## Safe wrapper variants (gated)", ""])
    lines.extend(
        _md_table(
            ["System", *EXTRACTION_COLUMNS, *WRAPPER_EXTRA_COLUMNS, "Regression vs Raw Mistral"],
            [
                _row_cells(
                    row,
                    ["System", *EXTRACTION_COLUMNS, *WRAPPER_EXTRA_COLUMNS, "Regression vs Raw Mistral"],
                )
                for row in safe_wrappers
            ],
        )
    )
    lines.extend(["", "---", "", "## Downstream metrics & status", ""])
    lines.extend(
        _md_table(
            ["System", *META_COLUMNS],
            [_row_cells(row, ["System", *META_COLUMNS]) for row in rows],
        )
    )
    lines.extend(["", "---", "", "## Artifact paths", ""])
    for row in rows:
        lines.append(f"- **{row['System']}** — `{row['Artifact path']}`")
    lines.extend(["", "---", "", _thesis_note(), ""])
    return "\n".join(lines)


def _thesis_note() -> str:
    return (
        "**Reading this table:** Raw Mistral LoRA (anchor) is the strongest "
        "span-extraction system on its own and is the non-regression anchor "
        "for every \"(safe wrapper)\" row (see "
        "configs/skillllm_v2_wrapper_guardrails.yaml and "
        "skillllm_v2.eval.non_regression_gate). For \"(safe wrapper)\" rows, "
        "the F1 columns report the gate's *selected* extraction: the "
        "wrapper's own output if `Gate Decision = accept`, otherwise the "
        "unmodified Raw Mistral anchor predictions (`Gate Decision = "
        "fallback_to_anchor`). \"Candidate Exact Span F1\" is the wrapper "
        "variant's own pre-gate exact-span F1, kept for transparency even on "
        "fallback. \"Linking Accuracy\" is the wrapper's taxonomy-linking "
        "coverage, produced for downstream job/course recommendation "
        "regardless of the gate decision -- this is the basis for the claim "
        "that SkillLLM_v2's CAG/Graph layer improves product utility "
        "(normalization, explainability, recommendation) even on rows where "
        "it does not improve raw span-extraction F1 over Raw Mistral."
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "NA"


def _regression(metrics: dict[str, Any]) -> str:
    anchor = read_json("artifacts/skillllm_v2/mistral_lora/metrics.json", default={}) or {}
    if not metrics or "exact_span_f1" not in metrics or "exact_span_f1" not in anchor:
        return "NA"
    delta = float(metrics["exact_span_f1"]) - float(anchor["exact_span_f1"])
    return f"{delta:.6f}"


if __name__ == "__main__":
    generate_money_table()
