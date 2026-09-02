"""linking_error_analysis.py

Measures taxonomy-linking *coverage* (the fraction of predicted skill
candidates that receive a non-null ``canonical_skill_id``/``canonical_skill_name``)
before and after switching :class:`skillllm_v2.normalize.taxonomy_linker.TaxonomyLinker`
from its old 4-entry hardcoded fallback table to the alias-table-backed,
grouped-entry table built by ``skillllm_v2.normalize.skill_alias_builder``
(``data/skillllm_v2/taxonomy/skill_aliases.jsonl``, 696 canonical entries).

Linking is attach-only: ``TaxonomyLinker.link_candidates()`` never edits
``surface_form``/``start``/``end``/``label``/``confidence``/``source``, so this
analysis only re-attaches ``canonical_skill_id``/``canonical_skill_name``/
``metadata.taxonomy_link`` in memory -- it never mutates extraction-stage
predictions or extraction F1.

Outputs (under ``reports/skillllm_v2/``):
  - ``linking_error_analysis.json`` -- per-stage coverage stats, link-method
    and per-label breakdowns.
  - ``linking_error_analysis.md``   -- human-readable summary.
  - ``linking_error_cases.jsonl``   -- correctly-extracted (TP) spans that the
    new alias-table linker still fails to link, for future alias-table
    extension.

Usage:
    python -m skillllm_v2.eval.linking_error_analysis
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from skillllm_v2.eval.extraction_eval import _key
from skillllm_v2.io_utils import read_json, read_jsonl, write_json, write_jsonl, write_text
from skillllm_v2.normalize import TaxonomyLinker
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

DATASET_ROOT = "data/source/skillspan_hf"

STAGES = [
    (
        "mistral_gliner_taxonomy",
        "Mistral + GLiNER + Taxonomy",
        "artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl",
        "reports/skillllm_v2/mistral_gliner_taxonomy_metrics.json",
    ),
    (
        "full_cag_graph",
        "Full CAG/Graph",
        "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl",
        "reports/skillllm_v2/full_cag_graph_metrics.json",
    ),
]

MAX_ERROR_CASES_PER_STAGE = 200
TOP_UNLINKED_SURFACES = 20


def _strip_link(item: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``item`` with any prior taxonomy link removed."""
    stripped = dict(item)
    stripped["canonical_skill_id"] = None
    stripped["canonical_skill_name"] = None
    metadata = dict(item.get("metadata") or {})
    metadata.pop("taxonomy_link", None)
    stripped["metadata"] = metadata
    return stripped


def analyze_stage(
    stage_id: str,
    label: str,
    predictions_path: str,
    before_metrics_path: str,
    linker: TaxonomyLinker,
    dataset_root: str = DATASET_ROOT,
) -> dict[str, Any] | None:
    rows = read_jsonl(predictions_path)
    if not rows:
        return None

    docs = load_skillspan_split(Path(dataset_root), "test", limit=None)
    gold_by_id = {doc["id"]: doc["gold_spans"] for doc in docs}

    before_metrics = read_json(before_metrics_path, default={})
    before_coverage = before_metrics.get("linking_accuracy")

    total = 0
    after_linked = 0
    tp_total = 0
    tp_linked = 0
    method_counts: Counter[str] = Counter()
    label_total: Counter[str] = Counter()
    label_linked: Counter[str] = Counter()
    unlinked_tp_surfaces: Counter[str] = Counter()
    error_cases: list[dict[str, Any]] = []

    for row in rows:
        gold_keys = {_key(span) for span in gold_by_id.get(row["id"], [])}
        relinked = linker.link_candidates([_strip_link(item) for item in row["predictions"]])
        for item in relinked:
            total += 1
            item_label = item.get("label", "?")
            label_total[item_label] += 1
            link = (item.get("metadata") or {}).get("taxonomy_link")
            linked = bool(item.get("canonical_skill_name"))
            method_counts[link["link_method"] if linked and link else "none"] += 1
            if linked:
                after_linked += 1
                label_linked[item_label] += 1

            if _key(item) in gold_keys:
                tp_total += 1
                if linked:
                    tp_linked += 1
                else:
                    unlinked_tp_surfaces[item["surface_form"].lower()] += 1
                    if len(error_cases) < MAX_ERROR_CASES_PER_STAGE:
                        error_cases.append(
                            {
                                "stage": stage_id,
                                "doc_id": row["id"],
                                "surface_form": item["surface_form"],
                                "label": item_label,
                                "source": item.get("source"),
                                "start": item.get("start"),
                                "end": item.get("end"),
                            }
                        )

    return {
        "stage_id": stage_id,
        "label": label,
        "total_predictions": total,
        "before_linking_coverage": before_coverage,
        "after_linking_coverage": after_linked / total if total else 0.0,
        "tp_count": tp_total,
        "tp_linking_coverage_after": tp_linked / tp_total if tp_total else 0.0,
        "link_method_counts": dict(method_counts),
        "label_totals": dict(label_total),
        "label_linked_after": dict(label_linked),
        "top_unlinked_tp_surfaces": unlinked_tp_surfaces.most_common(TOP_UNLINKED_SURFACES),
        "error_cases": error_cases,
    }


def run(dataset_root: str = DATASET_ROOT) -> dict[str, Any]:
    linker = TaxonomyLinker()
    stage_results = []
    all_error_cases: list[dict[str, Any]] = []
    for stage_id, label, predictions_path, before_metrics_path in STAGES:
        result = analyze_stage(stage_id, label, predictions_path, before_metrics_path, linker, dataset_root)
        if result is None:
            continue
        all_error_cases.extend(result["error_cases"])
        stage_results.append(result)

    payload = {
        "alias_table_entries": len(linker.entries),
        "stages": stage_results,
    }
    write_json("reports/skillllm_v2/linking_error_analysis.json", payload)
    write_jsonl("reports/skillllm_v2/linking_error_cases.jsonl", all_error_cases)
    write_text("reports/skillllm_v2/linking_error_analysis.md", _render_markdown(payload))
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Taxonomy Linking Error Analysis",
        "",
        f"Alias-table-backed `TaxonomyLinker` loaded {payload['alias_table_entries']} canonical "
        "entries from `data/skillllm_v2/taxonomy/skill_aliases.jsonl` (grouped by "
        "`canonical_skill_id`, taxonomy priority ESCO > ONET > LIGHTCAST > CUSTOM > "
        "SKILLSPAN_GOLD), replacing the previous 4-entry hardcoded fallback.",
        "",
        "Linking is attach-only: re-linking never changes `surface_form`/`start`/`end`/"
        "`label`/`source`, so it cannot affect extraction Exact/Token/Soft/Hard Skill F1 -- "
        "only the separate `linking_accuracy` (coverage) metric.",
        "",
    ]
    for stage in payload["stages"]:
        before = stage["before_linking_coverage"]
        after = stage["after_linking_coverage"]
        before_str = f"{before:.6f}" if isinstance(before, (int, float)) else "n/a"
        lines += [
            f"## {stage['label']} (`{stage['stage_id']}`)",
            "",
            f"- Total predicted spans: {stage['total_predictions']}",
            f"- Linking coverage (old 4-entry fallback): {before_str}",
            f"- Linking coverage (new {payload['alias_table_entries']}-entry alias table): {after:.6f}",
            f"- Correctly-extracted (TP) spans: {stage['tp_count']}",
            f"- TP linking coverage (new table): {stage['tp_linking_coverage_after']:.6f}",
            "",
            "### Link method breakdown (new table)",
            "",
            "| Method | Count |",
            "|---|---|",
        ]
        for method, count in sorted(stage["link_method_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {method} | {count} |")

        lines += [
            "",
            "### Per-label coverage (new table)",
            "",
            "| Label | Total | Linked | Coverage |",
            "|---|---|---|---|",
        ]
        for item_label, total in sorted(stage["label_totals"].items()):
            linked = stage["label_linked_after"].get(item_label, 0)
            coverage = linked / total if total else 0.0
            lines.append(f"| {item_label} | {total} | {linked} | {coverage:.6f} |")

        if stage["top_unlinked_tp_surfaces"]:
            lines += [
                "",
                "### Most common unlinked TP surface forms (candidates for alias-table extension)",
                "",
                "| Surface form | Count |",
                "|---|---|",
            ]
            for surface, count in stage["top_unlinked_tp_surfaces"]:
                lines.append(f"| {surface} | {count} |")
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- `linking_error_cases.jsonl` lists every correctly-extracted (TP) span across all "
        "stages above that the new alias-table linker still fails to attach a "
        "`canonical_skill_id` to (capped at "
        f"{MAX_ERROR_CASES_PER_STAGE} per stage). These are the highest-value additions for "
        "`MANUAL_ALIAS_PAIRS` in `skillllm_v2.normalize.skill_alias_builder` or for an "
        "external ESCO/O*NET/Lightcast export.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze taxonomy-linking coverage before/after the alias-table-backed TaxonomyLinker")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    args = parser.parse_args()

    payload = run(args.dataset_root)
    for stage in payload["stages"]:
        print(
            f"{stage['label']}: coverage {stage['before_linking_coverage']} -> "
            f"{stage['after_linking_coverage']:.6f} (TP coverage {stage['tp_linking_coverage_after']:.6f})"
        )
    print("Wrote reports/skillllm_v2/linking_error_analysis.json")
    print("Wrote reports/skillllm_v2/linking_error_analysis.md")
    print("Wrote reports/skillllm_v2/linking_error_cases.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
