"""Write a review summary of the LLM syllabus extraction.

Usage (from SkillLLM_v2_stage2/):
    python scripts/summarize_syllabi_extraction.py

Reads inputs/syllabi_parsed/*.json and writes inputs/syllabi_parsed/SUMMARY.md with:
  1. an overview table (one row per syllabus) to check course IDs, titles and prereqs,
  2. every answer that failed or only approximately passed verification, with its quote,
  3. the full skill list per syllabus.
"""

import json
from pathlib import Path

STAGE2_DIR = Path(__file__).resolve().parent.parent
PARSED_DIR = STAGE2_DIR / "inputs" / "syllabi_parsed"
SUMMARY_FILE = PARSED_DIR / "SUMMARY.md"


def cell(value):
    """Format a value for a Markdown table cell."""
    if value is None or value == []:
        return "—"
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    return str(value).replace("|", "/").replace("\n", " ")


def flagged_items(extracted, verification):
    """Yield (field, status, value, quote) for every answer that isn't an exact match."""
    if verification["course_id"] != "exact":
        yield "course_id", verification["course_id"], extracted.get("course_id"), (extracted.get("evidence") or {}).get("course_id")
    if extracted.get("course_title") and verification["course_title"] != "exact":
        yield "course_title", verification["course_title"], extracted.get("course_title"), None
    for code, status in zip(extracted.get("cross_listed") or [], verification["cross_listed"]):
        if status != "exact":
            yield "cross_listed", status, code, None
    for prereq, status in zip(extracted.get("prerequisites") or [], verification["prerequisites"]):
        for part in ("course", "evidence"):
            if status[part] != "exact":
                yield f"prerequisite {part}", status[part], prereq.get("course"), prereq.get("evidence")
    for requirement, status in zip(extracted.get("other_requirements") or [], verification["other_requirements"]):
        if status != "exact":
            yield "other_requirement", status, requirement.get("text"), requirement.get("evidence")
    for field in ("learning_outcomes", "topics"):
        for text, status in zip(extracted.get(field) or [], verification[field]):
            if status != "exact":
                yield field, status, None, text
    for skill, status in zip(extracted.get("skills") or [], verification["skills"]):
        if status["evidence"] != "exact":
            yield "skill evidence", status["evidence"], skill.get("skill"), skill.get("evidence")


def main():
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PARSED_DIR.glob("*.json"))]

    totals = {"exact": 0, "approximate": 0, "not_found": 0}
    for record in records:
        for key in totals:
            totals[key] += record["verification"]["summary"][key]
    checked = sum(totals.values())

    lines = [
        "# Syllabus extraction summary",
        "",
        f"- Syllabi extracted: **{len(records)}**",
        f"- Answers checked against syllabus text: **{checked}**",
        f"  - exact: {totals['exact']} ({totals['exact'] / max(1, checked):.0%})",
        f"  - approximate: {totals['approximate']} ({totals['approximate'] / max(1, checked):.0%})",
        f"  - not found: {totals['not_found']} ({totals['not_found'] / max(1, checked):.0%})",
        "",
        "How to verify: open the syllabus in `inputs/syllabi/` (or its text in `inputs/syllabi_text/`)",
        "and check each row. Section 2 lists only the answers the code could not confirm word-for-word.",
        "",
        "## 1. Overview",
        "",
        "| # | File | Course ID | Cross-listed | Title | Units | Prerequisites | Other requirements | Outcomes | Topics | Skills (explicit / inferred) | Not found |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for number, record in enumerate(records, 1):
        e, v = record["extracted"], record["verification"]
        skills = e.get("skills") or []
        explicit = sum(s.get("type") == "explicit" for s in skills)
        lines.append(
            f"| {number} | {cell(record['file'])} | {cell(e.get('course_id'))} | {cell(e.get('cross_listed'))} "
            f"| {cell(e.get('course_title'))} | {cell(e.get('units'))} "
            f"| {cell([p.get('course') for p in e.get('prerequisites') or []])} "
            f"| {cell([r.get('text') for r in e.get('other_requirements') or []])} "
            f"| {len(e.get('learning_outcomes') or [])} | {len(e.get('topics') or [])} "
            f"| {explicit} / {len(skills) - explicit} | {v['summary']['not_found']} |"
        )

    lines += ["", "## 2. Answers to check by hand", ""]
    any_flagged = False
    for record in records:
        items = list(flagged_items(record["extracted"], record["verification"]))
        if not items:
            continue
        any_flagged = True
        lines.append(f"### {record['file']}")
        lines.append("")
        for field, status, value, quote in items:
            value_part = f" **{cell(value)}**" if value else ""
            quote_part = f" — “{cell(quote)}”" if quote else ""
            lines.append(f"- `{status}` {field}:{value_part}{quote_part}")
        lines.append("")
    if not any_flagged:
        lines.append("Every answer matched the syllabus text exactly.")
        lines.append("")

    lines += ["## 3. Skills per syllabus", ""]
    for record in records:
        e = record["extracted"]
        lines.append(f"### {record['file']} — {cell(e.get('course_id'))} {cell(e.get('course_title'))}")
        lines.append("")
        lines.append("| Skill | Type | Level | Evidence | Reason (inferred only) |")
        lines.append("|---|---|---|---|---|")
        for skill in e.get("skills") or []:
            lines.append(
                f"| {cell(skill.get('skill'))} | {cell(skill.get('type'))} | {cell(skill.get('level'))} "
                f"| {cell(skill.get('evidence'))} | {cell(skill.get('reason'))} |"
            )
        lines.append("")

    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {SUMMARY_FILE} ({len(records)} syllabi)")


if __name__ == "__main__":
    main()
