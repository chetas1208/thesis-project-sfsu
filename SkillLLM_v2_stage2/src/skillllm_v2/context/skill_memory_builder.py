from __future__ import annotations

from typing import Any

from skillllm_v2.schema import normalize_text


def build_skill_memory(profile_summary: dict[str, Any]) -> dict[str, Any]:
    skills = profile_summary.get("skills", [])
    by_label: dict[str, list[dict[str, Any]]] = {"HARD_SKILL": [], "SOFT_SKILL": []}
    alias_index: dict[str, str] = {}
    for skill in skills:
        label = skill.get("label", "HARD_SKILL")
        by_label.setdefault(label, []).append(skill)
        canonical = skill.get("canonical_skill_id") or normalize_text(skill.get("surface_form", ""))
        alias_index[normalize_text(skill.get("surface_form", ""))] = canonical
        if skill.get("canonical_skill_name"):
            alias_index[normalize_text(skill["canonical_skill_name"])] = canonical
    return {
        "user_id": profile_summary.get("user_id"),
        "generated_at": profile_summary.get("generated_at"),
        "skill_count": len(skills),
        "skills_by_label": by_label,
        "alias_index": alias_index,
    }
