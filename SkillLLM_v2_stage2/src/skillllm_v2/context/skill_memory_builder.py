from __future__ import annotations

from typing import Any

from skillllm_v2.schema import normalize_text


def skill_key(skill: dict[str, Any]) -> str:
    """Stable reference to a skill fact: its canonical id, else its normalized surface form."""
    return str(skill.get("canonical_skill_id") or normalize_text(skill.get("surface_form", "")))


def build_skill_memory(profile_summary: dict[str, Any], inline_skills: bool = False) -> dict[str, Any]:
    """Index of the profile's skills.

    By default ``skills_by_label`` holds *references* (skill keys, see :func:`skill_key`) into
    ``profile_summary["skills"]`` instead of a second copy of every skill with all its evidence: the old layout
    stored each skill twice and made the packet ~2x larger than necessary. Pass ``inline_skills=True`` to get the
    legacy layout (full skill dicts per label).
    """
    skills = profile_summary.get("skills", [])
    by_label: dict[str, list[Any]] = {"HARD_SKILL": [], "SOFT_SKILL": []}
    alias_index: dict[str, str] = {}
    for skill in skills:
        label = skill.get("label", "HARD_SKILL")
        by_label.setdefault(label, []).append(skill if inline_skills else skill_key(skill))
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
