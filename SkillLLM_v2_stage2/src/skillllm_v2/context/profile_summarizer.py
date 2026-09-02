from __future__ import annotations

from typing import Any

from skillllm_v2.context.profile_schema import Evidence, ProfileSkillFact, ProfileSummary, utc_now_iso
from skillllm_v2.schema import normalize_text


def summarize_profile(
    profile_text: str,
    extracted_skills: list[dict[str, Any]] | None = None,
    user_id: str = "demo_user",
    target_roles: list[str] | None = None,
    missing_skills: list[str] | None = None,
    course_history: list[str] | None = None,
    preferences: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build an evidence-backed profile summary from explicit extracted skills.

    The summarizer intentionally does not infer hidden skills or sensitive
    attributes. Every skill fact comes from a provided extraction span.
    """
    generated_at = timestamp or utc_now_iso()
    facts: dict[str, ProfileSkillFact] = {}
    for skill in extracted_skills or []:
        surface = str(skill.get("surface_form") or skill.get("text") or "").strip()
        if not surface:
            continue
        key = str(skill.get("canonical_skill_id") or normalize_text(surface))
        evidence_text = str(skill.get("evidence") or _snippet(profile_text, skill) or surface)
        fact = facts.get(key)
        evidence = Evidence(
            source=str(skill.get("source") or "+".join(skill.get("sources", [])) or "unknown"),
            text=evidence_text,
            confidence=float(skill.get("confidence", 0.0)),
            timestamp=generated_at,
        )
        if fact is None:
            fact = ProfileSkillFact(
                surface_form=surface,
                label=str(skill.get("label") or "HARD_SKILL"),
                canonical_skill_id=skill.get("canonical_skill_id"),
                canonical_skill_name=skill.get("canonical_skill_name"),
                confidence=float(skill.get("confidence", 0.0)),
                evidence=[evidence],
                metadata={"source_offsets": [[skill.get("start"), skill.get("end")]]},
            )
            facts[key] = fact
        else:
            fact.confidence = max(fact.confidence, float(skill.get("confidence", 0.0)))
            fact.evidence.append(evidence)
            fact.metadata.setdefault("source_offsets", []).append([skill.get("start"), skill.get("end")])

    summary = ProfileSummary(
        user_id=user_id,
        generated_at=generated_at,
        skills=sorted(facts.values(), key=lambda item: item.surface_form.lower()),
        target_roles=target_roles or [],
        missing_skills=missing_skills or [],
        course_history=course_history or [],
        preferences=preferences or {},
        conflict_notes=_find_conflicts(facts.values()),
    )
    return summary.to_dict()


def _snippet(text: str, skill: dict[str, Any], window: int = 80) -> str:
    start, end = skill.get("start"), skill.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].strip()


def _find_conflicts(facts: list[ProfileSkillFact] | Any) -> list[str]:
    by_name: dict[str, set[str]] = {}
    for fact in facts:
        by_name.setdefault(normalize_text(fact.surface_form), set()).add(fact.label)
    return [
        f"Conflicting labels for {name}: {', '.join(sorted(labels))}"
        for name, labels in sorted(by_name.items())
        if len(labels) > 1
    ]
