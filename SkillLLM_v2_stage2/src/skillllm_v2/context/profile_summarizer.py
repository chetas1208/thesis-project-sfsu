from __future__ import annotations

from typing import Any, Callable

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
    clock: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Build an evidence-backed profile summary from explicit extracted skills.

    The summarizer intentionally does not infer hidden skills or sensitive
    attributes. Every skill fact comes from a provided extraction span.

    Determinism
    -----------
    * The output does not depend on the order of ``extracted_skills``: extractions that map to one skill fact are
      sorted canonically before the fact is built (representative surface form / label / canonical name come from
      the highest-confidence extraction, ties broken by (source, surface, offsets, evidence)); evidence entries and
      ``metadata.source_offsets`` are sorted; facts are sorted by (surface form, key).
    * ``generated_at`` / ``Evidence.timestamp`` are the only wall-clock fields. Pass ``timestamp`` (a fixed string)
      or ``clock`` (a zero-argument callable returning an ISO string) to control them. ``content_hash`` in
      ``build_cached_context_packet`` ignores both.

    De-duplication
    --------------
    Repeated (skill, evidence text) pairs are merged into one :class:`Evidence` whose confidence is the maximum and
    whose ``source`` joins the distinct sources with ``+``. ``metadata.mentions`` keeps the number of extractions
    that supported the fact, so the repetition signal is not lost.

    Conflicts
    ---------
    HARD/SOFT label conflicts are detected on the *raw* extractions (before merging by canonical id), grouped by
    normalized surface form and by canonical id, so a conflict between two extractions that merge into one fact is
    still reported. Contradictory date ranges in the profile text are NOT detected (no robust date parser is
    bundled; free-text date ranges such as "2018 to 2021" are ambiguous), a documented limitation.
    """
    generated_at = timestamp or (clock() if clock is not None else utc_now_iso())
    groups: dict[str, list[dict[str, Any]]] = {}
    for skill in extracted_skills or []:
        surface = str(skill.get("surface_form") or skill.get("text") or "").strip()
        if not surface:
            continue
        key = str(skill.get("canonical_skill_id") or normalize_text(surface))
        groups.setdefault(key, []).append(skill)

    facts: dict[str, ProfileSkillFact] = {}
    for key, items in groups.items():
        facts[key] = _build_fact(profile_text, items, generated_at)

    summary = ProfileSummary(
        user_id=user_id,
        generated_at=generated_at,
        skills=sorted(facts.values(), key=lambda item: (item.surface_form.lower(), item.key())),
        target_roles=target_roles or [],
        missing_skills=missing_skills or [],
        course_history=course_history or [],
        preferences=preferences or {},
        conflict_notes=_find_conflicts_raw(extracted_skills or []),
    )
    return summary.to_dict()


def _surface(skill: dict[str, Any]) -> str:
    return str(skill.get("surface_form") or skill.get("text") or "").strip()


def _confidence(skill: dict[str, Any]) -> float:
    return float(skill.get("confidence", 0.0))


def _source(skill: dict[str, Any]) -> str:
    return str(skill.get("source") or "+".join(sorted(str(s) for s in skill.get("sources", []))) or "unknown")


def _rep_sort_key(skill: dict[str, Any], profile_text: str) -> tuple:
    start, end = skill.get("start"), skill.get("end")
    return (
        -_confidence(skill),
        _source(skill),
        _surface(skill),
        start if isinstance(start, int) else -1,
        end if isinstance(end, int) else -1,
        str(skill.get("evidence") or ""),
        str(skill.get("label") or ""),
        str(skill.get("canonical_skill_name") or ""),
    )


def _build_fact(profile_text: str, items: list[dict[str, Any]], generated_at: str) -> ProfileSkillFact:
    ordered = sorted(items, key=lambda s: _rep_sort_key(s, profile_text))
    rep = ordered[0]
    by_text: dict[str, dict[str, Any]] = {}
    offsets: set[tuple[Any, Any]] = set()
    for skill in ordered:
        text = str(skill.get("evidence") or _snippet(profile_text, skill) or _surface(skill))
        slot = by_text.setdefault(text, {"confidence": 0.0, "sources": set()})
        slot["confidence"] = max(slot["confidence"], _confidence(skill))
        slot["sources"].add(_source(skill))
        offsets.add((skill.get("start"), skill.get("end")))
    evidence = [
        Evidence(source="+".join(sorted(slot["sources"])), text=text, confidence=slot["confidence"], timestamp=generated_at)
        for text, slot in sorted(by_text.items(), key=lambda kv: (-kv[1]["confidence"], kv[0]))
    ]
    return ProfileSkillFact(
        surface_form=_surface(rep),
        label=str(rep.get("label") or "HARD_SKILL"),
        canonical_skill_id=rep.get("canonical_skill_id"),
        canonical_skill_name=rep.get("canonical_skill_name"),
        confidence=max(_confidence(s) for s in ordered),
        evidence=evidence,
        metadata={
            "source_offsets": [list(o) for o in sorted(offsets, key=lambda o: (o[0] if isinstance(o[0], int) else -1, o[1] if isinstance(o[1], int) else -1))],
            "mentions": len(ordered),
        },
    )


def _snippet(text: str, skill: dict[str, Any], window: int = 80) -> str:
    start, end = skill.get("start"), skill.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return ""
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].strip()


def _find_conflicts_raw(extractions: list[dict[str, Any]]) -> list[str]:
    """HARD/SOFT label conflicts on raw extractions, before any merging by canonical id.

    Two groupings are checked: by normalized surface form (same words, different label) and by canonical skill id
    (different surface forms of one skill labelled differently). Only explicit labels count, a missing label is not
    evidence of a conflict.
    """
    by_surface: dict[str, set[str]] = {}
    by_id: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for skill in extractions:
        label = skill.get("label")
        surface = _surface(skill)
        if not label or not surface:
            continue
        by_surface.setdefault(normalize_text(surface), set()).add(str(label))
        cid = skill.get("canonical_skill_id")
        if cid:
            name = str(skill.get("canonical_skill_name") or cid)
            slot = by_id.setdefault(str(cid), (set(), set(), set()))
            slot[0].add(name)
            slot[1].add(str(label))
            slot[2].add(normalize_text(surface))
    notes = {f"Conflicting labels for {name}: {', '.join(sorted(labels))}" for name, labels in by_surface.items() if len(labels) > 1}
    for cid, (names, labels, surfaces) in by_id.items():
        if len(labels) > 1 and len(surfaces) > 1:  # a single surface form is already reported by the surface grouping
            name = min(names)  # order-independent choice when extractions disagree on the canonical name
            notes.add(f"Conflicting labels for {normalize_text(name) or cid}: {', '.join(sorted(labels))}")
    return sorted(notes)


def _find_conflicts(facts: list[ProfileSkillFact] | Any) -> list[str]:
    """Legacy fact-level conflict check (kept for backward compatibility; merged facts carry one label each)."""
    by_name: dict[str, set[str]] = {}
    for fact in facts:
        by_name.setdefault(normalize_text(fact.surface_form), set()).add(fact.label)
    return [
        f"Conflicting labels for {name}: {', '.join(sorted(labels))}"
        for name, labels in sorted(by_name.items())
        if len(labels) > 1
    ]
