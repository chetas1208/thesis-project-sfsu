from __future__ import annotations

from typing import Any, Callable, Mapping

from skillllm_v2.context.budget import json_tokens, skill_importance
from skillllm_v2.context.profile_summarizer import summarize_profile
from skillllm_v2.context.skill_memory_builder import build_skill_memory, skill_key
from skillllm_v2.schema import normalize_text

SNIPPET_CHARS = 120

_RAG_SOURCES = ["job_postings", "scraped_descriptions"]
_RAG_REASON = "Job postings are large and change frequently, so they should be retrieved rather than cached."
_GRAPH_PATHS = ["user -> skill -> missing_skill -> job -> course"]
_GRAPH_REASON = "Relational explanation belongs in the graph layer, not in extraction."


def compact_context(
    raw_profile_text: str,
    extracted_skills: list[dict[str, Any]],
    job_target: str | None = None,
    course_catalog_summaries: list[dict[str, Any]] | None = None,
    taxonomy_aliases: dict[str, str] | None = None,
    user_id: str = "demo_user",
    budget_tokens: int | None = None,
    *,
    reserve_tokens: int = 0,
    timestamp: str | None = None,
    clock: Callable[[], str] | None = None,
    source_reliability: Mapping[str, float] | None = None,
    label_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Package an evidence-backed profile as a cacheable context payload.

    ``budget_tokens=None`` (default): lossless packaging, every skill with all its (de-duplicated) evidence, the
    explanatory ``reason`` strings and every caller-supplied alias / catalog entry. The only differences from the
    legacy output are the fixes: skills appear once in ``skill_memory`` (as references), repeated evidence is merged,
    ordering is canonical and conflicts are detected before merging.

    ``budget_tokens=N``: the returned payload serializes to at most ``N - reserve_tokens`` proxy tokens
    (:func:`skillllm_v2.context.budget.count_tokens`) whenever that is feasible; the result carries a ``budget``
    report (``met`` is False only if even the skeleton with zero skills does not fit). Skills are ranked by
    :func:`skillllm_v2.context.budget.skill_importance` and kept greedily; each kept skill first gets its cheapest
    form (one evidence snippet), then leftover budget upgrades the most important skills to more evidence. Evidence
    text is only ever truncated to a substring of the original evidence, never rewritten. Lower-priority material
    (catalog summaries and aliases not tied to a kept skill) is added only if budget remains. The terse form drops
    the ``reason`` strings, empty containers, identity alias entries and per-evidence timestamps.
    """
    summary = summarize_profile(
        raw_profile_text,
        extracted_skills=extracted_skills,
        user_id=user_id,
        target_roles=[job_target] if job_target else [],
        timestamp=timestamp,
        clock=clock,
    )
    if budget_tokens is None:
        memory = build_skill_memory(summary)
        stable_context = {
            "profile_summary": summary,
            "skill_memory": memory,
            "taxonomy_aliases": taxonomy_aliases or {},
            "course_catalog_summary": course_catalog_summaries or [],
        }
        return {
            "stable_cag_context": stable_context,
            "dynamic_rag_policy": {"rag_sources": list(_RAG_SOURCES), "reason": _RAG_REASON},
            "graph_retrieval_policy": {"paths": list(_GRAPH_PATHS), "reason": _GRAPH_REASON},
            "conflict_notes": summary.get("conflict_notes", []),
        }
    if not isinstance(budget_tokens, int) or isinstance(budget_tokens, bool) or budget_tokens <= 0:
        raise ValueError("budget_tokens must be a positive integer or None")
    return _compact_to_budget(
        summary,
        taxonomy_aliases or {},
        list(course_catalog_summaries or []),
        job_target,
        budget_tokens,
        reserve_tokens,
        source_reliability,
        label_weights,
    )


# ------------------------------------------------------------------------------------------------ budgeted path


def _snippet(evidence: str, surface: str, limit: int = SNIPPET_CHARS) -> str:
    """A substring of ``evidence`` (<= limit chars), centred on the surface form when it occurs."""
    if len(evidence) <= limit:
        return evidence
    i = evidence.lower().find(surface.lower())
    if i < 0:
        return evidence[:limit]
    lo = max(0, min(i - limit // 4, len(evidence) - limit))
    return evidence[lo : lo + limit]


def _variants(skill: dict[str, Any]) -> list[dict[str, Any]]:
    """Representations of one skill, index 0 = richest ... last = cheapest. All text is a substring of the input.

    From the second form on, the default label (HARD_SKILL) is omitted (readers default to it), and the cheapest form
    also omits the per-evidence confidence (the skill-level ``confidence`` is the max over its evidence).
    """
    base = {k: skill[k] for k in ("surface_form", "label", "canonical_skill_id", "canonical_skill_name", "confidence") if skill.get(k) is not None}
    if base.get("canonical_skill_name") == base.get("surface_form"):
        del base["canonical_skill_name"]
    lean = {k: v for k, v in base.items() if not (k == "label" and v == "HARD_SKILL")}
    evidence = [{k: e[k] for k in ("source", "text", "confidence") if k in e} for e in skill.get("evidence", [])]
    meta = skill.get("metadata", {}) or {}
    surface = str(skill.get("surface_form", ""))
    full = {**base, "evidence": evidence}
    if meta:
        full["metadata"] = meta
    two = {**lean, "evidence": evidence[:2]}
    if meta.get("mentions") is not None:
        two["metadata"] = {"mentions": meta["mentions"]}
    one = {**lean, "evidence": [dict(e, text=_snippet(str(e.get("text", "")), surface)) for e in evidence[:1]]}
    tiny = {**lean, "evidence": [{k: v for k, v in dict(e, text=_snippet(str(e.get("text", "")), surface, 60)).items() if k != "confidence"} for e in evidence[:1]]}
    out = [full]
    for cand in (two, one, tiny):
        if cand != out[-1]:
            out.append(cand)
    return out


def _alias_ties(skill: dict[str, Any]) -> set[str]:
    names = {normalize_text(str(skill.get("surface_form", ""))), normalize_text(str(skill.get("canonical_skill_name") or "")), normalize_text(str(skill.get("canonical_skill_id") or ""))}
    return {n for n in names if n}


def _compact_to_budget(
    summary: dict[str, Any],
    aliases: dict[str, str],
    catalog: list[dict[str, Any]],
    job_target: str | None,
    budget: int,
    reserve: int,
    source_reliability: Mapping[str, float] | None,
    label_weights: Mapping[str, float] | None,
) -> dict[str, Any]:
    limit = budget - max(0, reserve)
    skills = summary.get("skills", [])
    target_terms = frozenset(normalize_text(job_target).split()) if job_target else frozenset()
    ranked = sorted(
        skills,
        key=lambda s: (-skill_importance(s, target_terms=target_terms, source_reliability=source_reliability, label_weights=label_weights), str(s.get("surface_form", "")).lower(), skill_key(s)),
    )
    variants = {skill_key(s): _variants(s) for s in skills}
    by_key = {skill_key(s): s for s in skills}
    level: dict[str, int] = {}  # key -> index into variants[key] (higher index = cheaper)
    cat_kept: list[int] = []
    extra_alias: list[str] = []
    notes_all = list(summary.get("conflict_notes", []))
    notes_kept: list[str] = list(notes_all)

    def tied_aliases(keys: list[str]) -> dict[str, str]:
        ties: set[str] = set()
        for k in keys:
            ties |= _alias_ties(by_key[k])
        return {a: c for a, c in aliases.items() if normalize_text(a) in ties or normalize_text(c) in ties}

    def build(final: bool = False, used: int = 0) -> dict[str, Any]:
        kept = [skill_key(s) for s in skills if skill_key(s) in level]  # summary order = canonical order
        kept_facts = [variants[k][level[k]] for k in kept]
        shown_aliases = {**tied_aliases(kept), **{a: aliases[a] for a in extra_alias}}
        profile: dict[str, Any] = {"user_id": summary.get("user_id"), "generated_at": summary.get("generated_at"), "skills": kept_facts}
        for field in ("target_roles", "missing_skills", "course_history", "preferences"):  # conflict_notes live once, top-level
            if summary.get(field):
                profile[field] = summary[field]
        # skill_memory is a derived index (labels, aliases): in budget mode only the count is kept, the rest is
        # rebuilt on demand with build_skill_memory(profile_summary).
        stable: dict[str, Any] = {"profile_summary": profile, "skill_memory": {"skill_count": len(kept)}}
        if shown_aliases:
            stable["taxonomy_aliases"] = shown_aliases
        if cat_kept:
            stable["course_catalog_summary"] = [catalog[i] for i in sorted(cat_kept)]
        return {
            "stable_cag_context": stable,
            "conflict_notes": notes_kept,
            "budget": {"limit": budget, "used": used, "kept": len(kept), "total": len(skills), "met": True, **({"notes_omitted": len(notes_all) - len(notes_kept)} if len(notes_kept) < len(notes_all) else {})},
        }

    def exact() -> int:
        return json_tokens(build())

    # Conflict notes get first claim on the budget, but at most 30% of what the skeleton leaves, so a profile with many
    # conflicts cannot crowd out every skill; the rest are counted in budget.notes_omitted.
    notes_kept.clear()
    skeleton = exact()
    allowance = 0.3 * max(0, limit - skeleton)
    spent = 0
    for note in notes_all:
        cost = json_tokens(note) + 1
        if spent + cost > allowance:
            break
        notes_kept.append(note)
        spent += cost
    base_cost = exact()
    used = base_cost
    if base_cost <= limit:
        # additive marginal costs, exact check afterwards
        def skill_cost(key: str, lvl: int) -> int:
            return json_tokens(variants[key][lvl]) + 1

        for skill in ranked:  # pass 1: admit cheapest form, in importance order
            key = skill_key(skill)
            lvl = len(variants[key]) - 1
            tied = tied_aliases([key]) if aliases else {}
            extra = sum(json_tokens({a: c}) - 2 + 1 for a, c in tied.items())
            cost = skill_cost(key, lvl) + extra
            if used + cost <= limit:
                level[key] = lvl
                used += cost
        for skill in ranked:  # pass 2: upgrade the most important skills with the leftover budget
            key = skill_key(skill)
            while key in level and level[key] > 0:
                delta = skill_cost(key, level[key] - 1) - skill_cost(key, level[key])
                if used + delta > limit:
                    break
                level[key] -= 1
                used += delta
        for idx, item in enumerate(catalog):  # extras: catalog summaries, then aliases not tied to a kept skill
            cost = json_tokens(item) + 1
            if used + cost <= limit:
                cat_kept.append(idx)
                used += cost
        shown = tied_aliases(list(level))
        for alias in sorted(aliases):
            if alias in shown:
                continue
            cost = json_tokens({alias: aliases[alias]}) - 2 + 1
            if used + cost <= limit:
                extra_alias.append(alias)
                used += cost
        # exact repair: the additive model can be off by a few tokens (brackets/commas)
        order_low_first = list(reversed(ranked))
        while exact() > limit and (cat_kept or extra_alias or level):
            if extra_alias:
                extra_alias.pop()
            elif cat_kept:
                cat_kept.pop()
            else:
                for victim in order_low_first:
                    if skill_key(victim) in level:
                        del level[skill_key(victim)]
                        break
    payload = build()
    total = json_tokens(payload)
    payload["budget"]["used"] = total
    payload["budget"]["met"] = total <= limit
    return payload
