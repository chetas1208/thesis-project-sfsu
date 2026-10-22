from __future__ import annotations

from typing import Any

from skillllm_v2.schema import normalize_text

LOW_CONFIDENCE = 0.5  # matched skills whose best evidence is below this are called out in the limitations


def _names(skill: dict[str, Any]) -> set[str]:
    """Normalized names a profile skill can be referred to by (surface form, canonical name, canonical id)."""
    raw = (skill.get("surface_form"), skill.get("canonical_skill_name"), skill.get("canonical_skill_id"))
    return {normalize_text(str(x)) for x in raw if x}


def _ranked_evidence(skill: dict[str, Any]) -> list[dict[str, Any]]:
    """Evidence items with text, best first (confidence desc, then text for determinism)."""
    items = [e for e in skill.get("evidence", []) if str(e.get("text", "")).strip()]
    return sorted(items, key=lambda e: (-float(e.get("confidence", 0.0)), str(e.get("text"))))


def generate_recommendation_explanation(
    user_profile: dict[str, Any],
    job_match: dict[str, Any] | None = None,
    skill_gaps: list[dict[str, Any]] | None = None,
    course_recommendations: list[dict[str, Any]] | None = None,
    evidence_per_skill: int = 1,
) -> dict[str, Any]:
    """Explain a job match from the user's profile, the match result, the gaps and the recommended courses.

    ``evidence`` lists, as ``"<surface form>: <evidence text>"`` lines, the profile evidence for the skills the
    explanation is *about*: matched skills, missing skills and gap skills (at most ``evidence_per_skill`` per skill,
    best first). Unrelated profile skills are no longer dumped into it. Skills the explanation talks about but the
    profile holds no evidence for are reported separately instead of being left silent or invented:

    * ``missing_skill_evidence``: one entry per missing skill, with its profile evidence lines (usually none) and the
      honest note ``"no evidence in profile"`` when there is none;
    * ``matched_without_evidence``: matched skills with no evidence text in the profile.

    ``limitations`` are statements computed from this input (which missing skills no recommended course covers,
    which matched skills rest on weak evidence, whether a match result was supplied at all), each checkable against
    the other fields; the old fixed boilerplate strings are gone. Existing keys keep their meaning.
    """
    matched = list(job_match.get("matched_skills", [])) if job_match else []
    missing = list(job_match.get("missing_skills", [])) if job_match else []
    gaps = skill_gaps or []
    courses = course_recommendations or []
    profile_skills = [s for s in user_profile.get("skills", []) if isinstance(s, dict)]
    gap_skills = [str(g.get("skill")) for g in gaps if g.get("skill")]

    index: dict[str, list[dict[str, Any]]] = {}
    for skill in profile_skills:
        for name in _names(skill):
            index.setdefault(name, []).append(skill)

    def lines_for(name: str) -> list[str]:
        out: list[str] = []
        seen: set[int] = set()
        for skill in index.get(normalize_text(name), []):
            if id(skill) in seen:
                continue
            seen.add(id(skill))
            for item in _ranked_evidence(skill)[: max(0, evidence_per_skill)]:
                out.append(f"{skill.get('surface_form')}: {item.get('text')}")
        return out

    evidence: list[str] = []
    matched_without_evidence: list[str] = []
    best_conf: dict[str, float] = {}
    for name in matched:
        lines = lines_for(name)
        if not lines:
            matched_without_evidence.append(name)
        for line in lines:
            if line not in evidence:
                evidence.append(line)
        confs = [float(e.get("confidence", 0.0)) for s in index.get(normalize_text(name), []) for e in _ranked_evidence(s)[:1]]
        if confs:
            best_conf[name] = max(confs)
    missing_skill_evidence: list[dict[str, Any]] = []
    for name in missing:
        lines = lines_for(name)
        for line in lines:
            if line not in evidence:
                evidence.append(line)
        missing_skill_evidence.append({"skill": name, "evidence": lines, "note": "" if lines else "no evidence in profile"})
    for name in gap_skills:  # gap skills that are neither matched nor missing (weak / outdated / uncertain ...)
        if name in matched or name in missing:
            continue
        for line in lines_for(name):
            if line not in evidence:
                evidence.append(line)

    covered: set[str] = set()
    course_fields_seen = False
    for course in courses:
        for key in ("teaches_skills", "covered_missing_skills", "covers_skills"):
            if key in course:
                course_fields_seen = True
                covered |= {normalize_text(str(s)) for s in course.get(key) or []}
    limitations: list[str] = []
    if job_match is None:
        limitations.append("No job match result was supplied, so matched and missing skills and the confidence are empty or zero.")
    else:
        no_ev = [m["skill"] for m in missing_skill_evidence if m["note"]]
        if no_ev:
            limitations.append(f"Missing skills are judged by name against the profile's skill list; no evidence in profile for: {', '.join(no_ev)}.")
        if matched_without_evidence:
            limitations.append(f"Matched skills with no evidence text in the profile: {', '.join(matched_without_evidence)}.")
        weak = sorted(name for name, conf in best_conf.items() if conf < LOW_CONFIDENCE)
        if weak:
            limitations.append(f"Matched skills resting on low-confidence evidence (< {LOW_CONFIDENCE}): {', '.join(weak)}.")
        if missing and course_fields_seen:
            uncovered = [m for m in missing if normalize_text(m) not in covered]
            if uncovered:
                limitations.append(f"No recommended course covers: {', '.join(uncovered)}.")
        elif missing and not courses:
            limitations.append(f"No course recommendations were supplied for {len(missing)} missing skill(s).")
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "gap_types": [gap.get("gap_type") for gap in gaps],
        "recommended_learning_path": [course.get("course_name") for course in courses],
        "evidence": evidence,
        "missing_skill_evidence": missing_skill_evidence,
        "matched_without_evidence": matched_without_evidence,
        "confidence": float(job_match.get("match_score", 0.0)) if job_match else 0.0,
        "limitations": limitations,
    }
