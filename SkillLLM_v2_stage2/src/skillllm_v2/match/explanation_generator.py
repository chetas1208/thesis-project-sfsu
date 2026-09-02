from __future__ import annotations

from typing import Any


def generate_recommendation_explanation(
    user_profile: dict[str, Any],
    job_match: dict[str, Any] | None = None,
    skill_gaps: list[dict[str, Any]] | None = None,
    course_recommendations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    matched = job_match.get("matched_skills", []) if job_match else []
    missing = job_match.get("missing_skills", []) if job_match else []
    courses = course_recommendations or []
    evidence = []
    for skill in user_profile.get("skills", []):
        for item in skill.get("evidence", [])[:1]:
            evidence.append(f"{skill.get('surface_form')}: {item.get('text')}")
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "gap_types": [gap.get("gap_type") for gap in skill_gaps or []],
        "recommended_learning_path": [course.get("course_name") for course in courses],
        "evidence": evidence,
        "confidence": float(job_match.get("match_score", 0.0)) if job_match else 0.0,
        "limitations": [
            "Recommendation quality depends on available job/course labels.",
            "Extraction fallback uses Raw Mistral when wrapper candidates fail non-regression gates.",
        ],
    }
