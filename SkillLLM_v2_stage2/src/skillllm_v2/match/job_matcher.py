from __future__ import annotations

from typing import Any


class JobMatcher:
    def score(self, user_profile: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        user_skills = set(user_profile.get("skills", []))
        required = set(job.get("required_skills", []))
        preferred = set(job.get("preferred_skills", []))
        matched_required = sorted(required & user_skills)
        matched_preferred = sorted(preferred & user_skills)
        missing = sorted(required - user_skills)
        required_coverage = len(matched_required) / max(1, len(required))
        preferred_coverage = len(matched_preferred) / max(1, len(preferred))
        semantic_role_fit = float(job.get("semantic_role_fit", 0.0))
        experience_level_fit = float(job.get("experience_level_fit", 0.0))
        user_preference_fit = float(job.get("user_preference_fit", 0.0))
        critical_penalty = len(missing) / max(1, len(required))
        score = 0.45 * required_coverage + 0.2 * preferred_coverage + 0.15 * semantic_role_fit + 0.1 * experience_level_fit + 0.1 * user_preference_fit - 0.25 * critical_penalty
        score = max(0.0, min(1.0, score))
        return {
            "job_id": job["job_id"],
            "match_score": score,
            "matched_skills": sorted(set(matched_required + matched_preferred)),
            "missing_skills": missing,
            "critical_missing_skills": missing,
            "evidence": [f"required coverage={required_coverage:.2f}", f"preferred coverage={preferred_coverage:.2f}"],
            "recommendation_reason": "Strong fit" if score >= 0.7 else "Fit depends on closing missing skill gaps",
        }
