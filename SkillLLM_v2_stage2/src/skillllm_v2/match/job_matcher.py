from __future__ import annotations

from typing import Any

from skillllm_v2.match.skill_keys import SkillKeyResolver


class JobMatcher:
    """Required/preferred skill coverage score.

    Skills are compared through :class:`skillllm_v2.match.skill_keys.SkillKeyResolver`: case-, whitespace- and
    punctuation-insensitive (symbol-bearing names such as ``c++`` / ``.net`` / ``node.js`` stay distinct) and
    alias-aware, so ``python == Python`` and ``SQL == structured query language``. Pass ``alias_resolver`` /
    ``aliases`` to use a specific alias table; by default the taxonomy alias table is used when it can be loaded
    (``use_default_aliases=False`` restricts matching to literal keys).

    The user profile's ``skills`` may be strings or dicts (``surface_form`` / ``name`` / ``canonical_skill_name`` /
    ``canonical_skill_id``); an optional ``skill_entries`` list of such dicts is merged in. Output schema is unchanged.
    """

    def __init__(
        self,
        alias_resolver: Any | None = None,
        *,
        aliases: dict[str, str] | None = None,
        use_default_aliases: bool = True,
        key_resolver: SkillKeyResolver | None = None,
    ) -> None:
        self.keys = key_resolver or SkillKeyResolver(alias_resolver, aliases=aliases, use_default_aliases=use_default_aliases)

    def score(self, user_profile: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
        held = self.keys.index(user_profile.get("skills"), user_profile.get("skill_entries"))
        required = self.keys.unique(job.get("required_skills"))
        preferred = self.keys.unique(job.get("preferred_skills"))
        matched_required = [name for name, keys in required if keys & held]
        matched_preferred = [name for name, keys in preferred if keys & held]
        missing = sorted(name for name, keys in required if not keys & held)
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
