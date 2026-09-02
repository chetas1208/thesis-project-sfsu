from __future__ import annotations

from typing import Any


class GapDetector:
    def detect(self, user_profile: dict[str, Any], job: dict[str, Any]) -> list[dict[str, Any]]:
        user_skills = set(user_profile.get("skills", []))
        weak_skills = set(user_profile.get("weak_skills", []))
        outdated = set(user_profile.get("outdated_skills", []))
        uncertain = set(user_profile.get("uncertain_skills", []))
        gaps: list[dict[str, Any]] = []
        for skill in job.get("required_skills", []):
            if skill not in user_skills:
                gaps.append({"skill": skill, "gap_type": "missing_critical"})
            elif skill in weak_skills:
                gaps.append({"skill": skill, "gap_type": "weak_skill"})
            elif skill in outdated:
                gaps.append({"skill": skill, "gap_type": "outdated_skill"})
            elif skill in uncertain:
                gaps.append({"skill": skill, "gap_type": "uncertain_skill"})
        for skill in job.get("preferred_skills", []):
            if skill not in user_skills:
                gaps.append({"skill": skill, "gap_type": "missing_preferred"})
        transferable = user_profile.get("transferable_skills", {})
        for gap in gaps:
            if gap["skill"] in transferable:
                gap["gap_type"] = "covered_by_transferable_skill"
                gap["transferable_from"] = transferable[gap["skill"]]
        return gaps
