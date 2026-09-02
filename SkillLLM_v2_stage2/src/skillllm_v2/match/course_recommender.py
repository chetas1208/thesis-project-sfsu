from __future__ import annotations

from typing import Any


class CourseRecommender:
    def recommend(self, gaps: list[dict[str, Any]], courses: list[dict[str, Any]], user_profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        user_profile = user_profile or {}
        gap_skills = {gap["skill"] for gap in gaps}
        user_skills = set(user_profile.get("skills", []))
        recommendations = []
        for course in courses:
            covers = sorted(gap_skills & set(course.get("teaches_skills", [])))
            if not covers:
                continue
            prerequisites = set(course.get("prerequisite_skills", []))
            prerequisite_fit = len(prerequisites & user_skills) / max(1, len(prerequisites))
            gap_coverage = len(covers) / max(1, len(gap_skills))
            level_fit = float(course.get("level_fit", 1.0))
            goal_fit = float(course.get("goal_alignment", 0.8))
            workload_penalty = float(course.get("workload_penalty", 0.0))
            score = max(0.0, min(1.0, 0.45 * gap_coverage + 0.25 * prerequisite_fit + 0.15 * level_fit + 0.15 * goal_fit - workload_penalty))
            recommendations.append(
                {
                    "course_id": course["course_id"],
                    "course_name": course.get("course_name", course["course_id"]),
                    "covers_skills": covers,
                    "gap_coverage": gap_coverage,
                    "match_score": score,
                    "why_recommended": f"Covers {len(covers)} missing skill(s): {', '.join(covers)}",
                }
            )
        return sorted(recommendations, key=lambda item: item["match_score"], reverse=True)
