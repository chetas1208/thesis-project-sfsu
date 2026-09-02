from __future__ import annotations

from typing import Any


class SkillGraphRetriever:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def skills_for_user(self, user_id: str) -> list[str]:
        return [target for _, target, data in self.graph.out_edges(user_id, data=True) if data.get("type") == "HAS_SKILL"]

    def required_skills_for_job(self, job_id: str) -> list[str]:
        return [target for _, target, data in self.graph.out_edges(job_id, data=True) if data.get("type") == "REQUIRES"]

    def courses_for_skill(self, skill_id: str) -> list[str]:
        return [source for source, _, data in self.graph.in_edges(skill_id, data=True) if data.get("type") == "TEACHES"]
