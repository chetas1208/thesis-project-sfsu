from __future__ import annotations

from typing import Any


class SkillGraphRetriever:
    """Typed neighbour lookups on a skill graph.

    Every method first checks that the node exists. networkx treats a ``str`` that is not a node as an *iterable of
    characters* (``out_edges("14")`` yields the edges of nodes ``"1"`` and ``"4"``), which used to leak other users'
    skills for an unknown id; unknown nodes now return ``[]``. Results are de-duplicated, order-preserving.
    """

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def _has(self, node: str) -> bool:
        try:
            return bool(self.graph.has_node(node))
        except TypeError:  # unhashable id
            return False

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        return list(dict.fromkeys(items))

    def _out(self, node: str, edge_type: str) -> list[str]:
        if not self._has(node):
            return []
        return self._unique([target for _, target, data in self.graph.out_edges(node, data=True) if data.get("type") == edge_type])

    def _in(self, node: str, edge_type: str) -> list[str]:
        if not self._has(node):
            return []
        return self._unique([source for source, _, data in self.graph.in_edges(node, data=True) if data.get("type") == edge_type])

    def skills_for_user(self, user_id: str) -> list[str]:
        return self._out(user_id, "HAS_SKILL")

    def required_skills_for_job(self, job_id: str) -> list[str]:
        return self._out(job_id, "REQUIRES")

    def courses_for_skill(self, skill_id: str) -> list[str]:
        return self._in(skill_id, "TEACHES")
