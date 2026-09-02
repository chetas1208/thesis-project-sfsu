from __future__ import annotations

from typing import Any

from .schema import EdgeType, NodeType


class SkillGraphBuilder:
    def __init__(self) -> None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("networkx is required for the Stage-2 graph layer") from exc
        self.nx = nx
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, node_type: NodeType, **properties: Any) -> None:
        self.graph.add_node(node_id, type=node_type.value, **properties)

    def add_edge(self, source: str, target: str, edge_type: EdgeType, **properties: Any) -> None:
        self.graph.add_edge(source, target, type=edge_type.value, **properties)

    def add_user_skill(self, user_id: str, skill_id: str, confidence: float = 1.0) -> None:
        self.add_node(user_id, NodeType.USER)
        self.add_node(skill_id, NodeType.SKILL)
        self.add_edge(user_id, skill_id, EdgeType.HAS_SKILL, confidence=confidence)

    def add_job_requirement(self, job_id: str, skill_id: str, required: bool = True) -> None:
        self.add_node(job_id, NodeType.JOB)
        self.add_node(skill_id, NodeType.SKILL)
        self.add_edge(job_id, skill_id, EdgeType.REQUIRES if required else EdgeType.PREFERS)

    def add_course_skill(self, course_id: str, skill_id: str) -> None:
        self.add_node(course_id, NodeType.COURSE)
        self.add_node(skill_id, NodeType.SKILL)
        self.add_edge(course_id, skill_id, EdgeType.TEACHES)
