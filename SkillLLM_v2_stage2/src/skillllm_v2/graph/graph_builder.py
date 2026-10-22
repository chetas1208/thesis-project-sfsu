from __future__ import annotations

import logging
from typing import Any, Callable

from .schema import EdgeType, NodeType

logger = logging.getLogger(__name__)

# Property keys that carry "no information" when empty: re-ingesting an edge never overwrites a real value with one.
_EMPTY = (None, "")


class SkillGraphBuilder:
    """Small typed graph of users, jobs, courses and skills.

    Behaviour guarantees (see docs/BUGFIX_LOG.md):

    * **Idempotent ingestion.** There is at most one edge per ``(source, target, edge type)``: the networkx edge key
      is the edge type, so adding the same relation again updates the existing edge instead of creating a parallel
      one. Re-ingesting identical data leaves the graph unchanged. Last write wins for each property, except that an
      empty value (``None`` / ``""``) never overwrites a non-empty one.
    * **No silent retyping.** ``add_node`` on an existing node with a *different* node type keeps the existing type
      and records a warning (``strict=True``: raises ``ValueError``). Calling ``add_node`` again with the same type
      merges the new properties in. An untyped placeholder is upgraded to a real type.
    * **Unknown-node edges are flagged.** ``add_edge`` with an endpoint that was never added still creates an untyped
      placeholder node (legacy behaviour, relied on by older callers) but marks it ``placeholder=True``, logs a
      warning and records it in ``self.warnings``; ``strict=True`` raises ``ValueError`` instead and adds nothing.
    * **Provenance.** Every edge created by the ``add_*`` helpers carries the keys ``CourseGraphBuilder`` uses
      (HAS_SKILL: source, confidence, evidence_text, last_seen; REQUIRES/PREFERS: importance, requirement_strength,
      source_sentence; TEACHES: confidence, depth, learning_outcome_evidence, source), populated from the arguments
      when given; unknown values are ``""`` / ``None``, never invented. ``last_seen`` is only written when a time is
      known (explicit argument or injected clock): an edge without it is honestly "time unknown", not stamped now.
    * **Deterministic time.** ``last_seen`` is the explicit argument, else ``clock()`` if a clock was injected, else
      absent. The builder never reads the wall clock on its own.
    """

    def __init__(self, strict: bool = False, clock: Callable[[], str] | None = None) -> None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("networkx is required for the Stage-2 graph layer") from exc
        self.nx = nx
        self.graph = nx.MultiDiGraph()
        self.strict = strict
        self.clock = clock
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ internals
    def _flag(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(message)

    def _now(self, explicit: str | None) -> str | None:
        if explicit is not None:
            return explicit
        return self.clock() if self.clock is not None else None

    # ------------------------------------------------------------------ nodes / edges
    def add_node(self, node_id: str, node_type: NodeType, **properties: Any) -> None:
        if self.graph.has_node(node_id):
            current = self.graph.nodes[node_id].get("type")
            if current is not None and current != node_type.value:
                message = f"node {node_id!r} already has type {current!r}; refusing to retype it to {node_type.value!r}"
                if self.strict:
                    raise ValueError(message)
                self._flag(message)
                node_type_value = current
            else:
                node_type_value = node_type.value
            if current is None:
                self.graph.nodes[node_id].pop("placeholder", None)
            self.graph.add_node(node_id, type=node_type_value, **properties)
            return
        self.graph.add_node(node_id, type=node_type.value, **properties)

    def add_edge(self, source: str, target: str, edge_type: EdgeType, /, **properties: Any) -> None:
        missing = [n for n in (source, target) if not self.graph.has_node(n)]
        if missing:
            message = f"edge {source!r} -[{edge_type.value}]-> {target!r} references unknown node(s) {missing!r}"
            if self.strict:
                raise ValueError(message)
            self._flag(message)
            for node in missing:
                self.graph.add_node(node, placeholder=True)
        key = edge_type.value
        if self.graph.has_edge(source, target, key):
            existing = self.graph[source][target][key]
            properties = {k: v for k, v in properties.items() if v not in _EMPTY or existing.get(k) in _EMPTY}
        self.graph.add_edge(source, target, key=key, type=edge_type.value, **properties)

    def add_user_skill(
        self,
        user_id: str,
        skill_id: str,
        confidence: float = 1.0,
        evidence_text: str = "",
        source: str = "unknown",
        last_seen: str | None = None,
        **properties: Any,
    ) -> None:
        self.add_node(user_id, NodeType.USER)
        self.add_node(skill_id, NodeType.SKILL)
        seen = self._now(last_seen)
        if seen is not None:  # unknown time is omitted rather than faked (and the builder never reads the wall clock)
            properties["last_seen"] = seen
        self.add_edge(
            user_id,
            skill_id,
            EdgeType.HAS_SKILL,
            confidence=float(confidence),
            evidence_text=evidence_text,
            source=source,
            **properties,
        )

    def add_job_requirement(
        self,
        job_id: str,
        skill_id: str,
        required: bool = True,
        importance: float = 1.0,
        source_sentence: str = "",
        **properties: Any,
    ) -> None:
        self.add_node(job_id, NodeType.JOB)
        self.add_node(skill_id, NodeType.SKILL)
        self.add_edge(
            job_id,
            skill_id,
            EdgeType.REQUIRES if required else EdgeType.PREFERS,
            importance=float(importance),
            requirement_strength="required" if required else "preferred",
            source_sentence=source_sentence,
            **properties,
        )

    def add_course_skill(
        self,
        course_id: str,
        skill_id: str,
        confidence: float = 1.0,
        learning_outcome_evidence: str = "",
        depth: str | None = None,
        source: str = "unknown",
        **properties: Any,
    ) -> None:
        self.add_node(course_id, NodeType.COURSE)
        self.add_node(skill_id, NodeType.SKILL)
        self.add_edge(
            course_id,
            skill_id,
            EdgeType.TEACHES,
            confidence=float(confidence),
            learning_outcome_evidence=learning_outcome_evidence,
            depth=depth,
            source=source,
            **properties,
        )
