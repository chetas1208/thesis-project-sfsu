"""NetworkX implementation of :class:`GraphStore`.

Built on the existing, bug-fixed pieces (see docs/BUGFIX_LOG.md): ``SkillGraphBuilder`` holds the graph (idempotent
edges keyed by type, no silent retyping), ``SkillGraphRetriever`` answers neighbour lookups (unknown ids return
``[]``) and ``explain_skill_path`` explains paths. This store is always buildable offline and is the source of truth
the Neo4j backend is checked against and falls back to.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from .graph_builder import SkillGraphBuilder
from .graph_retriever import SkillGraphRetriever
from .schema import EdgeType, NodeType
from .skill_graph_explainer import explain_skill_path
from .store import (
    LABELS,
    PROJECT,
    REL_TYPES,
    EdgeRecord,
    NodeRecord,
    StoreResult,
    chain_from_primitives,
    normalize_path_result,
    validate_project,
)


class LocalGraphStore:
    name = "local"

    def __init__(self, project: str = PROJECT, loader: Callable[["LocalGraphStore"], None] | None = None) -> None:
        self.project = validate_project(project)
        self.builder = SkillGraphBuilder()
        self._loader = loader
        self._loaded = loader is None

    # ------------------------------------------------------------------ internals
    @property
    def graph(self) -> Any:
        """The underlying ``networkx.MultiDiGraph`` (read it; write through ``upsert_*``)."""
        self._ensure_loaded()
        return self.builder.graph

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._loaded = True  # set first: the loader itself writes through upsert_*
            assert self._loader is not None
            self._loader(self)

    def ensure_loaded(self) -> None:
        self._ensure_loaded()

    def _result(self, value: Any, started: float) -> StoreResult:
        return StoreResult(value=value, backend="local", elapsed_ms=(time.perf_counter() - started) * 1000.0)

    def _in_project(self, node: str) -> bool:
        g = self.builder.graph
        return g.has_node(node) and g.nodes[node].get("project", self.project) == self.project

    # ------------------------------------------------------------------ writes
    def upsert_nodes(self, nodes: Iterable[NodeRecord], build_id: str, project: str | None = None) -> StoreResult:
        t0 = time.perf_counter()
        proj = validate_project(self.project if project is None else project)
        n = 0
        for rec in nodes:
            props = {k: v for k, v in rec.props.items() if v not in (None, "")}
            self.builder.add_node(rec.id, NodeType(rec.label), **{**props, "project": proj, "build_id": build_id})
            n += 1
        return self._result({"nodes": n}, t0)

    def upsert_edges(self, edges: Iterable[EdgeRecord], build_id: str, project: str | None = None) -> StoreResult:
        t0 = time.perf_counter()
        proj = validate_project(self.project if project is None else project)
        written = skipped = 0
        g = self.builder.graph
        for rec in edges:
            if not (g.has_node(rec.src) and g.has_node(rec.dst)):
                skipped += 1  # same as the Neo4j MATCH ... MERGE: an edge never creates a node
                continue
            props = {k: v for k, v in rec.props.items() if v not in (None, "")}
            self.builder.add_edge(rec.src, rec.dst, EdgeType(rec.type), **{**props, "project": proj, "build_id": build_id})
            written += 1
        return self._result({"edges": written, "skipped": skipped}, t0)

    def clear_namespace(self, project: str | None = None) -> StoreResult:
        t0 = time.perf_counter()
        proj = validate_project(self.project if project is None else project)
        g = self.builder.graph
        doomed = [n for n, d in g.nodes(data=True) if d.get("project") == proj]
        g.remove_nodes_from(doomed)
        return self._result({"nodes_deleted": len(doomed)}, t0)

    # ------------------------------------------------------------------ reads
    def user_skills(self, user_id: str) -> StoreResult:
        t0 = time.perf_counter()
        g = self.graph
        value = sorted(SkillGraphRetriever(g).skills_for_user(user_id)) if self._in_project(user_id) else []
        return self._result(value, t0)

    def job_skills(self, job_id: str, strength: str = "required") -> StoreResult:
        t0 = time.perf_counter()
        g = self.graph
        if not self._in_project(job_id):
            return self._result([], t0)
        if strength not in ("required", "preferred", "all"):
            raise ValueError("strength must be required|preferred|all")
        types = {"required": ("REQUIRES",), "preferred": ("PREFERS",), "all": ("REQUIRES", "PREFERS")}[strength]
        out = {t for _, t, d in g.out_edges(job_id, data=True) if d.get("type") in types}
        return self._result(sorted(out), t0)

    def skill_gap(self, user_id: str, job_id: str) -> StoreResult:
        t0 = time.perf_counter()
        have = set(self.user_skills(user_id).value)
        required = self.job_skills(job_id, "required").value
        preferred = self.job_skills(job_id, "preferred").value
        matched = [s for s in required if s in have]
        value = {
            "user_id": user_id,
            "job_id": job_id,
            "required": required,
            "matched": matched,
            "missing": [s for s in required if s not in have],
            "preferred": preferred,
            "preferred_missing": [s for s in preferred if s not in have],
            "coverage": (len(matched) / len(required)) if required else None,
        }
        return self._result(value, t0)

    def courses_teaching(self, skill_id: str) -> StoreResult:
        t0 = time.perf_counter()
        g = self.graph
        rows = []
        if self._in_project(skill_id):
            for course in SkillGraphRetriever(g).courses_for_skill(skill_id):
                edge = g.get_edge_data(course, skill_id, key="TEACHES") or {}
                rows.append({"course_id": course, "depth": edge.get("depth"), "confidence": edge.get("confidence")})
        return self._result(sorted(rows, key=lambda r: r["course_id"]), t0)

    def prerequisite_chain(self, course_id: str, max_depth: int = 4) -> StoreResult:
        t0 = time.perf_counter()
        g = self.graph
        retriever = SkillGraphRetriever(g)

        def prereqs(course_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
            out: dict[str, list[dict[str, Any]]] = {}
            for cid in course_ids:
                if not g.has_node(cid):
                    continue
                for _, skill, data in g.out_edges(cid, data=True):
                    if data.get("type") == "REQUIRES_PREREQUISITE":
                        out.setdefault(cid, []).append(
                            {"skill_id": skill, "level": data.get("level"), "strictness": data.get("strictness")})
            return out

        def teachers(skill_ids: list[str]) -> dict[str, list[str]]:
            return {s: retriever.courses_for_skill(s) for s in skill_ids}

        if not self._in_project(course_id):
            return self._result({"course_id": course_id, "levels": [], "max_depth": int(max_depth), "truncated": False,
                                 "skill_ids": []}, t0)
        return self._result(chain_from_primitives(course_id, max_depth, prereqs, teachers), t0)

    def explain_path(self, source: str, target: str, max_hops: int = 4) -> StoreResult:
        t0 = time.perf_counter()
        g = self.graph
        if not (self._in_project(source) and self._in_project(target)):
            return self._result(normalize_path_result(source, target, False, [], [], max_hops, False, reason="unknown node"), t0)
        res = explain_skill_path(g, source, target, max_hops=max_hops)
        if not res["found"]:
            value = normalize_path_result(source, target, False, [], [], max_hops, False, reason=res["reason"] or "no path")
        elif res["hops"] > max_hops:
            value = normalize_path_result(source, target, False, [], [], max_hops, True, hops_if_truncated=res["hops"],
                                          reason=f"shortest path has {res['hops']} hops, more than max_hops={max_hops}")
        else:
            value = normalize_path_result(source, target, True, res["path"], res["edges"], max_hops, False)
        return self._result(value, t0)

    def stats(self) -> StoreResult:
        t0 = time.perf_counter()
        g = self.graph
        nodes = {label: 0 for label in LABELS}
        for _, d in g.nodes(data=True):
            if d.get("project", self.project) == self.project and d.get("type") in nodes:
                nodes[d["type"]] += 1
        edges = {t: 0 for t in REL_TYPES}
        for _, _, d in g.edges(data=True):
            if d.get("project", self.project) == self.project and d.get("type") in edges:
                edges[d["type"]] += 1
        return self._result({"project": self.project, "nodes": nodes, "edges": edges,
                             "total_nodes": sum(nodes.values()), "total_edges": sum(edges.values())}, t0)

    def health(self) -> StoreResult:
        return StoreResult(value={"ok": True, "backend": "local"}, backend="local")

    def close(self) -> None:
        return None

    def __enter__(self) -> "LocalGraphStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
