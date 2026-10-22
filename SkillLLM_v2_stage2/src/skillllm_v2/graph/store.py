"""Backend-neutral graph store contract (Neo4j or local NetworkX).

Every operation returns a :class:`StoreResult` that says which backend answered (``backend``), whether a failure of
the preferred backend made the answer come from the other one (``fell_back``) and a redacted ``error`` string.

Id convention (global, so labels never collide): ``user:<id>``, ``job:<id>``, ``skill:<id>``, ``course:<id>``.
Namespace: every node and relationship carries ``project`` (default ``skillllm_v2``) and ``build_id``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

PROJECT = "skillllm_v2"
TEST_PROJECT = "skillllm_v2_test"
_PROJECT_RE = re.compile(r"^skillllm_v2(_test(_[a-z0-9]{1,16})?)?$")

LABELS = ("User", "Job", "Skill", "Course")
REL_TYPES = ("HAS_SKILL", "REQUIRES", "PREFERS", "TEACHES", "REQUIRES_PREREQUISITE", "RELATED_TO")
# (type) -> (source label, target label)
REL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "HAS_SKILL": ("User", "Skill"),
    "REQUIRES": ("Job", "Skill"),
    "PREFERS": ("Job", "Skill"),
    "TEACHES": ("Course", "Skill"),
    "REQUIRES_PREREQUISITE": ("Course", "Skill"),
    "RELATED_TO": ("Skill", "Skill"),
}
ID_PREFIX = {"User": "user:", "Job": "job:", "Skill": "skill:", "Course": "course:"}

#: refuse to write when the planned graph (plus what is already stored) exceeds this many nodes
MAX_NODES = 50_000


class GraphStoreError(RuntimeError):
    """Base error. Messages are always redacted."""


class SizeGuardError(GraphStoreError):
    """The planned graph exceeds the node budget."""


class GraphConnectionError(GraphStoreError):
    """Unreachable / TLS / DNS / service-unavailable."""


class GraphAuthError(GraphStoreError):
    """Authentication or authorization failure."""


class GraphTimeoutError(GraphStoreError):
    """A connection or transaction timed out."""


class GraphQueryError(GraphStoreError):
    """The server rejected a query or a constraint was violated."""


def node_id(label: str, raw: str) -> str:
    """``node_id('Skill', 'python') == 'skill:python'`` (idempotent if already prefixed)."""
    prefix = ID_PREFIX[label]
    raw = str(raw)
    return raw if raw.startswith(prefix) else prefix + raw


def label_of(identifier: str) -> str | None:
    for label, prefix in ID_PREFIX.items():
        if identifier.startswith(prefix):
            return label
    return None


def validate_project(project: str) -> str:
    """Only ``skillllm_v2`` and ``skillllm_v2_test[_suffix]`` namespaces may be written or cleared."""
    if not isinstance(project, str) or not _PROJECT_RE.match(project):
        raise GraphStoreError(f"refusing namespace {project!r}: must be 'skillllm_v2' or 'skillllm_v2_test[_x]'")
    return project


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class NodeRecord:
    label: str
    id: str
    props: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown node label {self.label!r}")
        if not self.id.startswith(ID_PREFIX[self.label]):
            raise ValueError(f"{self.label} id must start with {ID_PREFIX[self.label]!r}: {self.id!r}")


@dataclass(frozen=True)
class EdgeRecord:
    type: str
    src: str
    dst: str
    props: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.type not in REL_ENDPOINTS:
            raise ValueError(f"unknown relationship type {self.type!r}")
        s_label, d_label = REL_ENDPOINTS[self.type]
        if label_of(self.src) != s_label or label_of(self.dst) != d_label:
            raise ValueError(f"{self.type} must go {s_label} -> {d_label}: {self.src!r} -> {self.dst!r}")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.type, self.src, self.dst)


@dataclass
class GraphPlan:
    """What a build would write: nodes and edges, de-duplicated by key (later non-empty props win)."""

    nodes: dict[str, NodeRecord] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], EdgeRecord] = field(default_factory=dict)

    def add_node(self, rec: NodeRecord) -> None:
        old = self.nodes.get(rec.id)
        if old is None:
            self.nodes[rec.id] = NodeRecord(rec.label, rec.id, dict(rec.props))
            return
        merged = dict(old.props)
        merged.update({k: v for k, v in rec.props.items() if v not in (None, "", [])})
        self.nodes[rec.id] = NodeRecord(old.label, old.id, merged)

    def add_edge(self, rec: EdgeRecord) -> None:
        old = self.edges.get(rec.key)
        if old is None:
            self.edges[rec.key] = EdgeRecord(rec.type, rec.src, rec.dst, dict(rec.props))
            return
        merged = dict(old.props)
        merged.update({k: v for k, v in rec.props.items() if v not in (None, "", [])})
        self.edges[rec.key] = EdgeRecord(old.type, old.src, old.dst, merged)

    def extend(self, nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord]) -> None:
        for n in nodes:
            self.add_node(n)
        for e in edges:
            self.add_edge(e)

    def counts(self) -> dict[str, Any]:
        node_counts = {label: 0 for label in LABELS}
        for n in self.nodes.values():
            node_counts[n.label] += 1
        edge_counts = {t: 0 for t in REL_TYPES}
        dangling = 0
        for e in self.edges.values():
            edge_counts[e.type] += 1
            if e.src not in self.nodes or e.dst not in self.nodes:
                dangling += 1
        return {
            "nodes": node_counts,
            "edges": edge_counts,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "dangling_edges": dangling,
        }

    def check_size(self, max_nodes: int = MAX_NODES, existing_nodes: int = 0) -> None:
        total = len(self.nodes) + max(0, existing_nodes)
        if total > max_nodes:
            raise SizeGuardError(f"planned graph has {total} nodes (> hard limit {max_nodes}); refusing to write")


# --------------------------------------------------------------------------- results
@dataclass
class StoreResult:
    """Outcome of one store operation.

    ``backend``   which backend produced ``value`` ('neo4j' or 'local').
    ``fell_back`` True if the preferred backend failed (or was known down) and the other one answered.
    ``error``     redacted description of the failure that caused a fallback, or of the failure itself when
                  ``ok`` is False (no backend answered). ``None`` on a clean answer.
    """

    value: Any
    backend: str
    fell_back: bool = False
    error: str | None = None
    elapsed_ms: float = 0.0
    ok: bool = True

    def __post_init__(self) -> None:
        if self.backend not in ("neo4j", "local"):
            raise ValueError(f"backend must be 'neo4j' or 'local', not {self.backend!r}")

    def meta(self) -> dict[str, Any]:
        return {"backend": self.backend, "fell_back": self.fell_back, "error": self.error, "ok": self.ok,
                "elapsed_ms": round(self.elapsed_ms, 3)}


# --------------------------------------------------------------------------- contract
@runtime_checkable
class GraphStore(Protocol):
    """Operations every backend (and the auto/mirror compositions) provides."""

    name: str

    def upsert_nodes(self, nodes: Iterable[NodeRecord], build_id: str, project: str | None = None) -> StoreResult: ...
    def upsert_edges(self, edges: Iterable[EdgeRecord], build_id: str, project: str | None = None) -> StoreResult: ...
    def user_skills(self, user_id: str) -> StoreResult: ...
    def job_skills(self, job_id: str, strength: str = "required") -> StoreResult: ...
    def skill_gap(self, user_id: str, job_id: str) -> StoreResult: ...
    def courses_teaching(self, skill_id: str) -> StoreResult: ...
    def prerequisite_chain(self, course_id: str, max_depth: int = 4) -> StoreResult: ...
    def explain_path(self, source: str, target: str, max_hops: int = 4) -> StoreResult: ...
    def stats(self) -> StoreResult: ...
    def health(self) -> StoreResult: ...
    def clear_namespace(self, project: str | None = None) -> StoreResult: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------- shared query semantics
def chain_from_primitives(course_id: str, max_depth: int, prereq_skills, teachers) -> dict[str, Any]:
    """Transitive prerequisite chain, identical for every backend.

    ``prereq_skills(course_ids) -> {course_id: [{"skill_id", "level", "strictness"}]}`` and
    ``teachers(skill_ids) -> {skill_id: [course_id, ...]}`` are the only backend-specific pieces.

    Level 1 are the skills the course itself requires (``REQUIRES_PREREQUISITE``). Level k+1 are the skills required
    by the courses that teach a level-k skill (``TEACHES``), never revisiting a course (cycle safe). The result lists
    each skill once, at the first (shallowest) level it appears. ``truncated`` is True when the walk stopped at
    ``max_depth`` while further teacher courses still existed.
    """
    seen_courses = {course_id}
    seen_skills: set[str] = set()
    levels: list[dict[str, Any]] = []
    frontier = [course_id]
    truncated = False
    for depth in range(1, max(1, int(max_depth)) + 1):
        reqs = prereq_skills(frontier)
        entries: list[dict[str, Any]] = []
        new_skills: list[str] = []
        for cid in frontier:
            for item in sorted(reqs.get(cid, []), key=lambda r: r["skill_id"]):
                if item["skill_id"] in seen_skills:
                    continue
                seen_skills.add(item["skill_id"])
                new_skills.append(item["skill_id"])
                entries.append({"skill_id": item["skill_id"], "required_by": cid, "level": item.get("level"),
                                "strictness": item.get("strictness")})
        if not entries:
            break
        taught = teachers(new_skills)
        next_frontier: list[str] = []
        for entry in entries:
            entry["taught_by"] = sorted(taught.get(entry["skill_id"], []))
            for cid in entry["taught_by"]:
                if cid not in seen_courses:
                    seen_courses.add(cid)
                    next_frontier.append(cid)
        levels.append({"depth": depth, "skills": sorted(entries, key=lambda e: e["skill_id"])})
        if not next_frontier:
            break
        if depth == max_depth:
            further = prereq_skills(sorted(next_frontier))
            truncated = any(further.get(cid) for cid in next_frontier)
            break
        frontier = sorted(next_frontier)
    return {"course_id": course_id, "levels": levels, "max_depth": int(max_depth), "truncated": truncated,
            "skill_ids": sorted(seen_skills)}


def normalize_path_result(source: str, target: str, found: bool, nodes: list[str], edges: list[dict[str, Any]],
                          max_hops: int, truncated: bool, hops_if_truncated: int | None = None,
                          reason: str = "") -> dict[str, Any]:
    """Canonical explain_path payload shared by backends."""
    return {
        "source": source,
        "target": target,
        "found": found,
        "path": nodes if found else [],
        "edges": edges if found else [],
        "hops": (len(edges) if found else hops_if_truncated),
        "max_hops": max_hops,
        "truncated": truncated,
        "reason": reason,
        "explanation": _render(source, edges) if found else "No graph path found within max_hops.",
    }


def _render(source: str, edges: list[dict[str, Any]]) -> str:
    parts = [source]
    for e in edges:
        if e["direction"] == "reverse":
            parts.append(f"<-[{e['relation']}]- {e['to']}")
        else:
            parts.append(f"-[{e['relation']}]-> {e['to']}")
    return " ".join(parts)
