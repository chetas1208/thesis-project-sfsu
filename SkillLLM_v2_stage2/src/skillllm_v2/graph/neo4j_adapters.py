"""Adapters from the existing NetworkX graphs to backend-neutral records, plus the curated, synthetic-only build plan.

Read-only with respect to the course-side code: :class:`CourseGraphBuilder` / :class:`SkillGraphBuilder` graphs are only
*read* (``.graph``); this module never edits them. Only whitelisted, non-free-text properties are mapped (ids, names,
levels, numbers, flags, categorical strings); optional short ``evidence`` strings are kept only for synthetic sources.

What :func:`build_curated_plan` includes (everything else is deliberately NOT uploaded):

* skill names / aliases from ``resources/offline_aliases.jsonl`` (the curated alias set the offline linker uses);
* ``inputs/courses/course_catalog_large.jsonl`` rows whose ``source`` is ``synthetic``;
* ``inputs/courses/benchmark_scenarios.jsonl`` scenarios whose ``source`` is ``synthetic`` (scenarios whose job text is
  excerpted from a real posting are skipped; job text and gold labels are never uploaded);
* ``inputs/demo/*`` (profile text is not uploaded: only the curated skills it mentions).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from skillllm_v2.schema import graph_key, normalize_text

from .graph_builder import SkillGraphBuilder
from .schema import NodeType
from .store import (
    ID_PREFIX,
    LABELS,
    REL_TYPES,
    EdgeRecord,
    GraphPlan,
    GraphStoreError,
    NodeRecord,
    node_id,
)

ALLOWED_SOURCES = {"curated_aliases", "synthetic_catalog", "benchmark_synthetic", "demo", "derived_skill"}
FORBIDDEN_PROP_KEYS = {"text", "description", "sentence", "source_sentence", "resume", "posting", "job_text",
                       "learning_outcomes", "content", "snippet"}
FORBIDDEN_MARKERS = ("Extract_jobDesc", "Resume+Job", "SkillSpan", "data/source")
_MAX_STR = 300
_EVIDENCE_MAX = 200
_ALIAS_MAX = 25


# --------------------------------------------------------------------------- curated skill index
class CuratedSkillIndex:
    """Surface form -> (canonical id, canonical name) from the curated offline alias table."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.names: dict[str, str] = {}
        self.taxonomy: dict[str, str] = {}
        self.aliases: dict[str, list[str]] = {}
        self._lookup: dict[str, str] = {}
        for row in rows:
            cid = str(row["canonical_skill_id"])
            self.names.setdefault(cid, str(row.get("canonical_skill_name") or cid))
            self.taxonomy.setdefault(cid, str(row.get("taxonomy") or "CUSTOM"))
            alias = str(row.get("alias") or "")
            if alias and alias not in self.aliases.setdefault(cid, []):
                self.aliases[cid].append(alias)
            for key in (row.get("normalized_alias"), normalize_text(self.names[cid]), normalize_text(alias)):
                if key:
                    self._lookup.setdefault(str(key), cid)

    @classmethod
    def load(cls, path: Path | None = None) -> "CuratedSkillIndex":
        path = path or Path(__file__).resolve().parent.parent / "resources" / "offline_aliases.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            return cls(json.loads(line) for line in handle if line.strip())

    def lookup(self, surface: str) -> str | None:
        return self._lookup.get(normalize_text(surface))

    def resolve(self, surface: str) -> tuple[str, str]:
        """``(skill node id, display name)``: the curated id if known, else a key derived from the surface form."""
        cid = self.lookup(surface)
        if cid is not None:
            return node_id("Skill", cid), self.names[cid]
        return node_id("Skill", graph_key(surface) or normalize_text(surface) or surface), surface

    def find_in_text(self, text: str) -> list[str]:
        """Curated skill node ids mentioned in ``text`` (whole-word alias match). Used only for the demo profile."""
        norm = f" {normalize_text(text)} "
        hits: dict[str, None] = {}
        for key in sorted(self._lookup, key=len, reverse=True):
            if f" {key} " in norm:
                hits[node_id("Skill", self._lookup[key])] = None
        return list(hits)

    def skill_records(self) -> list[NodeRecord]:
        out = []
        for cid, name in self.names.items():
            aliases = [a for a in self.aliases.get(cid, []) if len(a) <= 80][:_ALIAS_MAX]
            out.append(NodeRecord("Skill", node_id("Skill", cid), {
                "name": name, "taxonomy": self.taxonomy.get(cid, "CUSTOM"), "aliases": aliases,
                "alias_count": len(self.aliases.get(cid, [])), "source": "curated_aliases", "synthetic": False}))
        return out


# --------------------------------------------------------------------------- graph -> records
_TYPE_TO_LABEL = {"User": "User", "Job": "Job", "Skill": "Skill", "Course": "Course"}


def _short(value: Any, limit: int = _EVIDENCE_MAX) -> str | None:
    text = str(value or "").strip()
    return text[:limit] if text else None


def records_from_graph(
    graph: Any,
    *,
    source: str,
    synthetic: bool,
    index: CuratedSkillIndex | None = None,
    include_evidence: bool = False,
    extra_node_props: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[NodeRecord], list[EdgeRecord], dict[str, int]]:
    """Map a ``SkillGraphBuilder`` / ``CourseGraphBuilder`` networkx graph to records (read-only).

    Node ids become ``user:`` / ``job:`` / ``skill:`` / ``course:`` ids; skills are canonicalised through ``index``
    (curated id when the name is a known alias). Returns ``(nodes, edges, skipped_counts)``.
    """
    index = index or CuratedSkillIndex([])
    skipped = {"nodes_untyped_or_other": 0, "edges_other_type": 0, "edges_dangling": 0}
    mapping: dict[Any, str] = {}
    nodes: dict[str, NodeRecord] = {}
    for key, data in graph.nodes(data=True):
        label = _TYPE_TO_LABEL.get(str(data.get("type", "")))
        if label is None:
            skipped["nodes_untyped_or_other"] += 1
            continue
        props: dict[str, Any] = {"source": source, "synthetic": bool(synthetic)}
        if label == "Skill":
            raw_name = str(data.get("canonical_skill_name") or data.get("name") or str(key).split(":", 1)[-1])
            nid, name = index.resolve(raw_name)
            props["name"] = name
            cid = nid[len(ID_PREFIX["Skill"]):]
            if index.lookup(raw_name) is not None:
                props["taxonomy"] = index.taxonomy.get(cid, "CUSTOM")
            else:
                props["taxonomy"] = str(data.get("taxonomy") or "CUSTOM")
                props["source"] = "derived_skill"
        else:
            raw = str(key)
            nid = node_id(label, raw)
            props["name"] = str(data.get("name") or data.get("title") or raw.split(":", 1)[-1])
            if label == "Course":
                for src, dst in (("title", "title"), ("level", "level"), ("provider", "provider")):
                    if data.get(src) not in (None, ""):
                        props[dst] = _short(data[src], 120)
                for src in ("duration_hours", "rating", "cost_usd"):
                    if isinstance(data.get(src), (int, float)) and not isinstance(data.get(src), bool):
                        props[src] = float(data[src])
            if label == "Job" and data.get("title"):
                props["title"] = _short(data["title"], 120)
        props.update((extra_node_props or {}).get(str(key), {}))
        mapping[key] = nid
        nodes[nid] = NodeRecord(label, nid, props)

    edges: dict[tuple[str, str, str], EdgeRecord] = {}
    for u, v, data in graph.edges(data=True):
        etype = str(data.get("type", ""))
        if etype not in REL_TYPES:
            skipped["edges_other_type"] += 1
            continue
        if u not in mapping or v not in mapping:
            skipped["edges_dangling"] += 1
            continue
        props = {"source": str(data.get("source") or source)}
        if etype == "HAS_SKILL":
            props.update(confidence=data.get("confidence"), last_seen=data.get("last_seen"),
                         mastery_level=data.get("mastery_level"))
            if include_evidence:
                props["evidence"] = _short(data.get("evidence_text"))
        elif etype in ("REQUIRES", "PREFERS"):
            props.update(importance=data.get("importance"), requirement_strength=data.get("requirement_strength"))
            if include_evidence:
                props["evidence"] = _short(data.get("source_sentence"))
        elif etype == "TEACHES":
            props.update(confidence=data.get("confidence"), depth=data.get("depth"), outcome_index=data.get("outcome_index"))
            if include_evidence:
                props["evidence"] = _short(data.get("learning_outcome_evidence"))
        elif etype == "REQUIRES_PREREQUISITE":
            props.update(level=data.get("level"), strictness=data.get("strictness"))
            if include_evidence:
                props["evidence"] = _short(data.get("prerequisite_evidence"))
        elif etype == "RELATED_TO":
            props.update(relation_type=data.get("relation_type"), weight=data.get("weight"))
        props = {k: v for k, v in props.items() if v not in (None, "")}
        try:
            rec = EdgeRecord(etype, mapping[u], mapping[v], props)
        except ValueError:
            skipped["edges_dangling"] += 1  # endpoint label does not fit the relationship (e.g. after canonicalising)
            continue
        old = edges.get(rec.key)
        if old is None:
            edges[rec.key] = rec
        else:
            old.props.update({k: val for k, val in props.items() if k not in old.props})
    return list(nodes.values()), list(edges.values()), skipped


def records_from_local_store(local: Any) -> tuple[list[NodeRecord], list[EdgeRecord]]:
    """Export a :class:`LocalGraphStore` back to records (used by mirror catch-up sync)."""
    g = local.graph
    drop = {"type", "project", "build_id", "placeholder"}
    nodes = [NodeRecord(d["type"], n, {k: v for k, v in d.items() if k not in drop})
             for n, d in g.nodes(data=True) if d.get("type") in LABELS]
    edges = []
    for u, v, d in g.edges(data=True):
        if d.get("type") in REL_TYPES:
            edges.append(EdgeRecord(d["type"], u, v, {k: val for k, val in d.items() if k not in drop}))
    return nodes, edges


# --------------------------------------------------------------------------- the curated plan
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cooccurrence_edges(plan: GraphPlan, min_shared: int = 2) -> list[EdgeRecord]:
    taught: dict[str, set[str]] = {}
    for e in plan.edges.values():
        if e.type == "TEACHES":
            taught.setdefault(e.dst, set()).add(e.src)
    skills = sorted(taught)
    out = []
    for i, a in enumerate(skills):
        for b in skills[i + 1:]:
            shared = len(taught[a] & taught[b])
            if shared >= min_shared:
                weight = shared / len(taught[a] | taught[b])
                out.append(EdgeRecord("RELATED_TO", a, b, {"relation_type": "co_taught", "weight": round(weight, 4),
                                                           "source": "synthetic_catalog"}))
    return out


def _add(plan: GraphPlan, nodes: list[NodeRecord], edges: list[EdgeRecord]) -> None:
    """Merge records; an existing Skill node (curated first) keeps its own name/source/taxonomy."""
    for n in nodes:
        if n.label == "Skill" and n.id in plan.nodes:
            continue
        plan.add_node(n)
    for e in edges:
        plan.add_edge(e)


def build_curated_plan(root: Path | str) -> tuple[GraphPlan, dict[str, Any]]:
    """Build the synthetic/curated-only plan from the repo at ``root``. Returns ``(plan, provenance)``."""
    from skillllm_v2.course.ingest_catalog import CourseCatalog
    from .course_graph_builder import CourseGraphBuilder

    root = Path(root)
    index = CuratedSkillIndex.load()
    plan = GraphPlan()
    prov: dict[str, Any] = {"included": {}, "skipped": {}}
    _add(plan, index.skill_records(), [])
    prov["included"]["curated_aliases"] = {"skills": len(index.names), "alias_rows": sum(len(v) for v in index.aliases.values())}

    # ---- synthetic course catalogue (CourseGraphBuilder, read-only on its graph)
    catalog_path = root / "inputs" / "courses" / "course_catalog_large.jsonl"
    rows = _read_jsonl(catalog_path)
    synthetic_rows = [r for r in rows if str(r.get("source", "")).lower() == "synthetic"]
    if len(synthetic_rows) != len(rows):
        prov["skipped"]["catalog_non_synthetic"] = [r.get("course_id") for r in rows if r not in synthetic_rows]
    cb = CourseGraphBuilder()
    cb.ingest_catalog(CourseCatalog(synthetic_rows))
    nodes, edges, sk = records_from_graph(cb.graph, source="synthetic_catalog", synthetic=True, index=index, include_evidence=True)
    _add(plan, nodes, edges)
    prov["included"]["synthetic_course_catalog"] = {"file": "inputs/courses/course_catalog_large.jsonl", "courses": len(synthetic_rows), **sk}
    _add(plan, [], _cooccurrence_edges(plan))

    # ---- synthetic benchmark scenarios (SkillGraphBuilder); posting-derived scenarios are skipped
    scen_path = root / "inputs" / "courses" / "benchmark_scenarios.jsonl"
    scenarios = _read_jsonl(scen_path)
    used = [s for s in scenarios if str(s.get("source", "")).lower() == "synthetic" and not s.get("posting_provenance")]
    skipped = [s["scenario_id"] for s in scenarios if s not in used]
    sb = SkillGraphBuilder()
    extra: dict[str, dict[str, Any]] = {}
    for s in used:
        uid = f"user:{s['user_profile']['user_id']}"
        jid = f"job:{s['target_job']['job_id']}"
        sb.add_node(uid, NodeType.USER, name=s["user_profile"]["user_id"])
        extra[uid] = {"experience_level": str(s.get("experience_level") or "")}
        sb.add_node(jid, NodeType.JOB, name=str(s["target_job"].get("title") or s["target_job"]["job_id"]))
        for item in s["user_profile"].get("skills", []):
            name = str(item.get("canonical_skill_name") or item.get("surface_form"))
            sid, sname = index.resolve(name)
            sb.add_node(sid, NodeType.SKILL, name=sname)
            sb.add_user_skill(uid, sid, confidence=float(item.get("confidence", 1.0)), source="benchmark_synthetic")
        for name in s["target_job"].get("required_skills", []):
            sid, sname = index.resolve(str(name))
            sb.add_node(sid, NodeType.SKILL, name=sname)
            sb.add_job_requirement(jid, sid, required=True, importance=1.0)
        for name in s["target_job"].get("preferred_skills", []):
            sid, sname = index.resolve(str(name))
            sb.add_node(sid, NodeType.SKILL, name=sname)
            sb.add_job_requirement(jid, sid, required=False, importance=0.7)
    nodes, edges, sk = records_from_graph(sb.graph, source="benchmark_synthetic", synthetic=True, index=index, extra_node_props=extra)
    _add(plan, nodes, edges)
    prov["included"]["benchmark_scenarios_synthetic"] = {"file": "inputs/courses/benchmark_scenarios.jsonl", "scenarios": len(used), **sk}
    prov["skipped"]["benchmark_scenarios_real_posting_excerpt"] = skipped

    # ---- demo inputs
    demo = root / "inputs" / "demo"
    db = SkillGraphBuilder()
    if (demo / "user_profile.json").exists():
        profile = json.loads((demo / "user_profile.json").read_text(encoding="utf-8"))
        uid = f"user:{profile.get('user_id', 'demo-user')}"
        db.add_node(uid, NodeType.USER, name=str(profile.get("user_id", "demo-user")))
        for sid in index.find_in_text(str(profile.get("text", ""))):  # only curated skill ids leave the machine
            db.add_node(sid, NodeType.SKILL, name=index.names[sid[len(ID_PREFIX["Skill"]):]])
            db.add_user_skill(uid, sid, confidence=1.0, source="demo")
    if (demo / "jobs.jsonl").exists():
        for job in _read_jsonl(demo / "jobs.jsonl"):
            jid = f"job:{job['job_id']}"
            db.add_node(jid, NodeType.JOB, name=str(job["job_id"]))
            for required, key in ((True, "required_skills"), (False, "preferred_skills")):
                for name in job.get(key, []):
                    sid, sname = index.resolve(str(name))
                    db.add_node(sid, NodeType.SKILL, name=sname)
                    db.add_job_requirement(jid, sid, required=required, importance=1.0 if required else 0.7)
    nodes, edges, sk1 = records_from_graph(db.graph, source="demo", synthetic=True, index=index)
    _add(plan, nodes, edges)
    n_demo_courses = 0
    if (demo / "course_catalog.jsonl").exists():
        dcb = CourseGraphBuilder()
        demo_rows = _read_jsonl(demo / "course_catalog.jsonl")
        dcb.ingest_catalog(CourseCatalog(demo_rows))
        nodes, edges, _ = records_from_graph(dcb.graph, source="demo", synthetic=True, index=index, include_evidence=False)
        _add(plan, nodes, edges)
        n_demo_courses = len(demo_rows)
    prov["included"]["demo"] = {"dir": "inputs/demo", "courses": n_demo_courses, **sk1}
    prov["not_uploaded"] = [
        "SkillLLM_v2/Extract_jobDesc (real postings)", "Resume+Job Data (resumes)", "data/source and SkillSpan sentences",
        "job text / gold labels of any scenario", "course descriptions and learning-outcome sentences (evidence kept only "
        "as <=200-char strings on synthetic catalogue edges)", "demo profile free text"]
    return plan, prov


def assert_safe_plan(plan: GraphPlan) -> None:
    """Refuse a plan that could carry free text or real-corpus data. Raises :class:`GraphStoreError`."""
    problems: list[str] = []

    def check(where: str, props: dict[str, Any]) -> None:
        for key, value in props.items():
            if key in FORBIDDEN_PROP_KEYS:
                problems.append(f"{where}: forbidden property {key!r}")
            if isinstance(value, str):
                if len(value) > _MAX_STR:
                    problems.append(f"{where}.{key}: string longer than {_MAX_STR}")
                if any(m.lower() in value.lower() for m in FORBIDDEN_MARKERS):
                    problems.append(f"{where}.{key}: references a non-uploadable corpus")

    for n in plan.nodes.values():
        if n.props.get("source") not in ALLOWED_SOURCES:
            problems.append(f"node {n.id}: source {n.props.get('source')!r} not allowed")
        check(n.id, n.props)
    for e in plan.edges.values():
        if e.props.get("source") not in ALLOWED_SOURCES:
            problems.append(f"edge {e.key}: source {e.props.get('source')!r} not allowed")
        check(str(e.key), e.props)
    if problems:
        raise GraphStoreError("unsafe plan: " + "; ".join(problems[:8]))


def plan_content_hash(plan: GraphPlan) -> str:
    """Deterministic digest of the plan: the default ``build_id`` (identical data => identical id => idempotent)."""
    digest = hashlib.sha256()
    for nid in sorted(plan.nodes):
        n = plan.nodes[nid]
        digest.update(json.dumps([n.label, n.id, n.props], sort_keys=True, default=str).encode())
    for key in sorted(plan.edges):
        e = plan.edges[key]
        digest.update(json.dumps([e.type, e.src, e.dst, e.props], sort_keys=True, default=str).encode())
    return "b" + digest.hexdigest()[:12]


def load_plan_into(store: Any, plan: GraphPlan, build_id: str, project: str | None = None) -> None:
    """Write a plan to any store (nodes first, then edges)."""
    store.upsert_nodes(plan.nodes.values(), build_id, project)
    store.upsert_edges(plan.edges.values(), build_id, project)


def curated_local_loader(root: Path | str) -> Callable[[Any], None]:
    """``local_loader`` for the factory: builds the curated plan and loads it into the local store on first use."""
    def _load(store: Any) -> None:
        plan, _ = build_curated_plan(root)
        assert_safe_plan(plan)
        load_plan_into(store, plan, plan_content_hash(plan))
    return _load
