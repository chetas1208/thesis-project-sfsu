"""Knowledge-graph and graph/contextual-retrieval evals (module "graph").

Deterministic offline checks on SYNTHETIC worlds (users, jobs, courses, skills with fully known ground truth, built by
``make_world``) and on the tiny checked-in demo inputs.  No labels, no LLM, no network.

What is deterministic code vs heuristic
---------------------------------------
* Structural validity, provenance presence, construction fidelity, idempotence, determinism and path realisability are
  exact set/equality computations (no heuristics).
* Shortest-path correctness is checked against (a) hand-computed expected paths on toy graphs and (b) an independent
  BFS implemented here on adjacency lists (NOT networkx), so the oracle does not share code with the module under test.
* "Edge signature" and "provenance spec" tables are THIS eval's reading of the builders/architecture doc (which edge
  types connect which node types, which fields an edge should carry).  They are conventions of the eval, stated in the
  caveats, not a schema enforced by ``graph/schema.py`` (which only lists the enum values).
* Retrieval comparisons use token-overlap baselines (title-only keyword match, random) as trivial references.  Query
  gold is derived from the generator's spec, so it is label-free but synthetic; distractor courses that merely MENTION a
  skill in their description are injected on purpose to separate lexical from graph retrieval.
"""

from __future__ import annotations

import random
import re
from collections import Counter, deque
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple
from unittest import mock

from .contract import EvalContext, EvalResult, bootstrap_ci, paired_bootstrap

MODULE = "graph"
DATASET = "synthetic_worlds_v1 (seeded generator, evals/graph_evals.py) + inputs/demo"

# edge type -> (allowed source node types, allowed target node types)  [eval convention, see module docstring]
EDGE_SIGNATURE: Dict[str, Tuple[Set[str], Set[str]]] = {
    "HAS_SKILL": ({"User", "Profile"}, {"Skill"}),
    "MISSING_SKILL": ({"User", "Profile"}, {"Skill"}),
    "REQUIRES": ({"Job", "Occupation"}, {"Skill"}),
    "PREFERS": ({"Job", "Occupation"}, {"Skill"}),
    "TEACHES": ({"Course"}, {"Skill"}),
    "REQUIRES_PREREQUISITE": ({"Course"}, {"Skill", "Prerequisite"}),
    "RELATED_TO": ({"Skill", "SkillCluster"}, {"Skill", "SkillCluster"}),
    "SIMILAR_TO": ({"Skill", "SkillCluster"}, {"Skill", "SkillCluster"}),
    "BELONGS_TO": ({"Skill"}, {"SkillCluster"}),
}
# edge type -> required provenance keys (origin / confidence-like / evidence-like) that CourseGraphBuilder advertises
PROVENANCE_SPEC: Dict[str, Tuple[str, ...]] = {
    "HAS_SKILL": ("source", "confidence", "evidence_text", "last_seen"),
    "REQUIRES": ("importance", "requirement_strength", "source_sentence"),
    "PREFERS": ("importance", "requirement_strength", "source_sentence"),
    "TEACHES": ("confidence", "depth", "learning_outcome_evidence"),
    "REQUIRES_PREREQUISITE": ("prerequisite_evidence", "strictness", "level"),
    "RELATED_TO": ("relation_type", "weight"),
}
EVIDENCE_KEY = {"HAS_SKILL": "evidence_text", "REQUIRES": "source_sentence", "PREFERS": "source_sentence",
                "TEACHES": "learning_outcome_evidence", "REQUIRES_PREREQUISITE": "prerequisite_evidence"}

_FIXED = "2026-01-01T00:00:00+00:00"


@contextmanager
def fixed_graph_clock(ts: str = _FIXED) -> Iterator[None]:
    with mock.patch("skillllm_v2.graph.course_graph_builder._utc_now_iso", lambda: ts):
        yield


@contextmanager
def ticking_graph_clock() -> Iterator[None]:
    state = {"n": 0}

    def tick() -> str:
        state["n"] += 1
        return f"2026-01-01T00:{state['n'] // 60:02d}:{state['n'] % 60:02d}+00:00"

    with mock.patch("skillllm_v2.graph.course_graph_builder._utc_now_iso", tick):
        yield


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def _res(name: str, metric: str, vals: Sequence[float], ctx: EvalContext, details: Optional[Dict[str, Any]] = None,
         caveats: Optional[List[str]] = None, dataset: str = DATASET, status: str = "ok", reason: str = "") -> EvalResult:
    if not vals:
        return EvalResult(name, MODULE, metric, None, 0, dataset, status="skipped", reason=reason or "no units", details=details or {}, caveats=caveats or [])
    ci = bootstrap_ci(vals, n_boot=ctx.n_boot, seed=ctx.seed) if len(vals) > 1 else None
    return EvalResult(name, MODULE, metric, _mean(vals), len(vals), dataset, ci=ci, details=details or {}, caveats=caveats or [], status=status, reason=reason)


def norm_id(name: str) -> str:
    from skillllm_v2.schema import normalize_text

    return normalize_text(name)


# --------------------------------------------------------------------------------------------------------------
# Synthetic world with known ground truth
# --------------------------------------------------------------------------------------------------------------

TOPICS = ["Cloud Engineering", "Data Platforms", "Applied Analytics", "Modern Web", "Reliability", "Machine Intelligence", "Backend Systems",
          "Mobile Craft", "Security Essentials", "Delivery Practices"]
TITLE_FORMS = ["Foundations of {s}", "Applied {s}", "{s} in Practice", "Hands-on {s}", "{s} for Professionals"]
DESC_MENTIONS = ["Builds on your earlier work with {d}.", "Often paired with {d} in real projects.", "Not a {d} course, but {d} knowledge helps."]
DESC_GENERIC = ["Learn the fundamentals and apply them in production.", "Learn by doing with weekly projects.", "Designed for working engineers who want to learn more."]


def _skill_pool(rng: random.Random, n: int) -> List[Tuple[str, str]]:
    from .context_evals import HARD_SKILLS

    seen: Set[str] = set()
    pool: List[Tuple[str, str]] = []
    for _, name, _ in HARD_SKILLS:
        sid = norm_id(name)
        if sid and sid not in seen and len(sid) > 2:
            seen.add(sid)
            pool.append((sid, name))
    rng.shuffle(pool)
    return pool[:n]


def make_world(seed: int, idx: int = 0, n_skills: int = 40, n_courses: int = 30, n_jobs: int = 12, n_users: int = 12) -> Dict[str, Any]:
    rng = random.Random(f"world:{seed}:{idx}")
    skills = _skill_pool(rng, n_skills)
    names = dict(skills)
    ids = [s for s, _ in skills]
    courses: List[Dict[str, Any]] = []
    for i in range(n_courses):
        teaches = rng.sample(ids, rng.randint(1, 3))
        rest = [s for s in ids if s not in teaches]
        prereqs = rng.sample(rest, rng.randint(0, 2))
        distract = rng.sample([s for s in rest if s not in prereqs], 1 + (rng.random() < 0.5))
        if rng.random() < 0.4:
            title = f"{rng.choice(TOPICS)} Bootcamp {i}"  # generic title: names no skill
        else:
            title = rng.choice(TITLE_FORMS).format(s=names[teaches[0]]) + f" {i}"
        desc = rng.choice(DESC_GENERIC) + " " + " ".join(rng.choice(DESC_MENTIONS).format(d=names[d]) for d in distract)
        courses.append({"course_id": f"course-{i}", "title": title, "description": desc, "teaches_skills": [names[s] for s in teaches],
                        "prerequisites": [names[s] for s in prereqs], "learning_outcomes": [f"Apply {names[s]} in production" for s in teaches],
                        "level": rng.choice(["beginner", "intermediate", "advanced"]), "duration_hours": float(rng.choice([10, 20, 30])),
                        "_teaches": teaches, "_prereqs": prereqs})
    jobs: List[Dict[str, Any]] = []
    for j in range(n_jobs):
        req = rng.sample(ids, rng.randint(3, 5))
        pref = rng.sample([s for s in ids if s not in req], rng.randint(1, 2))
        jobs.append({"job_id": f"job-{j}", "title": f"Role {j}",
                     "required_skills": [{"skill_name": names[s], "importance": round(rng.uniform(0.6, 1.0), 2), "source_sentence": f"You must know {names[s]}."} for s in req],
                     "preferred_skills": [{"skill_name": names[s], "importance": 0.7, "source_sentence": f"{names[s]} is a plus."} for s in pref],
                     "_req": req, "_pref": pref})
    users: List[Dict[str, Any]] = []
    for u in range(n_users):
        have = rng.sample(ids, rng.randint(5, 10))
        users.append({"user_id": f"user-{u}", "target_roles": [f"Role {u % n_jobs}"],
                      "skills": [{"surface_form": names[s], "canonical_skill_name": names[s], "confidence": round(rng.uniform(0.5, 1.0), 2),
                                  "evidence": f"Worked with {names[s]} on a team project.", "mastery_level": rng.choice(["beginner", "intermediate", "proficient"])}
                                 for s in have], "_have": have})
    return {"dataset": "synthetic", "seed": seed, "idx": idx, "skills": skills, "names": names, "courses": courses, "jobs": jobs, "users": users}


def expected_edges(world: Dict[str, Any]) -> Counter:
    c: Counter = Counter()
    for u in world["users"]:
        for s in u["_have"]:
            c[(u["user_id"], s, "HAS_SKILL")] += 1
    for j in world["jobs"]:
        for s in j["_req"]:
            c[(j["job_id"], s, "REQUIRES")] += 1
        for s in j["_pref"]:
            c[(j["job_id"], s, "PREFERS")] += 1
    for co in world["courses"]:
        for s in co["_teaches"]:
            c[(co["course_id"], s, "TEACHES")] += 1
        for s in co["_prereqs"]:
            c[(co["course_id"], s, "REQUIRES_PREREQUISITE")] += 1
    return c


def build_course_graph(world: Dict[str, Any], order_seed: Optional[int] = None, repeat: int = 1):
    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder

    g = CourseGraphBuilder()
    users, jobs, courses = list(world["users"]), list(world["jobs"]), list(world["courses"])
    if order_seed is not None:
        r = random.Random(order_seed)
        r.shuffle(users), r.shuffle(jobs), r.shuffle(courses)
    for _ in range(repeat):
        for c in courses:
            g.ingest_course({k: v for k, v in c.items() if not k.startswith("_")})
        for j in jobs:
            g.ingest_job({k: v for k, v in j.items() if not k.startswith("_")})
        for u in users:
            g.ingest_user_profile({k: v for k, v in u.items() if not k.startswith("_")})
    return g


def build_skill_graph(world: Dict[str, Any]):
    from skillllm_v2.graph.graph_builder import SkillGraphBuilder
    from skillllm_v2.graph.schema import NodeType

    g = SkillGraphBuilder()
    for u in world["users"]:
        for s in u["_have"]:
            g.add_user_skill(u["user_id"], s, confidence=0.9)
    for j in world["jobs"]:
        for s in j["_req"]:
            g.add_job_requirement(j["job_id"], s, required=True)
        for s in j["_pref"]:
            g.add_job_requirement(j["job_id"], s, required=False)
    for c in world["courses"]:
        for s in c["_teaches"]:
            g.add_course_skill(c["course_id"], s)
    return g


def edge_multiset(graph: Any) -> Counter:
    return Counter((u, v, d.get("type")) for u, v, d in graph.edges(data=True))


def canonical_form(graph: Any, drop: Sequence[str] = ()) -> List[str]:
    """Order-independent serialisation of nodes and edges (sorted), optionally dropping volatile attribute keys."""
    def clean(d: Dict[str, Any]) -> str:
        return repr(sorted((k, repr(v)) for k, v in d.items() if k not in drop))

    nodes = sorted(f"N|{n}|{clean(d)}" for n, d in graph.nodes(data=True))
    edges = sorted(f"E|{u}|{v}|{clean(d)}" for u, v, d in graph.edges(data=True))
    return nodes + edges


# --------------------------------------------------------------------------------------------------------------
# Structural validity
# --------------------------------------------------------------------------------------------------------------


def structural_report(graph: Any) -> Dict[str, Any]:
    from skillllm_v2.graph.schema import EdgeType, NodeType

    node_types = {t.value for t in NodeType}
    edge_types = {t.value for t in EdgeType}
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    node_ok = sum(1 for _, d in graph.nodes(data=True) if d.get("type") in node_types)
    untyped = [n for n, d in graph.nodes(data=True) if "type" not in d]
    edge_ok = sig_ok = dangling = 0
    prov_all = prov_n = 0
    key_present: Dict[str, List[int]] = {}
    ev_nonempty: Dict[str, List[int]] = {}
    for u, v, d in graph.edges(data=True):
        t = d.get("type")
        edge_ok += int(t in edge_types)
        nu, nv = graph.nodes[u].get("type"), graph.nodes[v].get("type")
        dangling += int(nu is None or nv is None)
        if t in EDGE_SIGNATURE:
            s_src, s_dst = EDGE_SIGNATURE[t]
            sig_ok += int(nu in s_src and nv in s_dst)
        if t in PROVENANCE_SPEC:
            prov_n += 1
            has = [k in d for k in PROVENANCE_SPEC[t]]
            prov_all += int(all(has))
            for k, h in zip(PROVENANCE_SPEC[t], has):
                key_present.setdefault(f"{t}.{k}", []).append(int(h))
        if t in EVIDENCE_KEY:
            ev_nonempty.setdefault(t, []).append(int(bool(str(d.get(EVIDENCE_KEY[t], "")).strip())))
    skill_nodes = [(n, d) for n, d in graph.nodes(data=True) if d.get("type") == "Skill"]
    skill_attr_ok = sum(1 for n, d in skill_nodes if d.get("canonical_skill_id") == n and d.get("canonical_skill_name"))
    return {
        "n_nodes": n_nodes, "n_edges": n_edges, "node_type_valid": node_ok / max(1, n_nodes), "edge_type_valid": edge_ok / max(1, n_edges),
        "edge_signature_valid": sig_ok / max(1, n_edges), "dangling_edge_rate": dangling / max(1, n_edges), "untyped_nodes": len(untyped),
        "provenance_complete": prov_all / max(1, prov_n), "n_prov_edges": prov_n,
        "key_presence": {k: sum(v) / len(v) for k, v in sorted(key_present.items())},
        "evidence_nonempty": {k: sum(v) / len(v) for k, v in sorted(ev_nonempty.items())},
        "skill_node_attr_ok": skill_attr_ok / max(1, len(skill_nodes)),
    }


def load_demo_graph() -> Optional[Any]:
    """Build a CourseGraphBuilder from inputs/demo (user_profile.json, jobs.jsonl, course_catalog.jsonl) if present."""
    import json
    from pathlib import Path

    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder

    root = Path(__file__).resolve().parents[3] / "inputs" / "demo"
    if not (root / "jobs.jsonl").exists():
        return None
    g = CourseGraphBuilder()
    prof = json.loads((root / "user_profile.json").read_text())
    skills = [s.strip() for s in re.split(r"[,.]| and ", prof.get("text", "")) if s.strip()]
    g.ingest_user_profile({"user_id": prof.get("user_id", "demo"), "skills": [s for s in ("Python", "SQL", "Docker", "communication", "machine learning") if s.lower() in prof.get("text", "").lower()]})
    for line in (root / "jobs.jsonl").read_text().splitlines():
        if line.strip():
            g.ingest_job(json.loads(line))
    for line in (root / "course_catalog.jsonl").read_text().splitlines():
        if line.strip():
            g.ingest_course(json.loads(line))
    return g


def eval_structure(ctx: EvalContext, worlds: List[Dict[str, Any]]) -> List[EvalResult]:
    res: List[EvalResult] = []
    with fixed_graph_clock():
        cg = [structural_report(build_course_graph(w).graph) for w in worlds]
        sg = [structural_report(build_skill_graph(w).graph) for w in worlds]
        demo = load_demo_graph()
    rep_demo = structural_report(demo.graph) if demo is not None else None
    for key, label in (("node_type_valid", "nodes whose `type` is a NodeType"), ("edge_type_valid", "edges whose `type` is an EdgeType"),
                       ("edge_signature_valid", "edges whose endpoint node types match the expected signature"),
                       ("dangling_edge_rate", "edges with an untyped (implicitly created) endpoint"),
                       ("provenance_complete", "edges carrying every provenance key of PROVENANCE_SPEC")):
        res.append(_res(f"graph.course_builder.{key}", f"CourseGraphBuilder: share of {label}", [r[key] for r in cg], ctx,
                        details={"demo_inputs": rep_demo[key] if rep_demo else None},
                        caveats=["EDGE_SIGNATURE / PROVENANCE_SPEC are this eval's conventions derived from the builders and the architecture doc; PROVENANCE_SPEC is read off CourseGraphBuilder itself, so 1.0 for it is partly tautological: the informative numbers are the SkillGraphBuilder 0.0 and the empty-evidence rate on inputs/demo"]))
        res.append(_res(f"graph.skill_builder.{key}", f"SkillGraphBuilder: share of {label}", [r[key] for r in sg], ctx,
                        caveats=["SkillGraphBuilder (graph/graph_builder.py) records almost no provenance by design: only HAS_SKILL confidence"] if key == "provenance_complete" else []))
    agg_present: Dict[str, List[float]] = {}
    for r in cg:
        for k, v in r["key_presence"].items():
            agg_present.setdefault(k, []).append(v)
    agg_present_s: Dict[str, List[float]] = {}
    for r in sg:
        for k, v in r["key_presence"].items():
            agg_present_s.setdefault(k, []).append(v)
    res.append(_res("graph.course_builder.provenance_key_presence_min", "lowest per-key presence rate among spec keys (CourseGraphBuilder)", [min(r["key_presence"].values()) for r in cg], ctx,
                    details={"per_key_mean": {k: _mean(v) for k, v in sorted(agg_present.items())}, "skill_builder_per_key_mean": {k: _mean(v) for k, v in sorted(agg_present_s.items())}}))
    ev: Dict[str, List[float]] = {}
    for r in cg:
        for k, v in r["evidence_nonempty"].items():
            ev.setdefault(k, []).append(v)
    res.append(_res("graph.course_builder.evidence_nonempty_rate", "share of edges whose evidence field is a non-empty string (synthetic worlds supply evidence)", [_mean([r["evidence_nonempty"][k] for k in r["evidence_nonempty"]]) for r in cg], ctx,
                    details={"per_edge_type_mean": {k: _mean(v) for k, v in sorted(ev.items())},
                             "demo_inputs": rep_demo["evidence_nonempty"] if rep_demo else None},
                    caveats=["on inputs/demo, users and jobs list skills as plain strings so evidence_text/source_sentence are empty: provenance keys exist but carry no evidence (see demo_inputs)"]))
    res.append(_res("graph.course_builder.skill_node_attr_ok", "Skill nodes with canonical_skill_id == node id and a canonical_skill_name", [r["skill_node_attr_ok"] for r in cg], ctx))
    # builder accepts edges to unknown nodes silently?
    from skillllm_v2.graph.graph_builder import SkillGraphBuilder
    from skillllm_v2.graph.schema import EdgeType, NodeType

    b = SkillGraphBuilder()
    b.add_edge("ghost-course", "ghost-skill", EdgeType.TEACHES)
    rejects = float(b.graph.number_of_edges() == 0)
    rep = structural_report(b.graph)
    b.add_node("x", NodeType.USER)
    b.add_node("x", NodeType.SKILL)
    retyped = b.graph.nodes["x"]["type"] != NodeType.USER.value
    res.append(EvalResult("graph.builder_rejects_dangling_edge", MODULE, "add_edge to unknown nodes is rejected (1) or silently creates untyped placeholders (0)", rejects, 1, "probe on SkillGraphBuilder (deterministic)",
                          details={"dangling_edge_rate_after_probe": rep["dangling_edge_rate"], "untyped_nodes": rep["untyped_nodes"], "add_node_silently_retypes_existing_node": retyped},
                          caveats=["a typo in an id yields an untyped dangling node that no later check in the builders catches; add_node with an existing id and a different type overwrites the type without error"]))
    return res


def eval_fidelity_idempotence_determinism(ctx: EvalContext, worlds: List[Dict[str, Any]]) -> List[EvalResult]:
    prec, rec, exact = [], [], []
    prec_s, rec_s = [], []
    idem, same_clock, ticks, ticks_mod, order_inv = [], [], [], [], []
    dup_nodes_changed: List[float] = []
    for w in worlds:
        exp = expected_edges(w)
        with fixed_graph_clock():
            g1 = build_course_graph(w)
            g2 = build_course_graph(w)
            gs = build_skill_graph(w)
            g_twice = build_course_graph(w, repeat=2)
            g_shuf = build_course_graph(w, order_seed=7)
        act = edge_multiset(g1.graph)
        tp = sum((act & exp).values())
        prec.append(tp / max(1, sum(act.values())))
        rec.append(tp / max(1, sum(exp.values())))
        exact.append(float(act == exp))
        acts = edge_multiset(gs.graph)
        tps = sum((acts & exp).values())
        # SkillGraphBuilder has no prerequisite edges: score against the spec minus REQUIRES_PREREQUISITE
        exp_s = Counter({k: v for k, v in exp.items() if k[2] != "REQUIRES_PREREQUISITE"})
        tps = sum((acts & exp_s).values())
        prec_s.append(tps / max(1, sum(acts.values())))
        rec_s.append(tps / max(1, sum(exp_s.values())))
        idem.append(g1.graph.number_of_edges() / max(1, g_twice.graph.number_of_edges()))
        same_clock.append(float(canonical_form(g1.graph) == canonical_form(g2.graph)))
        with ticking_graph_clock():
            a = build_course_graph(w)
            b = build_course_graph(w)
        ticks.append(float(canonical_form(a.graph) == canonical_form(b.graph)))
        ticks_mod.append(float(canonical_form(a.graph, drop=("last_seen",)) == canonical_form(b.graph, drop=("last_seen",))))
        order_inv.append(float(canonical_form(g1.graph) == canonical_form(g_shuf.graph)))
    caveat_spec = ["expected edges come from the generator spec, so this is a build-correctness check, not a measure of real-world graph quality"]
    return [
        _res("graph.course_builder.edge_precision", "built edges that are in the spec (multiset)", prec, ctx, caveats=caveat_spec),
        _res("graph.course_builder.edge_recall", "spec edges present in the built graph (multiset)", rec, ctx, caveats=caveat_spec),
        _res("graph.course_builder.exact_edge_match", "built edge multiset == spec edge multiset", exact, ctx, caveats=caveat_spec),
        _res("graph.skill_builder.edge_precision", "SkillGraphBuilder edges in the spec (prerequisites excluded: unsupported)", prec_s, ctx),
        _res("graph.skill_builder.edge_recall", "SkillGraphBuilder spec edges present (prerequisites excluded)", rec_s, ctx),
        _res("graph.ingest_idempotence", "edges after one ingest / edges after ingesting the same data twice (1.0 = idempotent)", idem, ctx,
             caveats=["MultiDiGraph.add_edge always creates a parallel edge, so re-ingesting the catalog/profile/job doubles every edge (this value is ~0.5 for a non-idempotent builder)"]),
        _res("graph.determinism_same_clock", "identical canonical graph from two builds of the same input (clock frozen)", same_clock, ctx),
        _res("graph.determinism_across_clock_ticks", "identical canonical graph when the wall clock advances between builds", ticks, ctx,
             details={"identical_modulo_last_seen": _mean(ticks_mod)},
             caveats=["HAS_SKILL.last_seen defaults to utc-now at build time, so graphs are not byte-stable across seconds unless profiles provide last_seen; modulo that field builds are identical"]),
        _res("graph.order_invariance", "identical canonical graph when users/jobs/courses are ingested in a different order", order_inv, ctx),
    ]


# --------------------------------------------------------------------------------------------------------------
# Path-finding correctness
# --------------------------------------------------------------------------------------------------------------


def _toy_graph(edges: Sequence[Tuple[str, str]], extra_nodes: Sequence[str] = ()):
    from skillllm_v2.graph.graph_builder import SkillGraphBuilder
    from skillllm_v2.graph.schema import EdgeType, NodeType

    b = SkillGraphBuilder()
    for u, v in edges:
        b.add_node(u, NodeType.SKILL)
        b.add_node(v, NodeType.SKILL)
        b.add_edge(u, v, EdgeType.RELATED_TO, weight=1.0)
    for n in extra_nodes:
        b.add_node(n, NodeType.SKILL)
    return b.graph


def bfs_shortest(adj: Dict[str, List[str]], s: str, t: str) -> Optional[int]:
    """Independent oracle (no networkx): hop distance or None."""
    if s == t:
        return 0
    seen = {s}
    q = deque([(s, 0)])
    while q:
        x, d = q.popleft()
        for y in adj.get(x, []):
            if y == t:
                return d + 1
            if y not in seen:
                seen.add(y)
                q.append((y, d + 1))
    return None


def realisable(graph: Any, path: Sequence[str]) -> bool:
    return all(graph.has_edge(a, b) for a, b in zip(path, path[1:]))


# hand-computed toy cases: (name, edges, source, target, max_hops, expected_path or None for 'no path')
TOY_CASES: List[Tuple[str, List[Tuple[str, str]], str, str, int, Optional[List[str]]]] = [
    ("chain3", [("a", "b"), ("b", "c"), ("c", "d")], "a", "d", 4, ["a", "b", "c", "d"]),
    ("shortcut", [("a", "b"), ("b", "d"), ("a", "c"), ("c", "d"), ("a", "d")], "a", "d", 4, ["a", "d"]),
    ("direction", [("a", "b"), ("b", "c")], "c", "a", 4, None),
    ("unreachable", [("a", "b"), ("c", "d")], "a", "d", 4, None),
    ("self", [("a", "b")], "a", "a", 4, ["a"]),
    ("six_hops_cap4", [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g")], "a", "g", 4, ["a", "b", "c", "d", "e", "f", "g"]),
    ("exactly_cap", [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")], "a", "e", 4, ["a", "b", "c", "d", "e"]),
]


def run_toy_case(case: Tuple[str, List[Tuple[str, str]], str, str, int, Optional[List[str]]]) -> Dict[str, Any]:
    from skillllm_v2.graph.skill_graph_explainer import explain_skill_path

    name, edges, s, t, hops, expected = case
    g = _toy_graph(edges, extra_nodes=[s, t])
    out = explain_skill_path(g, s, t, max_hops=hops)
    path = out["path"]
    exact = (path == expected) if expected is not None else (path == [])
    endpoint_ok = (not path) or (path[0] == s and path[-1] == t)
    return {"name": name, "expected": expected, "got": path, "exact": exact, "endpoint_ok": endpoint_ok, "realisable": realisable(g, path),
            "confidence": out["confidence"]}


def eval_paths(ctx: EvalContext, worlds: List[Dict[str, Any]]) -> List[EvalResult]:
    from skillllm_v2.graph.skill_graph_explainer import explain_skill_path

    res: List[EvalResult] = []
    toys = [run_toy_case(c) for c in TOY_CASES]
    res.append(_res("graph.path_toy_exact_match", "hand-computed toy cases where explain_skill_path returns exactly the expected path", [float(t["exact"]) for t in toys], ctx,
                    details={"cases": toys},
                    caveats=["expected paths are hand-computed; 'six_hops_cap4' expects the full path because a path whose end is not the requested target is not an explanation of that target"]))
    # random directed graphs vs independent BFS
    n_graphs = 6 if ctx.smoke else 30
    ok_within, endpoint_beyond, real_all, found_when_reach, truncated_wrong = [], [], [], [], []
    for gi in range(n_graphs):
        rng = random.Random(f"rg:{ctx.seed}:{gi}")
        n = 14
        nodes = [f"s{i}" for i in range(n)]
        edges = [(a, b) for a in nodes for b in nodes if a != b and rng.random() < 0.14]
        g = _toy_graph(edges, extra_nodes=nodes)
        adj: Dict[str, List[str]] = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
        within, beyond, rl, fr, tw = [], [], [], [], []
        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                d = bfs_shortest(adj, a, b)
                out = explain_skill_path(g, a, b, max_hops=4)
                path = out["path"]
                if path:
                    rl.append(float(realisable(g, path) and path[0] == a))
                if d is None:
                    continue
                fr.append(float(bool(path)))
                if d <= 4:
                    within.append(float(len(path) - 1 == d and path[-1] == b and realisable(g, path)))
                else:
                    beyond.append(float(bool(path) and path[-1] == b))
                    tw.append(float(bool(path) and path[-1] != b))
        if within:
            ok_within.append(_mean(within))
        if beyond:
            endpoint_beyond.append(_mean(beyond))
        if tw:
            truncated_wrong.append(_mean(tw))
        if rl:
            real_all.append(_mean(rl))
        if fr:
            found_when_reach.append(_mean(fr))
    res.append(_res("graph.path_shortest_correct_within_hops", "reachable pairs with BFS distance <= max_hops: returned path is shortest, ends at the target and is realisable (independent BFS oracle)", ok_within, ctx,
                    details={"n_random_graphs": n_graphs, "nodes": 14, "edge_prob": 0.14}))
    res.append(_res("graph.path_found_when_reachable", "reachable pairs for which a non-empty path is returned", found_when_reach, ctx))
    res.append(_res("graph.path_ends_at_target_beyond_hops", "reachable pairs with distance > max_hops: returned path still ends at the target", endpoint_beyond, ctx,
                    details={"truncated_to_wrong_endpoint_rate": _mean(truncated_wrong) if truncated_wrong else None},
                    caveats=["explain_skill_path trims the path to max_hops+1 nodes but still labels it as the path from source to target: the returned `path` ends at an intermediate node while `target` says otherwise"]))
    res.append(_res("graph.explanation_path_realisable_rate", "non-empty returned paths whose consecutive nodes are connected by graph edges and start at the source", real_all, ctx,
                    caveats=["truncated paths are realisable as a prefix (each hop is a real edge) even though they miss the target"]))
    # builder-shaped worlds: can the directed shortest-path explainer connect a user to a job / course at all?
    conn_found, conn_expected, skill_hop = [], [], []
    for w in worlds:
        with fixed_graph_clock():
            cg = build_course_graph(w)
        g = cg.graph
        found = exp = hop1 = tot1 = 0
        for u in w["users"]:
            for j in w["jobs"]:
                shared = set(u["_have"]) & set(j["_req"])
                if shared:
                    exp += 1  # an explanatory path U -HAS_SKILL-> s <-REQUIRES- J exists (undirected through a shared skill)
                    found += int(bool(explain_skill_path(g, u["user_id"], j["job_id"], max_hops=4)["path"]))
        conn_found.append(found / exp if exp else float("nan"))
        conn_expected.append(float(exp > 0))
        for j in w["jobs"]:
            for s in j["_req"]:
                tot1 += 1
                hop1 += int(bool(explain_skill_path(g, j["job_id"], s)["path"]))
        skill_hop.append(hop1 / max(1, tot1))
    valid = [v for v in conn_found if v == v]
    res.append(_res("graph.explainer_connects_user_to_job", "user/job pairs sharing a skill (so an explanatory path exists) for which explain_skill_path(user, job) returns a path", valid, ctx,
                    details={"job_to_required_skill_found": _mean(skill_hop)},
                    caveats=["in builder-made graphs every edge points INTO a Skill node (User->Skill, Job->Skill, Course->Skill), so no directed path ever connects two non-skill nodes; explain_skill_path uses directed nx.shortest_path and returns 'No graph path found'. The documented path 'user -> skill -> missing_skill -> job -> course' is unreachable through this function"]))
    return res


# --------------------------------------------------------------------------------------------------------------
# Retrieval: graph retriever / contextual retriever vs trivial baselines
# --------------------------------------------------------------------------------------------------------------


def _rr(ranked: Sequence[str], gold: Set[str]) -> float:
    for i, x in enumerate(ranked, 1):
        if x in gold:
            return 1.0 / i
    return 0.0


def eval_graph_retriever(ctx: EvalContext, worlds: List[Dict[str, Any]]) -> List[EvalResult]:
    from skillllm_v2.graph.graph_retriever import SkillGraphRetriever
    from skillllm_v2.retrieval.contextual_retrieval import ContextualRetriever

    exact_user, exact_job, exact_course = [], [], []
    gap_graph_p, gap_graph_r, gap_flat_p, gap_flat_r, gap_title_p, gap_title_r = [], [], [], [], [], []
    ctx_rec, ctx_mrr, title_rec, title_mrr, rand_rec, rand_mrr = [], [], [], [], [], []
    k = 5
    for w in worlds:
        with fixed_graph_clock():
            sg = build_skill_graph(w)
        r = SkillGraphRetriever(sg.graph)
        users = {u["user_id"]: set(u["_have"]) for u in w["users"]}
        jobs = {j["job_id"]: set(j["_req"]) for j in w["jobs"]}
        teaches: Dict[str, Set[str]] = {}
        for c in w["courses"]:
            for s in c["_teaches"]:
                teaches.setdefault(s, set()).add(c["course_id"])
        exact_user.append(_mean([float(set(r.skills_for_user(u)) == s and len(r.skills_for_user(u)) == len(s)) for u, s in users.items()]))
        exact_job.append(_mean([float(set(r.required_skills_for_job(j)) == s and len(r.required_skills_for_job(j)) == len(s)) for j, s in jobs.items()]))
        exact_course.append(_mean([float(set(r.courses_for_skill(s)) == teaches.get(s, set())) for s in w["names"]]))
        # two-hop gap query: courses that teach a skill the user lacks for a job
        retr = ContextualRetriever([{k_: v for k_, v in c.items() if not k_.startswith("_")} for c in w["courses"]])
        p_g, r_g, p_f, r_f, p_t, r_t = [], [], [], [], [], []
        for u in w["users"]:
            j = w["jobs"][int(u["user_id"].split("-")[1]) % len(w["jobs"])]
            gold_skills = jobs[j["job_id"]] - users[u["user_id"]]
            if not gold_skills:
                continue
            gold = set().union(*[teaches.get(s, set()) for s in gold_skills])
            if not gold:
                continue
            # graph route: required(job) - skills(user) -> courses_for_skill
            miss = set(r.required_skills_for_job(j["job_id"])) - set(r.skills_for_user(u["user_id"]))
            got = set()
            for s in miss:
                got |= set(r.courses_for_skill(s))
            p_g.append(len(got & gold) / max(1, len(got)))
            r_g.append(len(got & gold) / len(gold))
            # flat route: ContextualRetriever with the missing skill names as the query, top-|gold| (R-precision), + title-only baseline
            q = " ".join(w["names"][s] for s in sorted(gold_skills))
            kk = len(gold)
            top = {h["record"]["course_id"] for h in retr.search(q, top_k=kk)}
            p_f.append(len(top & gold) / kk)
            r_f.append(len(top & gold) / kk)
            qt = set(re.sub(r"[\W_]+", " ", q.lower()).split())
            tscore = sorted(((len(qt & set(re.sub(r"[\W_]+", " ", c["title"].lower()).split())), c["course_id"]) for c in w["courses"]), key=lambda t: (-t[0], t[1]))
            toptitle = {cid for sc, cid in tscore[:kk] if sc > 0}
            p_t.append(len(toptitle & gold) / kk)
            r_t.append(len(toptitle & gold) / kk)
        if p_g:
            gap_graph_p.append(_mean(p_g)), gap_graph_r.append(_mean(r_g)), gap_flat_p.append(_mean(p_f)), gap_flat_r.append(_mean(r_f))
            gap_title_p.append(_mean(p_t)), gap_title_r.append(_mean(r_t))
        # single-skill course query: contextual retriever vs title-only baseline vs random
        rng = random.Random(f"cq:{ctx.seed}:{w['idx']}")
        c_r, c_m, t_r, t_m, rn_r, rn_m = [], [], [], [], [], []
        cids = [c["course_id"] for c in w["courses"]]
        for s, name in rng.sample(sorted(w["names"].items()), min(15, len(w["names"]))):
            gold = teaches.get(s, set())
            if not gold:
                continue
            q = f"learn {name}"
            ranked = [h["record"]["course_id"] for h in retr.search(q, top_k=k)]
            tq = set(re.sub(r"[\W_]+", " ", q.lower()).split())
            ts = sorted(((len(tq & set(re.sub(r"[\W_]+", " ", c["title"].lower()).split())), c["course_id"]) for c in w["courses"]), key=lambda t: (-t[0], t[1]))
            tr = [cid for sc, cid in ts if sc > 0][:k]
            rr = cids[:]
            rng.shuffle(rr)
            for rs, (a, b) in ((ranked, (c_r, c_m)), (tr, (t_r, t_m)), (rr[:k], (rn_r, rn_m))):
                a.append(len(set(rs) & gold) / min(len(gold), k))
                b.append(_rr(rs, gold))
        ctx_rec.append(_mean(c_r)), ctx_mrr.append(_mean(c_m)), title_rec.append(_mean(t_r)), title_mrr.append(_mean(t_m)), rand_rec.append(_mean(rn_r)), rand_mrr.append(_mean(rn_m))
    res: List[EvalResult] = []
    res.append(_res("graph.retriever_exact_user_skills", "SkillGraphRetriever.skills_for_user equals the spec skill set (exact, per user)", exact_user, ctx))
    res.append(_res("graph.retriever_exact_job_requirements", "SkillGraphRetriever.required_skills_for_job equals the spec", exact_job, ctx))
    res.append(_res("graph.retriever_exact_courses_for_skill", "SkillGraphRetriever.courses_for_skill equals the spec", exact_course, ctx))
    pbp = paired_bootstrap(gap_graph_p[: len(gap_flat_p)], gap_flat_p, n_boot=ctx.n_boot, seed=ctx.seed) if gap_graph_p else {}
    res.append(_res("graph.gap_course_precision_graph", "courses teaching a (job-required minus user-held) skill: precision of graph route (set-based)", gap_graph_p, ctx,
                    details={"graph_recall": _mean(gap_graph_r), "flat_contextual_r_precision": _mean(gap_flat_p), "flat_title_only_r_precision": _mean(gap_title_p),
                             "paired_graph_minus_flat_precision": pbp},
                    caveats=["synthetic: distractor courses mention gap skills in their description without teaching them, which is exactly what hurts lexical retrieval; the graph route reads the TEACHES edges directly, so a near-1.0 value shows the lookup is faithful to the spec, not that it ranks well",
                             "graph route returns an unranked set; flat routes are scored as R-precision at k=|gold|"]))
    res.append(_res("graph.gap_course_r_precision_flat_contextual", "same query via ContextualRetriever over course text, R-precision", gap_flat_p, ctx,
                    details={"title_only_baseline": _mean(gap_title_p)}))
    res.append(_res("graph.contextual_retriever_recall_at_5", "ContextualRetriever recall@5 for 'learn <skill>' -> courses that teach it", ctx_rec, ctx,
                    details={"title_only_baseline": _mean(title_rec), "random": _mean(rand_rec), "paired_vs_title": paired_bootstrap(ctx_rec, title_rec, n_boot=ctx.n_boot, seed=ctx.seed)},
                    caveats=["query word 'learn' also occurs in generic descriptions; the retriever scores overlap/len(query) with no IDF, so such words dilute the ranking",
                             "unit = world (mean over 15 sampled skills)"]))
    res.append(_res("graph.contextual_retriever_mrr", "ContextualRetriever MRR for 'learn <skill>'", ctx_mrr, ctx,
                    details={"title_only_baseline": _mean(title_mrr), "random": _mean(rand_mrr), "paired_vs_title": paired_bootstrap(ctx_mrr, title_mrr, n_boot=ctx.n_boot, seed=ctx.seed)}))
    res.append(_res("graph.title_baseline_recall_at_5", "title-only keyword baseline recall@5", title_rec, ctx))
    res.append(_res("graph.random_baseline_recall_at_5", "random course ranking recall@5 (negative control)", rand_rec, ctx))
    # unknown-node probe: ids made of single characters that happen to be node ids
    from skillllm_v2.graph.graph_builder import SkillGraphBuilder

    b = SkillGraphBuilder()
    for d in "0123456789":
        b.add_user_skill(d, f"skill-{d}")
    r = SkillGraphRetriever(b.graph)
    unknown = [f"{a}{c}" for a in "123" for c in "4567"]  # "14", "15", ... none is a node
    leaks = [float(bool(r.skills_for_user(u))) for u in unknown]
    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder

    cgb = CourseGraphBuilder()
    for d in "0123456789":
        cgb.add_user_skill(d, f"skill-{d}")
    safe = [float(bool(cgb.get_user_skills(u))) for u in unknown]
    res.append(_res("graph.retriever_unknown_node_leak_rate", "unknown user ids for which SkillGraphRetriever.skills_for_user returns another user's skills (0.0 is correct)", leaks, ctx,
                    details={"CourseGraphBuilder.get_user_skills_leak_rate": _mean(safe), "probe": "10 users named '0'..'9'; queries '14','15',...,'37'"},
                    dataset="probe (deterministic)",
                    caveats=["networkx treats a str nbunch that is not a node as an iterable of characters, so out_edges('14') yields the edges of nodes '1' and '4'"],
                    status="ok"))
    return res


# --------------------------------------------------------------------------------------------------------------
# Skill-id normalisation (graph node identity)
# --------------------------------------------------------------------------------------------------------------

DISTINCT_SKILL_NAMES: List[Tuple[str, ...]] = [
    ("C", "C++", "C#"), ("F#", "F"), (".NET", "NET"),
]


def eval_id_collisions(ctx: EvalContext) -> List[EvalResult]:
    groups = []
    for grp in DISTINCT_SKILL_NAMES:
        ids = [norm_id(x) for x in grp]
        groups.append({"names": list(grp), "ids": ids, "collides": len(set(ids)) < len(grp)})
    vals = [float(g["collides"]) for g in groups]
    return [_res("graph.skill_id_collision_rate", "groups of genuinely different skills (C/C++/C#, F#/F, .NET/NET) that normalize_text maps to one node id", vals, ctx,
                 details={"groups": groups}, dataset="probe: hand-picked distinct skill names (deterministic)",
                 caveats=["graph builders derive node ids with normalize_text, which strips punctuation: 'C++' and 'C#' become 'c', so distinct skills share one Skill node and edges/explanations silently merge them"])]


# --------------------------------------------------------------------------------------------------------------
# Checked-in fixtures (inputs/evals/)
# --------------------------------------------------------------------------------------------------------------


def fixture_payload(seed: int = 42) -> Dict[str, Any]:
    world = make_world(seed, 0, n_skills=12, n_courses=6, n_jobs=3, n_users=3)
    return {"dataset": "synthetic", "generator": "skillllm_v2.evals.graph_evals.make_world / TOY_CASES", "seed": seed,
            "note": "SYNTHETIC toy graphs with hand-computed shortest paths, and one small generated world",
            "toy_path_cases": [{"name": n, "edges": e, "source": s, "target": t, "max_hops": h, "expected_path": x} for n, e, s, t, h, x in TOY_CASES],
            "small_world": world}


def write_fixtures(root: Any, seed: int = 42) -> str:
    import json
    from pathlib import Path

    path = Path(root) / "inputs" / "evals" / "graph_cases_synthetic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture_payload(seed), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------------------------------


def make_worlds(ctx: EvalContext) -> List[Dict[str, Any]]:
    return [make_world(ctx.seed, i) for i in range(3 if ctx.smoke else 12)]


def run(ctx: EvalContext) -> List[EvalResult]:
    worlds = make_worlds(ctx)
    out: List[EvalResult] = []
    steps = [lambda: eval_structure(ctx, worlds), lambda: eval_fidelity_idempotence_determinism(ctx, worlds), lambda: eval_paths(ctx, worlds),
             lambda: eval_graph_retriever(ctx, worlds), lambda: eval_id_collisions(ctx)]
    names = ["eval_structure", "eval_fidelity_idempotence_determinism", "eval_paths", "eval_graph_retriever", "eval_id_collisions"]
    for nm, fn in zip(names, steps):
        try:
            out.extend(fn())
        except Exception as exc:
            out.append(EvalResult(f"graph.{nm}", MODULE, nm, None, 0, DATASET, status="failed", reason=f"{type(exc).__name__}: {exc}"))
    out.extend([
        EvalResult("graph.edge_truth_vs_human_annotation", MODULE, "precision/recall of extracted graph edges against human-annotated skill relations", None, 0, "none", status="skipped",
                   reason="no human-annotated user/job/course graph is available offline; fidelity is measured against the synthetic generator spec only"),
        EvalResult("graph.taxonomy_alignment", MODULE, "node identity agreement with an external taxonomy (ESCO/O*NET)", None, 0, "none", status="skipped",
                   reason="no external taxonomy snapshot is checked in; only normalize_text-based ids are exercised"),
    ])
    return out
