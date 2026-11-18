"""L08 graph + retrieval layer: knowledge-graph construction QA, external validity on public prerequisite data, and
retrieval baselines with a controlled RAG-vs-GraphRAG comparison (hypothesis H5).

    PYTHONPATH=src python -m skillllm_v2.evals.research.graph_retrieval [--smoke] [--seed 42] [--n-boot 2000]

Everything here is deterministic CPU code. Definitions were fixed in Decisions.md (D-A5-01..07) before any system was run.
Nothing is fitted: RRF uses k=60 with equal weights, the graph 2-hop discount is a fixed λ=0.5, BM25/MiniLM are off the shelf.

This module also holds the helpers shared with ``course_rec`` (L09): statistics, catalogue/scenario loading, the
retrieval systems, the knowledge-graph path checker and the generated scenario set SG-v1.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from skillllm_v2.evals.research.common import (EVALS_ROOT, REPO_ROOT, md_table, record, rel, sha256_file,
                                               write_layer)

LAYER = "L08_graph_retrieval"
CATALOG = REPO_ROOT / "inputs/courses/course_catalog_large.jsonl"
SCENARIOS = REPO_ROOT / "inputs/courses/benchmark_scenarios.jsonl"
LABELS = REPO_ROOT / "inputs/courses/relevance_labels.jsonl"
SPLIT = REPO_ROOT / "reports/skillllm_v2/recommender_split.json"
EXTERNAL = REPO_ROOT / "data/external"
SG_DIR = REPO_ROOT / "inputs/evals/research/L09_course_rec"
SG_SEED = 20261117
SG_N = 240
RRF_K = 60
GRAPH_LAMBDA = 0.5
POOL_B = 20
REL = 2.0
DATASET = "course_catalog_large.jsonl (171 SYNTHETIC courses) + benchmark_scenarios.jsonl + relevance_labels.jsonl (v1-rule)"
BASE_PROV = [CATALOG, SCENARIOS, LABELS, SPLIT]

REQ_W, PREF_W = 1.0, 0.7  # system-visible gap weights (same as the repo's GapEngine / existing gap-overlap baseline)


# =====================================================================================================================
# statistics (numpy; scenario = unit)
# =====================================================================================================================
def boot_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap of the mean over independent units."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_test(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 42, n_perm: int = 10000) -> Dict[str, Any]:
    """Paired comparison of mean(a - b): bootstrap CI + bootstrap p, and a sign-flip permutation p (exact null of symmetry).

    Bootstrap p = 2·min(P*(d̄ ≤ 0), P*(d̄ ≥ 0)) with +1 smoothing (identical systems → p = 1, not 0)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0:
        raise ValueError("empty paired sample")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_le = (np.sum(means <= 0) + 1) / (n_boot + 1)
    p_ge = (np.sum(means >= 0) + 1) / (n_boot + 1)
    p_boot = float(min(1.0, 2 * min(p_le, p_ge)))
    obs = abs(d.mean())
    if np.all(np.abs(d) < 1e-15):
        p_perm = 1.0
    else:
        rng2 = np.random.default_rng(seed + 1)
        signs = rng2.choice([-1.0, 1.0], size=(n_perm, d.size))
        p_perm = float((np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12) + 1) / (n_perm + 1))
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_boot": p_boot, "p_perm": p_perm,
            "p": max(p_boot, p_perm), "n": int(d.size), "wins": int(np.sum(d > 1e-12)), "losses": int(np.sum(d < -1e-12)),
            "ties": int(np.sum(np.abs(d) <= 1e-12))}


def holm(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def cluster_boot(units: Sequence[Any], stat, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """Percentile CI of stat(list_of_units) resampling units (each unit = a cluster of observations)."""
    rng = np.random.default_rng(seed)
    n = len(units)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(stat([units[i] for i in idx]))
    lo, hi = np.nanquantile(np.asarray(vals, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


# =====================================================================================================================
# ranking metrics (graded labels, relevant = grade ≥ 2)
# =====================================================================================================================
def _dedupe(ids: Sequence[str]) -> List[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def ndcg(ids: Sequence[str], grades: Mapping[str, float], k: int) -> Optional[float]:
    ideal = sorted((float(g) for g in grades.values()), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg <= 0:
        return None
    dcg = sum((2 ** float(grades.get(c, 0)) - 1) / math.log2(i + 2) for i, c in enumerate(_dedupe(ids)[:k]))
    return dcg / idcg


def precision(ids: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    return sum(1 for c in _dedupe(ids)[:k] if grades.get(c, 0) >= REL) / k


def recall(ids: Sequence[str], grades: Mapping[str, float], k: int) -> Optional[float]:
    tot = sum(1 for g in grades.values() if g >= REL)
    if tot == 0:
        return None
    return sum(1 for c in _dedupe(ids)[:k] if grades.get(c, 0) >= REL) / tot


def mrr(ids: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    for i, c in enumerate(_dedupe(ids)[:k], start=1):
        if grades.get(c, 0) >= REL:
            return 1.0 / i
    return 0.0


RETRIEVAL_METRICS = [("p5", "P@5"), ("r10", "Recall@10"), ("r20", "Recall@20"), ("mrr", "MRR@20"), ("ndcg5", "nDCG@5"),
                     ("ndcg10", "nDCG@10")]


def retrieval_metrics(ids: Sequence[str], grades: Mapping[str, float]) -> Dict[str, Optional[float]]:
    return {"p5": precision(ids, grades, 5), "r10": recall(ids, grades, 10), "r20": recall(ids, grades, 20),
            "mrr": mrr(ids, grades, 20), "ndcg5": ndcg(ids, grades, 5), "ndcg10": ndcg(ids, grades, 10)}


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> List[str]:
    """Reciprocal-rank fusion with equal weights; ties broken by id (deterministic)."""
    score: Dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, c in enumerate(_dedupe(r), start=1):
            score[c] += 1.0 / (k + i)
    return sorted(score, key=lambda c: (-score[c], c))


# =====================================================================================================================
# data loading
# =====================================================================================================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def norm(s: str) -> str:
    from skillllm_v2.schema import normalize_text
    return normalize_text(str(s))


def visible_gaps(user: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, float]:
    """System-visible gap: job required (1.0) / preferred (0.7) skills the user does not hold at non-beginner level.
    Identical to the existing benchmark's ``user_gap_weights`` (asserted in tests)."""
    have = {norm(s.get("canonical_skill_name") or s.get("surface_form") or "") for s in user.get("skills", [])
            if str(s.get("mastery_level", "proficient")).lower() != "beginner"}
    g: Dict[str, float] = {}
    for s in job.get("preferred_skills", []):
        g[norm(s)] = PREF_W
    for s in job.get("required_skills", []):
        g[norm(s)] = REQ_W
    return {k: v for k, v in g.items() if k and k not in have}


def gap_names(user: Dict[str, Any], job: Dict[str, Any]) -> List[str]:
    ids = visible_gaps(user, job)
    out = []
    for s in list(job.get("required_skills", [])) + list(job.get("preferred_skills", [])):
        if norm(s) in ids and s not in out:
            out.append(s)
    return out


def q_gap(sc: Dict[str, Any]) -> str:
    job = sc["target_job"]
    return f"{job['title']}. " + ", ".join(gap_names(sc["user_profile"], job))


def q_job(sc: Dict[str, Any]) -> str:
    job = sc["target_job"]
    return f"{job['title']}. {job['text']}"


def known_ids(sc: Dict[str, Any]) -> Set[str]:
    """K_s = gold known ∪ weak (the existing prerequisite convention; weak counts as prerequisite exposure)."""
    return {norm(s) for s in list(sc["gold"]["known"]) + list(sc["gold"]["weak"])}


def user_known_visible(sc: Dict[str, Any]) -> Set[str]:
    """Skills the user declares to the systems (any mastery) — what a system may use for prerequisite reasoning."""
    return {norm(s.get("canonical_skill_name") or s.get("surface_form") or "") for s in sc["user_profile"].get("skills", [])}


def grades_of(label_row: Dict[str, Any], key: str = "labels") -> Dict[str, float]:
    return {k: float(v) for k, v in (label_row.get(key) or {}).items()}


def needs_prereq_hop(sc: Dict[str, Any], label_row: Dict[str, Any], P: Mapping[str, Set[str]]) -> bool:
    """D-A5-02: ≥1 course with content-only grade ≥ 2 whose catalogue prerequisites are not all known/weak for the user."""
    K = known_ids(sc)
    return any(g >= 2 and not P.get(cid, set()) <= K for cid, g in (label_row.get("labels_content_only") or {}).items())


# =====================================================================================================================
# environment: catalogue, indices, pipeline (built once per process)
# =====================================================================================================================
class Env:
    """Catalogue, BM25, MiniLM and the repo's CourseRecommendationPipeline (≈2 min to build: dictionary extraction)."""

    def __init__(self) -> None:
        from skillllm_v2.course.ingest_catalog import CourseCatalog
        from skillllm_v2.eval import recommender_benchmark as rb

        os.chdir(REPO_ROOT)  # the pipeline resolves its alias table relative to the working directory
        self.rb = rb
        self.catalog = CourseCatalog.load_from_file(CATALOG)
        self.courses = self.catalog.list_courses()
        self.by_id = {c.course_id: c for c in self.courses}
        self.ids = [c.course_id for c in self.courses]
        self.T = {c.course_id: {norm(s) for s in c.teaches_skills if norm(s)} for c in self.courses}
        self.P = {c.course_id: {norm(s) for s in c.prerequisites if norm(s)} for c in self.courses}
        self.text = {c.course_id: rb.course_text(c) for c in self.courses}
        self.bm25 = rb.Bm25Index(self.catalog)
        self.dense = rb.DenseIndex(self.catalog)
        self._qcache: Dict[str, Dict[str, float]] = {}

    @cached_property
    def pipe(self):
        from skillllm_v2.pipeline.course_recommendation_pipeline import CourseRecommendationPipeline
        p = CourseRecommendationPipeline(catalog_path=str(CATALOG))
        self.base_graph = p.graph.graph.copy()
        return p

    # ---- rankings over the full catalogue ----------------------------------------------------------------------
    def rank_bm25(self, q: str) -> List[str]:
        return [c.course_id for _, c in self.bm25.rank(q)]

    def dense_scores(self, q: str) -> Dict[str, float]:
        if q not in self._qcache:
            self._qcache[q] = {c.course_id: float(s) for s, c in self.dense.rank(q)}
        return self._qcache[q]

    def rank_dense(self, q: str) -> List[str]:
        s = self.dense_scores(q)
        return sorted(s, key=lambda c: (-s[c], c))

    def rank_skill_overlap(self, gaps: Mapping[str, float]) -> List[str]:
        """Flat attribute match: Σ gap weight over the catalogue teaches_skills (only courses with score > 0)."""
        sc = {cid: sum(w for g, w in gaps.items() if g in T) for cid, T in self.T.items()}
        return sorted([c for c in sc if sc[c] > 0], key=lambda c: (-sc[c], c))

    def rank_random(self, sid: str) -> List[str]:
        rng = random.Random(int(hashlib.sha256(f"a5-random:{sid}".encode()).hexdigest()[:12], 16))
        ids = list(self.ids)
        rng.shuffle(ids)
        return ids

    # ---- per-scenario knowledge graph (base catalogue graph + this user + this job, as the pipeline builds it) -------
    def scenario_graph(self, sc: Dict[str, Any]):
        from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
        _ = self.pipe
        b = CourseGraphBuilder()
        b.graph = self.base_graph.copy()
        b.ingest_user_profile(copy.deepcopy(sc["user_profile"]))
        b.ingest_job(copy.deepcopy(sc["target_job"]))
        return b

    # ---- the repo's Full recommender ------------------------------------------------------------------------------
    def run_full(self, config, sc: Dict[str, Any], top_k: int = 10) -> Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]:
        p = self.pipe
        p.set_config(config)
        p.graph.graph = self.base_graph.copy()
        res = p.run(user_input=copy.deepcopy(sc["user_profile"]), job_input=copy.deepcopy(sc["target_job"]), top_k=top_k)
        return res["recommendations"], p.graph.graph, res["skill_gap_analysis"]

    def run_repo_retriever(self, sc: Dict[str, Any], pool: int = 40) -> List[str]:
        from skillllm_v2.recommend.course_retriever import CourseRetriever
        b = self.scenario_graph(sc)
        gv = self.pipe.gap_engine.compute(copy.deepcopy(sc["user_profile"]), copy.deepcopy(sc["target_job"]), graph=b)
        r = CourseRetriever(catalog=self.pipe.catalog, graph=b)
        return [c.course_id for c in r.retrieve(gap_vector=gv, target_job=sc["target_job"], top_k=pool)]


# =====================================================================================================================
# graph retrieval (multi-hop) and path validity
# =====================================================================================================================
def _ntype(G, n: str) -> Optional[str]:
    return G.nodes[n].get("type") if G.has_node(n) else None


def graph_retrieve(G, job_id: str, user_id: str, gaps: Mapping[str, float], lam: float = GRAPH_LAMBDA
                   ) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    """Hop 1: Job −REQUIRES/PREFERS→ gap Skill ←TEACHES− Course (score Σ w_g·conf).
    Hop 2: hop-1 Course −REQUIRES_PREREQUISITE→ Skill p (user has no HAS_SKILL edge to p) ←TEACHES− bridge Course
    (score λ·parent hop-1 score). Returns (ranking of courses with score > 0, paths per course)."""
    s: Dict[str, float] = defaultdict(float)
    paths: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    held = {v for _, v, d in G.out_edges(user_id, data=True) if d.get("type") == "HAS_SKILL"} if G.has_node(user_id) else set()
    job_rel = {}
    if G.has_node(job_id):
        for _, v, d in G.out_edges(job_id, data=True):
            if d.get("type") in ("REQUIRES", "PREFERS"):
                job_rel.setdefault(v, d.get("type"))
    s1: Dict[str, float] = defaultdict(float)
    for g, w in sorted(gaps.items()):
        if not G.has_node(g):
            continue
        for c, _, d in G.in_edges(g, data=True):
            if d.get("type") == "TEACHES" and _ntype(G, c) == "Course":
                s1[c] += w * float(d.get("confidence", 1.0))
                paths[c].append({"hop": 1, "edges": [(job_id, job_rel.get(g, "REQUIRES"), g), (c, "TEACHES", g)],
                                 "non_edges": [(user_id, "HAS_SKILL", g)] if g not in held else []})
    for c, v in s1.items():
        s[c] += v
    for c, v in sorted(s1.items()):
        for _, p, d in G.out_edges(c, data=True):
            if d.get("type") != "REQUIRES_PREREQUISITE" or p in held:
                continue
            for b, _, d2 in G.in_edges(p, data=True):
                if d2.get("type") == "TEACHES" and _ntype(G, b) == "Course" and b != c:
                    s[b] += lam * v
                    first = paths[c][0]["edges"]
                    paths[b].append({"hop": 2, "edges": list(first) + [(c, "REQUIRES_PREREQUISITE", p), (b, "TEACHES", p)],
                                     "non_edges": [(user_id, "HAS_SKILL", p)]})
    ranking = sorted([c for c in s if s[c] > 0], key=lambda c: (-s[c], c))
    return ranking, dict(paths)


def edge_valid(G, u: str, etype: str, v: str) -> Tuple[bool, str]:
    from skillllm_v2.evals.graph_evals import EDGE_SIGNATURE
    if not (G.has_node(u) and G.has_node(v)):
        return False, "missing_node"
    if not any(d.get("type") == etype for d in (G.get_edge_data(u, v) or {}).values()):
        return False, "missing_edge"
    src, dst = EDGE_SIGNATURE.get(etype, (set(), set()))
    if _ntype(G, u) not in src or _ntype(G, v) not in dst:
        return False, "type_mismatch"
    return True, "ok"


def path_valid(G, path: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons = []
    for (u, t, v) in path.get("edges", []):
        ok, why = edge_valid(G, u, t, v)
        if not ok:
            reasons.append(f"{why}:{u}-{t}->{v}")
    for (u, t, v) in path.get("non_edges", []):  # claims that an edge is ABSENT (e.g. user lacks the skill)
        if G.has_node(u) and G.has_node(v) and any(d.get("type") == t for d in (G.get_edge_data(u, v) or {}).values()):
            reasons.append(f"edge_present:{u}-{t}->{v}")
    return (not reasons), reasons


def full_explanation_paths(recs: Sequence[Dict[str, Any]], job_id: str, user_id: str, gap_types: Mapping[str, str]
                           ) -> List[Dict[str, Any]]:
    """Decompose the Full recommender's explanation packets into checkable KG paths (structured fields, not prose)."""
    out = []
    for r in recs:
        cid = r["course_id"]
        for e in r.get("evidence", []):
            g = e.get("canonical_skill_id")
            rel_ = "REQUIRES"
            et = e.get("evidence_type")
            lacks = [] if gap_types.get(g) == "weak_skill" else [(user_id, "HAS_SKILL", g)]
            if et == "text_mention_only":
                out.append({"course": cid, "kind": "text_mention", "edges": [(job_id, rel_, g)], "non_edges": lacks,
                            "non_graph_hops": 1})
            elif et == "enabling_prerequisite":
                via, p = e.get("via_course_id"), e.get("via_prerequisite_skill")
                out.append({"course": cid, "kind": "bridge", "edges": [(job_id, rel_, g), (via, "TEACHES", g),
                                                                        (via, "REQUIRES_PREREQUISITE", p), (cid, "TEACHES", p)],
                            "non_edges": lacks})
            else:
                out.append({"course": cid, "kind": "primary", "edges": [(job_id, rel_, g), (cid, "TEACHES", g)], "non_edges": lacks})
        if r.get("role") != "bridge":
            for u in r.get("unlocks", []) or []:
                out.append({"course": cid, "kind": "unlock", "edges": [(u["course_id"], "REQUIRES_PREREQUISITE", u["prerequisite_skill"]),
                                                                       (cid, "TEACHES", u["prerequisite_skill"])], "non_edges": []})
    return out


def path_valid_lenient_job(G, path: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Job→Skill hops are claimed as 'requires'; accept REQUIRES or PREFERS (type-consistent Job→Skill edge)."""
    p2 = dict(path)
    edges = []
    for (u, t, v) in path["edges"]:
        if t == "REQUIRES" and _ntype(G, u) == "Job":
            kinds = {d.get("type") for d in (G.get_edge_data(u, v) or {}).values()}
            if "PREFERS" in kinds and "REQUIRES" not in kinds:
                t = "PREFERS"
        edges.append((u, t, v))
    p2["edges"] = edges
    return path_valid(G, p2)


# =====================================================================================================================
# graph construction QA (part 1)
# =====================================================================================================================
def skill_prereq_pairs(G) -> Dict[Tuple[str, str], int]:
    """Skill-level prerequisite relation implied by the graph: p ⇒ t iff some Course TEACHES t and REQUIRES_PREREQUISITE p.
    Value = support (#courses)."""
    sup: Dict[Tuple[str, str], int] = Counter()
    for c, d in G.nodes(data=True):
        if d.get("type") != "Course":
            continue
        taught = {v for _, v, e in G.out_edges(c, data=True) if e.get("type") == "TEACHES"}
        req = {v for _, v, e in G.out_edges(c, data=True) if e.get("type") == "REQUIRES_PREREQUISITE"}
        for t in taught:
            for p in req:
                if p != t:
                    sup[(t, p)] += 1
    return dict(sup)


def graph_qa(env: Env) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import networkx as nx
    from skillllm_v2.evals.graph_evals import EDGE_SIGNATURE, EVIDENCE_KEY, PROVENANCE_SPEC
    from skillllm_v2.match.skill_keys import SkillKeyResolver

    G = env.base_graph
    pipe = env.pipe
    recs: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}
    prov = [CATALOG]
    common = dict(n_units=len(env.courses), unit="course", split="all", dataset="course_catalog_large.jsonl (171 SYNTHETIC courses)",
                  provenance=prov)
    ntypes = Counter(d.get("type") for _, d in G.nodes(data=True))
    etypes = Counter(d.get("type") for _, _, d in G.edges(data=True))
    info["node_types"], info["edge_types"] = dict(ntypes), dict(etypes)

    # (a) edge fidelity vs authored catalogue fields (expected 1.0: the graph is a projection of these fields)
    auth_T = {(cid, s) for cid, T in env.T.items() for s in T}
    auth_P = {(cid, s) for cid, P in env.P.items() for s in P}
    g_T = {(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "TEACHES"}
    g_P = {(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "REQUIRES_PREREQUISITE"}
    for name, auth, got in (("teaches", auth_T, g_T), ("prerequisite", auth_P, g_P)):
        p = len(auth & got) / max(1, len(got))
        r = len(auth & got) / max(1, len(auth))
        recs.append(record(LAYER, "kg_builder(as_wired)", f"{name}_edge_precision_vs_authored", p, direction="higher", **common,
                           caveats=["Graph edges are built from the same authored catalogue field they are compared with: 1.0 is by construction, not a quality result."],
                           details={"n_graph_edges": len(got), "n_authored": len(auth)}))
        recs.append(record(LAYER, "kg_builder(as_wired)", f"{name}_edge_recall_vs_authored", r, direction="higher", **common,
                           details={"n_graph_edges": len(got), "n_authored": len(auth)}))

    # (b) is the enrichment (dictionary extraction + taxonomy linker + PrerequisiteParser) wired into the graph?
    enriched = {ec["course_id"]: ec for ec in pipe.enriched_courses}
    teach_ev = Counter(d.get("learning_outcome_evidence") == env.by_id[u].title for u, v, d in G.edges(data=True) if d.get("type") == "TEACHES")
    enr_T = {(cid, norm(it.get("canonical_skill_id") or it.get("surface_form") or "")) for cid, ec in enriched.items() for it in ec.get("taught_skills_evidence", [])}
    enr_P = {(cid, norm(it.get("canonical_skill_id") or it.get("skill_name") or "")) for cid, ec in enriched.items() for it in ec.get("parsed_prerequisites", [])}
    wired_T = len(enr_T & g_T) / max(1, len(enr_T))
    wired_P = len(enr_P & g_P) / max(1, len(enr_P))
    extra_T = sorted(enr_T - g_T)
    only_T = enr_T - auth_T  # pairs the enrichment adds or renames (not in the authored field)
    only_P = enr_P - auth_P
    only_in_graph = (len(only_T & g_T) + len(only_P & g_P)) / max(1, len(only_T) + len(only_P))
    info["enrichment"] = {"enriched_teaches_pairs": len(enr_T), "enriched_prereq_pairs": len(enr_P), "share_enriched_teaches_in_graph": wired_T,
                          "share_enriched_prereq_in_graph": wired_P, "enrichment_only_pairs": len(only_T) + len(only_P),
                          "share_enrichment_only_pairs_in_graph": only_in_graph, "enriched_teaches_not_in_graph_examples": extra_T[:15],
                          "teaches_edges_with_course_title_as_evidence": teach_ev.get(True, 0), "teaches_edges_total": sum(teach_ev.values())}
    recs.append(record(LAYER, "kg_builder(as_wired)", "enrichment_only_pairs_reaching_graph", only_in_graph, direction="none", **common,
                       details=info["enrichment"],
                       caveats=["CourseGraphBuilder.ingest_course reads taught_skills_evidence / parsed_prerequisites from record.metadata, but "
                                "CourseSkillExtractor puts them at the top level of the enriched dict, so the extractor/linker/parser output never reaches the graph."]))
    # (c) quality of the (unwired) enrichment vs authored: parser + linker prerequisites, extractor taught skills
    resolver = SkillKeyResolver()

    def same(a: str, b: str) -> bool:
        return bool(resolver.keys_for_name(a) & resolver.keys_for_name(b)) or norm(a) == norm(b)

    pp_tp = pp_fp = pp_fn = 0
    pp_id_drift = []
    tt_tp = tt_fp = tt_fn = 0
    tt_drift = []
    for cid, ec in enriched.items():
        c = env.by_id[cid]
        parsed = [(it.get("skill_name") or "", it.get("canonical_skill_name") or "", it.get("canonical_skill_id") or "") for it in ec.get("parsed_prerequisites", [])]
        for a in c.prerequisites:
            hit = [p for p in parsed if same(p[1] or p[0], a)]
            pp_tp += 1 if hit else 0
            pp_fn += 0 if hit else 1
            for p in parsed:
                if norm(p[0]) == norm(a) and norm(p[2]) != norm(a):
                    pp_id_drift.append({"course": cid, "authored": a, "canonical_id": p[2], "canonical_name": p[1]})
        pp_fp += sum(1 for p in parsed if not any(same(p[1] or p[0], a) for a in c.prerequisites))
        taught = [(it.get("surface_form") or "", it.get("canonical_skill_name") or "", it.get("canonical_skill_id") or "", it.get("source")) for it in ec.get("taught_skills_evidence", [])]
        uniq = {}
        for t in taught:
            uniq.setdefault(norm(t[2] or t[1] or t[0]), t)
        for a in c.teaches_skills:
            hit = [t for t in uniq.values() if same(t[1] or t[0], a)]
            tt_tp += 1 if hit else 0
            tt_fn += 0 if hit else 1
        for t in uniq.values():
            if not any(same(t[1] or t[0], a) for a in c.teaches_skills):
                tt_fp += 1
            if t[3] == "declared" and norm(t[2]) != norm(t[0]):
                tt_drift.append({"course": cid, "authored": t[0], "canonical_id": t[2], "canonical_name": t[1]})
    for name, tp, fp, fn in (("prereq_parser+linker", pp_tp, pp_fp, pp_fn), ("course_skill_extractor", tt_tp, tt_fp, tt_fn)):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-12, p + r)
        for m, v in (("precision_vs_authored", p), ("recall_vs_authored", r), ("f1_vs_authored", f)):
            recs.append(record(LAYER, name, m, v, direction="higher", **common, details={"tp": tp, "fp": fp, "fn": fn},
                               caveats=["Authored catalogue fields are the reference; extra extracted skills that are genuinely taught but not listed count as FP (precision is a lower bound).",
                                        "Measured on the enrichment output, which is NOT wired into the graph (see enriched_teaches_pairs_reaching_graph)."]))
    uniq_drift = {(d["authored"], d["canonical_id"]): d for d in pp_id_drift + tt_drift}
    info["linker_id_drift"] = sorted(uniq_drift.values(), key=lambda d: d["authored"])
    # linker over-merges: distinct authored skill names that the linker maps to ONE canonical id (would collapse graph nodes if wired)
    canon_groups = defaultdict(set)
    for ec in enriched.values():
        for it in ec.get("taught_skills_evidence", []):
            if it.get("source") == "declared":
                canon_groups[norm(it.get("canonical_skill_id") or "")].add(it.get("surface_form"))
        for it in ec.get("parsed_prerequisites", []):
            canon_groups[norm(it.get("canonical_skill_id") or "")].add(it.get("skill_name"))
    merges = {k: sorted(v) for k, v in canon_groups.items() if len({norm(x) for x in v}) > 1}
    info["linker_over_merges"] = merges
    recs.append(record(LAYER, "taxonomy_linker", "over_merge_groups_on_authored_skills", float(len(merges)), direction="lower", **common,
                       details={"groups": merges, "authored_names_affected": sum(len(v) for v in merges.values())},
                       caveats=["Distinct authored catalogue skills collapsed onto one canonical id by the linker (e.g. several 'Apache X' tools → 'apache'); "
                                "harmless today only because the enrichment is not wired into the graph."]))
    n_auth_names = len({s for c in env.courses for s in c.teaches_skills + c.prerequisites})
    recs.append(record(LAYER, "taxonomy_linker", "canonical_id_differs_from_authored_name_share",
                       len({d["authored"] for d in uniq_drift.values()}) / max(1, n_auth_names), direction="none", **common,
                       details={"examples": list(uniq_drift.values())[:20], "n_distinct_authored_names": n_auth_names},
                       caveats=["Not every difference is an error (alias canonicalisation is intended); examples are listed for inspection."]))

    # (d) duplicate nodes: over-merge (distinct raw names → one node id) and under-merge (alias collisions between node ids)
    raw_names = {s for c in env.courses for s in c.teaches_skills + c.prerequisites}
    for sc in load_jsonl(SCENARIOS):
        raw_names |= set(sc["target_job"]["required_skills"]) | set(sc["target_job"]["preferred_skills"])
        raw_names |= {s.get("canonical_skill_name") or s.get("surface_form") for s in sc["user_profile"]["skills"]}
    by_id = defaultdict(set)
    for s in raw_names:
        by_id[norm(s)].add(s)
    over = {k: sorted(v) for k, v in by_id.items() if len({x.strip() for x in v}) > 1 and len({x.lower() for x in v}) > 1}
    skill_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Skill"]
    keymap = defaultdict(set)
    for n in skill_nodes:
        for k in resolver.keys_for_name(G.nodes[n].get("canonical_skill_name") or n):
            keymap[k].add(n)
    under = sorted({tuple(sorted(v)) for v in keymap.values() if len(v) > 1})
    info["duplicates"] = {"over_merged_ids": over, "alias_collisions_between_nodes": [list(u) for u in under]}
    recs.append(record(LAYER, "kg_builder(as_wired)", "over_merged_skill_ids", float(len(over)), direction="lower", **common,
                       details={"groups": over}, caveats=["normalize_text maps symbol-bearing names together (e.g. C++/C#/C → 'c'); a merged node conflates distinct skills."]))
    recs.append(record(LAYER, "kg_builder(as_wired)", "alias_duplicate_skill_node_groups", float(len(under)), direction="lower", **common,
                       details={"groups": [list(u) for u in under]}, caveats=["Pairs of distinct Skill nodes that the repo's own SkillKeyResolver treats as the same skill."]))

    # (e) orphans, dangling prerequisites, infeasible courses, cycles
    deg0 = [n for n in G.nodes if G.degree(n) == 0]
    no_teach = [c for c in env.ids if not env.T[c]]
    taught_any = set().union(*env.T.values())
    dangling = sorted({p for P in env.P.values() for p in P} - taught_any)
    known: Set[str] = set()
    feasible: Set[str] = set()
    changed = True
    while changed:
        changed = False
        for c in env.ids:
            if c not in feasible and env.P[c] <= known:
                feasible.add(c)
                known |= env.T[c]
                changed = True
    infeasible = sorted(set(env.ids) - feasible)
    sp = skill_prereq_pairs(G)
    D = nx.DiGraph()
    D.add_edges_from((p, t) for (t, p) in sp)
    sccs = [sorted(s) for s in nx.strongly_connected_components(D) if len(s) > 1]
    self_loops = sorted(c for c in env.ids if env.T[c] & env.P[c])
    info["structure"] = {"orphan_nodes": deg0, "courses_without_teaches": no_teach, "dangling_prereqs": dangling,
                         "infeasible_courses_from_empty_profile": infeasible, "skill_prereq_pairs": len(sp),
                         "nontrivial_sccs": sccs, "courses_teaching_own_prereq": self_loops,
                         "longest_prereq_chain_skills": (nx.dag_longest_path_length(D) if nx.is_directed_acyclic_graph(D) else None)}
    for m, v, dirn in (("orphan_nodes", len(deg0), "lower"), ("courses_without_teaches_edges", len(no_teach), "lower"),
                       ("dangling_prerequisite_skills", len(dangling), "lower"), ("infeasible_courses_from_empty_profile", len(infeasible), "lower"),
                       ("prerequisite_cycles_nontrivial_scc", len(sccs), "lower"), ("courses_teaching_own_prerequisite", len(self_loops), "lower")):
        recs.append(record(LAYER, "kg_builder(as_wired)", m, float(v), direction=dirn, **common, details={"items": info["structure"].get(
            {"orphan_nodes": "orphan_nodes", "courses_without_teaches_edges": "courses_without_teaches", "dangling_prerequisite_skills": "dangling_prereqs",
             "infeasible_courses_from_empty_profile": "infeasible_courses_from_empty_profile", "prerequisite_cycles_nontrivial_scc": "nontrivial_sccs",
             "courses_teaching_own_prerequisite": "courses_teaching_own_prereq"}[m])}))

    # (f) type consistency + provenance completeness (eval conventions from evals/graph_evals.py)
    bad_type = sum(1 for u, v, d in G.edges(data=True) if not (_ntype(G, u) in EDGE_SIGNATURE.get(d.get("type"), (set(), set()))[0]
                                                                 and _ntype(G, v) in EDGE_SIGNATURE.get(d.get("type"), (set(), set()))[1]))
    recs.append(record(LAYER, "kg_builder(as_wired)", "edge_type_consistency", 1 - bad_type / max(1, G.number_of_edges()), direction="higher",
                       **common, details={"type_inconsistent_edges": bad_type, "n_edges": G.number_of_edges()}))
    for et in ("TEACHES", "REQUIRES_PREREQUISITE"):
        E = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("type") == et]
        keys_ok = sum(1 for _, _, d in E if all(k in d for k in PROVENANCE_SPEC[et]))
        ev = EVIDENCE_KEY[et]
        nonempty = sum(1 for _, _, d in E if str(d.get(ev, "")).strip())
        if et == "TEACHES":
            informative = sum(1 for u, v, d in E if str(d.get(ev, "")).strip() and d.get(ev) != env.by_id[u].title)
        else:
            informative = sum(1 for u, v, d in E if str(d.get(ev, "")).strip() and norm(d.get(ev, "")) != v)
        for m, val in ((f"{et.lower()}_provenance_fields_present", keys_ok / max(1, len(E))),
                       (f"{et.lower()}_evidence_nonempty", nonempty / max(1, len(E))),
                       (f"{et.lower()}_evidence_informative", informative / max(1, len(E)))):
            recs.append(record(LAYER, "kg_builder(as_wired)", m, val, direction="higher", **common, details={"n_edges": len(E)},
                               caveats=["'informative' = evidence is not merely the course title (TEACHES) or the prerequisite's own name (REQUIRES_PREREQUISITE)."]))

    # (g) per-scenario user/job ingestion fidelity: weak (beginner) skills keep their mastery on HAS_SKILL edges?
    weak_total = weak_kept = 0
    for sc in load_jsonl(SCENARIOS):
        b = env.scenario_graph(sc)
        uid = sc["user_profile"]["user_id"]
        for s in sc["user_profile"]["skills"]:
            if str(s.get("mastery_level", "")).lower() == "beginner":
                weak_total += 1
                sid = norm(s.get("canonical_skill_name") or s.get("surface_form"))
                eds = (b.graph.get_edge_data(uid, sid) or {}).values()
                weak_kept += any(d.get("mastery_level") == "beginner" for d in eds)
    recs.append(record(LAYER, "kg_builder(as_wired)", "has_skill_mastery_fidelity_for_weak_skills", weak_kept / max(1, weak_total) if weak_total else None,
                       direction="higher", n_units=weak_total, unit="weak user skill", split="all", dataset="benchmark_scenarios.jsonl (48)",
                       provenance=[SCENARIOS], status="measured" if weak_total else "not_available", reason="" if weak_total else "no weak skills",
                       caveats=["ingest_user_profile does not pass mastery_level to add_user_skill, so every HAS_SKILL edge says 'proficient'."]))
    return recs, info


# =====================================================================================================================
# external validity (part 2): public prerequisite datasets
# =====================================================================================================================
def concept_norm(x: str) -> str:
    x = re.sub(r"\(.*?\)", " ", x.replace("_", " ")).lower()
    return re.sub(r"[^a-z0-9+#]+", " ", x).strip()


def load_cpr(root: Path = EXTERNAL / "cpr_recover") -> Dict[str, Any]:
    courses = {}
    with (root / "cs_courses.csv").open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                courses[row[0]] = row[1]
    edges = []
    with (root / "cs_edges.csv").open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                edges.append((row[0].strip(), row[1].strip()))  # row[1] is a prerequisite of row[0]
    pairs = []
    with (root / "cs_annotations.tsv").open(encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd)
        for row in rd:
            labs = [x.strip() for x in row[2:] if x.strip()]
            cnt = Counter(labs)
            maj = [l for l, n in cnt.items() if n >= 2]
            pairs.append({"a": row[0], "b": row[1], "labels": labs, "majority": maj[0] if maj else None})
    preqs = set()
    with (root / "cs_preqs.csv").open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                preqs.add((row[0], row[1]))
    return {"courses": courses, "edges": edges, "pairs": pairs, "preqs": preqs}


def cpr_graph(cpr: Dict[str, Any]):
    """Build the repo's CourseGraphBuilder graph from real courses (D-A5-06 E1)."""
    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
    concepts = sorted({p["a"] for p in cpr["pairs"]} | {p["b"] for p in cpr["pairs"]})
    pats = {c: re.compile(r"(?<![a-z0-9])" + re.escape(concept_norm(c)) + r"(?![a-z0-9])") for c in concepts if concept_norm(c)}
    teaches = {}
    for cid, desc in cpr["courses"].items():
        low = " " + re.sub(r"\s+", " ", desc.lower()) + " "
        teaches[cid] = sorted(c for c, pat in pats.items() if pat.search(low))
    req_courses = defaultdict(set)
    for a, b in cpr["edges"]:
        if a in teaches and b in teaches:
            req_courses[a].add(b)
    b = CourseGraphBuilder()
    for cid in sorted(teaches):
        prereq_concepts = sorted({x for pc in req_courses[cid] for x in teaches[pc]})
        b.ingest_course({"course_id": cid, "title": cpr["courses"][cid].split(".")[0][:120], "description": "",
                         "teaches_skills": teaches[cid], "prerequisites": prereq_concepts, "level": "intermediate"})
    df = Counter(c for cid in teaches for c in teaches[cid])
    return b.graph, teaches, df


def external_e1(n_boot: int, seed: int, env: Optional[Env]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = EXTERNAL / "cpr_recover"
    files = [root / f for f in ("cs_courses.csv", "cs_edges.csv", "cs_annotations.tsv", "cs_preqs.csv")]
    if not all(f.exists() for f in files):
        return [record(LAYER, "kg_prereq_inference", "cpr_f1", None, n_units=0, unit="concept pair", split="external", dataset="CPR-Recover",
                       direction="higher", status="not_available", reason="data/external/cpr_recover missing (see MANIFEST.jsonl)")], {}
    cpr = load_cpr(root)
    G, teaches, df = cpr_graph(cpr)
    sup_all = skill_prereq_pairs(G)
    # minus co-taught variant
    sup_mc: Dict[Tuple[str, str], int] = Counter()
    for c, d in G.nodes(data=True):
        if d.get("type") != "Course":
            continue
        taught = {v for _, v, e in G.out_edges(c, data=True) if e.get("type") == "TEACHES"}
        req = {v for _, v, e in G.out_edges(c, data=True) if e.get("type") == "REQUIRES_PREREQUISITE"} - taught
        for t in taught:
            for p in req:
                sup_mc[(t, p)] += 1
    # check our majority vote reproduces cs_preqs.csv
    maj_pos = {(p["a"], p["b"]) for p in cpr["pairs"] if p["majority"] == "1"} | {(p["b"], p["a"]) for p in cpr["pairs"] if p["majority"] == "2"}
    agree_preqs = len(maj_pos & cpr["preqs"]) / max(1, len(cpr["preqs"]))
    raw: Dict[Tuple[str, str], List[int]] = defaultdict(list)  # directed instance (target, prereq) -> gold labels from all rows
    for p in cpr["pairs"]:
        if p["majority"] is None:
            continue
        for t, q, lab in ((p["a"], p["b"], "1"), (p["b"], p["a"], "2")):
            raw[(t, q)].append(int(p["majority"] == lab))
    # the same unordered pair can be annotated in both orientations: deduplicate, drop conflicting duplicates
    inst = [{"t": t, "p": q, "y": ys[0], "cluster": min(t, q)} for (t, q), ys in sorted(raw.items()) if len(set(ys)) == 1]
    n_conflict = sum(1 for ys in raw.values() if len(set(ys)) > 1)
    nid = lambda c: norm(c)  # node ids in the builder graph are normalize_text(raw concept title)
    enc = None
    if env is not None:
        names = sorted({i["t"] for i in inst} | {i["p"] for i in inst})
        emb = env.dense.model.encode([concept_norm(n) for n in names], normalize_embeddings=True, show_progress_bar=False) if env.dense.model is not None else None
        if emb is not None:
            enc = {n: emb[i] for i, n in enumerate(names)}
    dfn = {c: df.get(c, 0) for c in {i["t"] for i in inst} | {i["p"] for i in inst}}
    rng = random.Random(seed)
    scorers = {
        "graph_rule(repo semantics)": lambda i: float(sup_all.get((nid(i["t"]), nid(i["p"])), 0)),
        "graph_rule_minus_cotaught": lambda i: float(sup_mc.get((nid(i["t"]), nid(i["p"])), 0)),
        "graph_rule_reversed(sanity)": lambda i: float(sup_all.get((nid(i["p"]), nid(i["t"])), 0)),
        "frequency_direction": lambda i: float(dfn[i["p"]] > dfn[i["t"]]) + 1e-3 * (dfn[i["p"]] - dfn[i["t"]]),
        "random(prior)": lambda i: rng.random(),
    }
    if enc is not None:
        scorers["minilm_similarity×frequency_direction"] = lambda i: float(enc[i["t"]] @ enc[i["p"]]) * float(dfn[i["p"]] > dfn[i["t"]])
    binary_rule = {"graph_rule(repo semantics)", "graph_rule_minus_cotaught", "graph_rule_reversed(sanity)", "frequency_direction"}
    clusters = defaultdict(list)
    out_recs: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"n_courses": len(cpr["courses"]), "n_course_edges": len(cpr["edges"]), "n_annotated_pairs": len(cpr["pairs"]),
                               "n_pairs_with_majority": sum(1 for p in cpr["pairs"] if p["majority"]), "n_directed_instances": len(inst),
                               "n_positive_instances": sum(i["y"] for i in inst), "majority_vote_reproduces_cs_preqs": agree_preqs,
                               "n_conflicting_duplicate_instances_dropped": n_conflict,
                               "concepts_matched_in_any_description": sum(1 for c in dfn if dfn[c] > 0) / max(1, len(dfn)),
                               "mean_concepts_per_course": float(np.mean([len(v) for v in teaches.values()])), "systems": {}}
    for k, f in scorers.items():
        for i in inst:
            i[k] = f(i)
    for i in inst:
        clusters[i["cluster"]].append(i)
    units = list(clusters.values())

    def prf(insts, key, thr):
        tp = sum(1 for i in insts if i[key] >= thr and i["y"])
        fp = sum(1 for i in insts if i[key] >= thr and not i["y"])
        fn = sum(1 for i in insts if i[key] < thr and i["y"])
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn

    def ap(insts, key):
        s = sorted(insts, key=lambda i: -i[key])
        pos = sum(i["y"] for i in s)
        if pos == 0:
            return float("nan")
        hits, tot, i0 = 0, 0.0, 0
        # tie-aware AP: average precision over tie groups (each group's precision assigned to its positives)
        while i0 < len(s):
            j = i0
            while j < len(s) and s[j][key] == s[i0][key]:
                j += 1
            grp = s[i0:j]
            gp = sum(x["y"] for x in grp)
            for m in range(1, gp + 1):  # positives spread uniformly inside the tie group (expected-precision approximation)
                rank = i0 + m * len(grp) / max(1, gp)
                tot += (hits + m) / rank
            hits += gp
            i0 = j
        return tot / pos

    flat = lambda us: [i for u in us for i in u]
    for k in scorers:
        a = ap(inst, k)
        a_ci = cluster_boot(units, lambda us: ap(flat(us), k), n_boot=min(n_boot, 1000), seed=seed)
        row = {"ap": a, "ap_ci": a_ci}
        prov = files
        cav = ["Concept→course TEACHES edges come from exact name matching of the ANNOTATED concept vocabulary in course descriptions (oracle vocabulary).",
               "Candidate pairs in CPR-Recover were drawn from course dependencies, so recall is optimistic; precision is the informative number.",
               "CI: cluster bootstrap; cluster = alphabetically first concept of the pair (both directions of a pair and pairs sharing that concept resample together)."]
        out_recs.append(record(LAYER, k, "cpr_average_precision", a, n_units=len(units), unit="concept cluster", split="external",
                               dataset="CPR-Recover (Liang et al. 2017), real CS courses, 11 US universities", direction="higher", ci=a_ci,
                               ci_method="cluster bootstrap over concepts (B≤1000, seed 42)", provenance=prov, caveats=cav,
                               details={"prior_positive_rate": summary["n_positive_instances"] / max(1, len(inst))}))
        if k in binary_rule:
            thr = 1.0 if k.startswith("graph") else 0.5
            p, r, f, tp, fp, fn = prf(inst, k, thr)
            ci_p = cluster_boot(units, lambda us: prf(flat(us), k, thr)[0], n_boot=min(n_boot, 1000), seed=seed)
            ci_r = cluster_boot(units, lambda us: prf(flat(us), k, thr)[1], n_boot=min(n_boot, 1000), seed=seed)
            ci_f = cluster_boot(units, lambda us: prf(flat(us), k, thr)[2], n_boot=min(n_boot, 1000), seed=seed)
            row.update({"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "ci_p": ci_p, "ci_r": ci_r, "ci_f": ci_f})
            for m, v, ci in (("cpr_precision", p, ci_p), ("cpr_recall", r, ci_r), ("cpr_f1", f, ci_f)):
                out_recs.append(record(LAYER, k, m, v, n_units=len(units), unit="concept cluster", split="external",
                                       dataset="CPR-Recover (Liang et al. 2017), real CS courses, 11 US universities", direction="higher", ci=ci,
                                       ci_method="cluster bootstrap over concepts (B≤1000, seed 42)", provenance=prov, caveats=cav,
                                       details={"threshold": f"support ≥ {thr:g}" if k.startswith("graph") else "df(p) > df(t)", "tp": tp, "fp": fp, "fn": fn}))
        summary["systems"][k] = row
    # paired cluster-bootstrap differences in F1: repo graph rule vs its reverse and vs the frequency heuristic
    base = "graph_rule(repo semantics)"
    summary["paired_f1"] = {}
    for other in ("graph_rule_reversed(sanity)", "frequency_direction"):
        thr_o = 1.0 if other.startswith("graph") else 0.5
        diff = lambda us, o=other, t=thr_o: prf(flat(us), base, 1.0)[2] - prf(flat(us), o, t)[2]
        rng_b = np.random.default_rng(seed)
        B = min(n_boot, 1000)
        ds = []
        for _ in range(B):
            idx = rng_b.integers(0, len(units), size=len(units))
            ds.append(diff([units[j] for j in idx]))
        ds = np.asarray(ds)
        d0 = diff(units)
        lo, hi = np.quantile(ds, [0.025, 0.975])
        p = float(min(1.0, 2 * min((np.sum(ds <= 0) + 1) / (B + 1), (np.sum(ds >= 0) + 1) / (B + 1))))
        summary["paired_f1"][other] = {"delta": d0, "ci": [float(lo), float(hi)], "p": p}
        out_recs.append(record(LAYER, f"{base} vs {other}", "cpr_delta_f1", d0, n_units=len(units), unit="concept cluster", split="external",
                               dataset="CPR-Recover (Liang et al. 2017), real CS courses, 11 US universities", direction="higher", ci=(float(lo), float(hi)),
                               ci_method=f"paired cluster bootstrap over concepts (B={B}, seed {seed})", provenance=files, details={"p": p}))
    return out_recs, summary


def external_e2(G_syn) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synthetic catalogue's skill-prerequisite pairs vs independent annotations on overlapping concepts (descriptive)."""
    syn = {(t, p) for (t, p) in skill_prereq_pairs(G_syn)}
    names = {}
    for n, d in G_syn.nodes(data=True):
        if d.get("type") == "Skill":
            names[n] = concept_norm(d.get("canonical_skill_name") or n)
    syn_c = {(names.get(t, t), names.get(p, p)) for t, p in syn}
    out: Dict[str, Any] = {}
    gold_sets: Dict[str, Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[str]]] = {}
    # CPR majority
    cpr_root = EXTERNAL / "cpr_recover"
    if (cpr_root / "cs_annotations.tsv").exists():
        cpr = load_cpr(cpr_root)
        pos, neg, voc = set(), set(), set()
        for p in cpr["pairs"]:
            a, b = concept_norm(p["a"]), concept_norm(p["b"])
            voc |= {a, b}
            if p["majority"] == "1":
                pos.add((a, b)); neg.add((b, a))
            elif p["majority"] == "2":
                pos.add((b, a)); neg.add((a, b))
            elif p["majority"] == "3":
                neg.add((a, b)); neg.add((b, a))
        gold_sets["CPR-Recover"] = (pos, neg, voc)
    al_root = EXTERNAL / "al_cpl/data"
    if al_root.exists():
        pos, neg, voc = set(), set(), set()
        for f in sorted(al_root.glob("*.pairs")):
            pre = {tuple(concept_norm(x) for x in l.strip().split(",")[:2]) for l in (f.with_suffix(".preqs")).open() if l.strip()}
            for l in f.open():
                if not l.strip():
                    continue
                a, b = [concept_norm(x) for x in l.strip().split(",")[:2]]
                voc |= {a, b}
                (pos if (a, b) in pre else neg).add((a, b))
        gold_sets["AL-CPL"] = (pos, neg, voc)
    lb = EXTERNAL / "lecturebank/LB-Paper"
    if (lb / "208topics.csv").exists():
        tid = {row[0]: concept_norm(row[1]) for row in csv.reader((lb / "208topics.csv").open(encoding="utf-8")) if row}
        pos, neg = set(), set()
        for row in csv.reader((lb / "prerequisite_annotation.csv").open(encoding="utf-8")):
            if len(row) >= 3 and row[0] in tid and row[1] in tid:
                src, tgt = tid[row[0]], tid[row[1]]  # src is a prerequisite of tgt
                (pos if row[2].strip() == "1" else neg).add((tgt, src))
        gold_sets["LectureBank1"] = (pos, neg, set(tid.values()))
        t2 = {}
        for line in (lb / "LectureBank2/data/322topics_final.tsv").open(encoding="utf-8"):
            if "|" in line:
                i, n = line.rstrip("\n").split("|", 1)
                t2[i] = concept_norm(n)
        pos, neg = set(), set()
        for row in csv.reader((lb / "LectureBank2/data/final_new_annotation.csv").open(encoding="utf-8")):
            if len(row) >= 3 and row[0] in t2 and row[1] in t2:
                (pos if row[2].strip() == "1" else neg).add((t2[row[1]], t2[row[0]]))
        gold_sets["LectureBank2"] = (pos, neg, set(t2.values()))
    recs = []
    for name, (pos, neg, voc) in gold_sets.items():
        syn_in = {(t, p) for (t, p) in syn_c if t in voc and p in voc}
        agree = syn_in & pos
        contra = syn_in & neg
        unann = syn_in - pos - neg
        ext_pos_in = {(t, p) for (t, p) in pos if t in set(names.values()) and p in set(names.values())}
        recalled = ext_pos_in & syn_c
        out[name] = {"overlap_concepts": len(voc & set(names.values())), "synthetic_pairs_in_vocab": len(syn_in), "agree": sorted(agree),
                     "contradict": sorted(contra), "unannotated": len(unann), "external_positive_pairs_in_overlap": sorted(ext_pos_in),
                     "external_positives_implied_by_synthetic": sorted(recalled)}
        judged = len(agree) + len(contra)
        recs.append(record(LAYER, "synthetic_catalogue_prereq_structure", f"agreement_with_{name}", (len(agree) / judged) if judged else None,
                           n_units=judged, unit="annotated skill pair", split="external", dataset=name, direction="higher",
                           status="measured" if judged else "not_available", reason="" if judged else "no synthetic prerequisite pair falls on an annotated concept pair",
                           provenance=[CATALOG], details=out[name],
                           caveats=["Descriptive only: very few concept pairs overlap; name matching is exact after normalisation."]))
    return recs, out


# =====================================================================================================================
# generated scenario set SG-v1 (D-A5-07) — shared with L09
# =====================================================================================================================
AREA_TITLES = {"data": ["Data Analyst", "Data Engineer"], "ml": ["Machine Learning Engineer", "Data Scientist"],
               "prog": ["Software Developer", "Backend Engineer"], "web": ["Web Developer", "Frontend Engineer"],
               "devops": ["DevOps Engineer", "Site Reliability Engineer"], "cloud": ["Cloud Engineer", "Cloud Architect"],
               "sec": ["Security Analyst", "Security Engineer"], "soft": ["Project Coordinator", "Team Lead"]}


def text_labels(open_needs: Mapping[str, float], catalog_courses: Sequence[Any], resolver=None) -> Dict[str, int]:
    """V-text (D-A5-05): grade from mentions of open gold needs in title + description + outcomes; ignores teaches_skills,
    prerequisites."""
    if resolver is None:
        from skillllm_v2.match.skill_keys import SkillKeyResolver
        resolver = SkillKeyResolver()
    keys = {n: {k for k in ({norm(n)} | {norm(x) for x in resolver.keys_for_name(n)}) if k} for n in open_needs}
    out = {}
    for c in catalog_courses:
        text = " " + norm(" ".join([c.title, c.description] + list(c.learning_outcomes))) + " "
        mass, n_hit = 0.0, 0
        for need, w in open_needs.items():
            if any(f" {k} " in text for k in keys[need]):
                mass += w
                n_hit += 1
        g = 3 if (mass >= 3 and n_hit >= 2) else 2 if mass >= 2 else 1 if mass >= 1 else 0
        if g:
            out[c.course_id] = g
    return out


def generate_sg_v1(catalog_courses: Sequence[Any], n: int = SG_N, seed: int = SG_SEED) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Seeded SYNTHETIC scenarios from the catalogue vocabulary only; labels from the unchanged v1 rule + V-text."""
    from skillllm_v2.course.ingest_catalog import CourseCatalog
    from skillllm_v2.eval.benchmark_scenarios_spec import S
    from skillllm_v2.eval.build_relevance_labels import label_scenario
    cat = CourseCatalog(list(catalog_courses))
    rng = random.Random(seed)
    areas = sorted({c.metadata.get("area") for c in catalog_courses if c.metadata.get("area")})
    pool = {a: sorted({s for c in catalog_courses if c.metadata.get("area") == a for s in c.teaches_skills}) for a in areas}
    found = {a: sorted({s for c in catalog_courses if c.metadata.get("area") == a and not c.prerequisites for s in c.teaches_skills}) for a in areas}
    scens, labels = [], []
    for k in range(n):
        area = rng.choice(areas)
        title = rng.choice(AREA_TITLES.get(area, [f"{area.title()} Specialist"]))
        cand = list(pool[area])
        rng.shuffle(cand)
        n_req, n_pref, n_impl = rng.randint(1, 4), rng.randint(0, 2), rng.randint(0, 1)
        req, pref, impl = cand[:n_req], cand[n_req:n_req + n_pref], cand[n_req + n_pref:n_req + n_pref + n_impl]
        bg = rng.choice(areas)
        fb = list(found[bg]) or list(pool[bg])
        rng.shuffle(fb)
        known = [s for s in fb[:rng.randint(1, 4)] if s not in req + pref + impl]
        if rng.random() < 0.3 and len(req) > 1:
            known.append(rng.choice(req))
        weak = []
        if rng.random() < 0.2:
            opts = [s for s in req + pref if s not in known]
            if opts:
                weak = [rng.choice(opts)]
        sid = f"sg-{k + 1:03d}"
        text = (f"We are hiring a {title}. Required skills: {', '.join(req)}. "
                f"Nice to have: {', '.join(pref) if pref else 'none listed'}.")
        spec = S(sid=sid, background=f"{bg} background", level="mid", known=", ".join(known), job_title=title, text=text,
                 req=", ".join(req), pref=", ".join(pref), implicit=", ".join(impl), weak=", ".join(weak), source="generated_sg_v1")
        lab = label_scenario(spec, cat)
        skills = [{"surface_form": s, "canonical_skill_name": s, "confidence": 1.0} for s in known]
        skills += [{"surface_form": s, "canonical_skill_name": s, "confidence": 1.0, "mastery_level": "beginner"} for s in weak]
        open_needs = {s: w for s, w in lab["needs"].items() if s not in set(known)}
        scens.append({"scenario_id": sid, "title": f"{bg} background -> {title}", "source": "generated_sg_v1", "area": area,
                      "user_profile": {"user_id": f"user-{sid}", "skills": skills, "target_roles": [title]},
                      "target_job": {"job_id": f"job-{sid}", "title": title, "text": text, "required_skills": req, "preferred_skills": pref},
                      "gold": {"needs": lab["needs"], "implicit_not_given_to_systems": impl, "known": known, "weak": weak}})
        labels.append({"scenario_id": sid, "label_version": "v1-rule", "labels": {c: g for c, g in lab["final"].items() if g > 0},
                       "labels_content_only": {c: g for c, g in lab["content_only"].items() if g > 0},
                       "labels_text": text_labels(open_needs, catalog_courses)})
    return scens, labels


def load_or_build_sg(env: Env, smoke: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    scens, labels = generate_sg_v1(env.courses)
    SG_DIR.mkdir(parents=True, exist_ok=True)
    ps, pl = SG_DIR / "sg_v1_scenarios.jsonl", SG_DIR / "sg_v1_labels.jsonl"
    body_s = "\n".join(json.dumps(r, ensure_ascii=False) for r in scens) + "\n"
    body_l = "\n".join(json.dumps(r, ensure_ascii=False) for r in labels) + "\n"
    for p, body in ((ps, body_s), (pl, body_l)):
        if not p.exists() or p.read_text(encoding="utf-8") != body:
            p.write_text(body, encoding="utf-8")
    if smoke:
        scens, labels = scens[:24], labels[:24]
    return scens, labels, [ps, pl]


# =====================================================================================================================
# retrieval evaluation (part 3)
# =====================================================================================================================
SYS_ORDER = ["random", "bm25(q_job)", "dense(q_job)", "bm25", "dense", "skill_overlap", "graph_multihop", "hybrid(bm25+dense)",
             "hybrid+skill_overlap", "hybrid+graph", "repo_CourseRetriever", "pool_RAG(B=20,rerank=dense)", "pool_GraphRAG(B=20,rerank=dense)"]


def scenario_rankings(env: Env, sc: Dict[str, Any], with_repo: bool = True) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    sid = sc["scenario_id"]
    gaps = visible_gaps(sc["user_profile"], sc["target_job"])
    qg, qj = q_gap(sc), q_job(sc)
    b = env.scenario_graph(sc)
    G = b.graph
    R: Dict[str, List[str]] = {}
    R["random"] = env.rank_random(sid)
    R["bm25(q_job)"] = env.rank_bm25(qj)
    R["dense(q_job)"] = env.rank_dense(qj)
    R["bm25"] = env.rank_bm25(qg)
    R["dense"] = env.rank_dense(qg)
    R["skill_overlap"] = env.rank_skill_overlap(gaps)
    g_rank, g_paths = graph_retrieve(G, sc["target_job"]["job_id"], sc["user_profile"]["user_id"], gaps)
    R["graph_multihop"] = g_rank
    R["hybrid(bm25+dense)"] = rrf([R["bm25"], R["dense"]])
    R["hybrid+skill_overlap"] = rrf([R["bm25"], R["dense"], R["skill_overlap"]])
    R["hybrid+graph"] = rrf([R["bm25"], R["dense"], R["graph_multihop"]])
    if with_repo:
        R["repo_CourseRetriever"] = env.run_repo_retriever(sc)
    # pool variant: identical budget B and identical final scorer (MiniLM cosine to q_gap)
    hyb = R["hybrid(bm25+dense)"]
    ds = env.dense_scores(qg)
    rag_pool = hyb[:POOL_B]
    seeds = hyb[:POOL_B // 2]
    gscore = {c: (len(g_rank) - i) for i, c in enumerate(g_rank)}
    expansion = set()
    held = user_known_visible(sc)
    for s in seeds:
        sg = env.T[s] & set(gaps)
        if not sg:
            continue
        for c2 in env.ids:
            if c2 in seeds:
                continue
            if env.T[c2] & sg:
                expansion.add(c2)
            if env.T[c2] & (env.P[s] - held):
                expansion.add(c2)
    exp_sorted = sorted(expansion, key=lambda c: (-gscore.get(c, 0), c))
    gr_pool = list(seeds) + exp_sorted[: POOL_B - len(seeds)]
    for c in hyb:
        if len(gr_pool) >= POOL_B:
            break
        if c not in gr_pool:
            gr_pool.append(c)
    R["pool_RAG(B=20,rerank=dense)"] = sorted(rag_pool, key=lambda c: (-ds[c], c))
    R["pool_GraphRAG(B=20,rerank=dense)"] = sorted(gr_pool, key=lambda c: (-ds[c], c))
    meta = {"n_gaps": len(gaps), "graph_paths": g_paths, "graph": G, "pool_expansion_size": len(expansion)}
    return R, meta


def evaluate_set(env: Env, scens: Sequence[Dict[str, Any]], labels: Mapping[str, Dict[str, Any]], with_repo: bool = True,
                 with_full_paths: bool = True) -> Dict[str, Any]:
    from skillllm_v2.recommend.config import DEFAULT_CONFIG
    per: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(dict)  # label_variant/system -> sid -> metrics
    strata: Dict[str, Dict[str, Any]] = {}
    pathval: Dict[str, Dict[str, Any]] = {}
    for sc in scens:
        sid = sc["scenario_id"]
        lab = labels[sid]
        R, meta = scenario_rankings(env, sc, with_repo=with_repo)
        variants = {"v1": grades_of(lab, "labels"), "content_only": grades_of(lab, "labels_content_only")}
        if "labels_text" in lab:
            variants["v_text"] = grades_of(lab, "labels_text")
        for vname, gr in variants.items():
            for s, ids in R.items():
                per[f"{vname}|{s}"][sid] = retrieval_metrics(ids, gr)
        strata[sid] = {"n_gaps": meta["n_gaps"], "multi_gap": meta["n_gaps"] >= 2, "needs_prereq_hop": needs_prereq_hop(sc, lab, env.P),
                       "pool_expansion_size": meta["pool_expansion_size"]}
        # path validity: graph retriever paths of its top-10, and the Full recommender's explanation paths
        G = meta["graph"]
        gp = [p for c in R["graph_multihop"][:10] for p in meta["graph_paths"].get(c, [])]
        ok = [path_valid(G, p)[0] for p in gp]
        pv = {"graph_retriever_paths": len(gp), "graph_retriever_valid": sum(ok)}
        if with_full_paths:
            recs, Gf, gap = env.run_full(DEFAULT_CONFIG, sc)
            gtypes = {g["canonical_skill_id"]: g["gap_type"] for g in gap["gaps"]}
            fps = full_explanation_paths(recs, sc["target_job"]["job_id"], sc["user_profile"]["user_id"], gtypes)
            strict = [path_valid(Gf, p) for p in fps]
            lenient = [path_valid_lenient_job(Gf, p) for p in fps]
            pv.update({"full_paths": len(fps), "full_valid_strict": sum(1 for v, _ in strict if v), "full_valid": sum(1 for v, _ in lenient if v),
                       "full_text_only_paths": sum(1 for p in fps if p["kind"] == "text_mention"),
                       "full_invalid_reasons": Counter(r.split(":")[0] for v, rs in lenient for r in rs),
                       "full_path_list": [{"kind": p["kind"], "course": p["course"], "edges": [list(e) for e in p["edges"]], "valid_auto": v,
                                           "invalid_reasons": rs} for p, (v, rs) in zip(fps, lenient)]})
        pathval[sid] = pv
    return {"per": per, "strata": strata, "pathval": pathval}


def summarize(per: Mapping[str, Mapping[str, Dict[str, Optional[float]]]], ids: Sequence[str], system: str, metric: str, variant: str = "v1"
              ) -> List[float]:
    rows = per.get(f"{variant}|{system}", {})
    return [rows[i][metric] for i in ids if i in rows and rows[i][metric] is not None]


def paired_vals(per, ids, a: str, b: str, metric: str, variant: str = "v1") -> Tuple[List[float], List[float], List[str]]:
    A, B, keep = [], [], []
    ra, rb_ = per.get(f"{variant}|{a}", {}), per.get(f"{variant}|{b}", {})
    for i in ids:
        if i in ra and i in rb_ and ra[i][metric] is not None and rb_[i][metric] is not None:
            A.append(ra[i][metric]); B.append(rb_[i][metric]); keep.append(i)
    return A, B, keep


# =====================================================================================================================
# figures
# =====================================================================================================================
def _fig_bars(path: Path, rows: List[Dict[str, Any]], title: str, xlabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r["mean"] == r["mean"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(rows) + 1.2))
    y = np.arange(len(rows))
    means = [r["mean"] for r in rows]
    err = [[r["mean"] - r["lo"] for r in rows], [r["hi"] - r["mean"] for r in rows]]
    ax.barh(y, means, xerr=err, color="#4C78A8", alpha=0.85, capsize=3)
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    with path.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)


def _fig_forest(path: Path, rows: List[Dict[str, Any]], title: str, xlabel: str, ref: float = 0.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        ax.plot([r["lo"], r["hi"]], [i, i], color="#333")
        ax.plot([r["delta"]], [i], "o", color="#E45756")
    ax.axvline(ref, color="#888", ls="--", lw=1)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    with path.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# =====================================================================================================================
# main
# =====================================================================================================================
def run(seed: int = 42, n_boot: int = 2000, smoke: bool = False) -> Path:
    env = Env()
    _ = env.pipe
    out_dir = EVALS_ROOT / LAYER / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_head = 200 if smoke else max(n_boot, 10000)
    nb = 200 if smoke else n_boot
    split = json.loads(SPLIT.read_text())
    scen_all = {s["scenario_id"]: s for s in load_jsonl(SCENARIOS)}
    lab_all = {r["scenario_id"]: r for r in load_jsonl(LABELS)}
    test_ids = split["test"][:4] if smoke else split["test"]
    dev_ids = split["dev"][:4] if smoke else split["dev"]
    records: List[Dict[str, Any]] = []
    per_unit: List[Dict[str, Any]] = []

    # ---- 1. graph construction QA ------------------------------------------------------------------------------------
    qa_recs, qa_info = graph_qa(env)
    records += qa_recs

    # ---- 2. external validity --------------------------------------------------------------------------------------
    e1_recs, e1 = external_e1(nb, seed, env)
    records += e1_recs
    e2_recs, e2 = external_e2(env.base_graph)
    records += e2_recs
    records.append(record(LAYER, "retrieval_on_public_catalogue", "ndcg10", None, n_units=0, unit="query", split="external",
                          dataset="public course catalogue", direction="higher", status="not_available",
                          reason="No relevance judgments exist for job-gap queries on a public catalogue; creating them needs human (or LLM, secondary-only) labels. "
                                 "CPR-Recover's 654 real course descriptions are used instead for the prerequisite-semantics test (E1)."))

    # ---- 3. retrieval baselines, H5 ------------------------------------------------------------------------------------
    ids_eval = list(dict.fromkeys(test_ids + dev_ids))
    res = evaluate_set(env, [scen_all[i] for i in ids_eval], lab_all)
    per, strata, pathval = res["per"], res["strata"], res["pathval"]
    prov = BASE_PROV
    systems = [s for s in SYS_ORDER if f"v1|{s}" in per]
    table_rows = []
    for half, ids in (("test", test_ids), ("dev", dev_ids)):
        for s in systems:
            row = {"system": s, "half": half}
            for m, mname in RETRIEVAL_METRICS:
                vals = summarize(per, ids, s, m)
                head = m in ("r10", "ndcg10")
                mean, lo, hi = boot_ci(vals, n_boot=nb_head if head else nb, seed=seed)
                row[mname] = mean
                if half == "test":
                    records.append(record(LAYER, s, {"r10": "recall@10", "ndcg10": "ndcg@10", "p5": "precision@5", "r20": "recall@20",
                                                     "mrr": "mrr@20", "ndcg5": "ndcg@5"}[m], mean, n_units=len(vals), unit="scenario", split="test",
                                          dataset=DATASET, direction="higher", ci=(lo, hi),
                                          ci_method=f"percentile bootstrap over scenarios (B={nb_head if head else nb}, seed {seed})", provenance=prov,
                                          caveats=["Synthetic catalogue and scenarios; labels from the v1 rule over the same catalogue fields the skill-aware systems read (see L09 label audit)."],
                                          details={"headline": bool(head and s == "hybrid+graph"), "query": "q_job" if "q_job" in s else "q_gap",
                                                   "role": "proposed_graphrag" if s == "hybrid+graph" else "baseline"}))
            table_rows.append(row)
        for s in systems:
            for m, _ in RETRIEVAL_METRICS:
                for i in ids:
                    v = per[f"v1|{s}"].get(i, {}).get(m)
                    if v is not None and half == "test":
                        per_unit.append({"layer": LAYER, "family": "retrieval", "system": s, "metric": m, "unit_id": i, "value": v,
                                         **({"reference": "hybrid(bm25+dense)"} if s != "hybrid(bm25+dense)" else {})})

    # H5 family: (a) primary, (b) pool variant, (c) graph vs flat skill overlap — multi-gap test scenarios
    multi = [i for i in test_ids if strata[i]["multi_gap"]]
    fam = [("hybrid+graph", "hybrid(bm25+dense)", "primary: hybrid+graph − hybrid"),
           ("pool_GraphRAG(B=20,rerank=dense)", "pool_RAG(B=20,rerank=dense)", "pool variant: GraphRAG − RAG (B=20, dense rerank)"),
           ("hybrid+graph", "hybrid+skill_overlap", "graph vs flat skill overlap inside the same fusion")]
    h5 = []
    for a, b, label in fam:
        A, B, keep = paired_vals(per, multi, a, b, "ndcg10")
        t = paired_test(A, B, n_boot=nb_head, seed=seed) if keep else None
        sd = float(np.std(np.asarray(A) - np.asarray(B), ddof=1)) if len(keep) > 1 else float("nan")
        mde = (1.96 + 0.84) * sd / math.sqrt(len(keep)) if len(keep) > 1 else float("nan")  # two-sided α=.05, power .8 (normal approx.)
        h5.append({"a": a, "b": b, "label": label, "n": len(keep), "test": t, "sd_diff": sd, "mde80": mde})
    adj = holm([h["test"]["p"] if h["test"] else 1.0 for h in h5])
    for h, pa in zip(h5, adj):
        h["p_holm"] = pa
    prim = h5[0]
    if prim["test"] is None or prim["n"] == 0:
        verdict = "NOT TESTABLE"
    else:
        verdict = "SUPPORTED" if (prim["test"]["delta"] > 0 and prim["p_holm"] < 0.05) else "NOT SUPPORTED"
        if verdict == "NOT SUPPORTED" and prim["test"]["ci_low"] < 0 < prim["test"]["ci_high"]:
            verdict = "NOT SUPPORTED"  # contract: Δ>0 with Holm p<0.05 required; a CI spanning 0 is a failure to support
    rule = "H5: hybrid+graph vs hybrid (no graph) nDCG@10 on multi-gap (≥2 gaps) held-out queries: Δ > 0 with Holm p < 0.05"
    for h in h5:
        t = h["test"]
        if t is None:
            continue
        records.append(record(LAYER, f"{h['a']} vs {h['b']}", "delta_ndcg@10_multigap", t["delta"], n_units=h["n"], unit="scenario", split="test",
                              dataset=DATASET, direction="higher", ci=(t["ci_low"], t["ci_high"]), ci_method=f"paired bootstrap (B={nb_head}) + sign-flip permutation",
                              provenance=prov, details={"family": "H5", "p_boot": t["p_boot"], "p_perm": t["p_perm"], "p_raw": t["p"], "p_holm": h["p_holm"],
                                                        "wins": t["wins"], "losses": t["losses"], "ties": t["ties"], "comparison": h["label"],
                                                        "sd_paired_diff": h["sd_diff"], "mde_80pct_power": h["mde80"]}))
    pt = prim["test"]
    records.append(record(LAYER, "hybrid+graph vs hybrid(bm25+dense)", "H5_verdict", pt["delta"] if pt else None, n_units=prim["n"], unit="scenario",
                          split="test", dataset=DATASET, direction="higher", ci=(pt["ci_low"], pt["ci_high"]) if pt else None,
                          ci_method=f"paired bootstrap (B={nb_head}) + sign-flip permutation (10000)", provenance=prov,
                          status="measured" if pt else "not_available", reason="" if pt else "no multi-gap test scenarios",
                          caveats=["All 24 test scenarios have ≥4 system-visible gaps: the 1-gap stratum is empty on the original benchmark.",
                                   "Graph hop 1 equals flat skill overlap here because the KG's TEACHES edges are a 1:1 projection of the catalogue field."],
                          details={"verdict": verdict, "rule": rule, "p_raw": pt["p"] if pt else None, "p_holm": prim["p_holm"], "family": "H5",
                                   "p_boot": pt["p_boot"] if pt else None, "p_perm": pt["p_perm"] if pt else None,
                                   "family_members": [{"comparison": h["label"], "delta": h["test"]["delta"] if h["test"] else None, "p_holm": h["p_holm"]} for h in h5]}))
    # one (family="H5", bucket=<comparison>) group per family member so L10 re-tests exactly the declared pairs, Holm over the 3 members
    for (a, b, _lab), bucket in zip(fam, ("primary", "pool_variant", "graph_vs_flat_overlap")):
        for i in multi:
            for s in (a, b):
                v = per[f"v1|{s}"][i]["ndcg10"]
                if v is not None:
                    per_unit.append({"layer": LAYER, "family": "H5", "bucket": bucket, "system": s, "metric": "ndcg10", "unit_id": i, "value": v,
                                     **({"reference": b} if s == a else {})})

    # strata (test): needs-prereq-hop yes/no; label variants
    strat_rows = []
    for sname, cond in (("multi-gap (≥2)", lambda i: strata[i]["multi_gap"]), ("1-gap", lambda i: not strata[i]["multi_gap"]),
                        ("needs prerequisite hop", lambda i: strata[i]["needs_prereq_hop"]), ("no prerequisite hop", lambda i: not strata[i]["needs_prereq_hop"])):
        ids = [i for i in test_ids if cond(i)]
        for a, b, label in fam[:1] + fam[2:]:
            A, B, keep = paired_vals(per, ids, a, b, "ndcg10")
            if keep:
                t = paired_test(A, B, n_boot=nb, seed=seed)
                strat_rows.append({"name": f"{sname}: {label}", "n": len(keep), "delta": t["delta"], "lo": t["ci_low"], "hi": t["ci_high"], "p": t["p"]})
                records.append(record(LAYER, f"{a} vs {b}", f"delta_ndcg@10[{sname}]", t["delta"], n_units=len(keep), unit="scenario", split="test",
                                      dataset=DATASET, direction="higher", ci=(t["ci_low"], t["ci_high"]), ci_method=f"paired bootstrap (B={nb})",
                                      provenance=prov, details={"stratum": sname, "p": t["p"], "family": "H5-strata (descriptive, no Holm)"}))
            else:
                strat_rows.append({"name": f"{sname}: {label}", "n": 0, "delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")})
                records.append(record(LAYER, f"{a} vs {b}", f"delta_ndcg@10[{sname}]", None, n_units=0, unit="scenario", split="test", dataset=DATASET,
                                      direction="higher", status="not_available", reason=f"stratum '{sname}' is empty in the test half"))
    sens_rows = []
    for variant in ("content_only", "v_text"):
        A, B, keep = paired_vals(per, test_ids, "hybrid+graph", "hybrid(bm25+dense)", "ndcg10", variant)
        if keep:
            t = paired_test(A, B, n_boot=nb, seed=seed)
            sens_rows.append({"name": f"labels={variant}", "n": len(keep), "delta": t["delta"], "lo": t["ci_low"], "hi": t["ci_high"], "p": t["p"]})
            records.append(record(LAYER, "hybrid+graph vs hybrid(bm25+dense)", f"delta_ndcg@10[labels={variant}]", t["delta"], n_units=len(keep), unit="scenario",
                                  split="test", dataset=DATASET, direction="higher", ci=(t["ci_low"], t["ci_high"]), ci_method=f"paired bootstrap (B={nb})",
                                  provenance=prov, details={"p": t["p"], "family": "H5-label-sensitivity (descriptive)"}))

    # path validity (test)
    gp = sum(pathval[i]["graph_retriever_paths"] for i in test_ids)
    gv = sum(pathval[i]["graph_retriever_valid"] for i in test_ids)
    fpaths = sum(pathval[i].get("full_paths", 0) for i in test_ids)
    fval = sum(pathval[i].get("full_valid", 0) for i in test_ids)
    fstrict = sum(pathval[i].get("full_valid_strict", 0) for i in test_ids)
    per_sc_full = [pathval[i]["full_valid"] / pathval[i]["full_paths"] for i in test_ids if pathval[i].get("full_paths")]
    m_, lo_, hi_ = boot_ci(per_sc_full, n_boot=nb_head, seed=seed)
    reasons = sum((pathval[i].get("full_invalid_reasons", Counter()) for i in test_ids), Counter())
    records.append(record(LAYER, "SkillLLM_v2 Full recommender (explanation paths)", "path_validity", fval / max(1, fpaths), n_units=len(per_sc_full), unit="scenario",
                          split="test", dataset=DATASET, direction="higher", ci=(lo_, hi_), ci_method=f"bootstrap over scenarios of per-scenario validity (B={nb_head}); value = micro share",
                          provenance=prov, details={"headline": True, "n_paths": fpaths, "valid": fval, "valid_strict_REQUIRES_only": fstrict,
                                                    "per_scenario_mean": m_, "invalid_reasons": dict(reasons),
                                                    "text_only_paths": sum(pathval[i].get("full_text_only_paths", 0) for i in test_ids)},
                          caveats=["A path is valid iff every claimed edge exists with the claimed type and type-consistent endpoints, and every claimed absence (user lacks skill) holds.",
                                   "Job→Skill hops are worded 'requires'; REQUIRES or PREFERS edges are accepted (strict REQUIRES-only count in details).",
                                   "text_mention_only paths have a non-graph hop that cannot be checked as an edge (only their Job→Skill edge is checked)."]))
    records.append(record(LAYER, "graph_multihop (A5 retriever)", "path_validity", gv / max(1, gp), n_units=len(test_ids), unit="scenario", split="test",
                          dataset=DATASET, direction="higher", provenance=prov, details={"n_paths": gp, "valid": gv},
                          caveats=["Paths are read off the graph by construction; 1.0 is a sanity check of the checker, not a finding."]))
    for i in test_ids:
        if pathval[i].get("full_paths"):
            per_unit.append({"layer": LAYER, "family": "path_validity", "system": "SkillLLM_v2 Full", "metric": "path_validity", "unit_id": i,
                             "value": pathval[i]["full_valid"] / pathval[i]["full_paths"]})

    # ---- 3b. generated set SG-v1 (held out entirely) -----------------------------------------------------------------
    sg_s, sg_l, sg_prov = load_or_build_sg(env, smoke=smoke)
    sg_lab = {r["scenario_id"]: r for r in sg_l}
    sg = evaluate_set(env, sg_s, sg_lab, with_repo=False, with_full_paths=False)
    sg_ids = [s["scenario_id"] for s in sg_s]
    sg_rows = []
    sg_strata = []
    for sname, cond in (("all", lambda i: True), ("multi-gap (≥2)", lambda i: sg["strata"][i]["multi_gap"]),
                        ("1-gap", lambda i: not sg["strata"][i]["multi_gap"]),
                        ("needs prerequisite hop", lambda i: sg["strata"][i]["needs_prereq_hop"]),
                        ("no prerequisite hop", lambda i: not sg["strata"][i]["needs_prereq_hop"])):
        ids = [i for i in sg_ids if cond(i)]
        for a, b, label in fam:
            A, B, keep = paired_vals(sg["per"], ids, a, b, "ndcg10")
            if not keep:
                continue
            t = paired_test(A, B, n_boot=nb, seed=seed)
            sg_strata.append({"name": f"SG {sname}: {label}", "n": len(keep), "delta": t["delta"], "lo": t["ci_low"], "hi": t["ci_high"], "p": t["p"]})
            records.append(record(LAYER, f"{a} vs {b}", f"delta_ndcg@10[SG-v1 {sname}]", t["delta"], n_units=len(keep), unit="generated scenario",
                                  split="generated_heldout", dataset="SG-v1 (240 generated SYNTHETIC scenarios, D-A5-07)", direction="higher",
                                  ci=(t["ci_low"], t["ci_high"]), ci_method=f"paired bootstrap (B={nb}) + sign-flip permutation", provenance=sg_prov + [CATALOG],
                                  details={"p_boot": t["p_boot"], "p_perm": t["p_perm"], "p": t["p"], "family": "H5-SG (secondary; never pooled with test)"}))
    for s in systems:
        if f"v1|{s}" not in sg["per"]:
            continue
        mean, lo, hi = boot_ci(summarize(sg["per"], sg_ids, s, "ndcg10"), n_boot=nb, seed=seed)
        sg_rows.append({"name": s, "mean": mean, "lo": lo, "hi": hi})
        records.append(record(LAYER, s, "ndcg@10[SG-v1]", mean, n_units=len(sg_ids), unit="generated scenario", split="generated_heldout",
                              dataset="SG-v1 (240 generated SYNTHETIC scenarios, D-A5-07)", direction="higher", ci=(lo, hi),
                              ci_method=f"percentile bootstrap (B={nb})", provenance=sg_prov + [CATALOG]))
        for i in sg_ids:
            v = sg["per"][f"v1|{s}"][i]["ndcg10"]
            if v is not None:
                per_unit.append({"layer": LAYER, "family": "retrieval_SG-v1", "system": s, "metric": "ndcg10", "unit_id": i, "value": v,
                                 **({"reference": "hybrid(bm25+dense)"} if s != "hybrid(bm25+dense)" else {})})

    # ---- figures -------------------------------------------------------------------------------------------------------
    bars = []
    for s in systems:
        mean, lo, hi = boot_ci(summarize(per, test_ids, s, "ndcg10"), n_boot=nb, seed=seed)
        bars.append({"name": s, "mean": mean, "lo": lo, "hi": hi})
    _fig_bars(out_dir / "fig_retrieval_ndcg10_test.png", bars, "L08 retrieval, held-out test half (n=%d): nDCG@10, 95%% bootstrap CI" % len(test_ids), "nDCG@10")
    forest = [{"name": f"TEST {h['label']}", "n": h["n"], "delta": h["test"]["delta"], "lo": h["test"]["ci_low"], "hi": h["test"]["ci_high"], "p_holm": h["p_holm"]}
              for h in h5 if h["test"]]
    forest += [{**r, "p_holm": float("nan")} for r in strat_rows if r["n"]] + [{**r, "p_holm": float("nan")} for r in sens_rows]
    forest += [{**r, "p_holm": float("nan")} for r in sg_strata]
    forest = [{k: r.get(k) for k in ("name", "n", "delta", "lo", "hi", "p_holm")} for r in forest]
    if forest:
        _fig_forest(out_dir / "fig_h5_forest.png", forest, "H5: Δ nDCG@10 (graph − no graph), paired bootstrap 95% CI", "Δ nDCG@10")
    if sg_rows:
        _fig_bars(out_dir / "fig_retrieval_ndcg10_sg_v1.png", sg_rows, "L08 retrieval on SG-v1 (n=%d generated): nDCG@10" % len(sg_ids), "nDCG@10")
    if e1:
        e1_rows = [{"name": k, "mean": v["ap"], "lo": v["ap_ci"][0], "hi": v["ap_ci"][1]} for k, v in e1["systems"].items()]
        _fig_bars(out_dir / "fig_cpr_prereq_ap.png", e1_rows, "E1 CPR-Recover (real courses): concept-prerequisite average precision", "average precision")

    with (out_dir / "per_unit_scores.jsonl").open("w") as fh:
        for r in per_unit:
            fh.write(json.dumps(r) + "\n")
    with (out_dir / "paths_for_human_eval.jsonl").open("w") as fh:  # consumed by A6 (L11) if present
        for i in test_ids:
            for p in pathval[i].get("full_path_list", []):
                fh.write(json.dumps({"query_id": i, "system": "SkillLLM_v2 Full (DEFAULT_CONFIG)", "kind": p["kind"], "course_id": p["course"],
                                     "path": [{"src": u, "rel": t, "dst": v} for u, t, v in p["edges"]], "valid_auto": p["valid_auto"],
                                     "invalid_reasons_auto": p["invalid_reasons"]}) + "\n")
    (out_dir / "graph_qa_details.json").write_text(json.dumps(qa_info, indent=2, default=list))
    (out_dir / "external_validity_details.json").write_text(json.dumps({"E1_cpr": e1, "E2_overlap": e2}, indent=2, default=list))

    # ---- report -----------------------------------------------------------------------------------------------------------
    md = build_report(records, table_rows, h5, verdict, strat_rows, sens_rows, sg_strata, sg_rows, qa_info, e1, e2, test_ids, strata, pathval, smoke, nb_head)
    write_layer(LAYER, records, md, seed=seed, smoke=smoke, extra_meta={"n_boot": nb, "n_boot_headline": nb_head})
    return out_dir


def _f(x: Optional[float], d: int = 3) -> str:
    return "-" if x is None or x != x else f"{x:.{d}f}"


def build_report(records, table_rows, h5, verdict, strat_rows, sens_rows, sg_strata, sg_rows, qa, e1, e2, test_ids, strata, pathval, smoke, nb_head) -> str:
    R = {(r["system"], r["metric"]): r for r in records}
    L = ["# L08 — Knowledge graph construction, external validity, retrieval and H5", "",
         "Owner A5. Definitions pre-registered in `Decisions.md` D-A5-01..07 before any run. Unit = scenario; 95% percentile bootstrap CIs; "
         f"headline metrics B={nb_head}. Nothing is tuned (RRF k=60 equal weights, λ=0.5 fixed). All course/scenario data are SYNTHETIC "
         "except the external CPR-Recover / AL-CPL / LectureBank annotations.", ""]
    hp = h5[0]
    t = hp["test"]
    L += ["## Headline", "",
          f"- **H5 verdict: {verdict}.** hybrid+graph − hybrid nDCG@10 on the {hp['n']} multi-gap test scenarios = {_f(t['delta'] if t else None)} "
          f"[{_f(t['ci_low'] if t else None)}, {_f(t['ci_high'] if t else None)}], Holm p = {_f(hp['p_holm'])} (W/L/T {t['wins']}/{t['losses']}/{t['ties']})." if t else "- H5 not testable.",
          f"- Full recommender explanation-path validity (test): {_f(R.get(('SkillLLM_v2 Full recommender (explanation paths)', 'path_validity'), {}).get('value'))} "
          f"of {R.get(('SkillLLM_v2 Full recommender (explanation paths)', 'path_validity'), {}).get('details', {}).get('n_paths')} claimed paths.",
          f"- The knowledge graph is a 1:1 projection of the catalogue's authored `teaches_skills`/`prerequisites`; the extraction/linking/parsing "
          f"enrichment never reaches it: of the {qa['enrichment']['enrichment_only_pairs']} skill/prerequisite pairs the enrichment adds or renames, "
          f"{_f(qa['enrichment']['share_enrichment_only_pairs_in_graph'])} appear in the graph; all {qa['enrichment']['teaches_edges_total']} TEACHES edges "
          f"carry the course title as their only 'evidence' ({qa['enrichment']['teaches_edges_with_course_title_as_evidence']}).",
          f"- Taxonomy-linker over-merges on authored skills (would collapse distinct nodes if wired): {len(qa.get('linker_over_merges', {}))} groups, "
          f"e.g. {list(qa.get('linker_over_merges', {}).items())[:4]}.", ""]
    L += ["## 1. Graph construction QA (171-course catalogue graph)", "",
          f"Node types {qa['node_types']}; edge types {qa['edge_types']}.", ""]
    qa_rows = [{"system": r["system"], "metric": r["metric"], "value": r["value"]} for r in records if r["unit"] == "course" or r["metric"].startswith("has_skill")]
    L += [md_table(qa_rows, ["system", "metric", "value"]), ""]
    st = qa["structure"]
    L += [f"- Over-merged ids (distinct names → one node): {qa['duplicates']['over_merged_ids'] or 'none'}.",
          f"- Alias-duplicate Skill node groups (repo's own SkillKeyResolver says same skill): {qa['duplicates']['alias_collisions_between_nodes'] or 'none'}.",
          f"- Dangling prerequisite skills (no course teaches them): {st['dangling_prereqs'] or 'none'}; infeasible courses from an empty profile: {st['infeasible_courses_from_empty_profile'] or 'none'}; "
          f"non-trivial prerequisite SCCs: {st['nontrivial_sccs'] or 'none'}; courses teaching their own prerequisite: {st['courses_teaching_own_prereq'] or 'none'}; "
          f"skill-level prerequisite pairs: {st['skill_prereq_pairs']}; longest prerequisite chain (skills): {st['longest_prereq_chain_skills']}.",
          f"- Linker id drift examples (authored name → canonical id): {[(d['authored'], d['canonical_id']) for d in qa['linker_id_drift'][:12]]}.", ""]
    L += ["## 2. External validity on public prerequisite data", ""]
    if e1:
        L += [f"**E1 — CPR-Recover (real CS courses, 11 US universities; CC BY-NC-SA 4.0).** {e1['n_courses']} courses, {e1['n_course_edges']} course-prerequisite edges, "
              f"{e1['n_pairs_with_majority']} majority-labelled concept pairs → {e1['n_directed_instances']} directed instances ({e1['n_positive_instances']} positive). "
              f"Our majority vote reproduces {100 * e1['majority_vote_reproduces_cs_preqs']:.1f}% of the released cs_preqs.csv. "
              f"{100 * e1['concepts_matched_in_any_description']:.1f}% of annotated concepts were string-matched in ≥1 description.", "",
              "| system | AP [95% CI] | precision [CI] | recall [CI] | F1 [CI] |", "|---|---|---|---|---|"]
        for k, v in e1["systems"].items():
            pr = f"{_f(v.get('precision'))} [{_f(v.get('ci_p', (None, None))[0])}, {_f(v.get('ci_p', (None, None))[1])}]" if "precision" in v else "-"
            rc = f"{_f(v.get('recall'))} [{_f(v.get('ci_r', (None, None))[0])}, {_f(v.get('ci_r', (None, None))[1])}]" if "recall" in v else "-"
            f1 = f"{_f(v.get('f1'))} [{_f(v.get('ci_f', (None, None))[0])}, {_f(v.get('ci_f', (None, None))[1])}]" if "f1" in v else "-"
            L.append(f"| {k} | {_f(v['ap'])} [{_f(v['ap_ci'][0])}, {_f(v['ap_ci'][1])}] | {pr} | {rc} | {f1} |")
        L += ["", "Paired F1 differences (repo graph rule minus …, cluster bootstrap): " + "; ".join(
            f"{k}: {v['delta']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], p={v['p']:.3f}" for k, v in e1.get("paired_f1", {}).items()) + ". "
              f"Prior positive rate = {e1['n_positive_instances'] / max(1, e1['n_directed_instances']):.3f}; {e1.get('n_conflicting_duplicate_instances_dropped', 0)} "
              "conflicting duplicate directed instances (pair annotated in both orientations with different majorities) dropped."]
        L += ["", "Reading: the graph rule (p ⇒ t iff a course teaching t requires a course teaching p) is the repo's prerequisite semantics applied to real data; "
              "the reversed rule is a sanity check (a meaningful rule must beat its reverse). Concept→course edges use the annotated vocabulary (oracle) and "
              "candidate pairs were generated from course dependencies, so recall is optimistic.", ""]
    L += ["**E2 — does the synthetic catalogue's prerequisite structure agree with independent annotations?**", "",
          "| dataset | overlapping concepts | synthetic pairs in vocab | agree | contradict | unannotated | external positives in overlap | implied by synthetic |", "|---|---|---|---|---|---|---|---|"]
    for k, v in e2.items():
        L.append(f"| {k} | {v['overlap_concepts']} | {v['synthetic_pairs_in_vocab']} | {len(v['agree'])} | {len(v['contradict'])} | {v['unannotated']} | "
                 f"{len(v['external_positive_pairs_in_overlap'])} | {len(v['external_positives_implied_by_synthetic'])} |")
    L += ["", "Descriptive only (tiny overlap). Retrieval on a public course catalogue: **not available** (no relevance judgments for job-gap queries).", ""]
    L += ["## 3. Retrieval baselines (q_gap unless marked q_job; v1 labels, relevant = grade ≥ 2)", ""]
    for half in ("test", "dev"):
        rows = [r for r in table_rows if r["half"] == half]
        L += [f"### {half} half ({'held-out' if half == 'test' else 'TUNED-ON for the Full recommender; shown for context only'})", "",
              md_table(rows, ["system"] + [m for _, m in RETRIEVAL_METRICS]), ""]
    test_rows = [r for r in records if r["split"] == "test" and r["metric"] in ("ndcg@10", "recall@10") and r["ci_low"] is not None]
    L += ["95% CIs (test): " + "; ".join(f"{r['system']} {r['metric']} {_f(r['value'])} [{_f(r['ci_low'])}, {_f(r['ci_high'])}]" for r in test_rows), ""]
    L += ["## 4. Controlled RAG vs GraphRAG — H5", "", f"Rule: {R[('hybrid+graph vs hybrid(bm25+dense)', 'H5_verdict')]['details']['rule']}.", "",
          "| comparison (multi-gap test) | n | Δ nDCG@10 [95% CI] | p (max of boot, perm) | Holm p |", "|---|---|---|---|---|"]
    for h in h5:
        tt = h["test"]
        if tt:
            L.append(f"| {h['label']} | {h['n']} | {tt['delta']:+.3f} [{tt['ci_low']:+.3f}, {tt['ci_high']:+.3f}] | {tt['p']:.3f} | {h['p_holm']:.3f} |")
    L += ["", "Minimum detectable effect at 80% power (two-sided α = 0.05, normal approximation from the paired SD): " + "; ".join(
        f"{h['label']}: {_f(h.get('mde80'))} (SD of paired differences {_f(h.get('sd_diff'))})" for h in h5) + ".",
          "", "Reading the family: (a) adds a structured skill signal AND graph structure to a text-only hybrid, so a gain in (a) alone does not show that "
          "graph structure helps; (c) holds the structured skill signal fixed and isolates the graph-specific part (the prerequisite hop).",
          "", f"**Verdict: {verdict}.**", "", "Strata and label sensitivity (descriptive, no multiplicity correction):", "",
          "| stratum / variant | n | Δ [95% CI] | p |", "|---|---|---|---|"]
    for r in strat_rows + sens_rows + sg_strata:
        L.append(f"| {r['name']} | {r['n']} | {_f(r['delta'])} [{_f(r['lo'])}, {_f(r['hi'])}] | {_f(r['p'])} |")
    nh = sum(1 for i in test_ids if strata[i]["needs_prereq_hop"])
    L += ["", f"Test half: {len(test_ids)} scenarios, all multi-gap; {nh} need a prerequisite hop. SG-v1 (240 generated, held out, D-A5-07) populates the 1-gap "
          "and no-hop strata; it is author-generated and never pooled with the test half.", ""]
    if sg_rows:
        L += ["SG-v1 nDCG@10 by system: " + "; ".join(f"{r['name']} {_f(r['mean'])} [{_f(r['lo'])}, {_f(r['hi'])}]" for r in sg_rows), ""]
    pv = R.get(("SkillLLM_v2 Full recommender (explanation paths)", "path_validity"), {})
    L += ["## 5. Multi-hop path validity", "",
          f"- Full recommender explanation paths (test): {_f(pv.get('value'))} valid of {pv.get('details', {}).get('n_paths')} "
          f"(strict REQUIRES-only wording: {pv.get('details', {}).get('valid_strict_REQUIRES_only')}); invalid reasons: {pv.get('details', {}).get('invalid_reasons')}; "
          f"text-only paths (non-graph hop): {pv.get('details', {}).get('text_only_paths')}.",
          f"- A5 graph retriever paths: {_f(R.get(('graph_multihop (A5 retriever)', 'path_validity'), {}).get('value'))} (sanity check; by construction).", ""]
    L += ["## Caveats", "",
          "- Synthetic catalogue and scenarios; v1 labels read the same catalogue fields the skill-aware systems read (quantified in L09's label audit).",
          "- In this repo the graph's TEACHES edges equal the catalogue field, so graph hop 1 ≡ flat skill overlap; the only graph-specific signal is the prerequisite hop.",
          "- n = 24 test scenarios; scenarios share a catalogue and archetypes; bootstrap ignores label uncertainty.",
          "- External E1 uses exact name matching in place of Wikipedia-miner and the annotated vocabulary as an oracle concept list.", "",
          "## Reproduce", "", "```", "cd SkillLLM_v2_stage2 && PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.graph_retrieval --seed 42 --n-boot 2000", "```"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = run(seed=a.seed, n_boot=a.n_boot, smoke=a.smoke)
    print("wrote", rel(out))


if __name__ == "__main__":
    main()
