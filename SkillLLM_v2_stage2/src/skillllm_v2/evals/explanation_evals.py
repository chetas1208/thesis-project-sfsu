"""Explanation groundedness / faithfulness evals (module "explain"), no LLM.

Two explainers are audited:

* ``recommend.CourseExplanationGenerator`` (graph-backed course explanations), driven end to end on synthetic worlds
  through the real GapEngine -> CourseRetriever -> CourseRanker -> explainer pipeline;
* ``match.generate_recommendation_explanation`` (dict-in / dict-out job-match explanation), driven from structured
  inputs built from the same worlds.

The AUDIT is deterministic code (regex parsing of the generators' fixed sentence templates + exact graph/input lookups).
It is NOT a semantic judge: it can only verify claims it can parse, and it reports the share of explanation steps it
could not parse (``parse_coverage``) so that "0 unsupported claims" is never read as "everything verified".

Definitions
-----------
claim        one atomic assertion extracted from an explanation (job requires skill, course teaches skill, user lacks /
             has skill, prerequisite edge, outcome evidence text, gap label, ...).
supported    the claim maps to a graph edge/node attribute (course explainer) or to the structured inputs (match
             explainer).  unsupported = parsed but contradicted or absent.  Static boilerplate (the match explainer's
             ``limitations``) is counted separately as ``static`` and never as supported.
mutation     a controlled corruption of the graph (remove / swap an edge) or of the explanation (swap a skill or course);
             sensitivity = share of applicable mutations whose NEW unsupported claims the audit detects.  A control
             mutation on an unrelated edge must not trigger the audit (false-alarm rate).
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .contract import EvalContext, EvalResult, bootstrap_ci
from .graph_evals import build_course_graph, fixed_graph_clock, make_world

MODULE = "explain"
DATASET = "synthetic_worlds_v1 (seeded generator, evals/graph_evals.py)"

REQUIRED_TOP_FIELDS = ("course_id", "title", "rank", "score", "covered_missing_skills", "prerequisite_status", "evidence",
                       "explanation_path", "limitations", "next_profile_update_action")
REQUIRED_EVIDENCE_FIELDS = ("skill", "job_requirement_evidence", "gap_classification", "user_evidence", "course_learning_outcome_evidence")

Claim = Dict[str, Any]  # {"kind", "text", "status": supported|unsupported|unverifiable|static, "reason"}


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def _res(name: str, metric: str, vals: Sequence[float], ctx: EvalContext, details: Optional[Dict[str, Any]] = None,
         caveats: Optional[List[str]] = None, dataset: str = DATASET, status: str = "ok", reason: str = "") -> EvalResult:
    vals = [v for v in vals if v == v]
    if not vals:
        return EvalResult(name, MODULE, metric, None, 0, dataset, status="skipped", reason=reason or "no units", details=details or {}, caveats=caveats or [])
    ci = bootstrap_ci(vals, n_boot=ctx.n_boot, seed=ctx.seed) if len(vals) > 1 else None
    return EvalResult(name, MODULE, metric, _mean(vals), len(vals), dataset, ci=ci, details=details or {}, caveats=caveats or [], status=status, reason=reason)


# --------------------------------------------------------------------------------------------------------------
# Graph lookups used by the audit
# --------------------------------------------------------------------------------------------------------------


class GraphIndex:
    """Read-only lookups over a CourseGraphBuilder graph (title -> node, name -> skill id, typed edges)."""

    def __init__(self, graph: Any) -> None:
        self.g = graph
        self.title_to_nodes: Dict[str, List[str]] = {}
        self.name_to_skill: Dict[str, List[str]] = {}
        for n, d in graph.nodes(data=True):
            if d.get("type") in ("Course", "Job"):
                self.title_to_nodes.setdefault(str(d.get("title", n)), []).append(n)
            if d.get("type") == "Skill":
                self.name_to_skill.setdefault(str(d.get("canonical_skill_name", n)), []).append(n)

    def has_edge_type(self, u: str, v: str, types: Iterable[str]) -> bool:
        if not self.g.has_edge(u, v):
            return False
        ts = set(types)
        return any(d.get("type") in ts for d in self.g.get_edge_data(u, v).values())

    def edge_attrs(self, u: str, v: str, typ: str) -> List[Dict[str, Any]]:
        if not self.g.has_edge(u, v):
            return []
        return [d for d in self.g.get_edge_data(u, v).values() if d.get("type") == typ]

    def node_by_title(self, title: str, node_type: str) -> Optional[str]:
        for n in self.title_to_nodes.get(title, []):
            if self.g.nodes[n].get("type") == node_type:
                return n
        return None

    id_fallbacks = 0  # mentions that only resolved because the prose used the raw node id instead of canonical_skill_name
    mentions = 0

    def skill_by_name(self, name: str) -> Optional[str]:
        self.mentions += 1
        ids = self.name_to_skill.get(name)
        if ids:
            return ids[0]
        if name in self.g and self.g.nodes[name].get("type") == "Skill":
            self.id_fallbacks += 1
            return name
        return None


def _c(kind: str, text: str, status: str, reason: str = "", **extra: Any) -> Claim:
    d: Claim = {"kind": kind, "text": text, "status": status, "reason": reason}
    d.update(extra)
    return d


# --------------------------------------------------------------------------------------------------------------
# Audit of CourseExplanationGenerator output against the graph
# --------------------------------------------------------------------------------------------------------------

_RE_REQ = re.compile(r"^(?:Also enables: )?Target role '(.*)' requires '(.*)'$")
_RE_USER = re.compile(r"^User is (.*)$")
_RE_TEACH = re.compile(r"^Course '(.*)' teaches '(.*?)'(?: via outcome: \"(.*)\")?$")
_RE_TEXTREL = re.compile(r"^Course '(.*)' is textually related to '(.*?)' \(.*\); '(.*)' itself is not listed among the skills it teaches$")
_RE_PREREQ = re.compile(r"^Course '(.*)' requires prerequisite '(.*)'$")
_RE_COMES_FIRST = re.compile(r"^'(.*)' teaches '(.*?)' and therefore comes first \(.*\)$")
_RE_PREQ_OK = re.compile(r"^Prerequisite requirements satisfied: (.*)\.$")
_RE_PREQ_MISS = re.compile(r"^Caution: Missing prerequisite skill\(s\): (.*)\.$")
_RE_NOPREQ = re.compile(r"^Course has no prerequisite requirements\.$")
_RE_PAREN = re.compile(r"\s*\(taught by earlier recommended course [^)]*\)")


def audit_course_explanation(expl: Dict[str, Any], idx: GraphIndex, user_id: str, job_id: str, gap_by_id: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Claim-to-evidence audit of one explanation packet.  Returns claims plus completeness/consistency findings."""
    g = idx.g
    claims: List[Claim] = []
    course_id = expl.get("course_id")
    gap_by_id = gap_by_id or {}

    # ---- structured evidence entries
    for ev in expl.get("evidence", []) or []:
        sid = ev.get("canonical_skill_id")
        name = ev.get("skill", sid)
        et = ev.get("evidence_type")
        job_edges = idx.edge_attrs(job_id, sid, "REQUIRES") + idx.edge_attrs(job_id, sid, "PREFERS")
        claims.append(_c("ev.job_requires", f"{job_id} requires/prefers {name}", "supported" if job_edges else "unsupported",
                         "" if job_edges else "no REQUIRES/PREFERS edge", skill=sid))
        gc = str(ev.get("gap_classification", ""))
        has_user = idx.has_edge_type(user_id, sid, ["HAS_SKILL"])
        if gc in ("missing_critical", "missing_preferred"):
            want = "REQUIRES" if gc == "missing_critical" else "PREFERS"
            ok = (not has_user) and any(d.get("type") == want for d in job_edges) if job_edges else (not has_user)
            claims.append(_c("ev.user_lacks", f"user lacks {name} ({gc})", "supported" if ok else "unsupported",
                             "" if ok else ("user has HAS_SKILL edge" if has_user else f"job edge type != {want}"), skill=sid))
        elif gc in ("weak_skill", "outdated_skill", "uncertain_skill"):
            claims.append(_c("ev.user_has_weak", f"user has {name} but {gc}", "supported" if has_user else "unsupported",
                             "" if has_user else "no HAS_SKILL edge", skill=sid))
        elif gc == "covered_by_transferable_skill":
            claims.append(_c("ev.user_transferable", f"{name} covered by a transferable skill", "unverifiable", "transferable relation not in graph", skill=sid))
        if et == "text_mention_only":
            ok = not idx.has_edge_type(course_id, sid, ["TEACHES"])
            claims.append(_c("ev.text_only_negation", f"{course_id} does not list {name} as taught", "supported" if ok else "unsupported",
                             "" if ok else "TEACHES edge exists", skill=sid))
        elif et == "enabling_prerequisite":
            via = ev.get("via_course_id")
            ok = bool(via) and idx.has_edge_type(via, sid, ["TEACHES"])
            claims.append(_c("ev.enabling_via_teaches", f"{via} teaches {name}", "supported" if ok else "unsupported", "" if ok else "no TEACHES edge", skill=sid))
        else:
            ok = idx.has_edge_type(course_id, sid, ["TEACHES"])
            claims.append(_c("ev.course_teaches", f"{course_id} teaches {name}", "supported" if ok else "unsupported", "" if ok else "no TEACHES edge", skill=sid))
        # (placed per evidence entry below)
        # outcome text must be a recorded learning outcome / description of the course node
        outcome = str(ev.get("course_learning_outcome_evidence", ""))
        node = g.nodes.get(course_id, {})
        known = set(node.get("learning_outcomes") or []) | {node.get("description", "")}
        for d in idx.edge_attrs(course_id, sid, "TEACHES"):
            known.add(d.get("learning_outcome_evidence", ""))
        if et == "enabling_prerequisite":
            known |= set(g.nodes.get(course_id, {}).get("learning_outcomes") or [])
        claims.append(_c("ev.outcome_text", outcome[:60], "supported" if (outcome and outcome in known) else "unsupported",
                         "" if (outcome and outcome in known) else "outcome text not recorded on course node", skill=sid))
        # job evidence sentence must be the stored source_sentence or the documented fallback
        js = str(ev.get("job_requirement_evidence", ""))
        stored = {d.get("source_sentence", "") for d in job_edges}
        job_title = g.nodes.get(job_id, {}).get("title", job_id)
        ok = js in stored and bool(js) or js == f"Required by role {job_title}"
        claims.append(_c("ev.job_sentence", js[:60], "supported" if ok else "unsupported", "" if ok else "sentence not stored on job edge", skill=sid))
        # numeric user evidence "User has strength X (gap weight Y)" must match the gap vector when provided
        m = re.match(r"User has strength ([0-9.]+) \(gap weight: ([0-9.]+)\)", str(ev.get("user_evidence", "")))
        if m and sid in gap_by_id:
            gp = gap_by_id[sid]
            ok = abs(float(m.group(1)) - gp.user_strength) < 0.006 and abs(float(m.group(2)) - gp.gap_weight) < 0.006
            claims.append(_c("ev.user_strength", m.group(0), "supported" if ok else "unsupported", "" if ok else "differs from gap vector", skill=sid))

    # ---- next_profile_update_action must add exactly the skills the course TEACHES (graph edges)
    taught_ids = {t for _, t, d in g.out_edges(course_id, data=True) if d.get("type") == "TEACHES"}
    added = [x.get("canonical_skill_id") for x in ((expl.get("next_profile_update_action") or {}).get("skills_to_add") or [])]
    for sid in added:
        claims.append(_c("next.skill_added_is_taught", f"profile update adds {sid}", "supported" if sid in taught_ids else "unsupported", "" if sid in taught_ids else "course has no TEACHES edge to it", skill=sid))
    if set(added) != taught_ids:
        claims.append(_c("next.skills_added_complete", "profile update lists every taught skill", "unsupported", f"taught but not listed: {sorted(taught_ids - set(added))}"))

    # ---- explanation_path sentences
    n_steps = n_parsed = 0
    for step in expl.get("explanation_path", []) or []:
        for line in [step]:
            n_steps += 1
            parsed_any = False
            last_skill: Optional[str] = None
            last_req_type: Optional[str] = None
            for seg in (s.strip() for s in line.split(" -> ")):
                m = _RE_REQ.match(seg)
                if m:
                    parsed_any = True
                    jn = idx.node_by_title(m.group(1), "Job")
                    sk = idx.skill_by_name(m.group(2))
                    last_skill = sk
                    ok = bool(jn and sk and idx.has_edge_type(jn, sk, ["REQUIRES", "PREFERS"]))
                    claims.append(_c("path.job_requires", seg, "supported" if ok else "unsupported", "" if ok else "no job->skill edge", skill=sk))
                    continue
                m = _RE_USER.match(seg)
                if m:
                    parsed_any = True
                    state = m.group(1)
                    has_user = bool(last_skill and idx.has_edge_type(user_id, last_skill, ["HAS_SKILL"]))
                    if state.startswith("missing"):
                        ok = not has_user
                    elif state in ("weak skill", "outdated skill", "uncertain skill"):
                        ok = has_user
                    else:
                        ok = None  # transferable
                    claims.append(_c("path.user_state", seg, "unverifiable" if ok is None else ("supported" if ok else "unsupported"),
                                     "" if ok in (None, True) else "contradicts HAS_SKILL edges", skill=last_skill))
                    continue
                m = _RE_TEACH.match(seg)
                if m:
                    parsed_any = True
                    cn = idx.node_by_title(m.group(1), "Course")
                    sk = idx.skill_by_name(m.group(2))
                    ok = bool(cn and sk and idx.has_edge_type(cn, sk, ["TEACHES"]))
                    claims.append(_c("path.course_teaches", seg[:80], "supported" if ok else "unsupported", "" if ok else "no TEACHES edge", skill=sk))
                    continue
                m = _RE_TEXTREL.match(seg)
                if m:
                    parsed_any = True
                    cn = idx.node_by_title(m.group(1), "Course")
                    sk = idx.skill_by_name(m.group(3))
                    ok = bool(cn and sk and not idx.has_edge_type(cn, sk, ["TEACHES"]))
                    claims.append(_c("path.text_only_negation", seg[:80], "supported" if ok else "unsupported", "" if ok else "TEACHES edge exists", skill=sk))
                    continue
                m = _RE_PREREQ.match(seg)
                if m:
                    parsed_any = True
                    cn = idx.node_by_title(m.group(1), "Course")
                    sk = idx.skill_by_name(m.group(2))
                    ok = bool(cn and sk and idx.has_edge_type(cn, sk, ["REQUIRES_PREREQUISITE"]))
                    claims.append(_c("path.requires_prerequisite", seg[:80], "supported" if ok else "unsupported", "" if ok else "no prerequisite edge", skill=sk))
                    continue
                m = _RE_COMES_FIRST.match(seg)
                if m:
                    parsed_any = True
                    cn = idx.node_by_title(m.group(1), "Course")
                    sk = idx.skill_by_name(m.group(2))
                    ok = bool(cn and sk and idx.has_edge_type(cn, sk, ["TEACHES"]))
                    claims.append(_c("path.course_teaches", seg[:80], "supported" if ok else "unsupported", "" if ok else "no TEACHES edge", skill=sk))
                    continue
                m = _RE_PREQ_OK.match(seg) or _RE_PREQ_MISS.match(seg)
                if m:
                    parsed_any = True
                    for nm in [x.strip() for x in _RE_PAREN.sub("", m.group(1)).split(",") if x.strip()]:
                        sk = idx.skill_by_name(nm)
                        ok = bool(sk and idx.has_edge_type(course_id, sk, ["REQUIRES_PREREQUISITE"]))
                        claims.append(_c("path.prerequisite_listed", nm, "supported" if ok else "unsupported", "" if ok else "no prerequisite edge", skill=sk))
                    continue
                if _RE_NOPREQ.match(seg):
                    parsed_any = True
                    ok = not any(d.get("type") == "REQUIRES_PREREQUISITE" for _, _, d in g.out_edges(course_id, data=True))
                    claims.append(_c("path.no_prerequisites", seg, "supported" if ok else "unsupported", "" if ok else "prerequisite edges exist"))
            n_parsed += int(parsed_any)

    # ---- consistency + completeness
    consistency: List[Claim] = []
    ev_skill_ids = {e.get("canonical_skill_id") for e in expl.get("evidence", []) or []}
    cov = set(expl.get("covered_missing_skills") or [])
    # covered_missing_skills holds skill NAMES (or ids); each must match an evidence entry by name or id
    direct = {e.get("canonical_skill_id") for e in expl.get("evidence", []) or [] if not e.get("evidence_type")}
    ev_keys = ev_skill_ids | {e.get("skill") for e in expl.get("evidence", []) or []}
    consistency.append(_c("cons.covered_has_evidence", "every covered_missing skill has an evidence entry", "supported" if cov <= ev_keys else "unsupported",
                          "" if cov <= ev_keys else f"missing evidence for {sorted(cov - ev_keys)}"))
    ps = expl.get("prerequisite_status", {}) or {}
    for kind_name, plist in (("satisfied", ps.get("satisfied", []) or []), ("missing", ps.get("missing", []) or [])):
        for p in plist:
            pid = p.get("skill_id") if isinstance(p, dict) else None
            ok = bool(pid and idx.has_edge_type(course_id, pid, ["REQUIRES_PREREQUISITE"]))
            claims.append(_c(f"struct.prereq_{kind_name}", str(pid), "supported" if ok else "unsupported", "" if ok else "no prerequisite edge", skill=pid))
    # prose completeness: does the prerequisite sentence name every prerequisite that the structured status lists?
    prose = " ".join(str(x) for x in expl.get("explanation_path", []) or [])
    listed = [p for p in (ps.get("satisfied", []) or []) + (ps.get("missing", []) or []) if isinstance(p, dict)]
    omitted = [p for p in listed if str(p.get("canonical_skill_name") or p.get("skill_id")) not in prose]
    missing_names = []
    for p in ps.get("missing", []) or []:
        pid = p.get("skill_id") if isinstance(p, dict) else None
        missing_names.append(pid)
    ok = all(pid and idx.has_edge_type(course_id, pid, ["REQUIRES_PREREQUISITE"]) and not idx.has_edge_type(user_id, pid, ["HAS_SKILL"]) for pid in missing_names)
    consistency.append(_c("cons.prereq_missing_true", "prerequisites reported missing are real prerequisites the user lacks", "supported" if ok else "unsupported", "" if ok else "a 'missing' prerequisite is held or not a prerequisite"))
    flag = bool(ps.get("is_satisfied", True)) == (len(ps.get("missing", []) or []) == 0)
    consistency.append(_c("cons.prereq_flag", "is_satisfied equals 'no missing prerequisites'", "supported" if flag else "unsupported"))
    top_missing = [f for f in REQUIRED_TOP_FIELDS if f not in expl]
    ev_incomplete = sum(1 for e in expl.get("evidence", []) or [] for f in REQUIRED_EVIDENCE_FIELDS if not str(e.get(f, "")).strip())
    completeness = {"top_missing": top_missing, "evidence_fields_empty": ev_incomplete, "n_evidence": len(expl.get("evidence", []) or []),
                    "complete": (not top_missing) and ev_incomplete == 0 and bool(expl.get("evidence")) and bool(expl.get("explanation_path"))}
    completeness["prereq_prose_omits"] = len(omitted)
    completeness["prereq_listed"] = len(listed)
    return {"claims": claims, "consistency": consistency, "completeness": completeness, "n_steps": n_steps, "n_parsed": n_parsed,
            "direct_evidence_skills": sorted(x for x in direct if x)}


def unsupported_keys(audit: Dict[str, Any]) -> Set[Tuple[str, str]]:
    return {(c["kind"], c["text"]) for c in audit["claims"] + audit["consistency"] if c["status"] == "unsupported"}


# --------------------------------------------------------------------------------------------------------------
# Running the real pipeline on a synthetic world
# --------------------------------------------------------------------------------------------------------------


def run_course_pipeline(world: Dict[str, Any], n_users: int = 6, top_k: int = 5, config: Any = None) -> Tuple[Any, List[Dict[str, Any]]]:
    from skillllm_v2.course.ingest_catalog import CourseCatalog
    from skillllm_v2.recommend.course_ranker import CourseRanker
    from skillllm_v2.recommend.course_retriever import CourseRetriever
    from skillllm_v2.recommend.explanation_generator import CourseExplanationGenerator
    from skillllm_v2.recommend.config import LEGACY_CONFIG
    from skillllm_v2.recommend.gap_vector import GapEngine

    config = config or LEGACY_CONFIG
    with fixed_graph_clock():
        graph = build_course_graph(world)
    catalog = CourseCatalog([{k: v for k, v in c.items() if not k.startswith("_")} for c in world["courses"]])
    retriever = CourseRetriever(catalog=catalog, graph=graph)
    ranker, explainer, engine = CourseRanker(config=config), CourseExplanationGenerator(), GapEngine()
    items: List[Dict[str, Any]] = []
    for i, u in enumerate(world["users"][:n_users]):
        job = world["jobs"][i % len(world["jobs"])]
        user_profile = {k: v for k, v in u.items() if not k.startswith("_")}
        user_profile["experience_level"] = "intermediate"
        job_clean = {k: v for k, v in job.items() if not k.startswith("_")}
        gv = engine.compute(user_profile, job_clean, graph=graph)
        cands = retriever.retrieve(gv, job_clean, top_k=config.retrieval_pool)
        ranked = ranker.rank(candidates=[c.course for c in cands], user_profile=user_profile, target_job=job_clean, gap_vector=gv, graph=graph, top_k=top_k, catalog_courses=catalog.list_courses())
        expls = explainer.generate_recommendations_payload(ranked, user_profile, job_clean, gv, graph)
        for e in expls:
            items.append({"user_id": u["user_id"], "job_id": job["job_id"], "explanation": e, "gap_by_id": {g.canonical_skill_id: g for g in gv.gaps}})
    return graph.graph, items


# --------------------------------------------------------------------------------------------------------------
# Mutations
# --------------------------------------------------------------------------------------------------------------


def _remove_edges(graph: Any, u: str, v: str, types: Sequence[str]) -> int:
    n = 0
    if graph.has_edge(u, v):
        for key, d in list(graph.get_edge_data(u, v).items()):
            if d.get("type") in types:
                graph.remove_edge(u, v, key)
                n += 1
    return n


def graph_mutations(graph: Any, item: Dict[str, Any], rng: random.Random) -> List[Tuple[str, Any, Optional[str]]]:
    """(name, mutated graph, skill id the mutation concerns) for every mutation applicable to this explanation."""
    e = item["explanation"]
    cid, uid, jid = e["course_id"], item["user_id"], item["job_id"]
    out: List[Tuple[str, Any, Optional[str]]] = []
    direct = [x for x in e["evidence"] if not x.get("evidence_type")]
    skills_in_graph = [n for n, d in graph.nodes(data=True) if d.get("type") == "Skill"]
    if direct:
        sid = direct[0]["canonical_skill_id"]
        g1 = copy.deepcopy(graph)
        if _remove_edges(g1, cid, sid, ["TEACHES"]):
            out.append(("remove_teaches_edge", g1, sid))
        g2 = copy.deepcopy(graph)
        if _remove_edges(g2, jid, sid, ["REQUIRES", "PREFERS"]):
            out.append(("remove_job_requires_edge", g2, sid))
        g3 = copy.deepcopy(graph)
        if _remove_edges(g3, cid, sid, ["TEACHES"]):
            taught = {t for _, t, d in g3.out_edges(cid, data=True) if d.get("type") == "TEACHES"}
            alts = sorted(s for s in skills_in_graph if s not in taught and s != sid)
            if alts:
                g3.add_edge(cid, rng.choice(alts), type="TEACHES", depth="intermediate", confidence=1.0, learning_outcome_evidence="swapped", outcome_index=0)
                out.append(("swap_teaches_target", g3, sid))
        if direct[0].get("gap_classification", "").startswith("missing"):
            g4 = copy.deepcopy(graph)
            g4.add_edge(uid, sid, type="HAS_SKILL", confidence=1.0, evidence_text="injected", source="mutation", last_seen="2026-01-01", mastery_level="proficient")
            out.append(("add_user_has_skill_edge", g4, sid))
    pre = [p.get("skill_id") for p in (e.get("prerequisite_status", {}).get("satisfied", []) + e.get("prerequisite_status", {}).get("missing", [])) if isinstance(p, dict)]
    pre = [p for p in pre if p]
    if pre:
        g5 = copy.deepcopy(graph)
        if _remove_edges(g5, cid, pre[0], ["REQUIRES_PREREQUISITE"]):
            out.append(("remove_prerequisite_edge", g5, pre[0]))
    return out


def control_mutation(graph: Any, item: Dict[str, Any], rng: random.Random) -> Optional[Any]:
    """Remove one edge that the explanation does not touch (different course, different job, user, skills)."""
    e = item["explanation"]
    touched_skills = {x.get("canonical_skill_id") for x in e["evidence"]} | {p.get("skill_id") for p in (e.get("prerequisite_status", {}).get("satisfied", []) + e.get("prerequisite_status", {}).get("missing", [])) if isinstance(p, dict)}
    touched_nodes = {e["course_id"], item["user_id"], item["job_id"]} | {u.get("course_id") for u in e.get("unlocks", [])} | {x.get("via_course_id") for x in e["evidence"]}
    cands = [(u, v, k) for u, v, k in graph.edges(keys=True) if u not in touched_nodes and v not in touched_skills]
    if not cands:
        return None
    u, v, k = cands[rng.randrange(len(cands))]
    g = copy.deepcopy(graph)
    g.remove_edge(u, v, k)
    return g


def explanation_mutations(item: Dict[str, Any], idx: GraphIndex, rng: random.Random) -> List[Tuple[str, Dict[str, Any]]]:
    e = item["explanation"]
    out: List[Tuple[str, Dict[str, Any]]] = []
    names_all = sorted(idx.name_to_skill)
    direct = [x for x in e["evidence"] if not x.get("evidence_type")]
    if not direct:
        return out
    d0 = direct[0]
    name = d0["skill"]
    taught = {t for _, t, d in idx.g.out_edges(e["course_id"], data=True) if d.get("type") == "TEACHES"}
    required = {t for _, t, d in idx.g.out_edges(item["job_id"], data=True) if d.get("type") in ("REQUIRES", "PREFERS")}
    other = [n for n in names_all if (idx.skill_by_name(n) not in taught) and (idx.skill_by_name(n) not in required) and n != name]
    if other:
        alt = rng.choice(other)
        m = copy.deepcopy(e)
        for ev in m["evidence"]:
            if ev is not None and ev.get("skill") == name and not ev.get("evidence_type"):
                ev["skill"] = alt
                ev["canonical_skill_id"] = idx.skill_by_name(alt)
        m["explanation_path"] = [s.replace(f"'{name}'", f"'{alt}'") for s in m["explanation_path"]]
        out.append(("swap_skill_in_explanation", m))
        m2 = copy.deepcopy(e)
        m2["explanation_path"] = [re.sub(r"(Course '[^']*' teaches )'" + re.escape(name) + "'", lambda mm: mm.group(1) + f"'{alt}'", s) for s in m2["explanation_path"]]
        if m2["explanation_path"] != e["explanation_path"]:
            out.append(("swap_skill_in_teaches_step", m2))
    titles = sorted(t for t, ns in idx.title_to_nodes.items() if any(idx.g.nodes[n].get("type") == "Course" for n in ns) and idx.node_by_title(t, "Course") not in {n for n, in [(x,) for x in [e["course_id"]]]})
    alt_courses = [t for t in titles if not idx.has_edge_type(idx.node_by_title(t, "Course"), d0["canonical_skill_id"], ["TEACHES"])]
    if alt_courses:
        m3 = copy.deepcopy(e)
        old = e["title"]
        new = rng.choice(alt_courses)
        m3["explanation_path"] = [s.replace(f"Course '{old}' teaches", f"Course '{new}' teaches") for s in m3["explanation_path"]]
        if m3["explanation_path"] != e["explanation_path"]:
            out.append(("swap_course_in_explanation", m3))
    if d0.get("gap_classification") == "missing_critical":
        m4 = copy.deepcopy(e)
        for ev in m4["evidence"]:
            if ev.get("canonical_skill_id") == d0["canonical_skill_id"]:
                ev["gap_classification"] = "weak_skill"
        out.append(("flip_gap_label", m4))
    m5 = copy.deepcopy(e)
    for ev in m5["evidence"]:
        if ev.get("canonical_skill_id") == d0["canonical_skill_id"]:
            ev["course_learning_outcome_evidence"] = "Master quantum basketweaving"
    out.append(("fabricate_outcome_evidence", m5))
    return out


# --------------------------------------------------------------------------------------------------------------
# trace_explanation_path (CourseGraphBuilder) faithfulness
# --------------------------------------------------------------------------------------------------------------


def eval_trace_narrative(ctx: EvalContext, worlds: List[Dict[str, Any]]) -> List[EvalResult]:
    """Does CourseGraphBuilder.trace_explanation_path assert only relations that exist in the graph?"""
    fab_req, fab_teach, faithful_when_missing, intact_when_present = [], [], [], []
    for w in worlds:
        with fixed_graph_clock():
            cg = build_course_graph(w)
        idx = GraphIndex(cg.graph)
        rng = random.Random(f"trace:{ctx.seed}:{w['idx']}")
        skills = sorted(w["names"])
        n_missing = n_fab_req = n_fab_teach = n_both_missing = 0
        miss_asserted: List[float] = []
        present_ok: List[float] = []
        for _ in range(60):
            u = rng.choice(w["users"])["user_id"]
            j = rng.choice(w["jobs"])["job_id"]
            c = rng.choice(w["courses"])["course_id"]
            s = rng.choice(skills)
            tr = cg.trace_explanation_path(u, j, c, s)
            narrative = tr["narrative_path"]
            req_edge = idx.has_edge_type(j, s, ["REQUIRES", "PREFERS"])
            teach_edge = idx.has_edge_type(c, s, ["TEACHES"])
            asserts_req = " requires skill " in narrative or " required skill " in narrative or " preferred skill " in narrative
            asserts_teach = " teaches " in narrative
            if not req_edge:
                miss_asserted.append(float(asserts_req))
            if not teach_edge:
                miss_asserted.append(float(asserts_teach))
            if req_edge and teach_edge:
                present_ok.append(float(asserts_req and asserts_teach))
        if miss_asserted:
            faithful_when_missing.append(1.0 - _mean(miss_asserted))
        if present_ok:
            intact_when_present.append(_mean(present_ok))
    return [
        _res("explain.trace_path_faithful_when_edge_missing", "queries with an absent job->skill or course->skill edge where trace_explanation_path does NOT assert that relation", faithful_when_missing, ctx,
             details={"asserts_relation_when_edge_present": _mean(intact_when_present) if intact_when_present else None, "queries_per_world": 60},
             caveats=["trace_explanation_path builds its narrative from fixed templates and only reads the edge for optional text: with no REQUIRES edge it still writes 'Target role X required skill S', with no TEACHES edge it still writes 'Course C teaches S'. 0.0 means every missing edge is narrated as present",
                      "random (user, job, course, skill) quadruples: most are unrelated, which is exactly the situation an explanation must not fabricate relations for"]),
    ]


# --------------------------------------------------------------------------------------------------------------
# Course explainer evals
# --------------------------------------------------------------------------------------------------------------


def eval_course_explanations(ctx: EvalContext, worlds: List[Dict[str, Any]], config: Any = None, config_name: str = "legacy") -> List[EvalResult]:
    res: List[EvalResult] = []
    unsup_rate, claim_counts, parse_cov, complete, cons_rate, unver, prose_omit, id_fb = [], [], [], [], [], [], [], []
    by_kind: Dict[str, List[int]] = {}
    mut: Dict[str, List[float]] = {}
    mut_attr: Dict[str, List[float]] = {}
    ctrl_alarm: List[float] = []
    exp_mut: Dict[str, List[float]] = {}
    blind: Dict[str, List[float]] = {}
    n_expl = 0
    roles: Dict[str, int] = {}
    ev_types: Dict[str, int] = {}
    failures: List[str] = []
    examples: List[Dict[str, Any]] = []
    per_world_sens: Dict[str, List[float]] = {}
    per_world_sens_e: Dict[str, List[float]] = {}
    world_ctrl: List[float] = []
    for w in worlds:
        try:
            graph, items = run_course_pipeline(w, n_users=3 if ctx.smoke else 6, config=config)
        except Exception as exc:  # report, never hide
            failures.append(f"world {w['idx']}: {type(exc).__name__}: {exc}")
            continue
        idx = GraphIndex(graph)
        rng = random.Random(f"mut:{ctx.seed}:{w['idx']}")
        fb_n = fb_d = 0
        u_n = u_d = 0
        cons_n = cons_d = 0
        cov_n = cov_d = 0
        comp = []
        omit_n = omit_d = 0
        unv_n = 0
        wm: Dict[str, List[float]] = {}
        wme: Dict[str, List[float]] = {}
        wc: List[float] = []
        wb: Dict[str, List[float]] = {}
        for it in items:
            n_expl += 1
            roles[it['explanation'].get('role', '?')] = roles.get(it['explanation'].get('role', '?'), 0) + 1
            for x in it['explanation'].get('evidence', []):
                ev_types[str(x.get('evidence_type'))] = ev_types.get(str(x.get('evidence_type')), 0) + 1
            fb0 = (idx.id_fallbacks, idx.mentions)
            a = audit_course_explanation(it["explanation"], idx, it["user_id"], it["job_id"], it["gap_by_id"])
            fb_n += idx.id_fallbacks - fb0[0]
            fb_d += idx.mentions - fb0[1]
            base = unsupported_keys(a)
            checkable = [c for c in a["claims"] if c["status"] in ("supported", "unsupported")]
            u_n += sum(1 for c in checkable if c["status"] == "unsupported")
            u_d += len(checkable)
            unv_n += sum(1 for c in a["claims"] if c["status"] == "unverifiable")
            cons_n += sum(1 for c in a["consistency"] if c["status"] == "unsupported")
            cons_d += len(a["consistency"])
            cov_n += a["n_parsed"]
            cov_d += a["n_steps"]
            comp.append(float(a["completeness"]["complete"]))
            if a["completeness"]["prereq_listed"]:
                omit_d += 1
                omit_n += int(a["completeness"]["prereq_prose_omits"] > 0)
            for c in a["claims"]:
                if c["status"] == "unsupported":
                    by_kind.setdefault(c["kind"], [0])[0] += 1
                    if len(examples) < 6:
                        examples.append({"world": w["idx"], "course": it["explanation"]["course_id"], "claim": c})
            claim_counts.append(float(len(checkable)))
            # graph mutations
            for name, mg, sid in graph_mutations(graph, it, rng):
                after = unsupported_keys(audit_course_explanation(it["explanation"], GraphIndex(mg), it["user_id"], it["job_id"], it["gap_by_id"]))
                new = after - base
                wm.setdefault(name, []).append(float(bool(new)))
                sname = w["names"].get(sid, sid) if sid else ""
                mut_attr.setdefault(name, []).append(float(any((sid and sid in t) or (sname and sname in t) for _, t in new) or any(kind.startswith("cons.") or kind.startswith("path.prereq") for kind, _ in new)))
            cg = control_mutation(graph, it, rng)
            if cg is not None:
                after = unsupported_keys(audit_course_explanation(it["explanation"], GraphIndex(cg), it["user_id"], it["job_id"], it["gap_by_id"]))
                wc.append(float(bool(after - base)))
            for name, me in explanation_mutations(it, idx, rng):
                after = unsupported_keys(audit_course_explanation(me, idx, it["user_id"], it["job_id"], it["gap_by_id"]))
                wme.setdefault(name, []).append(float(bool(after - base)))
            for bname, bm in (("tamper_score", {"score": it["explanation"]["score"] + 0.3}), ("tamper_rank", {"rank": it["explanation"]["rank"] + 5}),
                              ("drop_limitations", {"limitations": []})):
                me = {**it["explanation"], **bm}
                after = unsupported_keys(audit_course_explanation(me, idx, it["user_id"], it["job_id"], it["gap_by_id"]))
                wb.setdefault(bname, []).append(float(bool(after - base)))
        id_fb.append(fb_n / max(1, fb_d))
        if u_d:
            unsup_rate.append(u_n / u_d)
        if cons_d:
            cons_rate.append(cons_n / cons_d)
        if cov_d:
            parse_cov.append(cov_n / cov_d)
        complete.append(_mean(comp))
        if omit_d:
            prose_omit.append(omit_n / omit_d)
        unver.append(unv_n)
        for k, v in wm.items():
            mut.setdefault(k, []).append(_mean(v))
        for k, v in wme.items():
            exp_mut.setdefault(k, []).append(_mean(v))
        for k, v in wb.items():
            blind.setdefault(k, []).append(_mean(v))
        if wc:
            world_ctrl.append(_mean(wc))
    if failures and not unsup_rate:
        return [EvalResult("explain.course_pipeline", MODULE, "pipeline run", None, 0, DATASET, status="failed", reason="; ".join(failures[:3]))]
    cave_audit = ["audit parses only the generator's fixed sentence templates; parse_coverage reports how many explanation steps were parsed",
                  "claims are verified against the same graph the explainer used (consistency with graph, not truth about the world)"]
    res.append(_res("explain.course.unsupported_claim_rate", "share of checkable claims (evidence entries + parsed path steps) with no supporting graph edge/attribute", unsup_rate, ctx,
                    details={"config": config_name, "n_explanations": n_expl, "role_counts": roles, "evidence_type_counts": ev_types, "mean_checkable_claims_per_explanation": _mean(claim_counts), "unsupported_by_kind": {k: v[0] for k, v in sorted(by_kind.items())},
                             "examples": examples, "pipeline_failures": failures}, caveats=cave_audit))
    res.append(_res("explain.course.supported_claim_rate", "1 - unsupported_claim_rate", [1.0 - x for x in unsup_rate], ctx, caveats=cave_audit))
    res.append(_res("explain.course.parse_coverage", "explanation_path steps the audit could parse into claims", parse_cov, ctx,
                    caveats=["steps not parsed (e.g. unfamiliar templates) are not verified; low coverage would make a low unsupported rate meaningless"]))
    res.append(_res("explain.course.completeness", "explanations with all required top-level fields and non-empty evidence fields (user skill / job evidence / gap / course outcome)", complete, ctx,
                    details={"required_top_fields": list(REQUIRED_TOP_FIELDS), "required_evidence_fields": list(REQUIRED_EVIDENCE_FIELDS)}))
    res.append(_res("explain.course.prereq_prose_omission_rate", "explanations whose prerequisite sentence leaves out a prerequisite listed in the structured prerequisite_status (only explanations with prerequisites)", prose_omit, ctx,
                    caveats=["the text says only 'Caution: Missing prerequisite skill(s): X' while satisfied prerequisites appear only in the structured field: a reader of the prose cannot tell what the user already satisfies"]))
    res.append(_res("explain.course.skill_mentions_as_raw_id", "skill mentions in explanation prose that only resolve through the normalized node id (e.g. 'junit', 'java') instead of the display name", id_fb, ctx,
                    caveats=["measured while parsing the clean explanations (graph mutations excluded); prerequisite names in 'Prerequisite requirements satisfied:' / bridge steps come from normalized ids, whereas covered skills use display names, so one explanation mixes both styles"]))
    res.append(_res("explain.course.consistency_violation_rate", "internal-consistency checks failed (covered skills have evidence; missing prerequisites are real and unheld; flag == no missing)", cons_rate, ctx))
    res.append(_res("explain.course.unverifiable_claims_per_world", "claims the audit cannot check against the graph (transferable-skill relations)", unver, ctx))
    # mutation sensitivity
    allm = [x for v in mut.values() for x in v]
    res.append(_res("explain.audit_sensitivity_graph_mutations", "graph corrupted (edge removed/swapped/injected) while the explanation is unchanged: audit reports >=1 NEW unsupported claim", allm, ctx,
                    details={"per_mutation": {k: {"mean_over_worlds": _mean(v), "n_worlds": len(v)} for k, v in sorted(mut.items())},
                             "attributed_to_mutated_skill_or_prereq": {k: _mean(v) for k, v in sorted(mut_attr.items())}},
                    caveats=["unit = world; each world contributes the mean over its applicable mutations", "mutations are applied to the explanation's own graph; removing the edge for a claim the audit never checks would be missed (that is what this measures)"]))
    for k, v in sorted(mut.items()):
        res.append(_res(f"explain.audit_sensitivity.{k}", f"graph mutation '{k}' detected by the audit", v, ctx))
    alle = [x for v in exp_mut.values() for x in v]
    res.append(_res("explain.audit_sensitivity_explanation_mutations", "explanation text corrupted (swapped skill/course, flipped gap label, fabricated outcome) against a clean graph: audit flags new unsupported claim", alle, ctx,
                    details={"per_mutation": {k: {"mean_over_worlds": _mean(v)} for k, v in sorted(exp_mut.items())}}))
    for k, v in sorted(exp_mut.items()):
        res.append(_res(f"explain.audit_sensitivity.{k}", f"explanation mutation '{k}' detected by the audit", v, ctx))
    res.append(_res("explain.audit_blind_spots", "mutations the audit cannot see by construction (score/rank tampering, limitations removed): share detected", [x for v in blind.values() for x in v], ctx,
                    details={"per_mutation": {k: _mean(v) for k, v in sorted(blind.items())}},
                    caveats=["documents the audit's limits: score/rank are not recomputed and limitation text is not verified, so tampering there is invisible; the sensitivity numbers above cover only the claim types the audit parses",
                             "mutations in the sensitivity tests target exactly the audited claim types, so ~1.0 there shows the audit works on what it checks, not that it checks everything"]))
    res.append(_res("explain.audit_false_alarm_control", "removing an UNRELATED edge makes the audit flag the explanation (0.0 is correct)", world_ctrl, ctx,
                    caveats=["negative control for the mutation test: the audit must react to relevant corruption only"]))
    res.append(_res("explain.audit_false_positive_clean", "clean explanation vs clean graph: share of explanations with any unsupported claim", [float(r > 0) for r in unsup_rate], ctx,
                    caveats=["per-world indicator (world has >=1 unsupported claim); use unsupported_claim_rate for the magnitude"]))
    return res


# --------------------------------------------------------------------------------------------------------------
# match.generate_recommendation_explanation
# --------------------------------------------------------------------------------------------------------------


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.lower()).strip()


def build_match_scenarios(world: Dict[str, Any], n_users: int = 6) -> List[Dict[str, Any]]:
    from skillllm_v2.context.profile_summarizer import summarize_profile
    from skillllm_v2.match.explanation_generator import generate_recommendation_explanation
    from skillllm_v2.match.gap_detector import GapDetector
    from skillllm_v2.match.job_matcher import JobMatcher

    names = world["names"]
    out: List[Dict[str, Any]] = []
    for i, u in enumerate(world["users"][:n_users]):
        job_w = world["jobs"][i % len(world["jobs"])]
        have_names = [names[s] for s in u["_have"]]
        text = " ".join(f"Worked with {n} on a team project." for n in have_names)
        extracted, pos = [], 0
        for n in have_names:
            st = text.find(n, pos)
            extracted.append({"surface_form": n, "label": "HARD_SKILL", "start": st, "end": st + len(n), "source": "mistral", "confidence": 0.9,
                              "evidence": f"Worked with {n} on a team project"})
            pos = st + len(n)
        with_clock = summarize_profile(text, extracted_skills=extracted, user_id=u["user_id"], timestamp="2026-01-01T00:00:00+00:00")
        job = {"job_id": job_w["job_id"], "required_skills": [names[s] for s in job_w["_req"]], "preferred_skills": [names[s] for s in job_w["_pref"]],
               "semantic_role_fit": 0.8, "experience_level_fit": 0.8, "user_preference_fit": 0.8}
        matcher_profile = {"skills": have_names}
        job_match = JobMatcher().score(matcher_profile, job)
        gaps = GapDetector().detect(matcher_profile, job)
        missing = set(job_match["missing_skills"]) | {g["skill"] for g in gaps}
        courses = [{"course_id": c["course_id"], "course_name": c["title"], "teaches_skills": c["teaches_skills"]}
                   for c in world["courses"] if set(c["teaches_skills"]) & missing][:3]
        expl = generate_recommendation_explanation(with_clock, job_match=job_match, skill_gaps=gaps, course_recommendations=courses)
        out.append({"user_id": u["user_id"], "profile": with_clock, "text": text, "job": job, "job_match": job_match, "gaps": gaps, "courses": courses,
                    "catalog": {c["title"]: c["teaches_skills"] for c in world["courses"]}, "explanation": expl, "have": have_names})
    return out


def audit_match_explanation(expl: Dict[str, Any], sc: Dict[str, Any]) -> Dict[str, Any]:
    have = {_norm(x) for x in sc["have"]}
    req = {_norm(x) for x in sc["job"]["required_skills"]}
    pref = {_norm(x) for x in sc["job"]["preferred_skills"]}
    claims: List[Claim] = []
    for s in expl.get("matched_skills", []):
        ok = _norm(s) in have and _norm(s) in (req | pref)
        claims.append(_c("match.matched_skill", s, "supported" if ok else "unsupported", "" if ok else "not held by user or not a job skill"))
    for s in expl.get("missing_skills", []):
        ok = _norm(s) in req and _norm(s) not in have
        claims.append(_c("match.missing_skill", s, "supported" if ok else "unsupported", "" if ok else "not required or held by user"))
    gaps = sc["gaps"]
    gt = expl.get("gap_types", [])
    ok_len = len(gt) == len(gaps)
    claims.append(_c("match.gap_count", f"{len(gt)} gap types", "supported" if ok_len else "unsupported", "" if ok_len else "differs from skill_gaps"))
    for g_claim, g_src in zip(gt, [x.get("gap_type") for x in gaps]):
        claims.append(_c("match.gap_type", str(g_claim), "supported" if g_claim == g_src else "unsupported", "" if g_claim == g_src else "differs from skill_gaps"))
    missing_set = {_norm(x) for x in sc["job_match"]["missing_skills"]} | {_norm(g["skill"]) for g in gaps}
    for title in expl.get("recommended_learning_path", []):
        teaches = {_norm(x) for x in sc["catalog"].get(title, [])}
        ok = bool(teaches & missing_set)
        claims.append(_c("match.course_for_gap", title, "supported" if ok else "unsupported", "" if ok else "course absent from catalog or teaches no missing skill"))
    evid = {f"{s.get('surface_form')}: {e.get('text')}" for s in sc["profile"].get("skills", []) for e in s.get("evidence", [])}
    relevant = 0
    for line in expl.get("evidence", []):
        ok = line in evid
        text = line.split(": ", 1)[1] if ": " in line else line
        in_src = text in sc["text"]
        claims.append(_c("match.evidence_line", line[:70], "supported" if (ok and in_src) else "unsupported", "" if (ok and in_src) else "not a profile evidence entry / not in source text"))
        skill = line.split(": ", 1)[0]
        relevant += int(_norm(skill) in (set(map(_norm, expl.get("matched_skills", []))) | set(map(_norm, expl.get("missing_skills", [])))))
    conf_ok = abs(float(expl.get("confidence", -1)) - float(sc["job_match"]["match_score"])) < 1e-9
    claims.append(_c("match.confidence", f"{expl.get('confidence')}", "supported" if conf_ok else "unsupported", "" if conf_ok else "differs from match_score"))
    for lim in expl.get("limitations", []):
        claims.append(_c("match.limitation", lim[:70], "static", "boilerplate string identical for every recommendation"))
    n_ev = len(expl.get("evidence", []))
    comp_fields = {"user_skill": bool(expl.get("matched_skills")) or bool(sc["have"]), "job_skill": bool(expl.get("matched_skills") or expl.get("missing_skills")),
                   "gap": "gap_types" in expl, "evidence": bool(n_ev)}
    return {"claims": claims, "evidence_relevance": relevant / n_ev if n_ev else float("nan"), "n_evidence": n_ev,
            "n_matched": len(expl.get("matched_skills", [])), "n_missing": len(expl.get("missing_skills", [])), "completeness": comp_fields,
            "missing_skill_evidence": 0}


def eval_match_explanations(ctx: EvalContext, worlds: List[Dict[str, Any]]) -> List[EvalResult]:
    from skillllm_v2.match.explanation_generator import generate_recommendation_explanation

    res: List[EvalResult] = []
    unsup, static_share, relevance, complete, ev_cover_missing = [], [], [], [], []
    mut: Dict[str, List[float]] = {}
    ctrl: List[float] = []
    for w in worlds:
        scs = build_match_scenarios(w, n_users=3 if ctx.smoke else 6)
        rng = random.Random(f"mm:{ctx.seed}:{w['idx']}")
        u_n = u_d = st = tot = 0
        rel, comp, mcov = [], [], []
        wm: Dict[str, List[float]] = {}
        wc: List[float] = []
        for sc in scs:
            a = audit_match_explanation(sc["explanation"], sc)
            base = {(c["kind"], c["text"]) for c in a["claims"] if c["status"] == "unsupported"}
            chk = [c for c in a["claims"] if c["status"] in ("supported", "unsupported")]
            u_n += sum(1 for c in chk if c["status"] == "unsupported")
            u_d += len(chk)
            st += sum(1 for c in a["claims"] if c["status"] == "static")
            tot += len(a["claims"])
            if a["evidence_relevance"] == a["evidence_relevance"]:
                rel.append(a["evidence_relevance"])
            comp.append(float(all(a["completeness"].values())))
            # does any evidence line speak to a MISSING skill (the job-side evidence)?
            mcov.append(float(bool(a["n_missing"]) and any(l.split(": ", 1)[0] and _norm(l.split(": ", 1)[0]) in {_norm(m) for m in sc["explanation"]["missing_skills"]} for l in sc["explanation"]["evidence"])))
            # mutations of the INPUTS the explanation was derived from (explanation text unchanged)
            def audit_with(mod: Dict[str, Any]) -> Set[Tuple[str, str]]:
                sc2 = {**sc, **mod}
                aa = audit_match_explanation(sc["explanation"], sc2)
                return {(c["kind"], c["text"]) for c in aa["claims"] if c["status"] == "unsupported"} - base

            if sc["explanation"]["matched_skills"]:
                s0 = sc["explanation"]["matched_skills"][0]
                wm.setdefault("remove_matched_skill_from_user", []).append(float(bool(audit_with({"have": [h for h in sc["have"] if h != s0]}))))
                job2 = {**sc["job"], "required_skills": [x for x in sc["job"]["required_skills"] if x != s0], "preferred_skills": [x for x in sc["job"]["preferred_skills"] if x != s0]}
                wm.setdefault("remove_matched_skill_from_job", []).append(float(bool(audit_with({"job": job2}))))
            if sc["explanation"]["missing_skills"]:
                m0 = sc["explanation"]["missing_skills"][0]
                wm.setdefault("user_gains_missing_skill", []).append(float(bool(audit_with({"have": sc["have"] + [m0]}))))
            if sc["explanation"]["recommended_learning_path"]:
                t0 = sc["explanation"]["recommended_learning_path"][0]
                cat2 = dict(sc["catalog"])
                cat2[t0] = []
                wm.setdefault("course_stops_teaching_gap_skill", []).append(float(bool(audit_with({"catalog": cat2}))))
            # control: drop an unrelated catalog course
            others = [t for t in sc["catalog"] if t not in sc["explanation"]["recommended_learning_path"]]
            if others:
                cat3 = dict(sc["catalog"])
                cat3.pop(others[rng.randrange(len(others))])
                wc.append(float(bool(audit_with({"catalog": cat3}))))
            # explanation-side corruption
            ex = copy.deepcopy(sc["explanation"])
            if ex["matched_skills"]:
                ex["matched_skills"][0] = "Quantum Basketweaving"
                aa = audit_match_explanation(ex, sc)
                wm.setdefault("swap_matched_skill_in_explanation", []).append(float(bool({(c["kind"], c["text"]) for c in aa["claims"] if c["status"] == "unsupported"} - base)))
            ex2 = copy.deepcopy(sc["explanation"])
            if ex2["evidence"]:
                ex2["evidence"][0] = ex2["evidence"][0] + " (fabricated)"
                aa = audit_match_explanation(ex2, sc)
                wm.setdefault("corrupt_evidence_line", []).append(float(bool({(c["kind"], c["text"]) for c in aa["claims"] if c["status"] == "unsupported"} - base)))
        if u_d:
            unsup.append(u_n / u_d)
        static_share.append(st / tot if tot else float("nan"))
        relevance.append(_mean(rel) if rel else float("nan"))
        complete.append(_mean(comp))
        ev_cover_missing.append(_mean(mcov))
        for k, v in wm.items():
            mut.setdefault(k, []).append(_mean(v))
        if wc:
            ctrl.append(_mean(wc))
    cave = ["inputs are structured dicts from the synthetic worlds via the real JobMatcher / GapDetector / summarize_profile; claims are verified against those same inputs",
            "the explainer copies its inputs, so a low unsupported rate shows faithful copying, not that matching was right"]
    res.append(_res("explain.match.unsupported_claim_rate", "match explanation: share of checkable claims (matched/missing skills, gap types, courses, evidence lines, confidence) not supported by the inputs", unsup, ctx, caveats=cave))
    res.append(_res("explain.match.static_boilerplate_share", "share of explanation claims that are static boilerplate (limitations), neither supported nor unsupported", static_share, ctx,
                    caveats=["limitations are two fixed strings; one asserts an extraction fallback policy unrelated to the recommended job, so it is unverifiable for every explanation"]))
    res.append(_res("explain.match.evidence_relevance", "evidence lines that concern a matched or missing skill of this job (rest = unrelated profile skills dumped into the explanation)", relevance, ctx,
                    caveats=["generate_recommendation_explanation lists the first evidence of EVERY profile skill, not only the matched ones; relevance is a heuristic share of lines whose skill is in matched/missing"]))
    res.append(_res("explain.match.missing_skill_has_evidence", "explanations with missing skills that include any evidence line about a missing skill", ev_cover_missing, ctx,
                    caveats=["evidence only covers user profile skills, so job-side gaps are never evidenced"]))
    res.append(_res("explain.match.completeness", "explanations with user skill, job skill, gap types and evidence all present", complete, ctx))
    allm = [x for v in mut.values() for x in v]
    res.append(_res("explain.match.audit_sensitivity", "inputs or explanation corrupted -> audit reports a new unsupported claim", allm, ctx,
                    details={"per_mutation": {k: _mean(v) for k, v in sorted(mut.items())}}))
    for k, v in sorted(mut.items()):
        res.append(_res(f"explain.match.audit_sensitivity.{k}", f"mutation '{k}' detected", v, ctx))
    res.append(_res("explain.match.audit_false_alarm_control", "dropping an unrelated catalog course makes the audit flag the explanation (0.0 correct)", ctrl, ctx))
    return res


def _legacy_subset(results: List[EvalResult]) -> List[EvalResult]:
    keep = ("unsupported_claim_rate", "parse_coverage", "completeness", "audit_sensitivity_graph_mutations", "audit_sensitivity_explanation_mutations")
    out = []
    for r in results:
        if any(r.eval_name.endswith(k) for k in keep):
            r.eval_name = r.eval_name.replace("explain.", "explain.legacy_cfg.", 1)
            out.append(r)
    return out


def run(ctx: EvalContext) -> List[EvalResult]:
    worlds = [make_world(ctx.seed, i) for i in range(2 if ctx.smoke else 8)]
    out: List[EvalResult] = []
    from skillllm_v2.recommend.config import DEFAULT_CONFIG

    for nm, fn in (("course", lambda c, w: eval_course_explanations(c, w, DEFAULT_CONFIG, "DEFAULT_CONFIG (strict_bridge, related fill)")),
                   ("course_legacy", lambda c, w: _legacy_subset(eval_course_explanations(c, w, None, "LEGACY_CONFIG"))),
                   ("trace", eval_trace_narrative), ("match", eval_match_explanations)):
        try:
            out.extend(fn(ctx, worlds))
        except Exception as exc:
            out.append(EvalResult(f"explain.{nm}", MODULE, nm, None, 0, DATASET, status="failed", reason=f"{type(exc).__name__}: {exc}"))
    out.extend([
        EvalResult("explain.semantic_faithfulness", MODULE, "NLI / LLM-judge faithfulness of explanation prose", None, 0, "none", status="skipped",
                   reason="needs a model-based judge; this suite audits claims structurally (edge/attribute lookup) with no LLM"),
        EvalResult("explain.human_usefulness", MODULE, "human rating of explanation helpfulness", None, 0, "none", status="skipped", reason="no human ratings available offline"),
    ])
    return out
