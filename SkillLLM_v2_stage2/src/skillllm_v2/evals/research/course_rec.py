"""L09 course recommendation layer: 8 systems + H6 ablation, WSGC@5 / prerequisite violations / learning-path validity,
explanation faithfulness (and whether "Explanation Completeness 1.0" is tautological), a label-independence audit, and the
held-out generated scenario set SG-v1.

    PYTHONPATH=src python -m skillllm_v2.evals.research.course_rec [--smoke] [--seed 42] [--n-boot 2000]

Definitions: Decisions.md D-A5-01..07 (fixed before any system was run). Shared helpers live in ``graph_retrieval``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, md_table, record, rel, write_layer
from skillllm_v2.evals.research.graph_retrieval import (BASE_PROV, CATALOG, LABELS, SCENARIOS, SPLIT, Env, _fig_bars, _fig_forest,
                                                       boot_ci, grades_of, holm, load_jsonl, load_or_build_sg, needs_prereq_hop, norm,
                                                       paired_test, q_gap, q_job, rrf, text_labels, user_known_visible, visible_gaps)

LAYER = "L09_course_rec"
DATASET = "course_catalog_large.jsonl (171 SYNTHETIC courses) + benchmark_scenarios.jsonl + relevance_labels.jsonl (v1-rule)"
SG_DATASET = "SG-v1 (240 generated SYNTHETIC scenarios, D-A5-07)"
K5, K10 = 5, 10

SYSTEMS = ["random", "lexical(BM25)", "semantic(MiniLM)", "skill_overlap", "skill_gap(greedy)", "hybrid(RRF)", "prerequisite_aware(greedy+filter)",
           "SkillLLM_v2 Full"]
EXTRA = ["Full-noPrereq (ablation)", "Full legacy (pre-tuning)", "lexical+filter", "semantic+filter", "hybrid+filter"]
H6_PAIRS = [("SkillLLM_v2 Full", "Full-noPrereq (ablation)", "primary: Full vs same ranker without prerequisite constraints/score"),
            ("prerequisite_aware(greedy+filter)", "skill_gap(greedy)", "greedy + filter vs greedy"),
            ("lexical+filter", "lexical(BM25)", "BM25 + filter vs BM25"),
            ("semantic+filter", "semantic(MiniLM)", "MiniLM + filter vs MiniLM"),
            ("hybrid+filter", "hybrid(RRF)", "hybrid + filter vs hybrid")]
METRICS = [("p5", "precision@5", "higher"), ("r5", "recall@5", "higher"), ("ndcg10", "ndcg@10", "higher"), ("mrr", "mrr@10", "higher"),
           ("wsgc5", "wsgc@5", "higher"), ("viol5", "prereq_violation_rate", "lower"), ("rsc5", "redundant_skill_coverage@5", "lower"),
           ("red5", "redundancy@5(existing)", "lower"), ("path_valid", "learning_path_validity", "higher"),
           ("path_repairable", "learning_path_repairable", "higher"), ("n_ret", "n_returned", "none")]
HEADLINE = {"wsgc5", "viol5", "ndcg10", "p5"}


# =====================================================================================================================
# metrics (D-A5-04)
# =====================================================================================================================
def wsgc_at_k(ids: Sequence[str], gold_gaps: Mapping[str, float], T: Mapping[str, Set[str]], k: int = K5) -> float:
    tot = sum(gold_gaps.values())
    if tot <= 0:
        return 1.0
    covered = set().union(*[T.get(c, set()) for c in ids[:k]]) if ids else set()
    return sum(w for g, w in gold_gaps.items() if g in covered) / tot


def violation_rate(ids: Sequence[str], K: Set[str], T: Mapping[str, Set[str]], P: Mapping[str, Set[str]], k: int = K5) -> float:
    top = list(ids[:k])
    if not top:
        return 0.0
    known, v = set(K), 0
    for c in top:
        if not P.get(c, set()) <= known:
            v += 1
        known |= T.get(c, set())
    return v / len(top)


def rsc_at_k(ids: Sequence[str], gold_gaps: Mapping[str, float], T: Mapping[str, Set[str]], k: int = K5) -> float:
    G = set(gold_gaps)
    inst = [T.get(c, set()) & G for c in ids[:k]]
    tot = sum(len(x) for x in inst)
    if tot == 0:
        return 0.0
    return (tot - len(set().union(*inst))) / tot


def path_validity(ids: Sequence[str], K: Set[str], T: Mapping[str, Set[str]], P: Mapping[str, Set[str]]) -> Tuple[float, float]:
    """(valid in the given order, repairable by reordering) for the whole returned list."""
    known, ok = set(K), True
    for c in ids:
        if not P.get(c, set()) <= known:
            ok = False
        known |= T.get(c, set())
    known, left, changed = set(K), list(ids), True
    while changed and left:
        changed = False
        for c in list(left):
            if P.get(c, set()) <= known:
                known |= T.get(c, set())
                left.remove(c)
                changed = True
    return float(ok), float(not left)


def list_metrics(ids: Sequence[str], grades: Mapping[str, float], gold_gaps: Mapping[str, float], K: Set[str], T, P, existing_red) -> Dict[str, Optional[float]]:
    from skillllm_v2.evals.research.graph_retrieval import mrr, ndcg, precision, recall
    pv, pr = path_validity(ids[:K10], K, T, P)
    return {"p5": precision(ids, grades, K5), "r5": recall(ids, grades, K5), "ndcg10": ndcg(ids, grades, K10), "mrr": mrr(ids, grades, K10),
            "wsgc5": wsgc_at_k(ids, gold_gaps, T), "viol5": violation_rate(ids, K, T, P), "rsc5": rsc_at_k(ids, gold_gaps, T), "red5": existing_red(ids),
            "path_valid": pv, "path_repairable": pr, "n_ret": float(len(ids[:K10]))}


# =====================================================================================================================
# explanation faithfulness (part 5)
# =====================================================================================================================
def _mentions(skill: str, text: str, resolver) -> bool:
    t = " " + norm(text) + " "
    keys = {norm(skill)} | {norm(k) for k in resolver.keys_for_name(skill)}
    return any(k and f" {k} " in t for k in keys)


def check_explanations(recs: Sequence[Dict[str, Any]], sc: Dict[str, Any], env: Env, G, resolver) -> List[Dict[str, Any]]:
    """Every claim in each explanation packet → {type, supported, detail}. Claims are read from the structured fields the
    explanation text is generated from (covered_missing_skills, evidence, prerequisite_status, unlocks, limitations)."""
    T, P = env.T, env.P
    job = sc["target_job"]
    req = {norm(s) for s in job.get("required_skills", [])}
    pref = {norm(s) for s in job.get("preferred_skills", [])}
    prof = {norm(s.get("canonical_skill_name") or s.get("surface_form") or ""): str(s.get("mastery_level", "proficient")).lower()
            for s in sc["user_profile"]["skills"]}
    gaps = visible_gaps(sc["user_profile"], job)
    rank = {r["course_id"]: i for i, r in enumerate(recs)}
    claims: List[Dict[str, Any]] = []

    def add(typ: str, ok: bool, detail: str, course: str) -> None:
        claims.append({"type": typ, "supported": bool(ok), "detail": detail, "course": course})

    for i, r in enumerate(recs):
        c = r["course_id"]
        crs = env.by_id[c]
        taught = T[c] | {v for _, v, d in G.out_edges(c, data=True) if d.get("type") == "TEACHES"}
        earlier = set().union(*[T[x["course_id"]] for x in recs[:i]]) if i else set()
        ev_by = {}
        for e in r.get("evidence", []):
            ev_by.setdefault(e.get("skill"), e)
        # (1) "covers missing skill X"
        for X in r.get("covered_missing_skills", []):
            e = ev_by.get(X, {})
            et = e.get("evidence_type")
            nx = norm(X)
            is_gap = nx in gaps or prof.get(nx) == "beginner"
            add("covers_gap_skill[strict field semantics]", is_gap and nx in taught, f"{c} covers {X} ({et or 'primary'})", c)
            if et == "text_mention_only":
                text = " ".join([crs.title, crs.description] + list(crs.learning_outcomes))
                tok = {t for sk in crs.teaches_skills for t in norm(sk).split() if len(t) >= 4}
                ok = is_gap and (_mentions(X, text, resolver) or any(t in tok for t in nx.split() if len(t) >= 4))
                add("covers_gap_skill[lenient: text-related]", ok, f"{c} text-related to {X}", c)
            elif et == "enabling_prerequisite":
                via, p = e.get("via_course_id"), norm(e.get("via_prerequisite_skill") or "")
                ok = is_gap and via in T and nx in T[via] and p in P[via] and p in taught
                add("covers_gap_skill[lenient: enables via bridge]", ok, f"{c} enables {X} via {via}/{p}", c)
            else:
                add("covers_gap_skill[lenient: teaches]", is_gap and nx in taught, f"{c} teaches {X}", c)
        # (2) job requirement + user gap classification + (3) outcome quote, per evidence entry
        for e in r.get("evidence", []):
            X = e.get("skill") or ""
            nx = norm(X)
            add("job_requires_skill", nx in req or nx in pref, f"job lists {X}", c)
            gc = e.get("gap_classification")
            held = nx in prof
            ok = {"missing_critical": (not held) and nx in req, "missing_preferred": (not held) and nx in pref and nx not in req,
                  "weak_skill": prof.get(nx) == "beginner"}.get(gc, False)
            add("user_gap_classification", ok, f"{X}: {gc}", c)
            if e.get("evidence_type") in (None, "enabling_prerequisite"):
                O = e.get("course_learning_outcome_evidence") or ""
                if e.get("evidence_type") == "enabling_prerequisite":
                    p = e.get("via_prerequisite_skill") or ""
                    add("outcome_quote_mentions_skill", _mentions(p, O, resolver), f"quote for prerequisite {p}: {O[:60]}", c)
                else:
                    add("outcome_quote_mentions_skill", _mentions(X, O, resolver), f"quote for {X}: {O[:60]}", c)
        # (4) prerequisite status
        ps = r.get("prerequisite_status", {})
        code = ps.get("status_code", "")
        if code == "satisfied_none_required":
            add("prerequisite_status", not P[c], f"{c} has no prerequisites", c)
        for p in ps.get("satisfied", []) or []:
            pid = norm(p.get("skill_id") or p.get("canonical_skill_name") or "") if isinstance(p, dict) else norm(str(p))
            by = str(p.get("satisfied_by") or "learner_profile") if isinstance(p, dict) else "learner_profile"
            if by.startswith("earlier_course:"):
                y = by.split(":", 1)[1]
                ok = pid in P[c] and y in rank and rank[y] < i and pid in T[y]
            else:
                ok = pid in P[c] and pid in prof
            add("prerequisite_satisfied_claim", ok, f"{c}: {pid} satisfied by {by}", c)
        if code == "all_satisfied":
            add("prerequisite_all_satisfied_claim", P[c] <= (set(prof) | earlier), f"{c}: all prerequisites satisfied", c)
        for p in ps.get("missing", []) or []:
            pid = norm(p.get("skill_id") or p.get("canonical_skill_name") or "") if isinstance(p, dict) else norm(str(p))
            add("prerequisite_missing_claim", pid in P[c] and pid not in prof, f"{c}: missing {pid}", c)
        # (5) unlock / bridge claims
        for u in r.get("unlocks", []) or []:
            uc, p = u["course_id"], norm(u["prerequisite_skill"])
            add("unlock_requires_prereq", uc in P and p in P[uc], f"{uc} requires {p}", c)
            add("unlock_this_teaches_prereq", p in taught, f"{c} teaches {p}", c)
            if u.get("in_list"):
                add("unlock_placement", uc in rank and rank[uc] > i and u.get("unlocked_course_rank") == rank[uc] + 1, f"{uc} later at {u.get('unlocked_course_rank')}", c)
            else:
                add("unlock_placement", not (uc in rank and rank[uc] > i), f"{uc} not later in list", c)
        # (6) limitations
        for lim in r.get("limitations", []):
            if lim.startswith("High workload"):
                add("limitation_workload", crs.duration_hours > 40.0, f"{c} {crs.duration_hours}h", c)
            elif lim.startswith("Contains partial skill overlap"):
                add("limitation_overlap", bool(T[c] & set(gaps) & earlier), f"{c} overlaps earlier picks on a gap skill", c)
            elif lim.startswith("Bridge course"):
                add("limitation_bridge", bool(r.get("unlocks")), f"{c} is a bridge", c)
            elif lim.startswith("Prerequisite gap"):
                add("limitation_prereq_gap", not P[c] <= (set(prof) | earlier), f"{c} has an unmet prerequisite", c)
    return claims


# =====================================================================================================================
# label-independence audit (part 6)
# =====================================================================================================================
def kappa(a: Sequence[int], b: Sequence[int], weights: Optional[str] = None) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b, weights=weights))


def visible_labels(sc: Dict[str, Any], catalog) -> Dict[str, Any]:
    """V-visible (D-A5-05): the unchanged v1 rule on system-visible needs (required=3, preferred=1, no implicit)."""
    from skillllm_v2.eval.benchmark_scenarios_spec import S
    from skillllm_v2.eval.build_relevance_labels import label_scenario
    sk = sc["user_profile"]["skills"]
    known = [s.get("canonical_skill_name") or s.get("surface_form") for s in sk if str(s.get("mastery_level", "")).lower() != "beginner"]
    weak = [s.get("canonical_skill_name") or s.get("surface_form") for s in sk if str(s.get("mastery_level", "")).lower() == "beginner"]
    job = sc["target_job"]
    spec = S(sid=sc["scenario_id"], background="", level="mid", known=", ".join(known), job_title=job["title"], text=job["text"],
             req=", ".join(job["required_skills"]), pref=", ".join(job["preferred_skills"]), implicit="", weak=", ".join(weak))
    return label_scenario(spec, catalog)


def label_audit(env: Env, scen: Mapping[str, Dict[str, Any]], labs: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    from scipy.stats import spearmanr
    src = (REPO_ROOT / "src/skillllm_v2/eval/build_relevance_labels.py").read_text()
    imports = sorted({(n.module if isinstance(n, ast.ImportFrom) else ",".join(a.name for a in n.names)) or ""
                      for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))})
    shared_code = [m for m in imports if any(x in m for x in ("recommend", "match", "pipeline", "graph", "ranker", "gap"))]
    v1, vc, vv, vt = [], [], [], []
    impl_share = []
    per_sc = {}
    for sid, sc in scen.items():
        lab = labs[sid]
        vis = visible_labels(sc, env.catalog)
        needs = sc["gold"]["needs"]
        known = set(sc["gold"]["known"])
        open_needs = {s: w for s, w in needs.items() if s not in known}
        txt = lab.get("labels_text") or text_labels(open_needs, env.courses)
        lab["labels_text"] = txt
        lab["labels_visible"] = {c: g for c, g in vis["final"].items() if g > 0}
        impl = set(sc["gold"].get("implicit_not_given_to_systems", []))
        tot = sum(open_needs.values())
        impl_share.append(sum(w for s, w in open_needs.items() if s in impl) / tot if tot else 0.0)
        a = [int(lab["labels"].get(c, 0)) for c in env.ids]
        per_sc[sid] = {"v1_rel": sum(1 for x in a if x >= 2), "visible_rel": sum(1 for c in env.ids if vis["final"][c] >= 2)}
        v1 += a
        vc += [int(lab["labels_content_only"].get(c, 0)) for c in env.ids]
        vv += [int(vis["final"][c]) for c in env.ids]
        vt += [int(txt.get(c, 0)) for c in env.ids]
    out = {"label_builder_imports": imports, "label_builder_imports_system_code": shared_code,
           "mean_implicit_share_of_open_need_mass": float(np.mean(impl_share)), "n_pairs": len(v1), "agreement": {}}
    for name, v in (("content_only", vc), ("v_visible", vv), ("v_text", vt)):
        out["agreement"][name] = {"kappa_rel_ge2": kappa([int(x >= 2) for x in v1], [int(x >= 2) for x in v]),
                                  "kappa_quadratic": kappa(v1, v, "quadratic"), "spearman": float(spearmanr(v1, v).correlation),
                                  "exact_grade_agreement": float(np.mean(np.asarray(v1) == np.asarray(v)))}
    out["shared_logic"] = [
        "Need/coverage: labels = exact-string intersection of catalogue teaches_skills with the need set; every skill-aware system (skill overlap, gap greedy, "
        "Full's gap_coverage and graph TEACHES edges) scores the same intersection.",
        "Prerequisite feasibility: labels give full grade only if P(c) ⊆ known ∪ weak and cap bridge-only courses at 1; the strict prerequisite mode "
        "(compat_basis='learner': prefer courses startable now), the '+ prereq filter' baselines and the violation metric implement the same catalogue rule.",
        "Weights differ: labels 3/2/1 (required/implicit/preferred) vs systems 1.0/0.7 (GapEngine double-applies 0.7 to preferred → 0.49).",
        "Independence that does exist: implicit needs are hidden from systems; labels never read any system output; no shared code (AST imports)."]
    return out


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).correlation)


# =====================================================================================================================
# per-scenario system lists
# =====================================================================================================================
def scenario_lists(env: Env, sc: Dict[str, Any], configs: Mapping[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[List[Dict[str, Any]], Any]]]:
    rb = env.rb
    sid = sc["scenario_id"]
    job = sc["target_job"]
    gaps = visible_gaps(sc["user_profile"], job)
    Kvis = user_known_visible(sc)
    L: Dict[str, List[str]] = {}
    L["random"] = env.rank_random(sid)[:K10]
    bm = env.rank_bm25(q_job(sc))
    de = env.rank_dense(q_job(sc))
    L["lexical(BM25)"] = bm[:K10]
    L["semantic(MiniLM)"] = de[:K10]
    L["skill_overlap"] = [c.course_id for _, c in rb.skill_overlap_rank(env.catalog, job)][:K10]
    greedy40 = [c.course_id for c in rb.gap_overlap_greedy(env.catalog, gaps, Kvis, False, 40)]
    L["skill_gap(greedy)"] = greedy40[:K10]
    hyb = rrf([env.rank_bm25(q_gap(sc)), env.rank_dense(q_gap(sc)), greedy40])
    L["hybrid(RRF)"] = hyb[:K10]
    L["prerequisite_aware(greedy+filter)"] = [c.course_id for c in rb.gap_overlap_greedy(env.catalog, gaps, Kvis, True, K10)]
    flt = lambda ids: [c.course_id for c in rb.apply_prereq_filter([env.by_id[i] for i in ids], Kvis, K10)]
    L["lexical+filter"] = flt(bm)
    L["semantic+filter"] = flt(de)
    L["hybrid+filter"] = flt(hyb)
    full: Dict[str, Tuple[List[Dict[str, Any]], Any]] = {}
    for name, cfg in configs.items():
        recs, G, _ = env.run_full(cfg, sc, top_k=K10)
        L[name] = [r["course_id"] for r in recs]
        full[name] = (recs, G)
    return L, full


def evaluate(env: Env, scens: Sequence[Dict[str, Any]], labs: Mapping[str, Dict[str, Any]], configs, resolver, explain: bool = True) -> Dict[str, Any]:
    from skillllm_v2.eval.course_recommendation_eval import explanation_completeness_rate, redundancy_at_k, weighted_gap_coverage_at_k
    T, P = env.T, env.P
    per: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(dict)
    claims: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    compl: Dict[str, Dict[str, float]] = defaultdict(dict)
    strata = {}
    lists_out = {}
    for sc in scens:
        sid = sc["scenario_id"]
        lab = labs[sid]
        known = set(sc["gold"]["known"])
        gold_gaps = {norm(s): float(w) for s, w in sc["gold"]["needs"].items() if s not in known}
        K = {norm(s) for s in list(sc["gold"]["known"]) + list(sc["gold"]["weak"])}
        assert K == user_known_visible(sc), f"{sid}: gold known/weak differ from the profile given to systems"
        gaps_list = [{"skill_name": s, "importance": float(w)} for s, w in sc["gold"]["needs"].items() if s not in known]

        def red(ids):
            return redundancy_at_k([env.rb.as_rec(env.by_id[c]) for c in ids], gaps_list, k=K5)

        L, full = scenario_lists(env, sc, configs)
        lists_out[sid] = L
        variants = {"v1": grades_of(lab, "labels"), "content_only": grades_of(lab, "labels_content_only")}
        for key, vname in (("labels_text", "v_text"), ("labels_visible", "v_visible")):
            if key in lab:
                variants[vname] = grades_of(lab, key)
        for s, ids in L.items():
            m = list_metrics(ids, variants["v1"], gold_gaps, K, T, P, red)
            chk = weighted_gap_coverage_at_k([env.rb.as_rec(env.by_id[c]) for c in ids], gaps_list, k=K5)
            assert abs(chk - m["wsgc5"]) < 1e-5, (sid, s, chk, m["wsgc5"])
            per[f"v1|{s}"][sid] = m
            for vname, gr in variants.items():
                if vname != "v1":
                    from skillllm_v2.evals.research.graph_retrieval import ndcg, precision
                    per[f"{vname}|{s}"][sid] = {"ndcg10": ndcg(ids, gr, K10), "p5": precision(ids, gr, K5)}
        strata[sid] = {"n_gaps": len(visible_gaps(sc["user_profile"], sc["target_job"])), "needs_prereq_hop": needs_prereq_hop(sc, lab, P)}
        if explain:
            for name, (recs, G) in full.items():
                claims[name][sid] = check_explanations(recs, sc, env, G, resolver)
                compl[name][sid] = explanation_completeness_rate(recs, k=K5)
            # tautology probes: (a) random courses with placeholder explanation fields; (b) Full's packets rotated across its own list
            rnd = [{**env.rb.as_rec(env.by_id[c]), "explanation": ["placeholder"], "evidence": ["placeholder"]} for c in L["random"]]
            compl["probe: random courses + placeholder fields"][sid] = explanation_completeness_rate(rnd, k=K5)
            recs, G = full["SkillLLM_v2 Full"]
            if len(recs) > 1:
                keys = ("covered_missing_skills", "evidence", "explanation_path", "prerequisite_status", "unlocks", "limitations")
                rot = [dict(r) for r in recs]
                for j, r in enumerate(rot):
                    src = recs[(j + 1) % len(recs)]
                    for k_ in keys:
                        r[k_] = src.get(k_)
                compl["probe: Full packets rotated to wrong courses"][sid] = explanation_completeness_rate(rot, k=K5)
                claims["probe: Full packets rotated to wrong courses"][sid] = check_explanations(rot, sc, env, G, resolver)
    return {"per": per, "claims": claims, "compl": compl, "strata": strata, "lists": lists_out}


def vals(per, ids, system, metric, variant="v1") -> Tuple[List[float], List[str]]:
    rows = per.get(f"{variant}|{system}", {})
    out, keep = [], []
    for i in ids:
        v = rows.get(i, {}).get(metric)
        if v is not None:
            out.append(v); keep.append(i)
    return out, keep


def paired(per, ids, a, b, metric, variant="v1"):
    ra, rb_ = per.get(f"{variant}|{a}", {}), per.get(f"{variant}|{b}", {})
    A, B = [], []
    for i in ids:
        x, y = ra.get(i, {}).get(metric), rb_.get(i, {}).get(metric)
        if x is not None and y is not None:
            A.append(x); B.append(y)
    return A, B


def faith_summary(claims_by_sid: Mapping[str, List[Dict[str, Any]]], ids: Sequence[str], strict: bool) -> Dict[str, Any]:
    def keep(c):
        if c["type"].startswith("covers_gap_skill["):
            return ("strict" in c["type"]) == strict
        return True
    rates, unsup, tot_s, tot = [], [], 0, 0
    by_type = defaultdict(lambda: [0, 0])
    for i in ids:
        cl = [c for c in claims_by_sid.get(i, []) if keep(c)]
        if not cl:
            continue
        s = sum(c["supported"] for c in cl)
        rates.append(s / len(cl)); unsup.append(len(cl) - s)
        tot_s += s; tot += len(cl)
        for c in cl:
            by_type[c["type"]][0] += c["supported"]; by_type[c["type"]][1] += 1
    return {"micro": tot_s / tot if tot else None, "per_scenario": rates, "unsupported_per_scenario": unsup, "n_claims": tot,
            "n_unsupported": tot - tot_s, "by_type": {k: {"supported": v[0], "total": v[1]} for k, v in sorted(by_type.items())}}


# =====================================================================================================================
# main
# =====================================================================================================================
def run(seed: int = 42, n_boot: int = 2000, smoke: bool = False) -> Path:
    from skillllm_v2.match.skill_keys import SkillKeyResolver
    from skillllm_v2.recommend.config import DEFAULT_CONFIG, LEGACY_CONFIG
    env = Env()
    _ = env.pipe
    resolver = SkillKeyResolver()
    out_dir = EVALS_ROOT / LAYER / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_head = 200 if smoke else max(n_boot, 10000)
    nb = 200 if smoke else n_boot
    configs = {"SkillLLM_v2 Full": DEFAULT_CONFIG, "Full-noPrereq (ablation)": DEFAULT_CONFIG.with_(prereq_mode="soft", w_prereq=0.0, w_unlock=0.0),
               "Full legacy (pre-tuning)": LEGACY_CONFIG}
    split = json.loads(SPLIT.read_text())
    scen = {s["scenario_id"]: s for s in load_jsonl(SCENARIOS)}
    labs = {r["scenario_id"]: r for r in load_jsonl(LABELS)}
    test_ids = split["test"][:4] if smoke else split["test"]
    dev_ids = split["dev"][:4] if smoke else split["dev"]
    audit_scen = {i: scen[i] for i in (test_ids + dev_ids)}
    audit = label_audit(env, audit_scen, labs)  # also attaches labels_text / labels_visible to labs
    res = evaluate(env, [scen[i] for i in test_ids + dev_ids], labs, configs, resolver)
    per, claims, compl, strata = res["per"], res["claims"], res["compl"], res["strata"]
    prov = BASE_PROV
    records: List[Dict[str, Any]] = []
    per_unit: List[Dict[str, Any]] = []
    allsys = SYSTEMS + EXTRA
    cav = ["Synthetic catalogue/scenarios; v1 labels read the same catalogue fields as skill-aware systems (see label audit).",
           "Prerequisite violation uses the same chained catalogue rule as the strict mode/filters: 0 for those systems is by construction."]
    table = {"test": [], "dev": []}
    for half, ids in (("test", test_ids), ("dev", dev_ids)):
        for s in allsys:
            row = {"system": s}
            for m, mname, d in METRICS:
                v, keep = vals(per, ids, s, m)
                mean, lo, hi = boot_ci(v, n_boot=nb_head if m in HEADLINE else nb, seed=seed)
                row[mname] = mean
                if half == "test":
                    records.append(record(LAYER, s, mname, mean, n_units=len(v), unit="scenario", split="test", dataset=DATASET, direction=d,
                                          ci=(lo, hi), ci_method=f"percentile bootstrap over scenarios (B={nb_head if m in HEADLINE else nb}, seed {seed})",
                                          provenance=prov, caveats=cav,
                                          details={"headline": bool(m in HEADLINE and s == "SkillLLM_v2 Full"), "role": "proposed" if s == "SkillLLM_v2 Full" else "baseline/ablation"}))
                    for i in keep:
                        per_unit.append({"layer": LAYER, "family": "rec", "system": s, "metric": m, "unit_id": i, "value": per[f"v1|{s}"][i][m],
                                         **({"reference": "SkillLLM_v2 Full"} if s != "SkillLLM_v2 Full" else {})})
            for vname in ("content_only", "v_text", "v_visible"):
                v, _ = vals(per, ids, s, "ndcg10", vname)
                row[f"ndcg@10[{vname}]"] = float(np.mean(v)) if v else None
            table[half].append(row)
    # reproduction of the stored benchmark-v2 test numbers (same systems, same labels, same split)
    v2p = REPO_ROOT / "reports/skillllm_v2/course_recommendation_benchmark_v2.json"
    repro = {}
    if v2p.exists() and not smoke:
        v2 = json.loads(v2p.read_text())["results"]["test"]["systems"]
        pairs = {"SkillLLM_v2 Full": "SkillLLM_v2 Full Graph Recommender", "Full legacy (pre-tuning)": "SkillLLM_v2 Full (legacy, pre-tuning)",
                 "prerequisite_aware(greedy+filter)": "Gap-Overlap Greedy + prereq filter", "skill_gap(greedy)": "Gap-Overlap Greedy (no graph)",
                 "skill_overlap": "Skill Overlap (user-agnostic)", "lexical(BM25)": "BM25 Keyword", "semantic(MiniLM)": "Dense Semantic",
                 "lexical+filter": "BM25 + prereq filter", "semantic+filter": "Dense + prereq filter"}
        for mine, theirs in pairs.items():
            for m, key in (("ndcg10", "ndcg10"), ("p5", "p5"), ("wsgc5", "gap_cov5"), ("viol5", "prereq_viol5")):
                v, _ = vals(per, test_ids, mine, m)
                d = float(np.mean(v)) - v2[theirs][key]["mean"]
                repro[f"{mine}|{m}"] = d
        mx = max(abs(x) for x in repro.values())
        records.append(record(LAYER, "reproduction vs course_recommendation_benchmark_v2.json", "max_abs_mean_difference", mx, n_units=len(test_ids),
                              unit="scenario", split="test", dataset=DATASET, direction="lower", provenance=prov + [v2p], details={"differences": repro},
                              caveats=["0 = the stored held-out benchmark numbers are reproduced exactly by this independent harness."]))
    # label-variant ndcg records (test)
    for s in allsys:
        for vname in ("content_only", "v_text", "v_visible"):
            v, _ = vals(per, test_ids, s, "ndcg10", vname)
            mean, lo, hi = boot_ci(v, n_boot=nb, seed=seed)
            records.append(record(LAYER, s, f"ndcg@10[labels={vname}]", mean, n_units=len(v), unit="scenario", split="test", dataset=DATASET,
                                  direction="higher", ci=(lo, hi), ci_method=f"percentile bootstrap (B={nb})", provenance=prov,
                                  details={"label_variant": vname, "family": "label-sensitivity"}))

    # ---- H6 --------------------------------------------------------------------------------------------------------------
    h6 = []
    for a, b, label in H6_PAIRS:
        Av, Bv = paired(per, test_ids, a, b, "viol5")
        An, Bn = paired(per, test_ids, a, b, "ndcg10")
        tv = paired_test(Av, Bv, n_boot=nb_head, seed=seed)
        tn = paired_test(An, Bn, n_boot=nb_head, seed=seed)
        h6.append({"a": a, "b": b, "label": label, "viol": tv, "ndcg": tn, "n": len(Av)})
    adj = holm([h["viol"]["p"] for h in h6])
    for h, pa in zip(h6, adj):
        h["p_holm_viol"] = pa
        records.append(record(LAYER, f"{h['a']} vs {h['b']}", "delta_prereq_violation_rate", h["viol"]["delta"], n_units=h["n"], unit="scenario", split="test",
                              dataset=DATASET, direction="lower", ci=(h["viol"]["ci_low"], h["viol"]["ci_high"]),
                              ci_method=f"paired bootstrap (B={nb_head}) + sign-flip permutation", provenance=prov,
                              details={"family": "H6", "p_raw": h["viol"]["p"], "p_holm": pa, "comparison": h["label"]}))
        records.append(record(LAYER, f"{h['a']} vs {h['b']}", "delta_ndcg@10", h["ndcg"]["delta"], n_units=h["n"], unit="scenario", split="test",
                              dataset=DATASET, direction="higher", ci=(h["ndcg"]["ci_low"], h["ndcg"]["ci_high"]),
                              ci_method=f"paired bootstrap (B={nb_head})", provenance=prov,
                              details={"family": "H6", "non_inferior_margin": -0.02, "non_inferior": h["ndcg"]["ci_low"] > -0.02, "p_raw": h["ndcg"]["p"],
                                       "comparison": h["label"]}))
    # H6 sensitivity: content-only labels (the v1 bridge cap rewards startable-now courses)
    h6_sens = {}
    for vname in ("content_only", "v_text"):
        A, B = paired(per, test_ids, "SkillLLM_v2 Full", "Full-noPrereq (ablation)", "ndcg10", vname)
        t = paired_test(A, B, n_boot=nb, seed=seed)
        h6_sens[vname] = t
        records.append(record(LAYER, "SkillLLM_v2 Full vs Full-noPrereq (ablation)", f"delta_ndcg@10[labels={vname}]", t["delta"], n_units=t["n"], unit="scenario",
                              split="test", dataset=DATASET, direction="higher", ci=(t["ci_low"], t["ci_high"]), ci_method=f"paired bootstrap (B={nb})",
                              provenance=prov, details={"family": "H6-label-sensitivity (descriptive)", "p": t["p"]}))

    p0 = h6[0]
    viol_ok = p0["viol"]["ci_high"] < 0
    ndcg_ok = p0["ndcg"]["ci_low"] > -0.02
    verdict = "SUPPORTED" if (viol_ok and ndcg_ok) else "NOT SUPPORTED"
    rule = "H6: prerequisite-aware vs same ranker without prerequisite constraints: violation rate lower with CI excluding 0, AND nDCG@10 non-inferior (lower CI bound of Δ > −0.02)"
    records.append(record(LAYER, "SkillLLM_v2 Full vs Full-noPrereq (ablation)", "H6_verdict", p0["viol"]["delta"], n_units=p0["n"], unit="scenario", split="test",
                          dataset=DATASET, direction="lower", ci=(p0["viol"]["ci_low"], p0["viol"]["ci_high"]),
                          ci_method=f"paired bootstrap (B={nb_head}) + sign-flip permutation (10000)", provenance=prov,
                          caveats=["The violation criterion is satisfied by construction when the strict mode enforces the same rule the metric checks; nDCG non-inferiority is the informative half."],
                          details={"verdict": verdict, "rule": rule, "p_raw": p0["viol"]["p"], "p_holm": p0["p_holm_viol"], "family": "H6",
                                   "violation_ci_excludes_0": viol_ok, "ndcg_delta": p0["ndcg"]["delta"], "ndcg_ci": [p0["ndcg"]["ci_low"], p0["ndcg"]["ci_high"]],
                                   "ndcg_non_inferior": ndcg_ok,
                                   "label_sensitivity": {k_: {"delta": t_["delta"], "ci": [t_["ci_low"], t_["ci_high"]], "non_inferior": t_["ci_low"] > -0.02}
                                                         for k_, t_ in h6_sens.items()},
                                   "secondary_pairs": [{"comparison": h["label"], "viol_delta": h["viol"]["delta"], "viol_ci": [h["viol"]["ci_low"], h["viol"]["ci_high"]],
                                                        "p_holm": h["p_holm_viol"], "ndcg_delta": h["ndcg"]["delta"], "ndcg_ci_low": h["ndcg"]["ci_low"]} for h in h6[1:]]}))
    for k_, h in enumerate(h6):
        for s in (h["a"], h["b"]):
            for m, fam_ in (("viol5", "H6"), ("ndcg10", "H6-ndcg10-noninferiority")):
                for i in test_ids:
                    v = per[f"v1|{s}"].get(i, {}).get(m)
                    if v is not None:
                        per_unit.append({"layer": LAYER, "family": fam_, "bucket": f"pair{k_}", "system": s, "metric": m, "unit_id": i, "value": v,
                                         **({"reference": h["b"]} if s == h["a"] else {})})
    # ---- D-B4-01: Full vs the other 7 primary systems (descriptive family) + dev-selected strongest graph-free baseline -----------------
    FULL = "SkillLLM_v2 Full"
    baselines = [s_ for s_ in SYSTEMS if s_ != FULL]
    dev_means = {s_: float(np.mean(vals(per, dev_ids, s_, "ndcg10")[0])) for s_ in baselines}
    strongest_dev = max(dev_means, key=dev_means.get)
    PRESPEC = "prerequisite_aware(greedy+filter)"
    vsb: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for m, mname, d in METRICS:
        if m not in ("ndcg10", "p5", "wsgc5", "viol5"):
            continue
        tests = {}
        for b_ in baselines:
            A, B = paired(per, test_ids, FULL, b_, m)
            tests[b_] = paired_test(A, B, n_boot=nb_head, seed=seed)
        adj_ = holm([tests[b_]["p"] for b_ in baselines])
        for b_, pa in zip(baselines, adj_):
            t = tests[b_]
            t["p_holm"] = pa
            t["verdict"] = ("Full significantly " + ("higher" if t["delta"] > 0 else "lower") if pa < 0.05 else "not distinguishable (Holm p >= 0.05)")
            records.append(record(LAYER, f"{FULL} vs {b_}", f"delta_{mname}[vs-baselines]", t["delta"], n_units=t["n"], unit="scenario", split="test",
                                  dataset=DATASET, direction=d, ci=(t["ci_low"], t["ci_high"]),
                                  ci_method=f"paired bootstrap (B={nb_head}) + sign-flip permutation (10000)", provenance=prov,
                                  details={"family": "L09-vs-baselines (descriptive, D-B4-01)", "p_raw": t["p"], "p_holm": pa, "holm_over": "7 baselines per metric",
                                           "wins": t["wins"], "losses": t["losses"], "ties": t["ties"], "verdict_text": t["verdict"],
                                           "is_prespecified_strongest": b_ == PRESPEC, "is_dev_selected_strongest": b_ == strongest_dev}))
        vsb[m] = tests
        for b_ in baselines:
            for i in test_ids:
                for s_ in (FULL, b_):
                    v = per[f"v1|{s_}"].get(i, {}).get(m)
                    if v is not None:
                        per_unit.append({"layer": LAYER, "family": f"L09-vs-baselines[{m}]", "bucket": b_, "system": s_, "metric": m, "unit_id": i,
                                         "value": v, **({"reference": b_} if s_ == FULL else {})})
    records.append(record(LAYER, "dev-selected strongest graph-free baseline", "dev_mean_ndcg@10", dev_means[strongest_dev], n_units=len(dev_ids), unit="scenario",
                          split="dev", dataset=DATASET, direction="higher", provenance=prov,
                          details={"selected": strongest_dev, "prespecified": PRESPEC, "dev_means": dev_means,
                                   "note": "selection uses the DEV (tuning) half only; Full is tuned on the same half, so dev numbers are optimistic context"}))

    # ---- explanation faithfulness + completeness tautology ------------------------------------------------------------------
    faith = {}
    for name in list(claims):
        for strict in (True, False):
            fs = faith_summary(claims[name], test_ids, strict)
            key = f"{name}|{'strict' if strict else 'lenient'}"
            faith[key] = fs
            if fs["micro"] is None:
                continue
            m, lo, hi = boot_ci(fs["per_scenario"], n_boot=nb_head if name == "SkillLLM_v2 Full" else nb, seed=seed)
            mu, ulo, uhi = boot_ci(fs["unsupported_per_scenario"], n_boot=nb, seed=seed)
            records.append(record(LAYER, name, f"explanation_faithfulness[{'strict' if strict else 'lenient'}]", fs["micro"], n_units=len(fs["per_scenario"]),
                                  unit="scenario", split="test", dataset=DATASET, direction="higher", ci=(lo, hi),
                                  ci_method=f"bootstrap over scenarios of per-scenario faithfulness (B={nb_head if name == 'SkillLLM_v2 Full' else nb}); value = micro rate",
                                  provenance=prov, details={"n_claims": fs["n_claims"], "n_unsupported": fs["n_unsupported"], "per_scenario_mean": m,
                                                            "by_type": fs["by_type"]},
                                  caveats=["Claims are read from the structured fields the explanation text is rendered from; each is checked against the catalogue, the job/profile and the per-scenario KG.",
                                           "strict = 'covered_missing_skills' read literally (the course teaches X); lenient = honours the packet's own evidence_type qualifier (text-related / enables via bridge)."]))
            records.append(record(LAYER, name, f"unsupported_claims_per_scenario[{'strict' if strict else 'lenient'}]", mu, n_units=len(fs["per_scenario"]), unit="scenario",
                                  split="test", dataset=DATASET, direction="lower", ci=(ulo, uhi), ci_method=f"percentile bootstrap (B={nb})", provenance=prov))
    for name, by in compl.items():
        v = [by[i] for i in test_ids if i in by]
        if v:
            mean, lo, hi = boot_ci(v, n_boot=nb, seed=seed)
            records.append(record(LAYER, name, "explanation_completeness@5(existing metric)", mean, n_units=len(v), unit="scenario", split="test", dataset=DATASET,
                                  direction="none", ci=(lo, hi), ci_method=f"percentile bootstrap (B={nb})", provenance=prov,
                                  caveats=["Existing metric checks field presence only: 'teaches_skills' and 'prerequisites' exist on every catalogue record, so 2 of its 4 conditions are always true."]))
    for i in test_ids:
        fs = [c for c in claims["SkillLLM_v2 Full"].get(i, []) if not c["type"].startswith("covers_gap_skill[") or "lenient" in c["type"]]
        if fs:
            per_unit.append({"layer": LAYER, "family": "faithfulness", "system": "SkillLLM_v2 Full", "metric": "explanation_faithfulness_lenient",
                             "unit_id": i, "value": sum(c["supported"] for c in fs) / len(fs)})
    unsup_examples = [c for i in test_ids for c in claims["SkillLLM_v2 Full"].get(i, []) if not c["supported"]][:40]

    # ---- label audit records -------------------------------------------------------------------------------------------------
    for vname, ag in audit["agreement"].items():
        for k, v in ag.items():
            records.append(record(LAYER, f"labels v1 vs {vname}", k, v, n_units=audit["n_pairs"], unit="(scenario, course) pair", split="test+dev",
                                  dataset=DATASET, direction="none", provenance=prov,
                                  details={"note": "agreement, not quality; pairs within a scenario are not independent (no CI)"}))
    ordering = {}
    sys_order = [s for s in allsys]
    base = [float(np.mean(vals(per, test_ids, s, "ndcg10")[0])) for s in sys_order]
    for vname in ("content_only", "v_text", "v_visible"):
        other = [float(np.mean(vals(per, test_ids, s, "ndcg10", vname)[0])) for s in sys_order]
        ordering[vname] = kendall_tau(base, other)
        records.append(record(LAYER, f"system ordering v1 vs {vname}", "kendall_tau_of_system_means", ordering[vname], n_units=len(sys_order), unit="system",
                              split="test", dataset=DATASET, direction="none", provenance=prov,
                              details={"systems": sys_order, "v1": base, vname: other}))
    # oracle that implements the label rule on system-visible inputs
    orc = []
    from skillllm_v2.evals.research.graph_retrieval import ndcg
    for i in test_ids:
        vg = grades_of(labs[i], "labels_visible")
        order = sorted(env.ids, key=lambda c: (-vg.get(c, 0), c))
        x = ndcg(order, grades_of(labs[i], "labels"), K10)
        if x is not None:
            orc.append(x)
    orc_m, orc_lo, orc_hi = boot_ci(orc, n_boot=nb, seed=seed)
    records.append(record(LAYER, "oracle: v1 rule applied to system-visible inputs", "ndcg@10", orc_m, n_units=len(orc), unit="scenario", split="test",
                          dataset=DATASET, direction="higher", ci=(orc_lo, orc_hi), ci_method=f"percentile bootstrap (B={nb})", provenance=prov,
                          caveats=["Ranks courses by V-visible grade (ties by id): what a system that merely re-implements the label rule on its own inputs scores against the gold labels."]))
    records.append(record(LAYER, "label_builder", "imports_system_code", float(len(audit["label_builder_imports_system_code"])), n_units=1, unit="module", split="all",
                          dataset="src/skillllm_v2/eval/build_relevance_labels.py", direction="lower", provenance=[REPO_ROOT / "src/skillllm_v2/eval/build_relevance_labels.py"],
                          details={"imports": audit["label_builder_imports"]}))
    records.append(record(LAYER, "gold needs", "implicit_share_of_open_need_mass", audit["mean_implicit_share_of_open_need_mass"], n_units=len(audit_scen),
                          unit="scenario", split="test+dev", dataset=DATASET, direction="none", provenance=prov))

    # ---- SG-v1 (held out) ------------------------------------------------------------------------------------------------------
    sg_s, sg_l, sg_prov = load_or_build_sg(env, smoke=smoke)
    sg_lab = {r["scenario_id"]: r for r in sg_l}
    sg = evaluate(env, sg_s, sg_lab, configs, resolver, explain=True)
    sg_ids = [s["scenario_id"] for s in sg_s]
    sg_rows = []
    for s in allsys:
        row = {"system": s}
        for m, mname, d in METRICS:
            v, keep = vals(sg["per"], sg_ids, s, m)
            mean, lo, hi = boot_ci(v, n_boot=nb, seed=seed)
            row[mname] = mean
            if m in HEADLINE or m in ("r5", "mrr", "path_valid"):
                records.append(record(LAYER, s, f"{mname}[SG-v1]", mean, n_units=len(v), unit="generated scenario", split="generated_heldout", dataset=SG_DATASET,
                                      direction=d, ci=(lo, hi), ci_method=f"percentile bootstrap (B={nb})", provenance=sg_prov + [CATALOG]))
            if m in ("ndcg10", "viol5", "wsgc5", "p5"):
                for i in keep:
                    per_unit.append({"layer": LAYER, "family": "rec_SG-v1", "system": s, "metric": m, "unit_id": i, "value": sg["per"][f"v1|{s}"][i][m]})
        sg_rows.append(row)
    sg_h6 = []
    for a, b, label in H6_PAIRS:
        for sname, cond in (("all", lambda i: True), ("needs prerequisite hop", lambda i: sg["strata"][i]["needs_prereq_hop"])):
            ids = [i for i in sg_ids if cond(i)]
            Av, Bv = paired(sg["per"], ids, a, b, "viol5")
            An, Bn = paired(sg["per"], ids, a, b, "ndcg10")
            if not Av:
                continue
            tv, tn = paired_test(Av, Bv, n_boot=nb, seed=seed), paired_test(An, Bn, n_boot=nb, seed=seed)
            sg_h6.append({"label": f"SG {sname}: {label}", "n": len(Av), "viol": tv, "ndcg": tn})
            records.append(record(LAYER, f"{a} vs {b}", f"delta_prereq_violation_rate[SG-v1 {sname}]", tv["delta"], n_units=len(Av), unit="generated scenario",
                                  split="generated_heldout", dataset=SG_DATASET, direction="lower", ci=(tv["ci_low"], tv["ci_high"]),
                                  ci_method=f"paired bootstrap (B={nb})", provenance=sg_prov, details={"family": "H6-SG (secondary)", "p": tv["p"]}))
            records.append(record(LAYER, f"{a} vs {b}", f"delta_ndcg@10[SG-v1 {sname}]", tn["delta"], n_units=len(An), unit="generated scenario",
                                  split="generated_heldout", dataset=SG_DATASET, direction="higher", ci=(tn["ci_low"], tn["ci_high"]),
                                  ci_method=f"paired bootstrap (B={nb})", provenance=sg_prov,
                                  details={"family": "H6-SG (secondary)", "p": tn["p"], "non_inferior": tn["ci_low"] > -0.02}))
    sg_vs = []
    for m_, mname_, d_ in METRICS:
        if m_ not in ("ndcg10", "p5", "wsgc5", "viol5"):
            continue
        ts = {}
        for b_ in (PRESPEC, "skill_gap(greedy)"):
            A, B = paired(sg["per"], sg_ids, FULL, b_, m_)
            ts[b_] = paired_test(A, B, n_boot=nb, seed=seed)
        for b_, pa in zip(ts, holm([t_["p"] for t_ in ts.values()])):
            t_ = ts[b_]
            t_["p_holm"] = pa
            sg_vs.append({"baseline": b_, "metric": m_, **t_})
            records.append(record(LAYER, f"{FULL} vs {b_}", f"delta_{mname_}[SG-v1 vs-baselines, post-hoc]", t_["delta"], n_units=t_["n"], unit="generated scenario",
                                  split="generated_heldout", dataset=SG_DATASET, direction=d_, ci=(t_["ci_low"], t_["ci_high"]),
                                  ci_method=f"paired bootstrap (B={nb}) + sign-flip permutation", provenance=sg_prov,
                                  details={"family": "L09-SG-vs-baselines (exploratory, post-hoc D-B4-02)", "p_raw": t_["p"], "p_holm": pa,
                                           "wins": t_["wins"], "losses": t_["losses"], "ties": t_["ties"]}))
    sg_faith = faith_summary(sg["claims"]["SkillLLM_v2 Full"], sg_ids, strict=False)
    sg_faith_s = faith_summary(sg["claims"]["SkillLLM_v2 Full"], sg_ids, strict=True)
    for tag, fs in (("lenient", sg_faith), ("strict", sg_faith_s)):
        if fs["micro"] is not None:
            m, lo, hi = boot_ci(fs["per_scenario"], n_boot=nb, seed=seed)
            records.append(record(LAYER, "SkillLLM_v2 Full", f"explanation_faithfulness[{tag}][SG-v1]", fs["micro"], n_units=len(fs["per_scenario"]),
                                  unit="generated scenario", split="generated_heldout", dataset=SG_DATASET, direction="higher", ci=(lo, hi),
                                  ci_method=f"bootstrap of per-scenario rate (B={nb}); value = micro", provenance=sg_prov, details={"by_type": fs["by_type"]}))

    # ---- figures ---------------------------------------------------------------------------------------------------------------
    for m, mname in (("ndcg10", "nDCG@10"), ("wsgc5", "WSGC@5"), ("viol5", "prerequisite violation rate@5")):
        rows = []
        for s in allsys:
            v, _ = vals(per, test_ids, s, m)
            mean, lo, hi = boot_ci(v, n_boot=nb, seed=seed)
            rows.append({"name": s, "mean": mean, "lo": lo, "hi": hi})
        _fig_bars(out_dir / f"fig_{m}_test.png", rows, f"L09 held-out test half (n={len(test_ids)}): {mname}, 95% CI", mname)
    forest = []
    for h in h6:
        forest.append({"name": f"viol Δ {h['label']}", "n": h["n"], "delta": h["viol"]["delta"], "lo": h["viol"]["ci_low"], "hi": h["viol"]["ci_high"]})
        forest.append({"name": f"nDCG Δ {h['label']}", "n": h["n"], "delta": h["ndcg"]["delta"], "lo": h["ndcg"]["ci_low"], "hi": h["ndcg"]["ci_high"]})
    for h in sg_h6:
        forest.append({"name": f"viol Δ {h['label']}", "n": h["n"], "delta": h["viol"]["delta"], "lo": h["viol"]["ci_low"], "hi": h["viol"]["ci_high"]})
        forest.append({"name": f"nDCG Δ {h['label']}", "n": h["n"], "delta": h["ndcg"]["delta"], "lo": h["ndcg"]["ci_low"], "hi": h["ndcg"]["ci_high"]})
    _fig_forest(out_dir / "fig_h6_forest.png", forest, "H6: Δ (prerequisite-aware − same ranker without), paired 95% CI", "Δ")
    ft = faith["SkillLLM_v2 Full|lenient"]["by_type"] if "SkillLLM_v2 Full|lenient" in faith else {}
    ft.update({k: v for k, v in faith.get("SkillLLM_v2 Full|strict", {}).get("by_type", {}).items() if "strict" in k})
    frows = [{"name": k, "mean": v["supported"] / v["total"], "lo": v["supported"] / v["total"], "hi": v["supported"] / v["total"], "n": v["total"]}
             for k, v in ft.items() if v["total"]]
    if frows:
        _fig_bars(out_dir / "fig_faithfulness_by_claim_type.png", frows, "Full recommender (test): share of claims supported, by claim type", "supported share")

    with (out_dir / "per_unit_scores.jsonl").open("w") as fh:
        for r in per_unit:
            fh.write(json.dumps(r) + "\n")
    (out_dir / "label_audit_details.json").write_text(json.dumps(audit, indent=2))
    (out_dir / "unsupported_claim_examples.json").write_text(json.dumps(unsup_examples, indent=2))
    (out_dir / "rankings_test.json").write_text(json.dumps({i: res["lists"][i] for i in test_ids}, indent=2))

    md = report(records, table, h6, verdict, h6_sens, faith, compl, audit, ordering, orc_m, (orc_lo, orc_hi), sg_rows, sg_h6, sg_faith, sg_faith_s,
                test_ids, strata, unsup_examples, nb_head, orc, vsb, strongest_dev, PRESPEC, dev_means, sg_vs)
    write_layer(LAYER, records, md, seed=seed, smoke=smoke, extra_meta={"n_boot": nb, "n_boot_headline": nb_head})
    return out_dir


def _f(x, d: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def report(records, table, h6, verdict, h6_sens, faith, compl, audit, ordering, orc_m, orc_ci, sg_rows, sg_h6, sg_faith, sg_faith_s, test_ids, strata,
           unsup, nb_head, orc, vsb, strongest_dev, prespec, dev_means, sg_vs) -> str:
    R = {(r["system"], r["metric"]): r for r in records}
    fs = faith.get("SkillLLM_v2 Full|lenient", {})
    fss = faith.get("SkillLLM_v2 Full|strict", {})
    rot = faith.get("probe: Full packets rotated to wrong courses|lenient", {})
    c_full = R.get(("SkillLLM_v2 Full", "explanation_completeness@5(existing metric)"), {}).get("value")
    c_rand = R.get(("probe: random courses + placeholder fields", "explanation_completeness@5(existing metric)"), {}).get("value")
    c_rot = R.get(("probe: Full packets rotated to wrong courses", "explanation_completeness@5(existing metric)"), {}).get("value")
    p0 = h6[0]
    f_full, f_rot = fs.get("micro"), rot.get("micro")
    taut = (c_rand is not None and c_rot is not None and f_full is not None and f_rot is not None
            and c_rand >= 0.99 and c_rot >= 0.99 and f_rot < f_full - 0.05)  # rule fixed in D-B4-01(3)
    taut_txt = "**the completeness metric is tautological** (rule of D-B4-01(3) met)" if taut else "tautology NOT demonstrated by the D-B4-01(3) rule"
    pf = vsb["ndcg10"][prespec]
    pfw = vsb["wsgc5"][prespec]
    L = ["# L09 — Course recommendation: systems, WSGC@5, prerequisites, explanation faithfulness, label audit, H6", "",
         "Owner A5. Definitions pre-registered in `Decisions.md` D-A5-01..07 (D-A5-04 = formal metric definitions incl. WSGC@5). "
         f"Primary = 24 held-out TEST scenarios; unit = scenario; headline CIs B={nb_head}. Catalogue and scenarios are SYNTHETIC.", "",
         "## Headline (test half, SkillLLM_v2 Full, mean [95% CI])", ""]
    for m in ("wsgc@5", "prereq_violation_rate", "ndcg@10", "precision@5"):
        r = R.get(("SkillLLM_v2 Full", m), {})
        L.append(f"- {m}: {_f(r.get('value'))} [{_f(r.get('ci_low'))}, {_f(r.get('ci_high'))}]")
    L += [f"- **H6 verdict: {verdict}.** Full − Full-noPrereq: violation Δ {p0['viol']['delta']:+.3f} [{p0['viol']['ci_low']:+.3f}, {p0['viol']['ci_high']:+.3f}] "
          f"(Holm p {p0['p_holm_viol']:.3f}); nDCG@10 Δ {p0['ndcg']['delta']:+.3f} [{p0['ndcg']['ci_low']:+.3f}, {p0['ndcg']['ci_high']:+.3f}] (margin −0.02).",
          f"- Explanation faithfulness (Full, test): lenient {_f(fs.get('micro'))} ({fs.get('n_unsupported')} unsupported of {fs.get('n_claims')} claims); "
          f"strict field semantics {_f(fss.get('micro'))}. Existing Explanation Completeness = {_f(c_full)} for Full, {_f(c_rand)} for random courses with placeholder "
          f"fields, {_f(c_rot)} for Full packets attached to the WRONG courses (faithfulness of those: {_f(rot.get('micro'))}) → {taut_txt}.",
          f"- Full vs the pre-specified strongest graph-free baseline `{prespec}` (test, n={pf['n']}): nDCG@10 Δ {pf['delta']:+.3f} [{pf['ci_low']:+.3f}, {pf['ci_high']:+.3f}], "
          f"Holm p {pf['p_holm']:.3f} → {pf['verdict']}; WSGC@5 Δ {pfw['delta']:+.3f} [{pfw['ci_low']:+.3f}, {pfw['ci_high']:+.3f}], Holm p {pfw['p_holm']:.3f}. "
          f"Strongest baseline chosen on DEV only: `{strongest_dev}` (dev nDCG@10 {dev_means[strongest_dev]:.3f}).",
          f"- Label audit: a ranker that just applies the label rule to the systems' own visible inputs reaches nDCG@10 {_f(orc_m)} [{_f(orc_ci[0])}, {_f(orc_ci[1])}] against the gold labels; "
          f"κ(v1, v_visible) = {_f(audit['agreement']['v_visible']['kappa_rel_ge2'])}, κ(v1, v_text) = {_f(audit['agreement']['v_text']['kappa_rel_ge2'])}.", ""]
    L += ["## 1. Systems (test half, held out)", ""]
    cols = ["system"] + [m for _, m, _ in METRICS] + ["ndcg@10[content_only]", "ndcg@10[v_text]", "ndcg@10[v_visible]"]
    L += [md_table(table["test"], cols), "", "CIs (test): " + "; ".join(
        f"{s} {m} {_f(R[(s, m)]['value'])} [{_f(R[(s, m)]['ci_low'])}, {_f(R[(s, m)]['ci_high'])}]" for s in ("SkillLLM_v2 Full", "prerequisite_aware(greedy+filter)", "hybrid(RRF)", "lexical(BM25)")
        for m in ("ndcg@10", "wsgc@5", "prereq_violation_rate") if (s, m) in R), ""]
    L += ["### Dev half (TUNED-ON for Full; context only)", "", md_table(table["dev"], cols), ""]
    L += ["## 2. H6 — prerequisite awareness", "", f"Rule: {R[('SkillLLM_v2 Full vs Full-noPrereq (ablation)', 'H6_verdict')]['details']['rule']}.", "",
          "| pair (test) | n | Δ violation [95% CI] | Holm p (violation family) | Δ nDCG@10 [95% CI] | non-inferior (−0.02)? |", "|---|---|---|---|---|---|"]
    for h in h6:
        L.append(f"| {h['label']} | {h['n']} | {h['viol']['delta']:+.3f} [{h['viol']['ci_low']:+.3f}, {h['viol']['ci_high']:+.3f}] | {h['p_holm_viol']:.3f} | "
                 f"{h['ndcg']['delta']:+.3f} [{h['ndcg']['ci_low']:+.3f}, {h['ndcg']['ci_high']:+.3f}] | {'yes' if h['ndcg']['ci_low'] > -0.02 else 'no'} |")
    L += ["", f"**Verdict: {verdict}** (primary pair). The violation half holds by construction for strict/filtered rankers; read the nDCG column. "
          "Label sensitivity of the primary nDCG Δ (post-hoc, D-B4-02): " + "; ".join(
              f"{k}: {t['delta']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] → non-inferiority (lower CI > −0.02) {'holds' if t['ci_low'] > -0.02 else 'FAILS'}"
              for k, t in h6_sens.items()) + ". The formal verdict is the pre-registered one (v1 labels); the nDCG half is therefore label-rule dependent whenever a variant FAILS.", ""]
    L += ["### Full vs the other 7 systems (descriptive, D-B4-01; Holm over the 7 baselines per metric; test half)", "",
          "| baseline | nDCG@10 Δ [CI] (Holm p) | P@5 Δ [CI] (Holm p) | WSGC@5 Δ [CI] (Holm p) | violation Δ [CI] (Holm p) |", "|---|---|---|---|---|"]
    for b_ in vsb["ndcg10"]:
        cells = []
        for m_ in ("ndcg10", "p5", "wsgc5", "viol5"):
            t = vsb[m_][b_]
            cells.append(f"{t['delta']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] ({t['p_holm']:.3f})")
        L.append(f"| {b_}{' (pre-specified strongest)' if b_ == prespec else ''}{' (dev-selected strongest)' if b_ == strongest_dev else ''} | " + " | ".join(cells) + " |")
    L.append("")
    if sg_h6:
        L += ["SG-v1 (240 generated, held out; secondary):", "", "| pair | n | Δ violation [CI] | Δ nDCG@10 [CI] |", "|---|---|---|---|"]
        for h in sg_h6:
            L.append(f"| {h['label']} | {h['n']} | {h['viol']['delta']:+.3f} [{h['viol']['ci_low']:+.3f}, {h['viol']['ci_high']:+.3f}] | "
                     f"{h['ndcg']['delta']:+.3f} [{h['ndcg']['ci_low']:+.3f}, {h['ndcg']['ci_high']:+.3f}] |")
        L.append("")
    if sg_vs:
        L += ["SG-v1 Full vs strongest graph-free / greedy baselines (exploratory, post-hoc D-B4-02; Holm over 2 per metric):", "",
              "| baseline | metric | n | Δ [CI] | Holm p |", "|---|---|---|---|---|"]
        for t in sg_vs:
            L.append(f"| {t['baseline']} | {t['metric']} | {t['n']} | {t['delta']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | {t['p_holm']:.3f} |")
        L.append("")
    L += ["## 3. Explanation faithfulness", "",
          "Every claim in each Full explanation packet is checked against the catalogue, the job/profile and the per-scenario knowledge graph "
          "(covered gap skill, job requirement, user gap class, quoted learning outcome mentions the skill, prerequisite status, unlock/bridge claims, limitations).", "",
          "| claim type (test, Full) | supported / total |", "|---|---|"]
    for k, v in sorted({**fs.get("by_type", {}), **{a: b for a, b in fss.get("by_type", {}).items() if "strict" in a}}.items()):
        L.append(f"| {k} | {v['supported']} / {v['total']} |")
    L += ["", "Faithfulness of the other Full variants (lenient): " + "; ".join(
        f"{k.split('|')[0]} {_f(v['micro'])}" for k, v in faith.items() if k.endswith("|lenient") and v.get("micro") is not None), "",
          f"SG-v1 Full faithfulness: lenient {_f(sg_faith.get('micro'))}, strict {_f(sg_faith_s.get('micro'))}.", "",
          f"**Is 'Explanation Completeness 1.0' tautological? {'Yes' if taut else 'Not demonstrated'} (rule D-B4-01(3)).** The existing metric (`course_recommendation_eval.explanation_completeness_rate`) only checks that four "
          "fields are non-empty; `teaches_skills` and `prerequisites` are present on every catalogue record, so two conditions always hold, and any list-valued "
          f"`explanation`/`evidence` satisfies the rest. Random courses with placeholder fields score {_f(c_rand)}; Full's own packets moved onto the wrong courses still score "
          f"{_f(c_rot)} while their faithfulness drops to {_f(rot.get('micro'))}.", "",
          "Unsupported-claim examples (first 12): " + "; ".join(f"[{c['type']}] {c['detail']}" for c in unsup[:12]), ""]
    ag = audit["agreement"]
    L += ["## 4. Label-independence audit", "",
          f"- Code: `build_relevance_labels.py` imports {audit['label_builder_imports']} — no recommender/graph code ({len(audit['label_builder_imports_system_code'])} system modules).",
          f"- Implicit needs (hidden from every system) carry {100 * audit['mean_implicit_share_of_open_need_mass']:.1f}% of the open-need mass on average.", ""]
    L += [f"- {s}" for s in audit["shared_logic"]] + [""]
    L += ["| v1 vs | κ (grade ≥ 2) | quadratic κ | Spearman ρ | exact agreement | Kendall τ of system ordering (test nDCG@10) |", "|---|---|---|---|---|---|"]
    for k, v in ag.items():
        L.append(f"| {k} | {_f(v['kappa_rel_ge2'])} | {_f(v['kappa_quadratic'])} | {_f(v['spearman'])} | {_f(v['exact_grade_agreement'])} | {_f(ordering.get(k))} |")
    L += ["", f"Oracle 'v1 rule on system-visible inputs' nDCG@10 vs gold (test): {_f(orc_m)} [{_f(orc_ci[0])}, {_f(orc_ci[1])}]. "
          "Interpretation: the gold is largely reconstructible from the systems' own inputs with the label rule; differences come mainly from implicit needs and "
          "weights. A system tuned toward this rule (as the Full recommender's dev-tuned strict prerequisite mode is) gains on v1 partly by matching the rule, "
          "which the content-only and V-text columns expose.", ""]
    L += ["## 5. Generated held-out set SG-v1 (240 scenarios; secondary)", "", md_table(sg_rows, ["system", "precision@5", "ndcg@10", "wsgc@5", "prereq_violation_rate",
                                                                                          "learning_path_validity", "n_returned"]), ""]
    L += ["## Caveats", "",
          "- Synthetic catalogue, scenarios and labels; human validation of labels is pending (L11).",
          "- n = 24 test scenarios; bootstrap ignores label uncertainty; scenarios share catalogue/archetypes.",
          "- SG-v1 is generated by the author's procedure (template job text favours lexical matching); it tests consistency of conclusions, not realism.",
          "- Violation rate and the strict/filter mechanisms share the same catalogue prerequisite rule (by-construction zeros).", "",
          "## Reproduce", "", "```", "cd SkillLLM_v2_stage2 && PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.course_rec --seed 42 --n-boot 2000", "```"]
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
