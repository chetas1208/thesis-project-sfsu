"""Job-matching evaluation on a deterministic synthetic benchmark.

Ground truth comes from *authored* role archetypes (inputs/evals/skill_universe.json) and an authored grading
rule over true skill sets (importance tiers + adjacency credit + core-skill caps).  It never uses the matcher's
own output.  Caveat (stated in every result): the matcher and the grading rule both reason about required-skill
coverage, so absolute numbers are optimistic; the informative quantities are the baselines, the robustness
deltas, the tie behaviour and the property checks.

Everything is stdlib + the existing ``skillllm_v2.eval.ranking_metrics`` (read-only) and runs on CPU in seconds.
"""

from __future__ import annotations

import json
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from skillllm_v2.eval.ranking_metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from skillllm_v2.evals.contract import EvalContext, EvalResult, bootstrap_ci, paired_bootstrap
from skillllm_v2.match.job_matcher import JobMatcher
from skillllm_v2.schema import normalize_text

MODULE = "match"
CORE_WEIGHT, IMPORTANT_WEIGHT, NICE_WEIGHT = 3.0, 2.0, 1.0
FIXTURE = ("inputs", "evals", "skill_universe.json")
CAVEATS = [
    "synthetic benchmark: authored role archetypes, not real postings",
    "ground-truth grading (tiered coverage + adjacency credit) shares the notion of required-skill coverage with the matcher, so absolute scores are optimistic",
]


# --------------------------------------------------------------------------------------------- fixtures
@dataclass
class Universe:
    roles: Dict[str, Dict[str, List[str]]]
    adjacency: Dict[str, set]  # skill -> set of adjacent skills (symmetric)
    aliases: Dict[str, List[str]]
    distractors: List[str]

    @property
    def all_skills(self) -> List[str]:
        seen: Dict[str, None] = {}
        for r in self.roles.values():
            for s in r["required"] + r["preferred"]:
                seen.setdefault(s, None)
        return sorted(seen)


def load_universe(ctx: EvalContext) -> Universe:
    path = ctx.path(*FIXTURE)
    if not path.exists():  # fall back to the repo copy when ctx.root is elsewhere
        path = Path(__file__).resolve().parents[3].joinpath(*FIXTURE)
    data = json.loads(path.read_text(encoding="utf-8"))
    adj: Dict[str, set] = {}
    for group in data["adjacency"]:
        for s in group:
            adj.setdefault(s, set()).update(x for x in group if x != s)
    return Universe(data["roles"], adj, data["aliases"], list(data["distractor_skills"]))


# --------------------------------------------------------------------------------------------- generation
def make_jobs(uni: Universe, rng: random.Random, n_variants: int) -> List[Dict[str, Any]]:
    """Jobs with truth tiers.  `tiers` maps skill -> weight (3 core, 2 important, 1 nice)."""
    pool = uni.all_skills
    jobs: List[Dict[str, Any]] = []
    for role in sorted(uni.roles):
        spec = uni.roles[role]
        for v in range(n_variants):
            req, pref = list(spec["required"]), list(spec["preferred"])
            if len(req) > 3 and rng.random() < 0.25:
                pref.append(req.pop())
            if rng.random() < 0.4:
                extra = rng.choice([s for s in pool if s not in req and s not in pref])
                pref.append(extra)
            tiers = {s: (CORE_WEIGHT if i < 2 else IMPORTANT_WEIGHT) for i, s in enumerate(req)}
            tiers.update({s: NICE_WEIGHT for s in pref})
            jobs.append({
                "job_id": f"{role}-{v}", "role": role, "required_skills": req, "preferred_skills": pref,
                "semantic_role_fit": 0.5, "experience_level_fit": 0.5, "user_preference_fit": 0.5,
                "_tiers": tiers, "_core": req[:2],
            })
    return jobs


def make_candidates(uni: Universe, rng: random.Random, n: int) -> List[Dict[str, Any]]:
    roles = sorted(uni.roles)
    pool = uni.all_skills
    out: List[Dict[str, Any]] = []
    for i in range(n):
        home = roles[i % len(roles)]
        spec = uni.roles[home]
        held: List[str] = []
        for s in spec["required"]:
            if rng.random() < 0.7:
                held.append(s)
            elif uni.adjacency.get(s) and rng.random() < 0.3:
                held.append(rng.choice(sorted(uni.adjacency[s])))
        for s in spec["preferred"]:
            if rng.random() < 0.5:
                held.append(s)
        for _ in range(rng.randint(0, 3)):
            held.append(rng.choice(pool))
        for _ in range(rng.randint(0, 2)):
            held.append(rng.choice(uni.distractors))
        skills = list(dict.fromkeys(held))
        if len(skills) < 2:
            skills = list(dict.fromkeys(skills + spec["required"][:2]))
        rng.shuffle(skills)
        out.append({"user_id": f"cand-{i:03d}", "home_role": home, "skills": skills})
    return out


def grade_pair(uni: Universe, cand_skills: Sequence[str], job: Dict[str, Any]) -> int:
    """Authored graded relevance 0..3 from TRUE skill sets."""
    have = set(cand_skills)

    def cred(s: str) -> float:
        if s in have:
            return 1.0
        return 0.5 if uni.adjacency.get(s, set()) & have else 0.0

    tiers = job["_tiers"]
    cov = sum(w * cred(s) for s, w in tiers.items()) / sum(tiers.values())
    grade = 3 if cov >= 0.85 else 2 if cov >= 0.6 else 1 if cov >= 0.35 else 0
    core = [cred(s) for s in job["_core"]]
    if any(c == 0.0 for c in core):
        grade = min(grade, 1)
    elif any(c < 1.0 for c in core):
        grade = min(grade, 2)
    return grade


def clean_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


# --------------------------------------------------------------------------------------------- systems
ScoreFn = Callable[[Dict[str, Any], Dict[str, Any]], float]


def matcher_score(user: Dict[str, Any], job: Dict[str, Any]) -> float:
    return float(JobMatcher().score(user, job)["match_score"])


def keyword_jaccard(user: Dict[str, Any], job: Dict[str, Any]) -> float:
    u = {normalize_text(s) for s in user["skills"]}
    j = {normalize_text(s) for s in job["required_skills"] + job["preferred_skills"]}
    return len(u & j) / max(1, len(u | j))


def keyword_count(user: Dict[str, Any], job: Dict[str, Any]) -> float:
    u = {normalize_text(s) for s in user["skills"]}
    j = {normalize_text(s) for s in job["required_skills"] + job["preferred_skills"]}
    return float(len(u & j))


def rank_ids(scores: Dict[str, float], tie: str = "id", rng: Optional[random.Random] = None,
             grades: Optional[Dict[str, float]] = None) -> List[str]:
    ids = list(scores)
    if tie == "id":
        return sorted(ids, key=lambda j: (-scores[j], j))
    if tie == "random":
        assert rng is not None
        rng.shuffle(ids)
        return sorted(ids, key=lambda j: -scores[j])  # stable: ties keep shuffled order
    if tie in ("pessimistic", "optimistic"):
        assert grades is not None
        sign = 1 if tie == "pessimistic" else -1
        return sorted(ids, key=lambda j: (-scores[j], sign * grades[j]))
    raise ValueError(tie)


def score_matrix(fn: ScoreFn, cands: Sequence[Dict[str, Any]], jobs: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    cj = [clean_job(j) for j in jobs]
    return [{j["job_id"]: fn(c, j) for j in cj} for c in cands]


def hit_at_1(ranked: Sequence[str], grades: Dict[str, float]) -> Optional[float]:
    """1 if the top-ranked job has the maximum authored grade; undefined (None) when no job reaches grade 2."""
    best = max(grades.values()) if grades else 0.0
    if best < 2.0 or not ranked:
        return None
    return 1.0 if grades.get(ranked[0], 0.0) >= best else 0.0


def metrics_for(ranked: Sequence[str], grades: Dict[str, float]) -> Dict[str, Optional[float]]:
    return {
        "ndcg_at_5": ndcg_at_k(ranked, grades, 5),
        "ndcg_at_10": ndcg_at_k(ranked, grades, 10),
        "map_at_10": average_precision_at_k(ranked, grades, 10, 2.0),
        "recall_at_5": recall_at_k(ranked, grades, 5, 2.0),
        "hit_at_1": hit_at_1(ranked, grades),
    }


def eval_system(mats: List[Dict[str, float]], grades_all: List[Dict[str, float]], tie: str = "id",
                seed: int = 0) -> Dict[str, List[Optional[float]]]:
    rng = random.Random(seed)
    out: Dict[str, List[Optional[float]]] = {"ndcg_at_5": [], "ndcg_at_10": [], "map_at_10": [], "recall_at_5": [], "hit_at_1": []}
    for sc, gr in zip(mats, grades_all):
        r = rank_ids(sc, tie, rng, gr)
        for k, v in metrics_for(r, gr).items():
            out[k].append(v)
    return out


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def _clean(vals: Sequence[Optional[float]]) -> List[float]:
    return [v for v in vals if v is not None]


def _res(name: str, metric: str, vals: Sequence[float], ctx: EvalContext, dataset: str, details: Optional[dict] = None,
         caveats: Optional[List[str]] = None, salt: int = 0, status: str = "ok", reason: str = "",
         module: str = MODULE) -> EvalResult:
    vals = list(vals)
    if not vals:
        return EvalResult(f"{module}.{name}", module, metric, None, 0, dataset, None, details or {}, caveats or [], "skipped",
                          reason or "no units")
    return EvalResult(f"{module}.{name}", module, metric, _mean(vals), len(vals), dataset,
                      bootstrap_ci(vals, ctx.n_boot, ctx.seed + salt), details or {}, caveats if caveats is not None else list(CAVEATS),
                      status, reason)


def _control(name: str, metric: str, value: float, passed: bool, expected: str, n: int, dataset: str,
             module: str = MODULE) -> EvalResult:
    return EvalResult(f"{module}.control.{name}", module, metric, float(value), n, dataset, None,
                      {"control": True, "passed": bool(passed), "expected": expected}, [])


def _rate_result(name: str, metric: str, viol: int, n: int, dataset: str, details: dict, caveats: List[str],
                 module: str = MODULE) -> EvalResult:
    rate = viol / n if n else None
    ci = None
    if n:
        ci = bootstrap_ci([1.0] * viol + [0.0] * (n - viol), 400, 7)
    d = dict(details)
    d.update({"direction": "lower", "violations": viol})
    return EvalResult(f"{module}.{name}", module, metric, rate, n, dataset, ci, d, caveats,
                      "ok" if n else "skipped", "" if n else "no cases")


# --------------------------------------------------------------------------------------------- perturbations
def perturb_skills(uni: Universe, skills: Sequence[str], kind: str, level: float, rng: random.Random) -> List[str]:
    if kind == "alias":
        out = []
        for s in skills:
            al = uni.aliases.get(s)
            out.append(rng.choice(al) if al and rng.random() < level else s)
        return out
    if kind == "case":
        return [s.lower() for s in skills]
    if kind == "noise":
        return list(skills) + [rng.choice(uni.distractors) for _ in range(int(level))]
    if kind == "missing":
        kept = [s for s in skills if rng.random() >= level]
        return kept or list(skills[:1])
    raise ValueError(kind)


# --------------------------------------------------------------------------------------------- property checks
def _rand_case(uni: Universe, rng: random.Random) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pool = uni.all_skills
    req = rng.sample(pool, rng.randint(1, 5))
    pref = rng.sample([s for s in pool if s not in req], rng.randint(0, 4))
    user_sk = rng.sample(pool, rng.randint(0, 8))
    job = {"job_id": "j", "required_skills": req, "preferred_skills": pref, "semantic_role_fit": round(rng.random(), 2),
           "experience_level_fit": round(rng.random(), 2), "user_preference_fit": round(rng.random(), 2)}
    return {"user_id": "u", "skills": user_sk}, job


def property_checks(uni: Universe, seed: int, n: int, dataset: str) -> List[EvalResult]:
    rng = random.Random(f"{seed}:match-props")
    m = JobMatcher()
    mono_v = bound_v = noise_v = case_v = alias_v = 0
    worst_mono = None
    case_example = None
    for _ in range(n):
        user, job = _rand_case(uni, rng)
        base = m.score(user, job)["match_score"]
        if not 0.0 <= base <= 1.0:
            bound_v += 1
        cand_add = [s for s in job["required_skills"] + job["preferred_skills"] if s not in user["skills"]]
        if cand_add:
            s = rng.choice(cand_add)
            after = m.score({"user_id": "u", "skills": user["skills"] + [s]}, job)["match_score"]
            if after < base - 1e-12:
                mono_v += 1
                worst_mono = worst_mono or {"added": s, "before": base, "after": after}
        noise = {"user_id": "u", "skills": user["skills"] + [rng.choice(uni.distractors)]}
        if abs(m.score(noise, job)["match_score"] - base) > 1e-12:
            noise_v += 1
        low = {"user_id": "u", "skills": [s.lower() for s in user["skills"]]}
        if user["skills"] and abs(m.score(low, job)["match_score"] - base) > 1e-12:
            case_v += 1
            case_example = case_example or {"user": user["skills"], "job_required": job["required_skills"],
                                            "score_orig": base, "score_lower": m.score(low, job)["match_score"]}
        # alias: replace one held-and-needed skill by its alias; a normalising matcher would keep the score
        held_needed = [s for s in user["skills"] if s in job["required_skills"] + job["preferred_skills"] and uni.aliases.get(s)]
        if held_needed:
            s = rng.choice(held_needed)
            aliased = {"user_id": "u", "skills": [uni.aliases[s][0] if x == s else x for x in user["skills"]]}
            if abs(m.score(aliased, job)["match_score"] - base) > 1e-12:
                alias_v += 1
    cav = ["property check over seeded random cases; expectation is 0 violations"]
    return [
        _rate_result("prop.score_in_unit_interval", "violation_rate", bound_v, n, dataset, {}, cav),
        _rate_result("prop.adding_required_or_preferred_skill_never_lowers_score", "violation_rate", mono_v, n, dataset,
                     {"example": worst_mono}, cav),
        _rate_result("prop.irrelevant_skill_invariance", "violation_rate", noise_v, n, dataset, {}, cav),
        _rate_result("prop.case_invariance", "violation_rate", case_v, n, dataset, {"example": case_example},
                     cav + ["REGRESSION GUARD (bug fixed, docs/BUGFIX_LOG_MATCH.md): JobMatcher compares skill strings exactly (case-sensitive) so 'python' != 'Python'"]),
        _rate_result("prop.alias_invariance", "violation_rate", alias_v, n, dataset, {},
                     cav + ["REGRESSION GUARD (bug fixed, docs/BUGFIX_LOG_MATCH.md): JobMatcher has no alias resolution; an alias surface form ('Postgres') scores as a missing skill"]),
    ]


def invariance_checks(uni: Universe, seed: int, n: int, dataset: str) -> List[EvalResult]:
    """Fairness-style sanity: score must not depend on attributes that are irrelevant to skills, nor on list order."""
    rng = random.Random(f"{seed}:match-inv")
    m = JobMatcher()
    max_delta = 0.0
    worst = None
    cases = 0
    for _ in range(n):
        user, job = _rand_case(uni, rng)
        base = m.score(user, job)["match_score"]
        variants = {
            "user_id": ({**user, "user_id": "someone-else"}, job),
            "protected_attrs": ({**user, "name": "Alex", "gender": rng.choice(["f", "m", "x"]), "age": rng.randint(18, 70),
                                 "nationality": "XX"}, job),
            "job_id_title": (user, {**job, "job_id": "zzz", "title": "Different Title", "company": "ACME"}),
            "skill_order": ({**user, "skills": rng.sample(user["skills"], len(user["skills"]))},
                            {**job, "required_skills": rng.sample(job["required_skills"], len(job["required_skills"])),
                             "preferred_skills": rng.sample(job["preferred_skills"], len(job["preferred_skills"]))}),
            "duplicate_user_skills": ({**user, "skills": user["skills"] + user["skills"]}, job),
        }
        for k, (u2, j2) in variants.items():
            d = abs(m.score(u2, j2)["match_score"] - base)
            cases += 1
            if d > max_delta:
                max_delta, worst = d, k
    res = EvalResult(f"{MODULE}.invariance.irrelevant_attributes_and_order", MODULE, "max_abs_score_delta", max_delta, cases,
                     dataset, None, {"direction": "lower", "worst_variant": worst,
                                     "variants": ["user_id", "protected_attrs(name/gender/age/nationality)", "job_id/title/company",
                                                  "skill list order", "duplicated user skills"]},
                     ["scores are invariant by construction if the matcher only reads skills; this guards regressions that would add leakage of protected attributes"])
    res2 = _control("invariance_zero_delta", "max_abs_score_delta", max_delta, max_delta <= 1e-9,
                    "score identical under irrelevant attribute / order changes", cases, dataset)
    return [res, res2]


# --------------------------------------------------------------------------------------------- hand-check
def _hand_ndcg_control(dataset: str) -> EvalResult:
    # grades 3,2,0,1 ; ranking [b,a,c,d]  -> DCG = (2^2-1)/log2(2) + (2^3-1)/log2(3) + 0 + (2^1-1)/log2(5)
    grades = {"a": 3, "b": 2, "c": 0, "d": 1}
    got = ndcg_at_k(["b", "a", "c", "d"], grades, 4)
    dcg = 3 / math.log2(2) + 7 / math.log2(3) + 0 + 1 / math.log2(5)
    idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4) + 0
    want = dcg / idcg
    return _control("metric_hand_computed_ndcg", "abs_error", abs(got - want), abs(got - want) < 1e-9,
                    "ndcg_at_k equals the closed-form hand computation", 1, dataset)


# --------------------------------------------------------------------------------------------- main
def run(ctx: EvalContext) -> List[EvalResult]:
    uni = load_universe(ctx)
    smoke = ctx.smoke
    n_c = 24 if smoke else 112
    n_v = 2 if smoke else 3
    rng = random.Random(f"{ctx.seed}:match-bench")
    jobs = make_jobs(uni, rng, n_v)
    cands = make_candidates(uni, rng, n_c)
    ds = f"synthetic-match-v1(seed={ctx.seed},candidates={n_c},jobs={len(jobs)})"
    grades_all = [{j["job_id"]: float(grade_pair(uni, c["skills"], j)) for j in jobs} for c in cands]
    results: List[EvalResult] = []

    mat = score_matrix(matcher_score, cands, jobs)
    kj = score_matrix(keyword_jaccard, cands, jobs)
    kc = score_matrix(keyword_count, cands, jobs)
    rrng = random.Random(f"{ctx.seed}:match-rand")
    rand_mats = [[{jid: rrng.random() for jid in g} for g in grades_all] for _ in range(10)]

    sys_m = {
        "matcher": eval_system(mat, grades_all),
        "keyword_jaccard": eval_system(kj, grades_all),
        "keyword_count": eval_system(kc, grades_all),
    }
    rand_runs = [eval_system(rm, grades_all) for rm in rand_mats]
    sys_m["random"] = {k: [_mean([run_[k][i] for run_ in rand_runs]) if rand_runs[0][k][i] is not None else None
                           for i in range(len(cands))] for k in rand_runs[0]}

    labels = {"matcher": "matcher", "keyword_jaccard": "baseline.keyword_jaccard", "keyword_count": "baseline.keyword_count",
              "random": "baseline.random"}
    for key, prefix in labels.items():
        for metric in ("ndcg_at_10", "ndcg_at_5", "map_at_10", "recall_at_5", "hit_at_1"):
            if key != "matcher" and metric not in ("ndcg_at_10", "map_at_10", "hit_at_1"):
                continue
            vals = _clean(sys_m[key][metric])
            extra = {"k": int(metric.rsplit("_", 1)[1]), "relevance_threshold_for_map_recall": 2, "grades": "0..3 authored",
                     "scenarios_total": len(sys_m[key][metric]), "dropped_undefined_scenarios": len(sys_m[key][metric]) - len(vals),
                     "dropped_rule": "candidates with no relevant job (ideal DCG = 0 / no grade>=2 job) are DROPPED from the mean (repo convention); trec_eval/ranx would count them as 0"}
            if metric == "map_at_10":
                extra["ap_convention"] = "AP@k divides by min(R,k) (src/skillllm_v2/eval/ranking_metrics.py); trec_eval/ranx divide by R -- see match.map_at_10_trec_convention"
            cav = list(CAVEATS) + (["random baseline = mean over 10 random score draws per candidate"] if key == "random" else [])
            results.append(_res(f"{prefix}.{metric}" if key != "matcher" else f"{metric}", metric, vals, ctx, ds, extra, cav,
                                salt=zlib.crc32(f"{key}:{metric}".encode()) % 97))

    # AP under the trec_eval convention (divide by R, undefined scenarios counted as 0) for comparability with ir_measures / ranx
    trec_ap, n_zero = [], 0
    for sc, gr in zip(mat, grades_all):
        R = sum(1 for g in gr.values() if g >= 2.0)
        if R == 0:
            trec_ap.append(0.0)
            n_zero += 1
            continue
        hits, tot = 0, 0.0
        for i, jid in enumerate(rank_ids(sc)[:10], start=1):
            if gr[jid] >= 2.0:
                hits += 1
                tot += hits / i
        trec_ap.append(tot / R)
    results.append(_res("map_at_10_trec_convention", "map_at_10_trec", trec_ap, ctx, ds,
                        {"ap_convention": "AP@10 / R (trec_eval, ranx); scenarios with R=0 counted as 0", "scenarios_total": len(trec_ap),
                         "zero_relevant_scenarios_counted_as_zero": n_zero, "relevance_threshold": 2},
                        list(CAVEATS) + ["differs from match.map_at_10 (min(R,k) divisor, zero-relevant scenarios dropped)"], salt=61))

    # paired comparisons over candidates where the metric is defined
    for base in ("keyword_jaccard", "keyword_count", "random"):
        a, b = sys_m["matcher"]["ndcg_at_10"], sys_m[base]["ndcg_at_10"]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if pairs:
            pb = paired_bootstrap([p[0] for p in pairs], [p[1] for p in pairs], ctx.n_boot, ctx.seed)
            results.append(EvalResult(f"{MODULE}.paired_vs_{base}.ndcg_at_10", MODULE, "delta_ndcg_at_10", pb["delta"], len(pairs), ds,
                                      (pb["ci_low"], pb["ci_high"]), {"p_value": pb["p"], "baseline": base}, list(CAVEATS)))

    # ties
    n_pairs = sum(len(g) for g in grades_all)
    zero = sum(1 for sc in mat for v in sc.values() if v == 0.0)
    results.append(_res("score_zero_clamp_rate", "share_of_scores_equal_0", [1.0 if v == 0.0 else 0.0 for sc in mat for v in sc.values()],
                        ctx, ds, {"direction": "lower", "pairs": n_pairs, "zero_scores": zero},
                        ["FINDING if high: the matcher clamps to 0, so weak matches are indistinguishable (ties)"], salt=3))
    tie_top10 = []
    for sc in mat:
        top = sorted(sc.values(), reverse=True)[:10]
        tie_top10.append(sum(1 for a, b in zip(top, top[1:]) if a == b) / 9.0)
    results.append(_res("tie_rate_top10", "adjacent_equal_score_share_in_top10", tie_top10, ctx, ds,
                        {"direction": "lower"}, ["ties are broken by job_id in the headline numbers"], salt=4))
    pess = eval_system(mat, grades_all, "pessimistic")["ndcg_at_10"]
    opt = eval_system(mat, grades_all, "optimistic")["ndcg_at_10"]
    rnd = eval_system(mat, grades_all, "random", seed=ctx.seed)["ndcg_at_10"]
    spread = [o - p for o, p in zip(opt, pess) if o is not None and p is not None]
    results.append(_res("tie_sensitivity_spread", "ndcg_at_10_optimistic_minus_pessimistic_tiebreak", spread, ctx, ds,
                        {"direction": "lower", "mean_pessimistic": _mean(_clean(pess)), "mean_optimistic": _mean(_clean(opt)),
                         "mean_random_tiebreak": _mean(_clean(rnd))},
                        ["a large spread means the reported nDCG depends on how a consumer breaks ties, which the matcher leaves undefined"],
                        salt=5))

    # robustness
    clean = sys_m["matcher"]["ndcg_at_10"]
    clean_k = sys_m["keyword_jaccard"]["ndcg_at_10"]

    def robust(name: str, kind: str, level: float, extra_job: Optional[Callable] = None) -> None:
        prng = random.Random(f"{ctx.seed}:match-rob:{name}")
        pc = [{**c, "skills": perturb_skills(uni, c["skills"], kind, level, prng)} for c in cands]
        pj = jobs
        if extra_job:
            jr = random.Random(f"{ctx.seed}:match-rob-jobs:{name}")
            pj = [extra_job(j, jr) for j in jobs]
        m_ = eval_system(score_matrix(matcher_score, pc, pj), grades_all)["ndcg_at_10"]
        k_ = eval_system(score_matrix(keyword_jaccard, pc, pj), grades_all)["ndcg_at_10"]
        idx = [i for i in range(len(cands)) if m_[i] is not None and clean[i] is not None]
        if not idx:
            return
        pb = paired_bootstrap([m_[i] for i in idx], [clean[i] for i in idx], ctx.n_boot, ctx.seed)
        pbk = paired_bootstrap([k_[i] for i in idx], [clean_k[i] for i in idx], ctx.n_boot, ctx.seed)
        results.append(EvalResult(
            f"{MODULE}.robust.{name}.ndcg_at_10_delta", MODULE, "ndcg_at_10_delta_vs_clean", pb["delta"], len(idx), ds,
            (pb["ci_low"], pb["ci_high"]),
            {"perturbation": kind, "level": level, "ndcg_at_10_perturbed": _mean([m_[i] for i in idx]),
             "ndcg_at_10_clean": _mean([clean[i] for i in idx]), "keyword_jaccard_delta": pbk["delta"],
             "keyword_jaccard_delta_ci": [pbk["ci_low"], pbk["ci_high"]], "p_value": pb["p"], "direction": "higher"},
            list(CAVEATS) + ["ground truth is held fixed; only the matcher's input surface is perturbed"]))

    robust("alias_25pct", "alias", 0.25)
    robust("alias_50pct", "alias", 0.5)
    robust("lowercase_user_skills", "case", 1.0)
    robust("noise_skills_3", "noise", 3)
    robust("noise_skills_8", "noise", 8)
    robust("missing_25pct", "missing", 0.25)
    robust("missing_50pct", "missing", 0.5)

    def noisy_fit(j, r):
        return {**j, "semantic_role_fit": r.random(), "experience_level_fit": r.random(), "user_preference_fit": r.random()}

    robust("random_job_fit_fields", "noise", 0, noisy_fit)

    # properties + invariance
    n_prop = 60 if smoke else 600
    results.extend(property_checks(uni, ctx.seed, n_prop, "synthetic-random-cases"))
    results.extend(invariance_checks(uni, ctx.seed, 40 if smoke else 300, "synthetic-random-cases"))

    # controls
    oracle = []
    for gr in grades_all:
        r = sorted(gr, key=lambda j: (-gr[j], j))
        oracle.append(ndcg_at_k(r, gr, 10))
    ov = _clean(oracle)
    results.append(_control("oracle_ranking_ndcg_at_10", "ndcg_at_10", _mean(ov), _mean(ov) > 0.999,
                            "ranking by ground-truth grades scores nDCG@10 = 1", len(ov), ds))
    rev = _clean([ndcg_at_k(sorted(gr, key=lambda j: (gr[j], j)), gr, 10) for gr in grades_all])
    rnd_mean = _mean(_clean(sys_m["random"]["ndcg_at_10"]))
    results.append(_control("anti_oracle_ndcg_at_10", "ndcg_at_10", _mean(rev), _mean(rev) < rnd_mean,
                            "worst-first ranking scores below the random baseline", len(rev), ds))
    # label shuffle: pair each candidate with ANOTHER candidate's grades
    shift = max(1, len(cands) // 3)
    shuf_vals = []
    for i, sc in enumerate(mat):
        g2 = grades_all[(i + shift) % len(cands)]
        r = rank_ids(sc)
        v = ndcg_at_k(r, g2, 10)
        if v is not None:
            shuf_vals.append(v)
    clean_mean = _mean(_clean(clean))
    shuf_mean = _mean(shuf_vals)
    results.append(_control("label_shuffle_ndcg_at_10", "ndcg_at_10", shuf_mean,
                            shuf_mean < clean_mean - 0.1 and shuf_mean < rnd_mean + 0.2,
                            "matcher ranking evaluated against another candidate's labels collapses toward the random baseline",
                            len(shuf_vals), ds))
    results.append(_control("random_baseline_below_matcher", "ndcg_at_10", rnd_mean, rnd_mean < clean_mean,
                            "random ranking scores below the matcher (benchmark has headroom)", len(cands), ds))
    results.append(_hand_ndcg_control(ds))
    again = eval_system(score_matrix(matcher_score, cands, jobs), grades_all)["ndcg_at_10"]
    results.append(_control("benchmark_determinism", "max_abs_diff", max(abs((a or 0) - (b or 0)) for a, b in zip(again, clean)),
                            all((a or 0) == (b or 0) for a, b in zip(again, clean)),
                            "re-running the benchmark with the same seed reproduces nDCG exactly", len(again), ds))
    return results
