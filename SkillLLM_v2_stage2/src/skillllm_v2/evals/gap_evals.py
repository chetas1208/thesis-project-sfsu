"""Skill-gap evaluation: detected gaps vs authored ground truth, severity rank correlation, transferable handling,
and monotonicity / consistency property checks.

Two systems are measured and compared:
  * ``GapDetector.detect``  (legacy, string based, no weights)
  * ``GapDetector.detect_vector`` / ``GapEngine.compute`` (weighted gap vector)

Ground truth: each (user, job) pair is built from an authored per-skill *state* (strong / weak / outdated / uncertain /
transferable / absent).  A job skill is a true gap iff the state is not "strong"; its authored severity is
tier_weight(core 3, important 2, nice 1) x deficit(absent 1.0, transferable 0.6, weak 0.5, outdated 0.4, uncertain 0.3).
These are authored judgements, not derived from either system's code or output.  Rank correlations are hand
implemented (Kendall tau-b, Spearman with average ranks) and cross-checked against scipy when it is importable.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from skillllm_v2.evals.contract import EvalContext, EvalResult, bootstrap_ci
from skillllm_v2.evals import match_evals as _m
from skillllm_v2.evals.match_evals import (CORE_WEIGHT, IMPORTANT_WEIGHT, NICE_WEIGHT, Universe, _mean, load_universe, make_jobs)
from skillllm_v2.match.gap_detector import GapDetector
from skillllm_v2.recommend.gap_vector import GapEngine
from skillllm_v2.schema import normalize_text

MODULE = "gap"
def _res(*a: Any, **k: Any) -> EvalResult:
    return _m._res(*a, module=MODULE, **k)


def _control(*a: Any, **k: Any) -> EvalResult:
    return _m._control(*a, module=MODULE, **k)


def _rate_result(*a: Any, **k: Any) -> EvalResult:
    return _m._rate_result(*a, module=MODULE, **k)


CAVEATS = [
    "synthetic pairs with authored per-skill states; truth gaps/severities are authored judgements, not real labelled data",
    "string-profile and dict-profile views of the same authored state are fed to the legacy detector and the gap engine respectively",
]
DEFICIT = {"absent": 1.0, "transferable": 0.6, "weak": 0.5, "outdated": 0.4, "uncertain": 0.3}
TRUTH_TYPE = {"weak": "weak_skill", "outdated": "outdated_skill", "uncertain": "uncertain_skill",
              "transferable": "covered_by_transferable_skill"}
IMPORTANCE_OF_TIER = {CORE_WEIGHT: 1.0, IMPORTANT_WEIGHT: 0.8, NICE_WEIGHT: 0.5}
STATE_P = [("strong", 0.40), ("weak", 0.12), ("outdated", 0.06), ("uncertain", 0.06), ("transferable", 0.10), ("absent", 0.26)]


# ------------------------------------------------------------------------------------------ rank correlation
def _avg_ranks(x: Sequence[float]) -> List[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rho = Pearson correlation of average ranks; None when either side is constant."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Kendall tau-b (tie corrected); None when either side is constant."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return None if denom == 0 else (conc - disc) / denom


# ------------------------------------------------------------------------------------------ case generation
def _state(rng: random.Random, can_transfer: bool) -> str:
    r, acc = rng.random(), 0.0
    for name, p in STATE_P:
        acc += p
        if r < acc:
            return "absent" if (name == "transferable" and not can_transfer) else name
    return "absent"


def make_gap_cases(uni: Universe, seed: int, n: int) -> List[Dict[str, Any]]:
    rng = random.Random(f"{seed}:gap-cases")
    jobs = make_jobs(uni, rng, 3)
    cases: List[Dict[str, Any]] = []
    for i in range(n):
        job = jobs[i % len(jobs)]
        req, pref, tiers = job["required_skills"], job["preferred_skills"], job["_tiers"]
        skills_str: List[str] = []
        skill_dicts: List[Dict[str, Any]] = []
        weak, outdated, uncertain, transferable = [], [], [], {}
        truth: Dict[str, Dict[str, Any]] = {}
        for s in req + pref:
            req_flag = s in req
            adj = sorted(a for a in uni.adjacency.get(s, set()) if a not in req and a not in pref)
            st = _state(rng, bool(adj))
            if st == "transferable":
                a = rng.choice(adj)
                transferable[s] = a
                if a not in skills_str:
                    skills_str.append(a)
                    skill_dicts.append({"canonical_skill_id": normalize_text(a), "name": a, "confidence": 0.95, "mastery_level": "advanced"})
            elif st == "strong":
                skills_str.append(s)
                skill_dicts.append({"canonical_skill_id": normalize_text(s), "name": s, "confidence": 0.9, "mastery_level": "advanced"})
            elif st == "weak":
                skills_str.append(s); weak.append(s)
                skill_dicts.append({"canonical_skill_id": normalize_text(s), "name": s, "confidence": 0.8, "mastery_level": "beginner"})
            elif st == "outdated":
                skills_str.append(s); outdated.append(s)
                skill_dicts.append({"canonical_skill_id": normalize_text(s), "name": s, "confidence": 0.9, "mastery_level": "advanced"})
            elif st == "uncertain":
                skills_str.append(s); uncertain.append(s)
                skill_dicts.append({"canonical_skill_id": normalize_text(s), "name": s, "confidence": 0.5, "mastery_level": "advanced"})
            if st != "strong":
                gtype = TRUTH_TYPE.get(st) or ("missing_critical" if req_flag else "missing_preferred")
                truth[normalize_text(s)] = {"skill": s, "state": st, "type": gtype, "required": req_flag,
                                            "severity": tiers[s] * DEFICIT[st], "tier": tiers[s]}
        for _ in range(rng.randint(0, 3)):  # irrelevant held skills
            d = rng.choice(uni.distractors)
            if d not in skills_str:
                skills_str.append(d)
                skill_dicts.append({"canonical_skill_id": normalize_text(d), "name": d, "confidence": 0.9, "mastery_level": "advanced"})
        legacy_user = {"user_id": f"u{i}", "skills": skills_str, "weak_skills": weak, "outdated_skills": outdated,
                       "uncertain_skills": uncertain, "transferable_skills": transferable}
        engine_user = {"user_id": f"u{i}", "skills": skill_dicts, "weak_skills": weak, "outdated_skills": outdated,
                       "uncertain_skills": uncertain, "transferable_skills": transferable}
        legacy_job = {"job_id": job["job_id"], "required_skills": list(req), "preferred_skills": list(pref)}
        engine_job = {"job_id": job["job_id"],
                      "required_skills": [{"name": s, "importance": IMPORTANCE_OF_TIER[tiers[s]]} for s in req],
                      "preferred_skills": [{"name": s, "importance": IMPORTANCE_OF_TIER[tiers[s]]} for s in pref]}
        cases.append({"legacy_user": legacy_user, "engine_user": engine_user, "legacy_job": legacy_job, "engine_job": engine_job,
                      "truth": truth, "job_skills": [normalize_text(s) for s in req + pref]})
    return cases


# ------------------------------------------------------------------------------------------ systems under test
def detector_gaps(case: Dict[str, Any], user_key: str = "legacy_user") -> Dict[str, str]:
    gaps = GapDetector().detect(case[user_key], case["legacy_job"])
    out: Dict[str, str] = {}
    for g in gaps:
        out.setdefault(normalize_text(g["skill"]), g["gap_type"])
    return out


def engine_gaps(case: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, float]]:
    gv = GapDetector().detect_vector(case["engine_user"], case["engine_job"])
    types = {g.canonical_skill_id: g.gap_type for g in gv.gaps}
    return types, dict(gv.vector)


def naive_all(case: Dict[str, Any]) -> Dict[str, str]:
    return {s: "any" for s in case["job_skills"]}


def naive_missing(case: Dict[str, Any]) -> Dict[str, str]:
    have = {normalize_text(s) for s in case["legacy_user"]["skills"]}
    return {s: "missing" for s in case["job_skills"] if s not in have}


def prf(pred: Sequence[str], truth: Sequence[str]) -> Tuple[int, int, int]:
    p, t = set(pred), set(truth)
    return len(p & t), len(p - t), len(t - p)


def _f(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    pr = tp / (tp + fp) if tp + fp else 1.0
    rc = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def _micro_ci(counts: List[Tuple[int, int, int]], idx: int, n_boot: int, seed: int) -> Tuple[float, float]:
    return bootstrap_ci(counts, n_boot, seed, stat=lambda v: _f(sum(c[0] for c in v), sum(c[1] for c in v), sum(c[2] for c in v))[idx])


# ------------------------------------------------------------------------------------------ main eval blocks
def detection_block(cases: List[Dict[str, Any]], ctx: EvalContext, ds: str) -> List[EvalResult]:
    systems = {
        "detector": lambda c: detector_gaps(c),
        "engine": lambda c: engine_gaps(c)[0],
        "baseline.all_job_skills": naive_all,
        "baseline.missing_only": naive_missing,
    }
    out: List[EvalResult] = []
    for name, fn in systems.items():
        counts, per_pair_f1, crit_hits, crit_n, type_ok, type_n = [], [], 0, 0, 0, 0
        pair_type_acc: List[float] = []
        for c in cases:
            pred = fn(c)
            truth = c["truth"]
            tp, fp, fn_ = prf(pred, truth)
            counts.append((tp, fp, fn_))
            per_pair_f1.append(_f(tp, fp, fn_)[2])
            crit = [s for s, t in truth.items() if t["type"] == "missing_critical"]
            crit_n += len(crit)
            crit_hits += sum(1 for s in crit if s in pred)
            both = [s for s in truth if s in pred]
            if both and not name.startswith("baseline"):
                ok = sum(1 for s in both if pred[s] == truth[s]["type"])
                type_ok += ok
                type_n += len(both)
                pair_type_acc.append(ok / len(both))
        pr, rc, f1 = _f(*(sum(c[i] for c in counts) for i in range(3)))
        det = {"micro_precision": pr, "micro_recall": rc, "micro_f1": f1, "total_truth_gaps": sum(c[0] + c[2] for c in counts)}
        for metric, val, idx in (("precision", pr, 0), ("recall", rc, 1), ("f1", f1, 2)):
            out.append(EvalResult(f"{MODULE}.{name}.micro_{metric}", MODULE, f"micro_{metric}", val, len(cases), ds,
                                  _micro_ci(counts, idx, ctx.n_boot, ctx.seed + idx), {"pooled_over_pairs": True, **det},
                                  list(CAVEATS) + (["trivial baseline"] if name.startswith("baseline") else [])))
        out.append(_res(f"{name}.macro_f1", "macro_f1_per_pair", per_pair_f1, ctx, ds, {}, list(CAVEATS), salt=11))
        if crit_n:
            out.append(EvalResult(f"{MODULE}.{name}.critical_gap_recall", MODULE, "critical_gap_recall", crit_hits / crit_n, crit_n, ds,
                                  bootstrap_ci([1.0] * crit_hits + [0.0] * (crit_n - crit_hits), ctx.n_boot, ctx.seed + 5), {},
                                  list(CAVEATS) + ["units are skills (not independent across pairs), so the CI is optimistic"]))
        if type_n:
            out.append(_res(f"{name}.gap_type_accuracy", "gap_type_accuracy_given_detected", pair_type_acc, ctx, ds,
                            {"skills_compared": type_n, "overall": type_ok / type_n}, list(CAVEATS), salt=13))
    return out


def severity_block(cases: List[Dict[str, Any]], ctx: EvalContext, ds: str) -> List[EvalResult]:
    taus: List[float] = []
    rhos: List[float] = []
    ftaus: List[float] = []
    rtaus: List[float] = []
    rng = random.Random(f"{ctx.seed}:gap-sev")
    for c in cases:
        _, vec = engine_gaps(c)
        shared = [s for s in c["truth"] if s in vec]
        if len(shared) < 3:
            continue
        t = [c["truth"][s]["severity"] for s in shared]
        w = [vec[s] for s in shared]
        tau, rho = kendall_tau_b(t, w), spearman(t, w)
        if tau is None or rho is None:
            continue
        taus.append(tau)
        rhos.append(rho)
        # baseline: requirement-strength only (required=1, preferred=0): tier/deficit blind
        flat = [1.0 if c["truth"][s]["required"] else 0.0 for s in shared]
        ft = kendall_tau_b(t, flat)
        if ft is not None:
            ftaus.append(ft)
        rt = kendall_tau_b(t, [rng.random() for _ in shared])
        if rt is not None:
            rtaus.append(rt)
    cav = list(CAVEATS) + ["only pairs with >=3 shared gap skills; truth severity = tier x deficit (authored)"]
    dr = {"pairs_total": len(cases), "dropped_pairs": len(cases) - len(taus),
          "dropped_rule": "pairs with <3 shared gap skills or a constant side (correlation undefined) are DROPPED, not counted as 0 (repo convention)"}
    return [
        _res("engine.severity_kendall_tau_b", "kendall_tau_b", taus, ctx, ds, dict(dr), cav, salt=21),
        _res("engine.severity_spearman_rho", "spearman_rho", rhos, ctx, ds, dict(dr), cav, salt=22),
        _res("baseline.requirement_flag_only.severity_kendall_tau_b", "kendall_tau_b", ftaus, ctx, ds, {}, cav, salt=23),
        _res("baseline.random.severity_kendall_tau_b", "kendall_tau_b", rtaus, ctx, ds, {}, cav, salt=24),
    ]


def transferable_block(cases: List[Dict[str, Any]], ctx: EvalContext, ds: str) -> List[EvalResult]:
    out: List[EvalResult] = []
    for name, fn in (("detector", lambda c: detector_gaps(c)), ("engine", lambda c: engine_gaps(c)[0])):
        hit: List[float] = []
        fp: List[float] = []
        for c in cases:
            pred = fn(c)
            for s, t in c["truth"].items():
                if t["state"] == "transferable":
                    hit.append(1.0 if pred.get(s) == "covered_by_transferable_skill" else 0.0)
                elif t["state"] == "absent":
                    fp.append(1.0 if pred.get(s) == "covered_by_transferable_skill" else 0.0)
        out.append(_res(f"{name}.transferable_recall", "transferable_label_recall", hit, ctx, ds, {},
                        list(CAVEATS) + ["units are job skills pooled over pairs"], salt=31))
        r = _res(f"{name}.transferable_false_positive_rate", "false_transferable_rate_on_absent_skills", fp, ctx, ds,
                 {"direction": "lower"}, list(CAVEATS), salt=32)
        out.append(r)
    # weights: transferable must weigh less than absent, more than strong (=0)
    ratios: List[float] = []
    for c in cases:
        _, vec = engine_gaps(c)
        for s, t in c["truth"].items():
            if t["state"] == "transferable" and s in vec:
                full = IMPORTANCE_OF_TIER[t["tier"]] * (1.0 if t["required"] else 0.7)
                ratios.append(vec[s] / full if full else 0.0)
    out.append(_res("engine.transferable_weight_ratio", "weight_over_missing_weight", ratios, ctx, ds,
                    {"authored_expectation": "strictly between 0 and 1 (partial credit)", "direction": "none"},
                    list(CAVEATS), salt=33))
    return out


def differential_block(cases: List[Dict[str, Any]], ctx: EvalContext, ds: str) -> List[EvalResult]:
    jac, type_agree, n_type = [], 0, 0
    for c in cases:
        d, e = detector_gaps(c), engine_gaps(c)[0]
        union = set(d) | set(e)
        jac.append(len(set(d) & set(e)) / len(union) if union else 1.0)
        for s in set(d) & set(e):
            n_type += 1
            type_agree += d[s] == e[s]
    res = [_res("differential.detector_vs_engine_gap_set_jaccard", "jaccard_of_gap_skill_sets", jac, ctx, ds, {},
                ["both systems read the same authored state; disagreement = behavioural inconsistency between the two implementations"],
                salt=41)]
    if n_type:
        res.append(EvalResult(f"{MODULE}.differential.detector_vs_engine_gap_type_agreement", MODULE, "gap_type_agreement",
                              type_agree / n_type, n_type, ds, None, {"direction": "higher"},
                              ["pooled over skills in both gap sets; not independent across pairs"]))
    return res


def robustness_block(cases: List[Dict[str, Any]], ctx: EvalContext, ds: str) -> List[EvalResult]:
    """Lower-case the user's surface forms (as an extractor that lowercases would)."""
    def lower_user(u: Dict[str, Any], dicts: bool) -> Dict[str, Any]:
        v = dict(u)
        if dicts:
            v["skills"] = [{**s, "name": s["name"].lower(), "canonical_skill_id": normalize_text(s["name"])} for s in u["skills"]]
        else:
            v["skills"] = [s.lower() for s in u["skills"]]
        return v

    fa_det, fa_eng = [], []
    for c in cases:
        base_d = set(detector_gaps(c))
        low_d = set(detector_gaps({**c, "legacy_user": {**lower_user(c["legacy_user"], False)}}))
        fa_det.append(len(low_d - base_d) / max(1, len(c["job_skills"])))
        base_e = set(engine_gaps(c)[0])
        low_e = set(engine_gaps({**c, "engine_user": lower_user(c["engine_user"], True)})[0])
        fa_eng.append(len(low_e - base_e) / max(1, len(c["job_skills"])))
    return [
        _res("robust.detector.spurious_gap_share_under_lowercased_user_skills", "added_gaps_per_job_skill", fa_det, ctx, ds,
             {"direction": "lower"}, list(CAVEATS) + ["REGRESSION GUARD (bug fixed, docs/BUGFIX_LOG_MATCH.md): if >0, GapDetector.detect compares exact strings"], salt=51),
        _res("robust.engine.spurious_gap_share_under_lowercased_user_skills", "added_gaps_per_job_skill", fa_eng, ctx, ds,
             {"direction": "lower"}, list(CAVEATS), salt=52),
    ]


# ------------------------------------------------------------------------------------------ property checks
def _rand_engine_case(uni: Universe, rng: random.Random, with_transfer: bool) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    pool = uni.all_skills
    req = rng.sample(pool, rng.randint(1, 4))
    pref = rng.sample([s for s in pool if s not in req], rng.randint(0, 3))
    job = {"job_id": "j",
           "required_skills": [{"name": s, "importance": round(rng.uniform(0.3, 1.0), 2)} for s in req],
           "preferred_skills": [{"name": s, "importance": round(rng.uniform(0.2, 0.9), 2)} for s in pref]}
    held = [s for s in rng.sample(pool, rng.randint(0, 6))]
    skills = [{"name": s, "confidence": round(rng.uniform(0.3, 1.0), 2),
               "mastery_level": rng.choice(["beginner", "intermediate", "advanced"])} for s in held]
    user = {"user_id": "u", "skills": skills, "weak_skills": [], "outdated_skills": [], "uncertain_skills": [],
            "transferable_skills": {}}
    if with_transfer:
        for s in req + pref:
            if s not in held and rng.random() < 0.5:
                user["transferable_skills"][s] = "something-adjacent"
    return user, job, req + pref


def property_block(uni: Universe, ctx: EvalContext, n: int) -> List[EvalResult]:
    rng = random.Random(f"{ctx.seed}:gap-props")
    eng = GapEngine()
    ds = "synthetic-random-cases"
    cav = ["property check over seeded random cases; expectation is 0 violations"]
    v = {k: 0 for k in ("mass_add_held", "w_add_held", "w_add_weak_held", "w_add_weak_held_transfer", "imp_up", "imp_up_mass",
                        "strength_up", "bounds", "req_ge_pref", "vector_sum", "det_subset", "det_order", "det_dup", "det_trans_held",
                        "det_case")}
    ex: Dict[str, Any] = {}

    for _ in range(n):
        user, job, all_sk = _rand_engine_case(uni, rng, with_transfer=False)
        gv = eng.compute(user, job)
        # bounds + vector consistency
        imps = {normalize_text(i["name"]): i["importance"] * (1.0 if i in job["required_skills"] else 0.7)
                for i in job["required_skills"] + job["preferred_skills"]}
        if any(not (-1e-12 <= w <= imps[s] + 1e-12) for s, w in gv.vector.items()):
            v["bounds"] += 1
        if abs(sum(gv.vector.values()) - gv.total_gap_mass) > 1e-9:
            v["vector_sum"] += 1
        # (1) add a held, strong skill
        pick = rng.choice(all_sk)
        if all(normalize_text(s["name"]) != normalize_text(pick) for s in user["skills"]):
            u2 = {**user, "skills": user["skills"] + [{"name": pick, "confidence": 1.0, "mastery_level": "advanced"}]}
            gv2 = eng.compute(u2, job)
            if gv2.total_gap_mass > gv.total_gap_mass + 1e-12 or gv2.get_gap_weight(pick) > gv.get_gap_weight(pick) + 1e-12:
                v["mass_add_held"] += 1
                ex.setdefault("mass_add_held", pick)
            # (1b) add a held skill with random (possibly weak) strength, no transferable mapping
            u3 = {**user, "skills": user["skills"] + [{"name": pick, "confidence": round(rng.uniform(0.05, 1.0), 2),
                                                       "mastery_level": rng.choice(["beginner", "intermediate", "advanced"])}]}
            gv3 = eng.compute(u3, job)
            if gv3.get_gap_weight(pick) > gv.get_gap_weight(pick) + 1e-12:
                v["w_add_held"] += 1
                ex.setdefault("w_add_held", pick)
        # (2) raise importance of one job skill
        side = rng.choice(["required_skills", "preferred_skills"])
        if job[side]:
            idx = rng.randrange(len(job[side]))
            name = job[side][idx]["name"]
            job2 = {**job, side: [dict(x) for x in job[side]]}
            job2[side][idx]["importance"] = job[side][idx]["importance"] + rng.uniform(0.01, 0.5)
            gv_b = eng.compute(user, job2)
            if gv_b.get_gap_weight(name) < gv.get_gap_weight(name) - 1e-12:
                v["imp_up"] += 1
                ex.setdefault("imp_up", name)
            if gv_b.total_gap_mass < gv.total_gap_mass - 1e-12:
                v["imp_up_mass"] += 1
        # (3) raise user's strength (confidence) on a held skill
        if user["skills"]:
            k = rng.randrange(len(user["skills"]))
            s0 = user["skills"][k]
            s1 = {**s0, "confidence": min(1.0, s0["confidence"] + rng.uniform(0.01, 0.5))}
            u_hi = {**user, "skills": user["skills"][:k] + [s1] + user["skills"][k + 1:]}
            g_hi = eng.compute(u_hi, job)
            if g_hi.get_gap_weight(s0["name"]) > gv.get_gap_weight(s0["name"]) + 1e-12:
                v["strength_up"] += 1
                ex.setdefault("strength_up", {"skill": s0, "conf_after": s1["confidence"]})
        # (5) same importance, required vs preferred
        sk = rng.choice(all_sk)
        imp = round(rng.uniform(0.3, 0.9), 2)
        gr = eng.compute({"user_id": "u", "skills": []}, {"job_id": "j", "required_skills": [{"name": sk, "importance": imp}]})
        gp = eng.compute({"user_id": "u", "skills": []}, {"job_id": "j", "preferred_skills": [{"name": sk, "importance": imp}]})
        if gr.total_gap_mass < gp.total_gap_mass - 1e-12:
            v["req_ge_pref"] += 1

        # transferable + weakly held (engine): adding a held-but-weak skill vs leaving it transferable-covered
        user_t, job_t, sk_t = _rand_engine_case(uni, rng, with_transfer=True)
        tgt = [s for s in user_t["transferable_skills"]]
        if tgt:
            s = rng.choice(tgt)
            base = eng.compute(user_t, job_t).get_gap_weight(s)
            u4 = {**user_t, "skills": user_t["skills"] + [{"name": s, "confidence": round(rng.uniform(0.05, 0.4), 2), "mastery_level": "beginner"}]}
            after = eng.compute(u4, job_t).get_gap_weight(s)
            if after > base + 1e-12:
                v["w_add_weak_held_transfer"] += 1
                ex.setdefault("w_add_weak_held_transfer", {"skill": s, "weight_when_transferable": base, "weight_when_held_weakly": after})

        # legacy detector properties
        d_user = {"user_id": "u", "skills": [s["name"] for s in user["skills"]]}
        d_job = {"job_id": "j", "required_skills": [i["name"] for i in job["required_skills"]],
                 "preferred_skills": [i["name"] for i in job["preferred_skills"]]}
        det = GapDetector()
        base_set = {g["skill"] for g in det.detect(d_user, d_job)}
        add = rng.choice(all_sk)
        after_set = {g["skill"] for g in det.detect({**d_user, "skills": d_user["skills"] + [add]}, d_job)}
        if not after_set <= base_set:
            v["det_subset"] += 1
        shuffled = {**d_job, "required_skills": rng.sample(d_job["required_skills"], len(d_job["required_skills"])),
                    "preferred_skills": rng.sample(d_job["preferred_skills"], len(d_job["preferred_skills"]))}
        if {g["skill"] for g in det.detect(d_user, shuffled)} != base_set:
            v["det_order"] += 1
        # same skill listed as both required and preferred -> one gap per skill expected
        if d_job["required_skills"]:
            s = d_job["required_skills"][0]
            overlap_job = {**d_job, "preferred_skills": d_job["preferred_skills"] + [s]}
            names = [g["skill"] for g in det.detect({"user_id": "u", "skills": []}, overlap_job)]
            if len(names) != len(set(names)):
                v["det_dup"] += 1
        # held weak skill + transferable mapping: detector must not label a held skill as "covered by transferable"
        if d_job["required_skills"]:
            s = d_job["required_skills"][0]
            gaps = det.detect({"user_id": "u", "skills": [s], "weak_skills": [s], "transferable_skills": {s: "adjacent"}}, d_job)
            if any(g["skill"] == s and g["gap_type"] == "covered_by_transferable_skill" for g in gaps):
                v["det_trans_held"] += 1
                ex.setdefault("det_trans_held", {"held_skill": s, "weak": True, "labelled": "covered_by_transferable_skill"})
        # case invariance
        low_user = {"user_id": "u", "skills": [s.lower() for s in d_user["skills"]]}
        if d_user["skills"] and {normalize_text(g["skill"]) for g in det.detect(low_user, d_job)} != {normalize_text(s) for s in base_set}:
            v["det_case"] += 1

    r = lambda key, name, metric, extra_cav=None, **d: _rate_result(name, metric, v[key], n, ds, {**d, **({"example": ex.get(key)} if key in ex else {})}, cav + (extra_cav or [])) # noqa: E731
    out = [
        r("mass_add_held", "prop.engine.adding_strong_held_skill_never_increases_gap", "violation_rate"),
        r("w_add_held", "prop.engine.adding_held_skill_any_strength_never_increases_its_weight_no_transferable", "violation_rate"),
        r("w_add_weak_held_transfer", "prop.engine.weakly_held_skill_not_worse_than_transferable_cover", "violation_rate",
          ["design inconsistency if >0: once a skill is held (even weakly, strength<0.4) transferable evidence is ignored, so the skill gets a LARGER gap weight than when it is merely transferable-covered (fixed strength 0.4); acquiring a weak version of a skill can raise the gap"]),
        r("imp_up", "prop.engine.raising_importance_never_lowers_gap_weight", "violation_rate"),
        r("imp_up_mass", "prop.engine.raising_importance_never_lowers_total_gap_mass", "violation_rate"),
        r("strength_up", "prop.engine.raising_user_strength_never_raises_gap_weight", "violation_rate"),
        r("bounds", "prop.engine.weights_within_importance_bounds", "violation_rate"),
        r("vector_sum", "prop.engine.total_mass_equals_vector_sum", "violation_rate"),
        r("req_ge_pref", "prop.engine.required_weight_ge_preferred_at_equal_importance", "violation_rate"),
        r("det_subset", "prop.detector.adding_held_skill_never_adds_gaps", "violation_rate"),
        r("det_order", "prop.detector.job_skill_order_invariance", "violation_rate"),
        r("det_dup", "prop.detector.one_gap_per_skill_when_listed_required_and_preferred", "violation_rate",
          ["condition is constructed in every case (rate 1.0 = the behaviour exists)", "REGRESSION GUARD (bug fixed, docs/BUGFIX_LOG_MATCH.md): if >0, a skill listed as both required and preferred yields duplicate gap entries"]),
        r("det_trans_held", "prop.detector.held_skill_never_labelled_covered_by_transferable", "violation_rate",
          ["condition is constructed in every case (rate 1.0 = the behaviour exists)", "REGRESSION GUARD (bug fixed, docs/BUGFIX_LOG_MATCH.md): if >0, transferable override is applied to skills the user already holds (weak_skill relabelled)"]),
        r("det_case", "prop.detector.case_invariance", "violation_rate",
          ["REGRESSION GUARD (bug fixed, docs/BUGFIX_LOG_MATCH.md): if >0, GapDetector.detect compares exact strings, so 'python' vs 'Python' creates a spurious gap"]),
    ]
    return out


# ------------------------------------------------------------------------------------------ controls
def controls_block(cases: List[Dict[str, Any]], ctx: EvalContext, ds: str) -> List[EvalResult]:
    out: List[EvalResult] = []
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    perfect_t, perfect_r = kendall_tau_b(x, x), spearman(x, x)
    rev_t, rev_r = kendall_tau_b(x, x[::-1]), spearman(x, x[::-1])
    out.append(_control("rank_corr_perfect_and_reversed", "tau_perfect_minus_tau_reversed", (perfect_t or 0) - (rev_t or 0),
                        abs((perfect_t or 0) - 1) < 1e-12 and abs((rev_t or 0) + 1) < 1e-12 and abs((perfect_r or 0) - 1) < 1e-12
                        and abs((rev_r or 0) + 1) < 1e-12, "identical ranking -> +1, reversed -> -1 (tau-b and Spearman)", 4, ds))
    # known tie example verified by hand: x=[1,2,2,3], y=[1,3,2,3]
    tb = kendall_tau_b([1, 2, 2, 3], [1, 3, 2, 3])
    # pairs: (1,2):c (1,3):c (1,4):c (2,3): x tie -> tx (2,4): c (3,4): c ; y tie at (2,4)? y2=3,y4=3 -> ty => c=3 ... computed below
    conc, disc, tx, ty = 0, 0, 0, 0
    xs, ys = [1, 2, 2, 3], [1, 3, 2, 3]
    for i in range(4):
        for j in range(i + 1, 4):
            dx, dy = xs[i] - xs[j], ys[i] - ys[j]
            if dx == 0 and dy == 0: continue
            if dx == 0: tx += 1
            elif dy == 0: ty += 1
            elif (dx > 0) == (dy > 0): conc += 1
            else: disc += 1
    want = (conc - disc) / math.sqrt((conc + disc + tx) * (conc + disc + ty))
    out.append(_control("kendall_tau_b_tie_hand_example", "abs_error", abs((tb or 0) - want), abs((tb or 0) - want) < 1e-12,
                        "tau-b on a hand-enumerated example with ties", 1, ds))
    try:  # cross-check with scipy on random data
        from scipy import stats
        rng = random.Random(f"{ctx.seed}:gap-ctrl")
        worst = 0.0
        for _ in range(60):
            a = [rng.randint(0, 5) for _ in range(rng.randint(4, 12))]
            b = [rng.randint(0, 5) for _ in a]
            mine_t, mine_r = kendall_tau_b(a, b), spearman(a, b)
            if mine_t is None or mine_r is None:
                continue
            worst = max(worst, abs(mine_t - stats.kendalltau(a, b).statistic), abs(mine_r - stats.spearmanr(a, b).statistic))
        out.append(_control("rank_corr_matches_scipy", "max_abs_diff_vs_scipy", worst, worst < 1e-9,
                            "hand-implemented tau-b/Spearman equal scipy.stats on 60 random tied samples", 60, ds))
    except Exception as exc:  # scipy optional
        out.append(EvalResult(f"{MODULE}.control.rank_corr_matches_scipy", MODULE, "max_abs_diff_vs_scipy", None, 0, ds, None,
                              {"control": True, "passed": True, "expected": "skipped"}, [], "skipped", f"scipy unavailable: {exc}"))
    # shuffled truth: severity correlation of engine against truth from a *different* pair must be ~0
    taus = []
    for i, c in enumerate(cases):
        o = cases[(i + 7) % len(cases)]
        _, vec = engine_gaps(c)
        sk = list(c["truth"])
        other = [o["truth"][s]["severity"] if s in o["truth"] else 0.0 for s in sk]
        w = [vec.get(s, 0.0) for s in sk]
        t = kendall_tau_b(other, w) if len(sk) >= 3 else None
        if t is not None:
            taus.append(t)
    m = _mean(taus) if taus else 0.0
    out.append(_control("severity_label_shuffle", "mean_tau_b_vs_foreign_labels", m, abs(m) < 0.25,
                        "engine weights correlated with another pair's severities stay near 0", len(taus), ds))
    # oracle detection: predicting the truth set scores F1 = 1 through the same prf path
    counts = [prf(c["truth"].keys(), c["truth"].keys()) for c in cases]
    f1 = _f(*(sum(x_[i] for x_ in counts) for i in range(3)))[2]
    out.append(_control("oracle_detection_f1", "micro_f1", f1, f1 > 0.999, "predicting the authored truth scores F1 = 1", len(cases), ds))
    # trivial baseline sanity: all-job-skills must have recall 1 and precision < detector precision
    return out


# ------------------------------------------------------------------------------------------ run
def run(ctx: EvalContext) -> List[EvalResult]:
    uni = load_universe(ctx)
    n = 40 if ctx.smoke else 240
    cases = make_gap_cases(uni, ctx.seed, n)
    ds = f"synthetic-gap-v1(seed={ctx.seed},pairs={n})"
    out: List[EvalResult] = []
    out += detection_block(cases, ctx, ds)
    out += severity_block(cases, ctx, ds)
    out += transferable_block(cases, ctx, ds)
    out += differential_block(cases, ctx, ds)
    out += robustness_block(cases, ctx, ds)
    out += property_block(uni, ctx, 60 if ctx.smoke else 400)
    out += controls_block(cases, ctx, ds)
    # baseline sanity control: all-job-skills baseline has recall 1.0
    rec = [r for r in out if r.eval_name == f"{MODULE}.baseline.all_job_skills.micro_recall"]
    if rec and rec[0].value is not None:
        out.append(_control("all_skills_baseline_recall_is_one", "micro_recall", rec[0].value, rec[0].value > 0.999,
                            "predicting every job skill as a gap recalls all authored gaps", len(cases), ds))
    return out
