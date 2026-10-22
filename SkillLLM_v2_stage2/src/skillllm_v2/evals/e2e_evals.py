"""End-to-end evaluation of the offline-runnable pipeline: (profile text, job text) -> skills -> match -> gaps -> courses.

Real components that run offline are exercised for real: ``SkillLLMv2Pipeline`` (with the deterministic *stub* Mistral
fallback and the dictionary extractor; no weights), ``JobMatcher``, ``GapDetector`` / ``GapEngine``,
``CourseRecommender``.  Nothing here depends on ``skillllm_v2.pipeline.inference``; it is probed and used only if it imports.

Three wiring modes isolate where errors enter:
  * ``pipeline`` -- user AND job skills come from the extractor (full end-to-end)
  * ``demo``     -- user skills from the extractor (canonical names), job skills authored (what ``smoke_runner.end_to_end_demo_smoke`` does)
  * ``oracle``   -- authored names on both sides (downstream stages only; extraction errors removed)

Measured: extraction vs ground truth, gap detection vs ground truth per mode, name-consistency loss, recommendation precision,
traceability (user skill -> job skill -> course skill -> prerequisite status), final-JSON schema validity, per-stage CPU
latency, determinism (in-process and across PYTHONHASHSEED), and graceful degradation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from skillllm_v2.evals.contract import EvalContext, EvalResult, bootstrap_ci
from skillllm_v2.schema import normalize_text

MODULE = "e2e"
CAVEATS = [
    "synthetic authored cases (12 in a full run, 5 in smoke) -- tiny n, wide CIs; not real resumes/postings",
    "offline: Mistral is the deterministic stub fallback (8 alias strings), GLiNER absent, so extraction numbers do NOT describe the trained model",
]
CUE = re.compile(r"(nice to have|preferred|bonus|a plus)", re.I)  # eval-side job-text splitter (not part of the system under test)
FIXTURE_DIR = ("inputs", "evals")


# ------------------------------------------------------------------------------------------------ fixtures
def _fixture_path(ctx: EvalContext, name: str) -> Path:
    p = ctx.path(*FIXTURE_DIR, name)
    return p if p.exists() else Path(__file__).resolve().parents[3].joinpath(*FIXTURE_DIR, name)


def load_cases(ctx: EvalContext) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], List[Dict[str, Any]]]:
    data = json.loads(_fixture_path(ctx, "e2e_cases.json").read_text(encoding="utf-8"))
    catalog = [json.loads(l) for l in _fixture_path(ctx, "e2e_catalog.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return data["cases"], data["equivalents"], catalog


def gt_key_map(equivalents: Dict[str, List[str]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for key, forms in equivalents.items():
        m[normalize_text(key)] = key
        for f in forms:
            m[normalize_text(f)] = key
    return m


def to_key(name: str, kmap: Dict[str, str]) -> str:
    return kmap.get(normalize_text(name), normalize_text(name))


# ------------------------------------------------------------------------------------------------ pipeline glue
def build_pipeline(**kw: Any) -> Any:
    from skillllm_v2.wrapper import SkillLLMv2Pipeline
    return SkillLLMv2Pipeline(**kw)


def skill_name(item: Dict[str, Any]) -> str:
    return item.get("canonical_skill_name") or item["surface_form"]


def extract_names(pipe: Any, text: str) -> List[Tuple[str, int]]:
    """[(name, start_offset)] de-duplicated, in text order."""
    seen, out = set(), []
    for it in sorted(pipe.extract(text), key=lambda x: (x.get("start") or 0)):
        n = skill_name(it)
        if normalize_text(n) not in seen:
            seen.add(normalize_text(n))
            out.append((n, it.get("start") or 0))
    return out


def split_job(named: List[Tuple[str, int]], text: str) -> Tuple[List[str], List[str]]:
    m = CUE.search(text)
    cut = m.start() if m else len(text) + 1
    return [n for n, s in named if s < cut], [n for n, s in named if s >= cut]


def _timed(bucket: Dict[str, List[float]], stage: str, fn: Callable[[], Any]) -> Any:
    t = time.perf_counter()
    out = fn()
    bucket.setdefault(stage, []).append((time.perf_counter() - t) * 1000.0)
    return out


def run_case(case: Dict[str, Any], mode: str, pipe: Any, catalog: List[Dict[str, Any]], lat: Optional[Dict[str, List[float]]] = None,
             ) -> Dict[str, Any]:
    from skillllm_v2.match.course_recommender import CourseRecommender
    from skillllm_v2.match.gap_detector import GapDetector
    from skillllm_v2.match.job_matcher import JobMatcher

    lat = lat if lat is not None else {}
    if mode == "oracle":
        user_names = list(case["user_skills"])
        req, pref = list(case["job_required"]), list(case["job_preferred"])
        extracted_user, extracted_job = [], []
    else:
        extracted_user = _timed(lat, "extract_profile", lambda: extract_names(pipe, case["profile_text"]))
        user_names = [n for n, _ in extracted_user]
        if mode == "pipeline":
            extracted_job = _timed(lat, "extract_job", lambda: extract_names(pipe, case["job_text"]))
            req, pref = split_job(extracted_job, case["job_text"])
        else:
            extracted_job = []
            req, pref = list(case["job_required"]), list(case["job_preferred"])
    user = {"user_id": f"user-{case['id']}", "skills": user_names}
    job = {"job_id": f"job-{case['id']}", "title": "Target Role", "required_skills": req, "preferred_skills": pref,
           "semantic_role_fit": 0.5, "experience_level_fit": 0.5, "user_preference_fit": 0.5}
    match = _timed(lat, "match", lambda: JobMatcher().score(user, job))
    gaps = _timed(lat, "gap_detect", lambda: GapDetector().detect(user, job))
    gv = _timed(lat, "gap_vector", lambda: GapDetector().detect_vector(user, job))
    recs_vec = _timed(lat, "recommend", lambda: CourseRecommender().recommend(gv, catalog, user, job))
    recs_demo = CourseRecommender().recommend(gaps, catalog, user)  # smoke_runner-style legacy path
    return {"case": case["id"], "user": user, "job": job, "match": match, "gaps": gaps, "gap_vector": gv.to_dict(),
            "recs_vector": recs_vec, "recs_demo": recs_demo,
            "final": {"user_skill_profile": user, "job_matches": [match], "skill_gaps": gaps,
                      "course_recommendations": recs_vec},
            "final_demo": {"user_skill_profile": user, "job_matches": [match], "skill_gaps": gaps,
                           "course_recommendations": recs_demo},
            "extracted_user": [n for n, _ in extracted_user], "extracted_job": [n for n, _ in extracted_job]}


# ------------------------------------------------------------------------------------------------ schema
def output_schema() -> Dict[str, Any]:
    num01 = {"type": "number", "minimum": 0, "maximum": 1}
    strs = {"type": "array", "items": {"type": "string"}}
    rec = {
        "type": "object",
        "required": ["course_id", "title", "rank", "score", "covered_missing_skills", "prerequisite_status", "evidence",
                     "explanation_path", "limitations", "next_profile_update_action"],
        "properties": {
            "course_id": {"type": "string", "minLength": 1}, "title": {"type": "string"},
            "rank": {"type": "integer", "minimum": 1}, "score": num01, "covered_missing_skills": strs,
            "prerequisite_status": {"type": "object", "required": ["is_satisfied", "status_code", "satisfied", "missing"],
                                    "properties": {"is_satisfied": {"type": "boolean"}, "status_code": {"type": "string"},
                                                   "satisfied": {"type": "array"}, "missing": {"type": "array"}}},
            "evidence": {"type": "array", "items": {"type": "object", "required": ["skill", "gap_classification"]}},
            "explanation_path": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "limitations": strs,
            "next_profile_update_action": {"type": "object", "required": ["trigger", "course_id", "skills_to_add"]},
        },
    }
    return {
        "type": "object",
        "required": ["user_skill_profile", "job_matches", "skill_gaps", "course_recommendations"],
        "properties": {
            "user_skill_profile": {"type": "object", "required": ["user_id", "skills"], "properties": {"skills": strs}},
            "job_matches": {"type": "array", "items": {"type": "object", "required": ["job_id", "match_score", "matched_skills", "missing_skills"],
                                                       "properties": {"match_score": num01, "matched_skills": strs, "missing_skills": strs}}},
            "skill_gaps": {"type": "array", "items": {"type": "object", "required": ["skill", "gap_type"]}},
            "course_recommendations": {"type": "array", "items": rec},
        },
    }


def validate_output(obj: Any) -> List[str]:
    """Returns a list of error strings ([] = valid).  Also enforces strict JSON (no NaN/Inf) and contiguous ranks."""
    errs: List[str] = []
    try:
        text = json.dumps(obj, allow_nan=False)
        json.loads(text)
    except (TypeError, ValueError) as exc:
        return [f"not strict-JSON serialisable: {exc}"]
    try:
        import jsonschema
        v = jsonschema.Draft7Validator(output_schema())
        errs.extend(f"{'/'.join(map(str, e.absolute_path))}: {e.message[:120]}" for e in v.iter_errors(obj))
    except ImportError:
        errs.append("jsonschema unavailable")
    recs = obj.get("course_recommendations", []) if isinstance(obj, dict) else []
    ranks = [r.get("rank") for r in recs if isinstance(r, dict)]
    if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
        errs.append(f"ranks not contiguous from 1: {ranks}")
    return errs


# ------------------------------------------------------------------------------------------------ traceability
def _prereq_id(p: Any) -> str:
    if isinstance(p, dict):
        return normalize_text(str(p.get("skill_id") or p.get("canonical_skill_id") or p.get("skill_name") or p.get("canonical_skill_name") or ""))
    return normalize_text(str(p))


def trace_failures(rec: Dict[str, Any], gap_set: set, job_set: set, user_known: set, catalog: Dict[str, Dict[str, Any]],
                   earlier: List[Dict[str, Any]]) -> List[str]:
    """Links: user gap -> job skill -> course skill -> prerequisite status.  Returns failure strings ([] = fully traceable)."""
    fails: List[str] = []
    covered = [normalize_text(s) for s in rec.get("covered_missing_skills", [])]
    course = catalog.get(rec.get("course_id", ""))
    if course is None:
        return ["course_id not in catalog"]
    teaches = {normalize_text(s) for s in course.get("teaches_skills", [])}
    prereqs = {normalize_text(s) for s in course.get("prerequisite_skills", [])}
    ev = rec.get("evidence", [])
    ev_by = {normalize_text(e.get("skill", "")): e for e in ev if isinstance(e, dict)}
    if not covered:
        fails.append("no covered_missing_skills")
    for s in covered:
        if s not in gap_set:
            fails.append(f"user->gap link broken: '{s}' is not a detected gap")
        if s not in job_set:
            fails.append(f"gap->job link broken: '{s}' not a job skill")
        e = ev_by.get(s)
        if e is None:
            fails.append(f"no evidence entry for '{s}'")
        elif s not in teaches and e.get("evidence_type") not in ("text_mention_only", "enabling_prerequisite"):
            fails.append(f"job->course link broken: course does not teach '{s}' and evidence not labelled")
    if not rec.get("explanation_path"):
        fails.append("empty explanation_path")
    ps = rec.get("prerequisite_status")
    if not isinstance(ps, dict):
        fails.append("prerequisite_status missing")
        return fails
    sat = {_prereq_id(p): p for p in ps.get("satisfied", [])}
    mis = {_prereq_id(p) for p in ps.get("missing", [])}
    earlier_taught = set()
    for r in earlier:
        c = catalog.get(r.get("course_id", ""))
        if c:
            earlier_taught |= {normalize_text(s) for s in c.get("teaches_skills", [])}
    for p in prereqs:
        if p in user_known:
            if p not in sat:
                fails.append(f"prerequisite '{p}' known to user but not reported satisfied")
        elif p in sat:
            by = str(sat[p].get("satisfied_by", "")) if isinstance(sat[p], dict) else ""
            if not by.startswith("earlier_course") or p not in earlier_taught:
                fails.append(f"prerequisite '{p}' reported satisfied without user skill or earlier course")
        elif p not in mis:
            fails.append(f"prerequisite '{p}' absent from both satisfied and missing")
    if bool(ps.get("is_satisfied")) != (len(mis) == 0):
        fails.append("is_satisfied inconsistent with missing list")
    return fails


def trace_rate(run_: Dict[str, Any], recs_key: str, catalog: Dict[str, Dict[str, Any]]) -> Tuple[int, int, List[str]]:
    recs = run_[recs_key]
    gv_keys = set(run_["gap_vector"]["vector"]) if recs_key == "recs_vector" else {normalize_text(g["skill"]) for g in run_["gaps"]}
    job_set = {normalize_text(s) for s in run_["job"]["required_skills"] + run_["job"]["preferred_skills"]}
    known = {normalize_text(s) for s in run_["user"]["skills"]}
    ok, msgs = 0, []
    for i, r in enumerate(recs):
        f = trace_failures(r, gv_keys, job_set, known, catalog, recs[:i])
        ok += not f
        msgs.extend(f[:2])
    return ok, len(recs), msgs


# ------------------------------------------------------------------------------------------------ metrics helpers
def _f1(pred: set, truth: set) -> Tuple[float, float, float]:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else (1.0 if not truth else 0.0)
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v) if v else float("nan")


def _res(name: str, metric: str, vals: Sequence[float], ctx: EvalContext, dataset: str, details: Optional[dict] = None,
         caveats: Optional[List[str]] = None, salt: int = 0) -> EvalResult:
    vals = list(vals)
    if not vals:
        return EvalResult(f"{MODULE}.{name}", MODULE, metric, None, 0, dataset, None, details or {}, caveats or [], "skipped", "no units")
    return EvalResult(f"{MODULE}.{name}", MODULE, metric, _mean(vals), len(vals), dataset, bootstrap_ci(vals, ctx.n_boot, ctx.seed + salt),
                      details or {}, caveats if caveats is not None else list(CAVEATS))


def _control(name: str, metric: str, value: float, passed: bool, expected: str, n: int, dataset: str) -> EvalResult:
    return EvalResult(f"{MODULE}.control.{name}", MODULE, metric, float(value), n, dataset, None,
                      {"control": True, "passed": bool(passed), "expected": expected}, [])


def _skipped(name: str, metric: str, dataset: str, reason: str) -> EvalResult:
    return EvalResult(f"{MODULE}.{name}", MODULE, metric, None, 0, dataset, None, {}, [], "skipped", reason)


def _strip_volatile(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _strip_volatile(v) for k, v in o.items() if "latency" not in k and "time" not in k}
    if isinstance(o, list):
        return [_strip_volatile(v) for v in o]
    return o


def digest_all(root: Path) -> str:
    """Deterministic digest of every case's outputs (used in-process and from subprocesses with different hash seeds)."""
    ctx = EvalContext(root=root)
    cases, _, catalog = load_cases(ctx)
    pipe = build_pipeline()
    payload = []
    for c in cases:
        r = run_case(c, "pipeline", pipe, catalog)
        payload.append(_strip_volatile({k: r[k] for k in ("extracted_user", "extracted_job", "match", "gaps", "gap_vector", "recs_vector", "recs_demo")}))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ degradation
class _Boom:
    def extract(self, text: str) -> list:
        raise RuntimeError("extractor unavailable")


def degradation_scenarios(catalog: List[Dict[str, Any]], case: Dict[str, Any]) -> List[Dict[str, Any]]:
    from skillllm_v2.match.course_recommender import CourseRecommender
    from skillllm_v2.match.gap_detector import GapDetector
    from skillllm_v2.match.job_matcher import JobMatcher

    def full(user_text: str = "I know Python and SQL.", job_text: str = "Requirements: Python, AWS and Docker.", pipe: Any = None,
             cat: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        pipe = pipe or build_pipeline()
        c = {**case, "profile_text": user_text, "job_text": job_text, "id": "deg"}
        return run_case(c, "pipeline", pipe, catalog if cat is None else cat)["final"]

    def only_downstream(user: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        m = JobMatcher().score(user, job)
        g = GapDetector().detect(user, job)
        gv = GapDetector().detect_vector(user, job)
        r = CourseRecommender().recommend(gv, catalog, user, job)
        return {"user_skill_profile": {"user_id": "u", "skills": []}, "job_matches": [m], "skill_gaps": g, "course_recommendations": r}

    ok_job = {"job_id": "j", "required_skills": ["Python"], "preferred_skills": []}
    long_text = ("I have worked with Python, SQL and Docker for many years. " * 400)
    return [
        {"name": "mistral_extractor_raises", "expect_graceful": True, "fn": lambda: full(pipe=build_pipeline(mistral=_Boom()))},
        {"name": "gliner_extractor_raises", "expect_graceful": True, "fn": lambda: full(pipe=build_pipeline(gliner=_Boom()))},
        {"name": "dictionary_extractor_raises", "expect_graceful": True, "fn": lambda: full(pipe=build_pipeline(dictionary=_Boom()))},
        {"name": "empty_course_catalog", "expect_graceful": True, "fn": lambda: full(cat=[])},
        {"name": "empty_profile_text", "expect_graceful": True, "fn": lambda: full(user_text="")},
        {"name": "job_text_without_skills", "expect_graceful": True, "fn": lambda: full(job_text="We are hiring a friendly colleague.")},
        {"name": "unicode_and_emoji_text", "expect_graceful": True,
         "fn": lambda: full(user_text="我会 Python 🐍 и SQL، communication ✓", job_text="Requirements: Python 🐍, Docker.")},
        {"name": "catalog_row_missing_fields", "expect_graceful": True,
         "fn": lambda: full(cat=[{"course_id": "x1", "course_name": "Broken"}, {"course_id": "x2", "course_name": "Ok",
                                                                                "teaches_skills": ["Docker"], "prerequisite_skills": []}])},
        {"name": "very_long_text_20k_chars", "expect_graceful": True, "fn": lambda: full(user_text=long_text)},
        {"name": "user_profile_without_skills_key", "expect_graceful": True, "fn": lambda: only_downstream({"user_id": "u"}, ok_job)},
        {"name": "job_without_skill_lists", "expect_graceful": True, "fn": lambda: only_downstream({"user_id": "u", "skills": ["Python"]}, {"job_id": "j"})},
        {"name": "job_without_job_id", "expect_graceful": False,
         "fn": lambda: only_downstream({"user_id": "u", "skills": ["Python"]}, {"required_skills": ["Python"]})},
        {"name": "null_skill_lists", "expect_graceful": False,
         "fn": lambda: only_downstream({"user_id": "u", "skills": None}, {"job_id": "j", "required_skills": None})},
    ]


# ------------------------------------------------------------------------------------------------ run
def run(ctx: EvalContext) -> List[EvalResult]:
    import warnings

    with warnings.catch_warnings():  # the pipeline warns once per call that it is on stub output; that is stated in the caveats
        warnings.filterwarnings("ignore", message=".*deterministic STUB.*", category=RuntimeWarning)
        return _run(ctx)


def _run(ctx: EvalContext) -> List[EvalResult]:
    cases, equivalents, catalog = load_cases(ctx)
    if ctx.smoke:
        cases = cases[:5]
    kmap = gt_key_map(equivalents)
    cat_by_id = {c["course_id"]: c for c in catalog}
    ds = f"synthetic-e2e-v1(cases={len(cases)})"
    out: List[EvalResult] = []

    t0 = time.perf_counter()
    pipe = build_pipeline()
    init_ms = (time.perf_counter() - t0) * 1000.0

    lat: Dict[str, List[float]] = {}
    runs: Dict[str, List[Dict[str, Any]]] = {}
    for mode in ("pipeline", "demo", "oracle"):
        runs[mode] = [run_case(c, mode, pipe, catalog, lat if mode == "pipeline" else None) for c in cases]

    # ---- extraction vs ground truth (pipeline mode)
    up, ur, uf, jp, jr, jf = [], [], [], [], [], []
    for c, r in zip(cases, runs["pipeline"]):
        pu = {to_key(n, kmap) for n in r["extracted_user"]}
        tu = {to_key(n, kmap) for n in c["user_skills"]}
        pj = {to_key(n, kmap) for n in r["extracted_job"]}
        tj = {to_key(n, kmap) for n in c["job_required"] + c["job_preferred"]}
        a, b, f = _f1(pu, tu); up.append(a); ur.append(b); uf.append(f)
        a, b, f = _f1(pj, tj); jp.append(a); jr.append(b); jf.append(f)
    ex_cav = list(CAVEATS) + ["equivalence-aware matching (e.g. 'structured query language' == SQL); per-case macro average"]
    out += [_res("extraction.profile_precision", "precision", up, ctx, ds, {}, ex_cav, 1),
            _res("extraction.profile_recall", "recall", ur, ctx, ds, {}, ex_cav, 2),
            _res("extraction.profile_f1", "f1", uf, ctx, ds, {}, ex_cav, 3),
            _res("extraction.job_precision", "precision", jp, ctx, ds, {}, ex_cav, 4),
            _res("extraction.job_recall", "recall", jr, ctx, ds, {}, ex_cav, 5),
            _res("extraction.job_f1", "f1", jf, ctx, ds, {}, ex_cav, 6)]

    # ---- gap detection vs ground truth per mode
    for mode in ("pipeline", "demo", "oracle"):
        gp, gr, gf = [], [], []
        for c, r in zip(cases, runs[mode]):
            truth = {to_key(s, kmap) for s in c["job_required"] + c["job_preferred"]} - {to_key(s, kmap) for s in c["user_skills"]}
            pred = {to_key(g["skill"], kmap) for g in r["gaps"]}
            a, b, f = _f1(pred, truth); gp.append(a); gr.append(b); gf.append(f)
        cav = list(CAVEATS) + [f"mode={mode}: " + {"pipeline": "extractor on both sides", "demo": "extractor user skills vs authored job skill names (smoke_runner demo wiring)",
                                                    "oracle": "authored names both sides (downstream only)"}[mode]]
        out += [_res(f"gap_{mode}.precision", "precision", gp, ctx, ds, {"mode": mode}, cav, 7),
                _res(f"gap_{mode}.recall", "recall", gr, ctx, ds, {"mode": mode}, cav, 8),
                _res(f"gap_{mode}.f1", "f1", gf, ctx, ds, {"mode": mode}, cav, 9)]

    # ---- name-consistency loss: GT-held job skills reported as gaps although the user (per extractor) holds them
    fg_demo, fg_oracle = [], []
    for mode, bucket in (("demo", fg_demo), ("oracle", fg_oracle)):
        for c, r in zip(cases, runs[mode]):
            held_gt = {to_key(s, kmap) for s in c["user_skills"]} & {to_key(s, kmap) for s in c["job_required"] + c["job_preferred"]}
            extracted_keys = {to_key(n, kmap) for n in r["user"]["skills"]}
            both = held_gt & extracted_keys  # skills the user really has AND the extractor found
            if both:
                flagged = {to_key(g["skill"], kmap) for g in r["gaps"]}
                bucket.append(len(both & flagged) / len(both))
    out.append(_res("name_consistency.false_gap_rate_demo_wiring", "share_of_held_and_extracted_skills_flagged_as_gaps", fg_demo, ctx, ds,
                    {"direction": "lower", "oracle_wiring_rate": _mean(fg_oracle) if fg_oracle else None},
                    list(CAVEATS) + ["FINDING if >0: extractor emits canonical_skill_name ('structured query language', 'Amazon Web Services') while jobs/catalog use surface names ('SQL', 'AWS'); the exact-string matcher/gap detector then reports a held skill as missing"], 10))

    # ---- course recommendation: precision@3 vs random, and gap->course linkage
    rng = random.Random(f"{ctx.seed}:e2e-rand")
    for mode in ("pipeline", "oracle"):
        prec, link = [], []
        base = []
        no_relevant = no_reach = 0
        for c, r in zip(cases, runs[mode]):
            truth_gaps = {to_key(s, kmap) for s in c["job_required"] + c["job_preferred"]} - {to_key(s, kmap) for s in c["user_skills"]}
            relevant = set()
            for co in catalog:
                teaches = {to_key(s, kmap) for s in co["teaches_skills"]}
                if teaches & truth_gaps:
                    relevant.add(co["course_id"])
            # enabling prerequisites of relevant courses the user lacks
            for cid in list(relevant):
                for p in cat_by_id[cid]["prerequisite_skills"]:
                    pk = to_key(p, kmap)
                    if pk not in {to_key(s, kmap) for s in c["user_skills"]}:
                        relevant |= {co["course_id"] for co in catalog if pk in {to_key(s, kmap) for s in co["teaches_skills"]}}
            top = r["recs_vector"][:3]
            if relevant:
                prec.append(sum(1 for x in top if x["course_id"] in relevant) / 3.0)
                if mode == "oracle":
                    base.append(len(relevant) / len(catalog))
            else:
                no_relevant += 1
            reach = [g for g in truth_gaps if any(g in {to_key(s, kmap) for s in co["teaches_skills"]} for co in catalog)]
            if not reach:
                no_reach += 1
            if reach:
                covered = {to_key(s, kmap) for x in r["recs_vector"][:5] for s in x["covered_missing_skills"]}
                link.append(len(set(reach) & covered) / len(reach))
        out.append(_res(f"recommend_{mode}.precision_at_3", "precision_at_3", prec, ctx, ds,
                        {"mode": mode, "random_baseline_expected": _mean(base) if base else None, "relevance": "teaches a true gap or is a missing prerequisite of such a course",
                         "cases_total": len(cases), "dropped_zero_relevant_cases": no_relevant,
                         "dropped_rule": "cases with no relevant course are DROPPED from the mean (repo convention), not scored 0; precision@3 divides by 3 (short lists penalised)"},
                        list(CAVEATS) + (["precision@3 has a ceiling below 1 when fewer than 3 relevant courses exist"]), 11))
        out.append(_res(f"recommend_{mode}.gap_to_course_linkage_at_5", "share_of_reachable_gaps_covered_by_top5", link, ctx, ds,
                        {"mode": mode, "cases_total": len(cases), "dropped_cases_without_reachable_gaps": no_reach},
                        list(CAVEATS) + ["reachable = some catalog course teaches the gap skill"], 12))
        if mode == "oracle" and base:
            out.append(_res("recommend_random_baseline.precision_at_3", "expected_precision_at_3_random", base, ctx, ds, {"note": "relevant share of catalog"}, list(CAVEATS), 13))

    # ---- traceability + schema per wiring/path
    for mode, key, path in (("pipeline", "recs_vector", "vector_path"), ("oracle", "recs_vector", "vector_path"), ("pipeline", "recs_demo", "demo_path")):
        per_case, ok_t, tot_t, msgs = [], 0, 0, []
        for r in runs[mode]:
            ok, n, m = trace_rate(r, key, cat_by_id)
            ok_t += ok; tot_t += n; msgs.extend(m)
            if n:
                per_case.append(ok / n)
        e = _res(f"traceability.{mode}.{path}", "fully_traceable_recommendation_share", per_case, ctx, ds,
                 {"mode": mode, "path": path, "recommendations": tot_t, "traceable": ok_t, "sample_failures": msgs[:5],
                  "links": "user gap -> job skill -> course skill -> prerequisite status"}, list(CAVEATS) + ["per-case mean over cases that produced >=1 recommendation"], 14)
        out.append(e)
    for mode, key, path in (("pipeline", "final", "vector_path"), ("pipeline", "final_demo", "demo_path"), ("oracle", "final", "vector_path")):
        valid, errs = [], []
        for r in runs[mode]:
            e_ = validate_output(r[key])
            valid.append(0.0 if e_ else 1.0)
            errs.extend(e_[:2])
        out.append(_res(f"schema_valid.{mode}.{path}", "valid_output_share", valid, ctx, ds,
                        {"mode": mode, "path": path, "sample_errors": errs[:5], "schema": "strict JSON, required keys, score in [0,1], contiguous ranks, per-recommendation minimum contract"},
                        list(CAVEATS) + ["schema is authored from the documented minimum JSON contract in recommend/explanation_generator.py"], 15))

    # ---- latency
    for stage, vals in lat.items():
        vals_sorted = sorted(vals)
        out.append(EvalResult(f"{MODULE}.latency.{stage}", MODULE, "mean_latency_ms", _mean(vals), len(vals), ds,
                              bootstrap_ci(vals, ctx.n_boot, ctx.seed + 16),
                              {"direction": "lower", "median_ms": vals_sorted[len(vals) // 2], "p95_ms": vals_sorted[min(len(vals) - 1, int(0.95 * len(vals)))],
                               "pipeline_init_ms": init_ms, "hardware_dependent": True},
                              ["CPU wall-clock on whatever machine ran the suite; first call includes warm-up; stub extractors, not model inference"]))
    total = [sum(lat[s][i] for s in lat) for i in range(len(next(iter(lat.values()))))] if lat else []
    out.append(_res("latency.total_per_case", "mean_latency_ms", total, ctx, ds, {"direction": "lower", "hardware_dependent": True},
                    ["sum of stage means per case, full-pipeline mode"], 17))

    # ---- determinism
    r2 = [run_case(c, "pipeline", pipe, catalog) for c in cases]
    same = [1.0 if json.dumps(_strip_volatile({k: a[k] for k in ("extracted_user", "extracted_job", "match", "gaps", "gap_vector", "recs_vector")}), sort_keys=True, default=str)
            == json.dumps(_strip_volatile({k: b[k] for k in ("extracted_user", "extracted_job", "match", "gaps", "gap_vector", "recs_vector")}), sort_keys=True, default=str) else 0.0
            for a, b in zip(runs["pipeline"], r2)]
    out.append(_res("determinism.in_process", "identical_output_share", same, ctx, ds, {}, ["two runs in one process"], 18))
    if not ctx.smoke:
        src = str(Path(__file__).resolve().parents[2])
        digests: Dict[str, str] = {}
        for hs in ("1", "424242"):
            env = {**os.environ, "PYTHONHASHSEED": hs, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
            try:
                cp = subprocess.run([sys.executable, "-c", f"from pathlib import Path; from skillllm_v2.evals.e2e_evals import digest_all; print(digest_all(Path({str(ctx.root.resolve())!r})))"],
                                    capture_output=True, text=True, timeout=300, env=env, cwd=str(ctx.root.resolve()))
                digests[hs] = cp.stdout.strip().splitlines()[-1] if cp.returncode == 0 and cp.stdout.strip() else f"error:{cp.stderr[-200:]}"
            except Exception as exc:  # pragma: no cover
                digests[hs] = f"error:{exc}"
        ok = len(set(digests.values())) == 1 and not any(v.startswith("error") for v in digests.values())
        out.append(EvalResult(f"{MODULE}.determinism.across_hash_seeds", MODULE, "identical_digest_across_PYTHONHASHSEED", 1.0 if ok else 0.0, len(digests), ds, None,
                              {"digests": digests, "direction": "higher"}, ["two fresh interpreter processes with different PYTHONHASHSEED values; catches set-iteration-order nondeterminism"]))
    else:
        out.append(_skipped("determinism.across_hash_seeds", "identical_digest_across_PYTHONHASHSEED", ds, "smoke mode skips subprocess runs"))

    # ---- graceful degradation
    scen = degradation_scenarios(catalog, cases[0])
    rows = []
    for s in scen:
        t = time.perf_counter()
        try:
            res = s["fn"]()
            errs = validate_output(res)
            rows.append({"name": s["name"], "expect_graceful": s["expect_graceful"], "graceful": not errs, "error": "; ".join(errs[:2]),
                         "seconds": round(time.perf_counter() - t, 3)})
        except Exception as exc:
            rows.append({"name": s["name"], "expect_graceful": s["expect_graceful"], "graceful": False,
                         "error": f"{type(exc).__name__}: {str(exc)[:100]}", "seconds": round(time.perf_counter() - t, 3)})
    expected = [r for r in rows if r["expect_graceful"]]
    out.append(EvalResult(f"{MODULE}.degradation.graceful_share", MODULE, "graceful_share_of_expected_scenarios",
                          sum(r["graceful"] for r in expected) / len(expected), len(expected), "synthetic-degradation-scenarios",
                          bootstrap_ci([1.0 if r["graceful"] else 0.0 for r in expected], ctx.n_boot, ctx.seed + 19),
                          {"scenarios": rows, "direction": "higher",
                           "informational_scenarios": [r for r in rows if not r["expect_graceful"]]},
                          ["graceful = returns schema-valid output without raising; scenarios marked informational are input-contract violations and are recorded but not scored",
                           "FINDING for every failing scenario: see details.scenarios"]))

    # ---- optional pipeline.inference
    try:
        import importlib
        mod = importlib.import_module("skillllm_v2.pipeline.inference")
        public = sorted(n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n)))
        out.append(EvalResult(f"{MODULE}.inference_module_present", MODULE, "module_importable", 1.0, 1, ds, None,
                              {"public_callables": [n for n in public if not n[0].isupper() or n == "SkillLLMV2Pipeline"][:20]},
                              ["module imported; only the generic probe below (completion, strict JSON, recall, latency, determinism) is run on it"]))
        out += _inference_probe(mod, cases, kmap, ctx, ds)
    except Exception as exc:
        out.append(_skipped("inference_module_present", "module_importable", ds, f"skillllm_v2.pipeline.inference not importable: {type(exc).__name__}: {str(exc)[:80]}"))

    out += _controls(runs, cases, catalog, cat_by_id, kmap, ds)
    return out



def _inference_probe(mod: Any, cases: List[Dict[str, Any]], kmap: Dict[str, str], ctx: EvalContext, ds: str) -> List[EvalResult]:
    """Generic probe of skillllm_v2.pipeline.inference.SkillLLMV2Pipeline (stub anchor): completes, strict JSON, skill recall, latency, determinism."""
    cls = getattr(mod, "SkillLLMV2Pipeline", None)
    if cls is None:
        return [_skipped("inference.probe", "completion_share", ds, "module has no SkillLLMV2Pipeline class")]
    try:
        pipe = cls({"inference": {"anchor": {"mode": "stub"}}})
    except Exception as exc:  # noqa: BLE001
        return [_skipped("inference.probe", "completion_share", ds, f"could not construct SkillLLMV2Pipeline: {type(exc).__name__}: {str(exc)[:100]}")]
    done, strict, recall, lat, same = [], [], [], [], []
    warm_ms = None
    for i, c in enumerate(cases):
        t = time.perf_counter()
        try:
            r1 = pipe.run(c["profile_text"], c["job_text"])
            ms = (time.perf_counter() - t) * 1000.0
            d1 = r1.to_dict() if hasattr(r1, "to_dict") else {}
            try:
                json.dumps(d1, allow_nan=False, default=str)
                strict.append(1.0)
            except (TypeError, ValueError):
                strict.append(0.0)
            done.append(1.0)
            names = {to_key(skill_name(x), kmap) for x in getattr(r1, "skills", [])}
            gt = {to_key(x, kmap) for x in c["user_skills"]}
            recall.append(len(names & gt) / len(gt) if gt else 1.0)
            if i == 0:
                warm_ms = ms
            else:
                lat.append(ms)
            r2 = pipe.run(c["profile_text"], c["job_text"])
            same.append(1.0 if [skill_name(x) for x in r1.skills] == [skill_name(x) for x in r2.skills] else 0.0)
        except Exception:  # noqa: BLE001
            done.append(0.0)
    cav = list(CAVEATS) + ["probe only: the inference pipeline's job-text parsing and course stage are not scored here"]
    out = [_res("inference.completion_share", "run_completes_share", done, ctx, ds, {"direction": "higher"}, cav, 31),
           _res("inference.strict_json_share", "to_dict_strict_json_share", strict, ctx, ds, {"direction": "higher"}, cav, 32),
           _res("inference.profile_skill_recall", "recall", recall, ctx, ds, {}, cav, 33),
           _res("inference.determinism_same_skills_twice", "identical_share", same, ctx, ds, {"direction": "higher"}, cav, 34)]
    if lat:
        out.append(_res("inference.latency_after_warmup", "mean_latency_ms", lat, ctx, ds, {"direction": "lower", "hardware_dependent": True, "first_call_ms": warm_ms},
                        ["first call (lazy loading) excluded and reported in details"], 35))
    return out

def _controls(runs: Dict[str, List[Dict[str, Any]]], cases: List[Dict[str, Any]], catalog: List[Dict[str, Any]], cat_by_id: Dict[str, Dict[str, Any]],
              kmap: Dict[str, str], ds: str) -> List[EvalResult]:
    out: List[EvalResult] = []
    # oracle wiring must reproduce the authored gap set exactly
    exact = 0
    for c, r in zip(cases, runs["oracle"]):
        truth = {to_key(s, kmap) for s in c["job_required"] + c["job_preferred"]} - {to_key(s, kmap) for s in c["user_skills"]}
        exact += {to_key(g["skill"], kmap) for g in r["gaps"]} == truth
    out.append(_control("oracle_wiring_reproduces_truth_gaps", "exact_match_share", exact / len(cases), exact == len(cases),
                        "authored names on both sides reproduce the authored gap set in every case (harness + matcher wiring sanity)", len(cases), ds))
    # schema validator negative controls
    sample = next((r["final"] for r in runs["oracle"] if r["final"]["course_recommendations"]), None)
    if sample is None:
        out.append(_skipped("control.schema_validator_rejects_corruption", "rejected_share", ds, "no sample output with recommendations"))
    else:
        corruptions: List[Callable[[Dict[str, Any]], None]] = [
            lambda o: o["course_recommendations"][0].pop("course_id"),
            lambda o: o["course_recommendations"][0].__setitem__("score", float("nan")),
            lambda o: o["course_recommendations"][0].__setitem__("rank", 0),
            lambda o: o["course_recommendations"][0].__setitem__("evidence", "not-a-list"),
            lambda o: o["course_recommendations"][0]["prerequisite_status"].__setitem__("is_satisfied", "yes"),
            lambda o: o.pop("job_matches"),
            lambda o: o["job_matches"][0].__setitem__("match_score", 1.7),
        ]
        rejected = 0
        for fn in corruptions:
            o = copy.deepcopy(sample)
            fn(o)
            rejected += bool(validate_output(o))
        clean_ok = not validate_output(sample)
        out.append(_control("schema_validator_rejects_corruption", "rejected_share", rejected / len(corruptions),
                            clean_ok and rejected == len(corruptions), "clean output validates; every one of 7 corruptions is rejected", len(corruptions), ds))
    # traceability checker negative controls
    r = next((r for r in runs["oracle"] if r["recs_vector"]), None)
    if r is None:
        out.append(_skipped("control.traceability_checker_flags_tampering", "flagged_share", ds, "no recommendation to tamper with"))
    else:
        gv_keys = set(r["gap_vector"]["vector"])
        job_set = {normalize_text(s) for s in r["job"]["required_skills"] + r["job"]["preferred_skills"]}
        known = {normalize_text(s) for s in r["user"]["skills"]}
        rec0 = r["recs_vector"][0]
        clean_fail = trace_failures(rec0, gv_keys, job_set, known, cat_by_id, [])
        t1 = copy.deepcopy(rec0); t1["covered_missing_skills"] = ["underwater basket weaving"]
        t2 = copy.deepcopy(rec0); t2["prerequisite_status"]["is_satisfied"] = not t2["prerequisite_status"]["is_satisfied"]
        t3 = copy.deepcopy(rec0); t3["course_id"] = "no-such-course"
        t4 = copy.deepcopy(rec0); t4["explanation_path"] = []
        flagged = sum(bool(trace_failures(t, gv_keys, job_set, known, cat_by_id, [])) for t in (t1, t2, t3, t4))
        out.append(_control("traceability_checker_flags_tampering", "flagged_share", flagged / 4, flagged == 4 and not clean_fail,
                            "an untampered recommendation passes; 4 tampered variants (foreign skill, flipped prerequisite flag, unknown course, empty path) are all flagged",
                            4, ds))
    # real recommendations must never cover a skill outside the job
    bad = 0
    tot = 0
    for r in runs["oracle"]:
        js = {normalize_text(s) for s in r["job"]["required_skills"] + r["job"]["preferred_skills"]}
        for rec in r["recs_vector"]:
            for s in rec["covered_missing_skills"]:
                tot += 1
                bad += normalize_text(s) not in js
    out.append(_control("no_recommendation_covers_foreign_skill", "foreign_skill_share", bad / tot if tot else 0.0, bad == 0,
                        "covered_missing_skills is always a subset of the job's skills", tot, ds))
    return out


if __name__ == "__main__":  # python -m skillllm_v2.evals.e2e_evals  -> prints the digest (used by tests)
    print(digest_all(Path(".")))
