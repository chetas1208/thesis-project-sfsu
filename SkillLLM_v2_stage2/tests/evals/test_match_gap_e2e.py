"""Tests for match_evals, gap_evals and e2e_evals: hand-checkable helpers, detection power of the property checks
(they must flag deliberately broken systems), and smoke runs of the three modules (controls must pass)."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from skillllm_v2.evals import e2e_evals as ee
from skillllm_v2.evals import gap_evals as ge
from skillllm_v2.evals import match_evals as me
from skillllm_v2.evals.contract import EvalContext
from skillllm_v2.match.job_matcher import JobMatcher
from skillllm_v2.recommend.gap_vector import GapEngine

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def ctx():
    return EvalContext(root=ROOT, seed=42, n_boot=200, smoke=True)


def _by_name(results):
    return {r.eval_name: r for r in results}


# ------------------------------------------------------------------ rank correlation helpers
def test_kendall_spearman_perfect_reversed_and_ties():
    x = [1, 2, 3, 4, 5]
    assert ge.kendall_tau_b(x, x) == pytest.approx(1.0)
    assert ge.kendall_tau_b(x, x[::-1]) == pytest.approx(-1.0)
    assert ge.spearman(x, x[::-1]) == pytest.approx(-1.0)
    assert ge.kendall_tau_b([1, 1, 1], [1, 2, 3]) is None  # constant side is undefined, not 0
    # hand example x=[1,2,2,3], y=[1,3,2,3]: C=4 D=0 Tx=1 Ty=1 -> 4/sqrt(5*5)
    assert ge.kendall_tau_b([1, 2, 2, 3], [1, 3, 2, 3]) == pytest.approx(4 / 5)


def test_rank_corr_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    import random
    rng = random.Random(1)
    for _ in range(20):
        a = [rng.randint(0, 4) for _ in range(9)]
        b = [rng.randint(0, 4) for _ in range(9)]
        if ge.kendall_tau_b(a, b) is None:
            continue
        assert ge.kendall_tau_b(a, b) == pytest.approx(stats.kendalltau(a, b).statistic)
        assert ge.spearman(a, b) == pytest.approx(stats.spearmanr(a, b).statistic)


# ------------------------------------------------------------------ authored grading
def test_grade_pair_rules(ctx):
    uni = me.load_universe(ctx)
    job = {"required_skills": ["Python", "SQL", "Docker"], "preferred_skills": ["AWS"],
           "_tiers": {"Python": 3.0, "SQL": 3.0, "Docker": 2.0, "AWS": 1.0}, "_core": ["Python", "SQL"]}
    assert me.grade_pair(uni, ["Python", "SQL", "Docker", "AWS"], job) == 3
    assert me.grade_pair(uni, ["Python", "Docker", "AWS"], job) <= 1  # a core skill is completely missing -> capped at 1
    assert me.grade_pair(uni, ["Python", "SQL"], job) == 2  # 6/9 coverage = 0.67
    assert me.grade_pair(uni, ["Photoshop"], job) == 0
    # adjacency credit: Azure is adjacent to AWS -> half credit for a nice-to-have, never a core skill
    assert me.grade_pair(uni, ["Python", "SQL", "Docker", "Azure"], job) == 3


def test_hit_at_1_undefined_without_relevant():
    assert me.hit_at_1(["a", "b"], {"a": 1, "b": 0}) is None
    assert me.hit_at_1(["a", "b"], {"a": 3, "b": 2}) == 1.0
    assert me.hit_at_1(["b", "a"], {"a": 3, "b": 2}) == 0.0


# ------------------------------------------------------------------ property checks have detection power
def test_match_property_checks_flag_case_sensitivity_and_pass_when_fixed(ctx, monkeypatch):
    uni = me.load_universe(ctx)
    res = _by_name(me.property_checks(uni, 7, 120, "t"))
    assert res["match.prop.case_invariance"].value == 0.0  # fixed: shared skill_key canonicalisation (docs/BUGFIX_LOG_MATCH.md bug 1); was 0.277
    assert res["match.prop.score_in_unit_interval"].value == 0.0

    class Normalising(JobMatcher):
        def score(self, user_profile, job):
            u = {**user_profile, "skills": [s.lower() for s in user_profile.get("skills", [])]}
            j = {**job, "required_skills": [s.lower() for s in job.get("required_skills", [])],
                 "preferred_skills": [s.lower() for s in job.get("preferred_skills", [])]}
            return super().score(u, j)

    monkeypatch.setattr(me, "JobMatcher", Normalising)
    res2 = _by_name(me.property_checks(uni, 7, 120, "t"))
    assert res2["match.prop.case_invariance"].value == 0.0


def test_match_invariance_detects_attribute_leak(ctx, monkeypatch):
    uni = me.load_universe(ctx)
    ok = me.invariance_checks(uni, 3, 20, "t")
    assert all(r.details.get("passed", True) for r in ok)

    class Leaky(JobMatcher):
        def score(self, user_profile, job):
            out = super().score(user_profile, job)
            if user_profile.get("gender") == "f":
                out["match_score"] = max(0.0, out["match_score"] - 0.05)
            return out

    monkeypatch.setattr(me, "JobMatcher", Leaky)
    bad = {r.eval_name: r for r in me.invariance_checks(uni, 3, 60, "t")}
    assert bad["match.control.invariance_zero_delta"].details["passed"] is False


def test_gap_property_checks_flag_broken_engine(ctx, monkeypatch):
    uni = me.load_universe(ctx)
    good = _by_name(ge.property_block(uni, ctx, 80))
    assert good["gap.prop.engine.raising_importance_never_lowers_gap_weight"].value == 0.0
    assert good["gap.prop.engine.adding_strong_held_skill_never_increases_gap"].value == 0.0
    # formerly documented bugs, fixed in docs/BUGFIX_LOG_MATCH.md (bugs 1-2); now regression guards
    assert good["gap.prop.detector.case_invariance"].value == 0.0
    assert good["gap.prop.detector.held_skill_never_labelled_covered_by_transferable"].value == 0.0

    class Broken(GapEngine):
        def compute(self, user_profile, job, graph=None):
            gv = super().compute(user_profile, job, graph)
            for g in gv.gaps:
                gv.vector[g.canonical_skill_id] = g.gap_weight = 1.0 / (0.1 + g.importance)  # decreasing in importance
            gv.total_gap_mass = sum(gv.vector.values())
            return gv

    monkeypatch.setattr(ge, "GapEngine", Broken)
    bad = _by_name(ge.property_block(uni, ctx, 80))
    assert bad["gap.prop.engine.raising_importance_never_lowers_gap_weight"].value > 0.2


def test_gap_truth_is_independent_and_consistent(ctx):
    uni = me.load_universe(ctx)
    cases = ge.make_gap_cases(uni, 42, 30)
    for c in cases:
        assert set(c["truth"]) <= set(c["job_skills"])
        for t in c["truth"].values():
            assert t["severity"] > 0 and t["state"] != "strong"
    # oracle predictions score F1 = 1 through the same scorer
    tp, fp, fn = ge.prf(cases[0]["truth"].keys(), cases[0]["truth"].keys())
    assert fp == 0 and fn == 0


# ------------------------------------------------------------------ module smoke runs
@pytest.mark.parametrize("mod,prefix", [(me, "match."), (ge, "gap."), (ee, "e2e.")])
def test_module_smoke_run_controls_pass_and_names_prefixed(mod, prefix, ctx):
    results = mod.run(ctx)
    assert results
    assert all(r.eval_name.startswith(prefix) and r.module == prefix[:-1] for r in results)
    assert not [r for r in results if r.status == "failed"]
    controls = [r for r in results if r.details.get("control")]
    assert controls
    assert all(r.details["passed"] for r in controls if r.status == "ok"), [r.eval_name for r in controls if not r.details["passed"]]
    for r in results:
        if r.value is not None:
            assert math.isfinite(r.value)
        if r.ci:
            assert r.ci[0] <= r.ci[1] + 1e-12


def test_match_smoke_is_deterministic_and_reports_ap_convention(ctx):
    a = {r.eval_name: (r.value, r.ci) for r in me.run(ctx)}
    b = {r.eval_name: (r.value, r.ci) for r in me.run(ctx)}
    assert a == b
    m = _by_name(me.run(ctx))
    assert "ap_convention" in m["match.map_at_10"].details
    assert "dropped_undefined_scenarios" in m["match.ndcg_at_10"].details
    # trec convention (divide by R, zero-relevant scenarios count as 0) can only lower MAP
    assert m["match.map_at_10_trec_convention"].value <= m["match.map_at_10"].value + 1e-12
    assert m["match.baseline.random.ndcg_at_10"].value < m["match.ndcg_at_10"].value


def test_match_beats_random_with_paired_ci(ctx):
    m = _by_name(me.run(ctx))
    p = m["match.paired_vs_random.ndcg_at_10"]
    assert p.ci[0] > 0.3


# ------------------------------------------------------------------ e2e helpers
def _sample_output(ctx):
    cases, _, catalog = ee.load_cases(ctx)
    pipe = ee.build_pipeline()
    for c in cases:
        r = ee.run_case(c, "oracle", pipe, catalog)
        if r["final"]["course_recommendations"]:
            return r, {x["course_id"]: x for x in catalog}
    raise AssertionError("no case produced recommendations")


def test_schema_validator_accepts_real_and_rejects_corrupt(ctx):
    r, _ = _sample_output(ctx)
    assert ee.validate_output(r["final"]) == []
    for mutate in (lambda o: o["course_recommendations"][0].pop("title"),
                   lambda o: o["course_recommendations"][0].__setitem__("score", float("inf")),
                   lambda o: o["course_recommendations"][0].__setitem__("rank", 5),
                   lambda o: o["job_matches"][0].__setitem__("match_score", -0.1)):
        o = copy.deepcopy(r["final"])
        mutate(o)
        assert ee.validate_output(o), "corruption not detected"


def test_traceability_flags_tampering(ctx):
    r, cat = _sample_output(ctx)
    gv = set(r["gap_vector"]["vector"])
    job = {ee.normalize_text(s) for s in r["job"]["required_skills"] + r["job"]["preferred_skills"]}
    known = {ee.normalize_text(s) for s in r["user"]["skills"]}
    rec = r["recs_vector"][0]
    assert ee.trace_failures(rec, gv, job, known, cat, []) == []
    t = copy.deepcopy(rec)
    t["covered_missing_skills"] = ["knitting"]
    assert ee.trace_failures(t, gv, job, known, cat, [])


def test_pipeline_wiring_reproduces_gap_truth_in_oracle_mode(ctx):
    cases, eq, catalog = ee.load_cases(ctx)
    kmap = ee.gt_key_map(eq)
    pipe = ee.build_pipeline()
    for c in cases:
        r = ee.run_case(c, "oracle", pipe, catalog)
        truth = {ee.to_key(s, kmap) for s in c["job_required"] + c["job_preferred"]} - {ee.to_key(s, kmap) for s in c["user_skills"]}
        assert {ee.to_key(g["skill"], kmap) for g in r["gaps"]} == truth


def test_e2e_degradation_records_extractor_failure_as_finding(ctx):
    r = _by_name(ee.run(ctx))["e2e.degradation.graceful_share"]
    names = {s["name"]: s for s in r.details["scenarios"]}
    assert "empty_course_catalog" in names and names["empty_course_catalog"]["graceful"]
    assert 0.0 <= r.value <= 1.0
