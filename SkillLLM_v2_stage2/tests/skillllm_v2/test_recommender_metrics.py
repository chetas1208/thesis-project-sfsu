import math

import pytest

from skillllm_v2.eval import ranking_metrics as rm
from skillllm_v2.eval.build_relevance_labels import content_grade, gold_needs, label_scenario
from skillllm_v2.eval.benchmark_scenarios_spec import S
from skillllm_v2.eval.recommender_benchmark import apply_prereq_filter
from skillllm_v2.course.ingest_catalog import CourseCatalog


GRADES = {"a": 3, "b": 2, "c": 1, "d": 0, "e": 2}


def test_ndcg_perfect_and_known_value():
    # ideal order: a(3), b(2), e(2) ...
    assert rm.ndcg_at_k(["a", "b", "e"], GRADES, 3) == pytest.approx(1.0)
    # hand computed: ranking [b, a]: dcg = 3/log2(2) + 7/log2(3); ideal (k=2) = 7/1 + 3/log2(3)
    dcg = 3 / math.log2(2) + 7 / math.log2(3)
    idcg = 7 / math.log2(2) + 3 / math.log2(3)
    assert rm.ndcg_at_k(["b", "a"], GRADES, 2) == pytest.approx(dcg / idcg)


def test_ndcg_zero_and_undefined():
    assert rm.ndcg_at_k(["d", "x"], GRADES, 5) == 0.0
    assert rm.ndcg_at_k(["a"], {"a": 0, "b": 0}, 5) is None  # no relevant course at all


def test_ndcg_ideal_uses_all_judged_courses_not_only_retrieved():
    # system retrieved only the grade-2 item; ideal has the grade-3 item first
    val = rm.ndcg_at_k(["b"], GRADES, 3)
    ideal = 7 / math.log2(2) + 3 / math.log2(3) + 3 / math.log2(4)
    assert val == pytest.approx((3 / math.log2(2)) / ideal)


def test_ndcg_ignores_duplicates_and_respects_k():
    assert rm.ndcg_at_k(["a", "a", "b"], GRADES, 2) == pytest.approx(rm.ndcg_at_k(["a", "b"], GRADES, 2))
    # relevant item beyond k contributes nothing
    assert rm.ndcg_at_k(["d", "d2", "a"], GRADES, 2) == 0.0


def test_precision_at_k_threshold():
    assert rm.precision_at_k(["a", "c", "d", "b", "x"], GRADES, 5, rel_threshold=2) == pytest.approx(2 / 5)
    assert rm.precision_at_k(["a", "c", "d", "b", "x"], GRADES, 5, rel_threshold=1) == pytest.approx(3 / 5)
    # short list is penalised (divides by k)
    assert rm.precision_at_k(["a"], GRADES, 5) == pytest.approx(1 / 5)


def test_recall_at_k():
    # relevant (>=2): a, b, e  -> 3 total
    assert rm.recall_at_k(["a", "d", "b"], GRADES, 3) == pytest.approx(2 / 3)
    assert rm.recall_at_k(["a", "d", "b"], GRADES, 1) == pytest.approx(1 / 3)
    assert rm.recall_at_k(["x"], {"x": 0}, 5) is None


def test_average_precision_known_cases():
    # relevant a,b,e. ranking [a, d, b]: P@1=1, P@3=2/3 -> (1 + 2/3)/min(3,10)=0.5556
    assert rm.average_precision_at_k(["a", "d", "b"], GRADES, 10) == pytest.approx((1 + 2 / 3) / 3)
    # perfect
    assert rm.average_precision_at_k(["a", "b", "e"], GRADES, 10) == pytest.approx(1.0)
    # nothing relevant retrieved
    assert rm.average_precision_at_k(["d", "c"], GRADES, 10) == 0.0
    # truncated normaliser min(R, k): k=2 with R=3 -> perfect top-2 gives 1.0
    assert rm.average_precision_at_k(["a", "b"], GRADES, 2) == pytest.approx(1.0)


def test_map_is_mean_of_ap():
    aps = [rm.average_precision_at_k(r, GRADES, 10) for r in (["a", "b", "e"], ["d", "c"])]
    assert sum(aps) / 2 == pytest.approx(0.5)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [0.1, 0.4, 0.5, 0.6, 0.9, 0.3]
    m1 = rm.bootstrap_mean_ci(vals, n_boot=2000, seed=1)
    m2 = rm.bootstrap_mean_ci(vals, n_boot=2000, seed=1)
    assert m1 == m2
    assert m1[1] <= m1[0] <= m1[2]
    # zero-variance sample -> degenerate interval
    assert rm.bootstrap_mean_ci([0.5] * 5, n_boot=500, seed=0) == (0.5, 0.5, 0.5)


def test_paired_bootstrap_detects_consistent_advantage_and_ties():
    a = [0.9, 0.8, 0.85, 0.95, 0.9, 0.8, 0.9, 0.85]
    b = [0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.45]
    r = rm.paired_bootstrap(a, b, n_boot=4000, seed=3)
    assert r["mean_diff"] > 0.3 and r["p_value"] < 0.01 and r["wins"] == 8 and r["losses"] == 0
    same = rm.paired_bootstrap(a, a, n_boot=1000, seed=3)
    assert same["mean_diff"] == 0 and same["p_value"] == 1.0 and same["ties"] == 8


def test_holm_adjust_monotone_and_known():
    adj = rm.holm_adjust([0.01, 0.04, 0.03])
    # sorted p: .01*3=.03 ; .03*2=.06 ; .04*1=.04 -> running max => .03,.06,.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)


# ---- label procedure -------------------------------------------------------------------------------
def _catalog():
    return CourseCatalog([
        {"course_id": "py", "title": "Python", "teaches_skills": ["Python"], "prerequisites": []},
        {"course_id": "docker", "title": "Docker", "teaches_skills": ["Docker"], "prerequisites": ["Linux"]},
        {"course_id": "k8s", "title": "K8s", "teaches_skills": ["Kubernetes"], "prerequisites": ["Docker"]},
        {"course_id": "mesh", "title": "Mesh", "teaches_skills": ["Service Mesh"], "prerequisites": ["Kubernetes"]},
        {"course_id": "linux", "title": "Linux", "teaches_skills": ["Linux"], "prerequisites": []},
        {"course_id": "wide", "title": "Wide", "teaches_skills": ["Kubernetes", "A", "B", "C"], "prerequisites": []},
    ])


def test_content_grade_rules():
    needs = {"Docker": 3, "CI/CD": 1}
    assert content_grade(["Docker"], needs)[0] == 3
    assert content_grade(["Docker", "X", "Y", "Z"], needs)[0] == 2  # m>=3 but low focus
    assert content_grade(["CI/CD"], needs)[0] == 1
    assert content_grade(["X"], needs)[0] == 0


def test_label_prerequisite_feasibility_and_enabler():
    sc = S("t", "bg", "mid", known="Python", job_title="J", text="x", req="Kubernetes", pref="", implicit="")
    lab = label_scenario(sc, _catalog())["final"]
    # K8s needs Docker which needs Linux (2 hops) -> not reachable within one bridge -> 0
    assert lab["k8s"] == 0
    # 'wide' has no prerequisite; covers need mass 3 with focus 0.25 -> grade 2
    assert lab["wide"] == 2
    assert lab["py"] == 0

    sc2 = S("t2", "bg", "mid", known="Python,Linux", job_title="J", text="x", req="Kubernetes", pref="", implicit="")
    lab2 = label_scenario(sc2, _catalog())["final"]
    # Docker is a one-step bridge -> k8s capped at 1, docker course is an enabler -> 1
    assert lab2["k8s"] == 1
    assert lab2["docker"] == 1

    sc3 = S("t3", "bg", "mid", known="Python,Linux,Docker", job_title="J", text="x", req="Kubernetes", pref="", implicit="")
    assert label_scenario(sc3, _catalog())["final"]["k8s"] == 3  # startable and focused


def test_gold_needs_weights():
    sc = S("t", "bg", "mid", known="", job_title="J", text="x", req="A,B", pref="C,A", implicit="D")
    assert gold_needs(sc) == {"A": 3, "B": 3, "C": 1, "D": 2}


def test_prereq_filter_is_chained():
    cat = _catalog()
    ranked = [cat.get_course(i) for i in ("k8s", "docker", "linux", "py")]
    kept = apply_prereq_filter(ranked, {"python"}, k=10)
    # k8s dropped (Docker unknown), docker dropped (Linux unknown), linux kept; then (list is single pass)
    assert [c.course_id for c in kept] == ["linux", "py"]
    kept2 = apply_prereq_filter(ranked, {"linux"}, k=10)
    assert [c.course_id for c in kept2] == ["docker", "linux", "py"]
