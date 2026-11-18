"""Tests for skillllm_v2.evals.research.course_rec (tiny synthetic fixtures; no pipeline build)."""

from types import SimpleNamespace

import pytest

from skillllm_v2.evals.research import course_rec as cr

T = {"a": {"x", "y"}, "b": {"y", "z"}, "c": {"p"}, "d": {"w"}}
P = {"a": set(), "b": {"p"}, "c": set(), "d": {"q"}}


def test_wsgc_uses_catalogue_field_and_weights():
    gaps = {"x": 3.0, "z": 1.0, "w": 2.0}
    assert cr.wsgc_at_k(["a"], gaps, T) == pytest.approx(3 / 6)
    assert cr.wsgc_at_k(["a", "b"], gaps, T) == pytest.approx(4 / 6)
    assert cr.wsgc_at_k([], gaps, T) == 0.0
    assert cr.wsgc_at_k(["a"], {}, T) == 1.0


def test_violation_rate_chained():
    assert cr.violation_rate(["b", "c"], set(), T, P) == 0.5  # b needs p before c teaches it
    assert cr.violation_rate(["c", "b"], set(), T, P) == 0.0
    assert cr.violation_rate(["d"], set(), T, P) == 1.0


def test_rsc_counts_repeat_teaching():
    gaps = {"y": 1.0, "x": 1.0}
    assert cr.rsc_at_k(["a", "b"], gaps, T) == pytest.approx(1 / 3)  # instances {x,y},{y}: 3 total, 2 distinct
    assert cr.rsc_at_k(["d"], gaps, T) == 0.0


def test_path_validity_order_vs_repairable():
    assert cr.path_validity(["c", "b"], set(), T, P) == (1.0, 1.0)
    assert cr.path_validity(["b", "c"], set(), T, P) == (0.0, 1.0)
    assert cr.path_validity(["d"], set(), T, P) == (0.0, 0.0)


def _toy_env():
    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
    b = CourseGraphBuilder()
    courses = {
        "c1": SimpleNamespace(course_id="c1", title="Docker", description="Containers", learning_outcomes=["Build Docker images", "Write YAML"],
                              teaches_skills=["Docker"], prerequisites=[], duration_hours=50.0),
        "c2": SimpleNamespace(course_id="c2", title="K8s", description="Clusters", learning_outcomes=["Operate clusters"],
                              teaches_skills=["Kubernetes"], prerequisites=["Docker"], duration_hours=20.0),
    }
    for c in courses.values():
        b.ingest_course({"course_id": c.course_id, "title": c.title, "teaches_skills": c.teaches_skills, "prerequisites": c.prerequisites})
    env = SimpleNamespace(T={k: {cr.norm(s) for s in v.teaches_skills} for k, v in courses.items()},
                          P={k: {cr.norm(s) for s in v.prerequisites} for k, v in courses.items()}, by_id=courses)
    sc = {"target_job": {"required_skills": ["Docker"], "preferred_skills": ["Kubernetes"]},
          "user_profile": {"skills": [{"surface_form": "Linux", "canonical_skill_name": "Linux"}]}}
    return env, sc, b.graph


def test_check_explanations_flags_false_and_supports_true_claims():
    env, sc, G = _toy_env()
    resolver = SimpleNamespace(keys_for_name=lambda n: set())
    recs = [
        {"course_id": "c1", "covered_missing_skills": ["Docker"], "role": "primary",
         "evidence": [{"skill": "Docker", "canonical_skill_id": "docker", "gap_classification": "missing_critical",
                       "course_learning_outcome_evidence": "Build Docker images"}],
         "prerequisite_status": {"status_code": "satisfied_none_required", "satisfied": [], "missing": []},
         "unlocks": [], "limitations": ["High workload commitment: 50h"]},
        {"course_id": "c2", "covered_missing_skills": ["Kubernetes", "Python"], "role": "primary",
         "evidence": [{"skill": "Kubernetes", "canonical_skill_id": "kubernetes", "gap_classification": "missing_critical",
                       "course_learning_outcome_evidence": "Operate clusters"}],
         "prerequisite_status": {"status_code": "all_satisfied", "satisfied": [{"skill_id": "docker", "satisfied_by": "earlier_course:c1"}], "missing": []},
         "unlocks": [], "limitations": ["High workload commitment: 20h"]},
    ]
    claims = cr.check_explanations(recs, sc, env, G, resolver)
    by = {}
    for c in claims:
        by.setdefault((c["course"], c["type"]), []).append(c["supported"])
    assert by[("c1", "covers_gap_skill[strict field semantics]")] == [True]
    assert by[("c1", "outcome_quote_mentions_skill")] == [True]
    assert by[("c1", "prerequisite_status")] == [True]
    assert by[("c1", "limitation_workload")] == [True]
    assert by[("c2", "covers_gap_skill[strict field semantics]")] == [True, False]  # Python is neither taught nor a gap
    assert by[("c2", "user_gap_classification")] == [False]  # Kubernetes is preferred, not critical
    assert by[("c2", "outcome_quote_mentions_skill")] == [False]  # quote does not mention Kubernetes
    assert by[("c2", "prerequisite_satisfied_claim")] == [True]  # docker taught by earlier c1
    assert by[("c2", "limitation_workload")] == [False]  # 20h is not > 40h
    fs = cr.faith_summary({"s": claims}, ["s"], strict=True)
    assert fs["n_unsupported"] == 4 and 0 < fs["micro"] < 1


def test_kappa_sanity():
    assert cr.kappa([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)
    assert cr.kappa([0, 1, 2, 3], [0, 1, 2, 3], "quadratic") == pytest.approx(1.0)
