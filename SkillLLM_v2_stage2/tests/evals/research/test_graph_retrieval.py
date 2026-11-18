"""Tests for skillllm_v2.evals.research.graph_retrieval (tiny synthetic fixtures; no pipeline build)."""

from types import SimpleNamespace

import pytest

from skillllm_v2.evals.research import graph_retrieval as gr


def test_paired_test_identical_systems_p_is_one():
    t = gr.paired_test([0.5, 0.2, 0.9], [0.5, 0.2, 0.9], n_boot=500)
    assert t["delta"] == 0 and t["p_boot"] == 1.0 and t["p_perm"] == 1.0 and t["ties"] == 3


def test_paired_test_clear_shift_is_significant():
    a = [0.6 + 0.01 * i for i in range(30)]
    b = [x - 0.2 for x in a]
    t = gr.paired_test(a, b, n_boot=2000)
    assert abs(t["delta"] - 0.2) < 1e-9 and t["ci_low"] > 0.19 and t["p"] < 0.01 and t["wins"] == 30


def test_boot_ci_and_holm():
    m, lo, hi = gr.boot_ci([1.0, 1.0, 1.0], n_boot=100)
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    assert gr.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_ranking_metrics_hand_values():
    grades = {"a": 3, "b": 2, "c": 1}
    ids = ["c", "a", "x", "b"]
    assert gr.precision(ids, grades, 5) == pytest.approx(2 / 5)
    assert gr.recall(ids, grades, 10) == 1.0
    assert gr.mrr(ids, grades, 10) == 0.5
    assert gr.ndcg(["a", "b", "c"], grades, 10) == pytest.approx(1.0)
    assert gr.recall(ids, {"c": 1}, 10) is None  # no grade>=2 item


def test_rrf_is_deterministic_and_rewards_agreement():
    out = gr.rrf([["a", "b", "c"], ["b", "a", "d"], ["b", "c"]])
    assert out[0] == "b" and set(out) == {"a", "b", "c", "d"}
    assert gr.rrf([["x", "y"], ["y", "x"]]) == ["x", "y"]  # tie -> id order


def test_visible_gaps_matches_existing_baseline_on_benchmark():
    from skillllm_v2.eval.recommender_benchmark import user_gap_weights
    for sc in gr.load_jsonl(gr.SCENARIOS):
        assert gr.visible_gaps(sc["user_profile"], sc["target_job"]) == user_gap_weights(sc["user_profile"], sc["target_job"])


def _toy_graph():
    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
    b = CourseGraphBuilder()
    b.ingest_course({"course_id": "c1", "title": "Advanced X", "teaches_skills": ["X"], "prerequisites": ["P"], "level": "advanced"})
    b.ingest_course({"course_id": "c2", "title": "Intro P", "teaches_skills": ["P"], "prerequisites": [], "level": "beginner"})
    b.ingest_course({"course_id": "c3", "title": "Unrelated", "teaches_skills": ["Z"], "prerequisites": [], "level": "beginner"})
    b.ingest_user_profile({"user_id": "u", "skills": [{"surface_form": "Q", "canonical_skill_name": "Q"}]})
    b.ingest_job({"job_id": "j", "title": "Job", "required_skills": ["X"], "preferred_skills": []})
    return b.graph


def test_graph_retrieve_two_hop_and_paths_valid():
    G = _toy_graph()
    ranking, paths = gr.graph_retrieve(G, "j", "u", {"x": 1.0})
    assert ranking == ["c1", "c2"]
    assert paths["c2"][0]["hop"] == 2
    for c in ranking:
        for p in paths[c]:
            assert gr.path_valid(G, p)[0]


def test_path_valid_detects_bad_edges():
    G = _toy_graph()
    ok, why = gr.path_valid(G, {"edges": [("c3", "TEACHES", "x")]})
    assert not ok and why[0].startswith("missing_edge")
    ok, why = gr.path_valid(G, {"edges": [("j", "TEACHES", "x")]})
    assert not ok  # Job has no TEACHES edge
    ok, why = gr.path_valid(G, {"edges": [], "non_edges": [("u", "HAS_SKILL", "q")]})
    assert not ok and why[0].startswith("edge_present")


def test_skill_prereq_pairs_and_concept_norm():
    G = _toy_graph()
    assert gr.skill_prereq_pairs(G) == {("x", "p"): 1}
    assert gr.concept_norm("Object-oriented_programming") == "object oriented programming"
    assert gr.concept_norm("Mercury_(planet)") == "mercury"


def test_text_labels_reads_text_not_skill_field():
    c = SimpleNamespace(course_id="c", title="Docker basics", description="Containers with Docker.", learning_outcomes=["Use SQL"],
                        teaches_skills=["Kubernetes"], prerequisites=[])
    resolver = SimpleNamespace(keys_for_name=lambda n: set())
    out = gr.text_labels({"Docker": 3, "Kubernetes": 1}, [c], resolver)
    assert out == {"c": 2}  # Docker mentioned (mass 3, one need) -> 2; Kubernetes only in teaches_skills -> ignored


def test_generated_set_is_deterministic_and_in_vocabulary():
    from skillllm_v2.course.ingest_catalog import CourseCatalog
    courses = CourseCatalog.load_from_file(gr.CATALOG).list_courses()
    s1, l1 = gr.generate_sg_v1(courses, n=6)
    s2, l2 = gr.generate_sg_v1(courses, n=6)
    assert s1 == s2 and l1 == l2
    vocab = {s for c in courses for s in c.teaches_skills}
    for sc in s1:
        assert set(sc["target_job"]["required_skills"]) <= vocab and 1 <= len(sc["target_job"]["required_skills"]) <= 4
        assert "Required skills" in sc["target_job"]["text"]
