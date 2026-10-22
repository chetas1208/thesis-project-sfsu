from __future__ import annotations

import json
from pathlib import Path

from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.eval.course_recommendation_eval import explanation_completeness_rate, prerequisite_violation_rate
from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.recommend.config import DEFAULT_CONFIG, LEGACY_CONFIG
from skillllm_v2.recommend.course_ranker import CourseRanker
from skillllm_v2.recommend.course_retriever import CourseRetriever
from skillllm_v2.recommend.explanation_generator import CourseExplanationGenerator
from skillllm_v2.recommend.gap_vector import GapEngine
from skillllm_v2.recommend.loop_eval import SPLIT_PATH, split_ids

ROOT = Path(__file__).resolve().parents[2]


def _course(cid: str, title: str, teaches: list[str], prereq: list[str], level: str = "intermediate") -> CourseRecord:
    return CourseRecord(
        course_id=cid, title=title, learning_outcomes=[f"Able to apply {', '.join(teaches)}"], teaches_skills=teaches,
        prerequisites=prereq, level=level, duration_hours=20.0,
    )


def _setup(courses: list[CourseRecord], user_skills: list[str], required: list[str], preferred: list[str] | None = None):
    catalog = CourseCatalog(courses)
    graph = CourseGraphBuilder()
    graph.ingest_catalog(catalog)
    user = {"user_id": "u", "skills": list(user_skills), "target_roles": ["Role"]}
    job = {"job_id": "j", "title": "Role", "required_skills": list(required), "preferred_skills": list(preferred or [])}
    gv = GapEngine().compute(user, job, graph=graph)
    cands = [c.course for c in CourseRetriever(catalog=catalog, graph=graph).retrieve(gv, job, top_k=40)]
    return catalog, graph, user, job, gv, cands


def _payload(ranked, user, job, gv, graph):
    return CourseExplanationGenerator().generate_recommendations_payload(ranked, user, job, gv, graph)


CHAIN = [
    _course("py", "Python Basics", ["Python"], [], "beginner"),
    _course("ml", "Machine Learning", ["Machine Learning"], ["Python"]),
    _course("dl", "Deep Learning", ["Deep Learning"], ["Machine Learning"], "advanced"),
]


def test_legacy_ranker_default_is_unchanged_and_pipeline_default_is_tuned() -> None:
    assert CourseRanker().config == LEGACY_CONFIG
    assert LEGACY_CONFIG.prereq_mode == "soft"
    assert DEFAULT_CONFIG.prereq_mode == "strict_bridge"
    assert DEFAULT_CONFIG.infer_user_level is False  # round 4 was rejected on dev


def test_default_config_equals_final_config_of_the_logged_loop() -> None:
    log = ROOT / "reports/skillllm_v2/recommender_loop_log.json"
    if not log.exists():  # pragma: no cover
        return
    assert DEFAULT_CONFIG.to_dict() == json.loads(log.read_text())["final_config"]


def test_strict_drop_never_recommends_a_course_before_its_prerequisite() -> None:
    _, graph, user, job, gv, cands = _setup(CHAIN, ["Python"], ["Deep Learning"])
    ranked = CourseRanker(config=LEGACY_CONFIG.with_(prereq_mode="strict_drop")).rank(cands, user, job, gv, graph=graph, top_k=5)
    assert [r.course_id for r in ranked] == []  # DL is blocked, and strict_drop has no bridge to offer
    soft = CourseRanker(config=LEGACY_CONFIG).rank(cands, user, job, gv, graph=graph, top_k=5)
    assert [r.course_id for r in soft] == ["dl"]  # soft mode recommends it anyway (the violation being fixed)


def test_strict_bridge_places_bridge_before_the_blocked_course_with_full_explanation() -> None:
    _, graph, user, job, gv, cands = _setup(CHAIN, ["Python"], ["Deep Learning"])
    cfg = DEFAULT_CONFIG
    ranked = CourseRanker(config=cfg).rank(cands, user, job, gv, graph=graph, top_k=5, catalog_courses=CHAIN)
    ids = [r.course_id for r in ranked]
    assert ids == ["ml", "dl"]
    assert ranked[0].role == "bridge" and ranked[1].role == "primary"
    recs = [{**p, "teaches_skills": next(c for c in CHAIN if c.course_id == p["course_id"]).teaches_skills,
             "prerequisites": next(c for c in CHAIN if c.course_id == p["course_id"]).prerequisites}
            for p in _payload(ranked, user, job, gv, graph)]
    assert explanation_completeness_rate(recs, k=5) == 1.0
    assert prerequisite_violation_rate(recs, {"python"}, k=5, allow_chained_learning=True) == 0.0
    # the bridge explains why it is there; DL reports its prerequisite as met by the earlier course
    assert any("comes first" in step for step in recs[0]["explanation_path"])
    assert any("earlier recommended course ml" in step for step in recs[1]["explanation_path"])
    assert recs[0]["unlocks"][0]["course_id"] == "dl" and recs[0]["unlocks"][0]["in_list"] is True


def test_strict_modes_do_not_use_courses_whose_prerequisite_nobody_teaches() -> None:
    courses = [_course("x", "Orphan Advanced", ["Quantum Skill"], ["Unobtainium"])]
    _, graph, user, job, gv, cands = _setup(courses, ["Python"], ["Quantum Skill"])
    for mode in ("strict_drop", "strict_bridge"):
        out = CourseRanker(config=LEGACY_CONFIG.with_(prereq_mode=mode)).rank(cands, user, job, gv, graph=graph, top_k=5, catalog_courses=courses)
        assert out == []


def test_related_text_fill_is_flagged_feasible_and_explained() -> None:
    courses = [
        _course("sec3", "Application Security", ["Application Security"], []),
        _course("sec8", "Security Operations", ["Security Operations"], []),
        _course("sec9", "Advanced Forensics", ["Digital Forensics"], ["Security Operations"], "advanced"),
    ]
    catalog, graph, user, job, gv, cands = _setup(courses, ["Python"], ["Application Security"])
    cfg = DEFAULT_CONFIG
    ranked = CourseRanker(config=cfg).rank(cands, user, job, gv, graph=graph, top_k=5, catalog_courses=courses)
    assert ranked[0].course_id == "sec3"
    fill = [r for r in ranked if r.role == "related_text"]
    assert [r.course_id for r in fill] == ["sec8"]  # shares the content word 'security'; sec9 is blocked / unrelated
    payload = _payload(ranked, user, job, gv, graph)
    assert all(p["evidence"] and p["explanation_path"] and p["covered_missing_skills"] for p in payload)
    assert any("Text-level relatedness only" in lim for lim in payload[1]["limitations"])
    off = CourseRanker(config=cfg.with_(fill_related=False)).rank(cands, user, job, gv, graph=graph, top_k=5, catalog_courses=courses)
    assert [r.course_id for r in off] == ["sec3"]


def test_mmr_redundancy_penalises_already_covered_gap_mass() -> None:
    courses = [
        _course("a", "Docker Intro", ["Docker"], [], "beginner"),
        _course("b", "Docker Deep Dive", ["Docker"], [], "intermediate"),
        _course("c", "Kubernetes", ["Kubernetes"], [], "intermediate"),
    ]
    _, graph, user, job, gv, cands = _setup(courses, ["Python"], ["Docker", "Kubernetes"])
    ranked = CourseRanker(config=DEFAULT_CONFIG.with_(w_redundancy=0.6)).rank(cands, user, job, gv, graph=graph, top_k=3, catalog_courses=courses)
    ids = [r.course_id for r in ranked]
    assert ids.index("c") < max(ids.index("a"), ids.index("b"))  # the unrelated gap is covered before the second Docker course
    assert ranked[-1].score_breakdown["redundancy_penalty"] > 0.0


def test_split_is_fixed_disjoint_and_matches_the_recorded_file() -> None:
    ids = [f"sc-{i:02d}" for i in range(1, 49)]
    dev, test = split_ids(ids)
    assert len(dev) == 24 and len(test) == 24 and not set(dev) & set(test) and set(dev) | set(test) == set(ids)
    assert split_ids(ids) == (dev, test)
    if SPLIT_PATH.exists():
        rec = json.loads(SPLIT_PATH.read_text())
        assert rec["dev"] == dev and rec["test"] == test


def test_recommender_code_has_no_label_inputs() -> None:
    src_dir = ROOT / "src/skillllm_v2/recommend"
    for name in ("course_ranker.py", "course_retriever.py", "explanation_generator.py", "gap_vector.py", "learning_path_planner.py", "config.py"):
        text = (src_dir / name).read_text(encoding="utf-8")
        assert "relevance_labels" not in text and "benchmark_scenarios" not in text and '["gold"]' not in text, name
    pipe = (ROOT / "src/skillllm_v2/pipeline/course_recommendation_pipeline.py").read_text(encoding="utf-8")
    assert "relevance_labels" not in pipe and '["gold"]' not in pipe
