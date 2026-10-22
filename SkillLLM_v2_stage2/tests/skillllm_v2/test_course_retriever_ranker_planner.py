from __future__ import annotations

from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.recommend.course_ranker import CourseRanker
from skillllm_v2.recommend.course_retriever import CourseRetriever
from skillllm_v2.recommend.explanation_generator import CourseExplanationGenerator
from skillllm_v2.recommend.gap_vector import GapEngine
from skillllm_v2.recommend.learning_path_planner import LearningPathPlanner


def test_retriever_ranker_planner_end_to_end() -> None:
    courses = [
        CourseRecord(
            course_id="c_python",
            title="Python Foundations",
            learning_outcomes=["Write python scripts"],
            teaches_skills=["Python"],
            prerequisites=[],
            level="beginner",
            duration_hours=15.0,
        ),
        CourseRecord(
            course_id="c_ml",
            title="Machine Learning with Scikit-Learn",
            learning_outcomes=["Train ML models"],
            teaches_skills=["Machine Learning", "Scikit-Learn"],
            prerequisites=["Python"],
            level="intermediate",
            duration_hours=25.0,
        ),
        CourseRecord(
            course_id="c_dl",
            title="Deep Learning with PyTorch",
            learning_outcomes=["Train neural networks with PyTorch"],
            teaches_skills=["Deep Learning", "PyTorch"],
            prerequisites=["Machine Learning"],
            level="advanced",
            duration_hours=35.0,
        ),
    ]

    catalog = CourseCatalog(courses)
    graph = CourseGraphBuilder()
    graph.ingest_catalog(catalog)

    user_profile = {
        "user_id": "alice",
        "skills": ["Python"],
        "experience_level": "intermediate",
    }
    target_job = {
        "job_id": "ai-researcher",
        "title": "AI Researcher",
        "required_skills": [
            {"skill_name": "Machine Learning", "importance": 1.0},
            {"skill_name": "PyTorch", "importance": 1.0},
        ],
        "preferred_skills": [],
    }

    # 1. Gap Vector
    gap_engine = GapEngine()
    gap_vector = gap_engine.compute(user_profile, target_job, graph=graph)
    assert "machine learning" in gap_vector.vector
    assert "pytorch" in gap_vector.vector

    # 2. Retriever
    retriever = CourseRetriever(catalog=catalog, graph=graph)
    candidates = retriever.retrieve(gap_vector, target_job, top_k=5)
    assert len(candidates) >= 2
    c_ids = [c.course_id for c in candidates]
    assert "c_ml" in c_ids or "c_dl" in c_ids

    # 3. Ranker
    ranker = CourseRanker()
    ranked = ranker.rank(
        candidates=[c.course for c in candidates],
        user_profile=user_profile,
        target_job=target_job,
        gap_vector=gap_vector,
        graph=graph,
        top_k=5,
    )
    assert len(ranked) >= 2
    # c_ml should have satisfied prerequisites (user has Python), whereas c_dl requires ML
    ml_rec = next(r for r in ranked if r.course_id == "c_ml")
    assert ml_rec.prerequisite_status["is_satisfied"] is True

    # 4. Learning Path Planner
    planner = LearningPathPlanner()
    path = planner.plan(
        recommended_courses=ranked,
        user_profile=user_profile,
        target_job=target_job,
        gap_vector=gap_vector,
        graph=graph,
    )
    assert path.total_courses >= 2
    ordered_ids = [c["course_id"] for c in path.ordered_courses]
    # In learning path, c_ml must precede c_dl because c_dl requires Machine Learning
    if "c_ml" in ordered_ids and "c_dl" in ordered_ids:
        assert ordered_ids.index("c_ml") < ordered_ids.index("c_dl")

    # 5. Explanations
    explainer = CourseExplanationGenerator()
    explanations = explainer.generate_recommendations_payload(
        ranked_courses=ranked,
        user_profile=user_profile,
        target_job=target_job,
        gap_vector=gap_vector,
        graph=graph,
    )
    assert len(explanations) == len(ranked)
    for exp in explanations:
        assert "course_id" in exp
        assert "title" in exp
        assert "rank" in exp
        assert "score" in exp
        assert "covered_missing_skills" in exp
        assert "prerequisite_status" in exp
        assert "evidence" in exp
        assert "explanation_path" in exp
        assert "limitations" in exp
        assert "next_profile_update_action" in exp
