from __future__ import annotations

from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.graph.schema import EdgeType, NodeType


def test_course_graph_construction_and_prerequisites() -> None:
    graph = CourseGraphBuilder()

    # Ingest user
    user_id = graph.ingest_user_profile({
        "user_id": "u1",
        "skills": ["Python", "SQL"],
    })
    assert user_id == "u1"
    user_skills = graph.get_user_skills("u1")
    assert len(user_skills) == 2
    user_sids = {s["canonical_skill_id"] for s in user_skills}

    # Ingest job
    job_id = graph.ingest_job({
        "job_id": "ml-engineer",
        "title": "Machine Learning Engineer",
        "required_skills": [
            {"skill_name": "Python", "importance": 1.0},
            {"skill_name": "PyTorch", "importance": 1.0},
        ],
        "preferred_skills": [
            {"skill_name": "Docker", "importance": 0.8},
        ],
    })
    assert job_id == "ml-engineer"
    reqs = graph.get_job_requirements("ml-engineer")
    assert len(reqs) == 3

    # Ingest course
    graph.ingest_course({
        "course_id": "dl-101",
        "title": "Deep Learning with PyTorch",
        "teaches_skills": ["PyTorch"],
        "prerequisites": ["Python"],
        "level": "intermediate",
    })

    # Evaluate prerequisites
    prereq_eval = graph.evaluate_prerequisites("dl-101", user_sids)
    assert prereq_eval["is_satisfied"] is True
    assert prereq_eval["compatibility_score"] == 1.0
    assert prereq_eval["violation_count"] == 0

    # Test course with missing prerequisite
    graph.ingest_course({
        "course_id": "adv-cuda",
        "title": "CUDA Optimization",
        "teaches_skills": ["CUDA"],
        "prerequisites": ["C++", "Hardware Architecture"],
    })
    prereq_eval2 = graph.evaluate_prerequisites("adv-cuda", user_sids)
    assert prereq_eval2["is_satisfied"] is False
    assert prereq_eval2["violation_count"] == 2

    # Trace explanation path
    trace = graph.trace_explanation_path(
        user_id="u1",
        job_id="ml-engineer",
        course_id="dl-101",
        target_skill_id="pytorch",
    )
    assert "narrative_path" in trace
    assert "Target role" in trace["narrative_path"]
    assert "teaches" in trace["narrative_path"]
