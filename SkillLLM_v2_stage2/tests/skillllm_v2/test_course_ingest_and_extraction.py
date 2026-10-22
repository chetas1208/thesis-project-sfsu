from __future__ import annotations

import tempfile
from pathlib import Path

from skillllm_v2.course.extract_course_skills import CourseSkillExtractor
from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.course.prerequisite_parser import PrerequisiteParser


def test_course_catalog_ingest_and_serialization() -> None:
    data = [
        {
            "course_id": "test-101",
            "title": "Test Course",
            "description": "A comprehensive test course.",
            "learning_outcomes": ["Learn Python", "Build models"],
            "teaches_skills": ["Python", "Machine Learning"],
            "prerequisites": ["Linear Algebra"],
            "level": "intermediate",
            "duration_hours": 30.0,
            "provider": "TestProvider",
            "rating": 4.8,
            "cost_usd": 0.0,
        }
    ]
    catalog = CourseCatalog(data)
    assert len(catalog) == 1
    rec = catalog.get_course("test-101")
    assert rec is not None
    assert rec.title == "Test Course"
    assert rec.duration_hours == 30.0

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        catalog.save_to_jsonl(tmp_path)
        loaded = CourseCatalog.load_from_file(tmp_path)
        assert len(loaded) == 1
        assert loaded.get_course("test-101").title == "Test Course"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_prerequisite_parser_various_formats() -> None:
    parser = PrerequisiteParser()

    # List of skills
    res1 = parser.parse(["Python", "Basic Linear Algebra"])
    assert len(res1) == 2
    assert res1[0].skill_name == "Python"
    assert res1[0].strictness == "strict"
    assert res1[1].level == "beginner"

    # None / empty patterns
    res_none = parser.parse("Prerequisites: None")
    assert len(res_none) == 0

    # Unstructured recommended background
    res2 = parser.parse("Recommended: prior experience with Docker and Linux commands.")
    assert len(res2) >= 1
    assert any("docker" in r.skill_name.lower() for r in res2)
    assert all(r.strictness == "recommended" for r in res2)


def test_course_skill_extractor() -> None:
    extractor = CourseSkillExtractor()
    course = {
        "course_id": "c-py",
        "title": "Python Programming Mastery",
        "learning_outcomes": ["Write clean and idiomatic Python scripts", "Build REST APIs with SQL databases"],
        "teaches_skills": ["Python"],
        "prerequisites": ["None"],
        "level": "beginner",
    }
    enriched = extractor.extract_from_course(course)
    assert "taught_skills_evidence" in enriched
    assert len(enriched["canonical_skills"]) >= 1
    assert "python" in enriched["canonical_skills"]
    assert enriched["canonical_prerequisites"] == []
