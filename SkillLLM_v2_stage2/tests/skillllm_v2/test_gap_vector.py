from __future__ import annotations

from skillllm_v2.recommend.gap_vector import GapEngine


def test_gap_vector_computation() -> None:
    engine = GapEngine(weak_threshold=0.6)

    profile = {
        "user_id": "u_test",
        "skills": [
            {"surface_form": "Python", "canonical_skill_id": "python", "confidence": 1.0, "mastery_level": "proficient"},
            {"surface_form": "SQL", "canonical_skill_id": "sql", "confidence": 0.4, "mastery_level": "beginner"},
        ],
        "transferable_skills": {
            "gcp": "Amazon Web Services",
        },
    }

    job = {
        "job_id": "data-eng",
        "required_skills": [
            {"skill_name": "Python", "importance": 1.0},
            {"skill_name": "SQL", "importance": 1.0},
            {"skill_name": "Apache Spark", "importance": 1.0},
        ],
        "preferred_skills": [
            {"skill_name": "Docker", "importance": 0.8},
            {"skill_name": "GCP", "importance": 0.7},
        ],
    }

    gv = engine.compute(profile, job)
    assert gv.user_id == "u_test"
    assert gv.job_id == "data-eng"

    # Spark is missing critical
    gap_types = {g.canonical_skill_id: g.gap_type for g in gv.gaps}
    assert gap_types.get("apache spark") == "missing_critical"
    # Docker is missing preferred
    assert gap_types.get("docker") == "missing_preferred"
    # SQL is weak skill (confidence 0.4 * beginner mastery 0.35 < 0.6)
    assert gap_types.get("sql") == "weak_skill"
    # GCP is covered by transferable skill
    assert gap_types.get("gcp") == "covered_by_transferable_skill"

    # Python should not be in gaps
    assert "python" not in gv.vector

    assert gv.total_gap_mass > 0.0
    assert gv.critical_count >= 1
    assert gv.preferred_count >= 1
    assert gv.weak_count >= 1
