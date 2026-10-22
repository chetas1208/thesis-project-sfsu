"""Regression tests for generate_recommendation_explanation (docs/BUGFIX_LOG.md, bug 12)."""

from __future__ import annotations

from skillllm_v2.match.explanation_generator import generate_recommendation_explanation

PROFILE = {
    "skills": [
        {"surface_form": "Python", "canonical_skill_name": "Python", "canonical_skill_id": "python", "confidence": 0.9,
         "evidence": [{"text": "Built ETL pipelines in Python", "confidence": 0.9}, {"text": "Python scripting", "confidence": 0.6}]},
        {"surface_form": "SQL", "canonical_skill_name": "Structured Query Language", "canonical_skill_id": "sql", "confidence": 0.4,
         "evidence": [{"text": "Wrote SQL reports", "confidence": 0.4}]},
        {"surface_form": "Excel", "canonical_skill_name": "Microsoft Excel", "canonical_skill_id": "excel", "confidence": 0.8,
         "evidence": [{"text": "Tracked budgets in Excel", "confidence": 0.8}]},
        {"surface_form": "Hiking", "evidence": [{"text": "Enjoys hiking", "confidence": 0.9}]},
    ]
}
JOB_MATCH = {"job_id": "j1", "match_score": 0.62, "matched_skills": ["Python", "SQL"], "missing_skills": ["Docker", "Kubernetes"]}
GAPS = [{"skill": "Docker", "gap_type": "missing_critical"}, {"skill": "Kubernetes", "gap_type": "missing_critical"}]
COURSES = [{"course_id": "c1", "course_name": "Docker Basics", "teaches_skills": ["Docker"]}]


def _explain(**kw):
    return generate_recommendation_explanation(PROFILE, JOB_MATCH, GAPS, COURSES, **kw)


def test_evidence_only_concerns_matched_and_missing_skills():
    out = _explain()
    assert out["evidence"] == ["Python: Built ETL pipelines in Python", "SQL: Wrote SQL reports"]
    assert not any(line.startswith(("Excel", "Hiking")) for line in out["evidence"])  # unrelated profile skills are not dumped


def test_old_one_line_per_profile_skill_dump_is_gone_but_keys_are_backward_compatible():
    out = _explain()
    for key in ("matched_skills", "missing_skills", "gap_types", "recommended_learning_path", "evidence", "confidence", "limitations"):
        assert key in out
    assert out["matched_skills"] == ["Python", "SQL"] and out["missing_skills"] == ["Docker", "Kubernetes"]
    assert out["gap_types"] == ["missing_critical", "missing_critical"] and out["recommended_learning_path"] == ["Docker Basics"]
    assert out["confidence"] == 0.62


def test_missing_skills_are_reported_honestly_as_having_no_evidence():
    out = _explain()
    assert out["missing_skill_evidence"] == [
        {"skill": "Docker", "evidence": [], "note": "no evidence in profile"},
        {"skill": "Kubernetes", "evidence": [], "note": "no evidence in profile"},
    ]
    assert not any("Docker" in line or "Kubernetes" in line for line in out["evidence"])  # nothing is invented for them


def test_missing_skill_with_real_profile_evidence_gets_it_without_the_no_evidence_note():
    profile = {"skills": PROFILE["skills"] + [{"surface_form": "Docker", "evidence": [{"text": "Heard of Docker at a meetup", "confidence": 0.2}]}]}
    out = generate_recommendation_explanation(profile, JOB_MATCH, GAPS, COURSES)
    entry = next(m for m in out["missing_skill_evidence"] if m["skill"] == "Docker")
    assert entry["evidence"] == ["Docker: Heard of Docker at a meetup"] and entry["note"] == ""
    assert "Docker: Heard of Docker at a meetup" in out["evidence"]


def test_evidence_lines_are_always_real_profile_evidence():
    real = {f"{s['surface_form']}: {e['text']}" for s in PROFILE["skills"] for e in s["evidence"]}
    assert set(_explain(evidence_per_skill=5)["evidence"]) <= real
    assert _explain(evidence_per_skill=5)["evidence"][:2] == ["Python: Built ETL pipelines in Python", "Python: Python scripting"]


def test_matching_works_through_canonical_name_and_id():
    job_match = {"matched_skills": ["Structured Query Language", "python"], "missing_skills": [], "match_score": 1.0}
    out = generate_recommendation_explanation(PROFILE, job_match, [], [])
    assert "SQL: Wrote SQL reports" in out["evidence"] and "Python: Built ETL pipelines in Python" in out["evidence"]


def test_limitations_are_specific_and_checkable():
    lims = _explain()["limitations"]
    text = " ".join(lims)
    assert "Recommendation quality depends on available job/course labels" not in text and "Extraction fallback" not in text  # boilerplate gone
    assert any("no evidence in profile for: Docker, Kubernetes" in s for s in lims)
    assert any("No recommended course covers: Kubernetes" in s for s in lims)  # Docker is covered by the supplied course, Kubernetes is not
    assert "Docker." not in " ".join(s for s in lims if "No recommended course covers" in s)
    assert any("low-confidence" in s and "SQL" in s and "Python" not in s.split("(")[-1] for s in lims)  # SQL evidence 0.4 < 0.5, Python 0.9


def test_limitations_empty_when_nothing_is_wrong():
    job_match = {"matched_skills": ["Python"], "missing_skills": [], "match_score": 0.9}
    assert generate_recommendation_explanation(PROFILE, job_match, [], [])["limitations"] == []


def test_matched_skill_without_evidence_text_is_listed_not_invented():
    profile = {"skills": [{"surface_form": "Python", "evidence": []}]}
    out = generate_recommendation_explanation(profile, {"matched_skills": ["Python"], "missing_skills": [], "match_score": 0.7}, [], [])
    assert out["evidence"] == [] and out["matched_without_evidence"] == ["Python"]
    assert any("no evidence text in the profile: Python" in s for s in out["limitations"])


def test_no_job_match_is_stated_and_inputs_are_not_mutated():
    import copy

    before = copy.deepcopy(PROFILE)
    out = generate_recommendation_explanation(PROFILE)
    assert out["confidence"] == 0.0 and out["evidence"] == [] and out["matched_skills"] == []
    assert any("No job match result" in s for s in out["limitations"])
    _explain()
    assert PROFILE == before


def test_no_courses_supplied_is_stated():
    out = generate_recommendation_explanation(PROFILE, JOB_MATCH, GAPS, [])
    assert any("No course recommendations were supplied for 2 missing" in s for s in out["limitations"])


def test_minimal_pipeline_style_profile_without_confidence_fields():
    profile = {"skills": [{"surface_form": "Python", "evidence": [{"text": "Python"}]}]}
    out = generate_recommendation_explanation(profile, {"matched_skills": ["Python"], "missing_skills": [], "match_score": 0.5}, [], [])
    assert out["evidence"] == ["Python: Python"]
