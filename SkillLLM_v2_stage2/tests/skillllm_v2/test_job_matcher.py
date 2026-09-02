from skillllm_v2.match import JobMatcher


def test_job_matcher() -> None:
    result = JobMatcher().score({"skills": ["Python"]}, {"job_id": "j1", "required_skills": ["Python", "AWS"], "preferred_skills": []})
    assert result["missing_skills"] == ["AWS"]
