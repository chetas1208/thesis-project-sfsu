from skillllm_v2.match import GapDetector


def test_gap_detector() -> None:
    gaps = GapDetector().detect({"skills": ["Python"]}, {"required_skills": ["Python", "AWS"], "preferred_skills": ["SQL"]})
    assert {gap["gap_type"] for gap in gaps} == {"missing_critical", "missing_preferred"}
