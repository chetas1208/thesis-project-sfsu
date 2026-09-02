from skillllm_v2.schemas import SkillCandidate, SystemBenchmarkResult, validate_skill_candidate, validate_system_result


def test_skill_candidate_schema() -> None:
    candidate = SkillCandidate("Python", "HARD_SKILL", 0, 6, "mistral", 0.9, "Python").to_dict()
    validate_skill_candidate(candidate)


def test_system_result_schema() -> None:
    result = SystemBenchmarkResult(system_name="BERT baseline", exact_span_f1=None).to_dict()
    validate_system_result(result)
