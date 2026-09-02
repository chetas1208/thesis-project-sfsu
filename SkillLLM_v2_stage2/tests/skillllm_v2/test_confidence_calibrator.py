from skillllm_v2.schemas import SkillCandidate
from skillllm_v2.verify import ConfidenceCalibrator


def test_calibrator_dictionary_threshold() -> None:
    candidate = SkillCandidate("Python", "HARD_SKILL", 0, 6, "dictionary", 0.5, "Python").to_dict()
    assert not ConfidenceCalibrator().accept(candidate)
