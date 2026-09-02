from skillllm_v2.schemas import SkillCandidate
from skillllm_v2.verify import SpanVerifier


def test_span_verifier_rejects_generic() -> None:
    candidate = SkillCandidate("experience", "HARD_SKILL", 0, 10, "dictionary", 0.9, "experience").to_dict()
    assert not SpanVerifier().verify(candidate)["is_valid_skill"]
