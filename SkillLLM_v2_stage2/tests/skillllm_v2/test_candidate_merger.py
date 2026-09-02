from skillllm_v2.extract import CandidateMerger
from skillllm_v2.schemas import SkillCandidate


def test_candidate_merger_sources() -> None:
    candidates = [
        SkillCandidate("AWS", "HARD_SKILL", 0, 3, "mistral", 0.8, "AWS").to_dict(),
        SkillCandidate("Amazon Web Services", "HARD_SKILL", 10, 29, "dictionary", 0.8, "Amazon Web Services").to_dict(),
    ]
    merged = CandidateMerger().merge(candidates)
    assert len(merged) == 1
    assert set(merged[0]["metadata"]["sources"]) == {"mistral", "dictionary"}
