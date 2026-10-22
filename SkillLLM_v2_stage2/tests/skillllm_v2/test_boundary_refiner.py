from skillllm_v2.schemas import SkillCandidate
from skillllm_v2.verify import BoundaryRefiner


def test_boundary_refiner_replaces_with_taxonomy_evidence() -> None:
    """Edits are applied only when the trimmed span has alias/taxonomy evidence."""
    text = "knowledge of AWS and strong communication skills"
    aws = SkillCandidate("knowledge of AWS", "HARD_SKILL", 0, 16, "mistral", 0.9, "knowledge of AWS").to_dict()
    comm_start = text.index("strong")
    comm = SkillCandidate("strong communication skills", "SOFT_SKILL", comm_start, len(text), "mistral", 0.9, "strong communication skills").to_dict()
    refined = BoundaryRefiner().refine(text, [aws, comm])
    assert [item["surface_form"] for item in refined] == ["AWS", "communication"]
    for item in refined:
        edit = item["metadata"]["boundary_edit"]
        assert edit["source"] == "boundary_refiner"
        assert edit["edit_type"] != "original"


def test_boundary_refiner_preserves_span_without_taxonomy_evidence() -> None:
    """A span with no leading/trailing filler match generates only an 'original' variant
    and is preserved unchanged (non-destructive), regardless of alias-table contents."""
    text = "I have a solid background in distributed systems design."
    start = text.index("background in distributed systems design")
    end = start + len("background in distributed systems design")
    candidate = SkillCandidate("background in distributed systems design", "HARD_SKILL", start, end, "mistral", 0.85, "background in distributed systems design").to_dict()
    refined = BoundaryRefiner().refine(text, [candidate])
    assert len(refined) == 1
    assert refined[0]["surface_form"] == "background in distributed systems design"
    assert refined[0]["start"] == start and refined[0]["end"] == end
    assert refined[0]["metadata"]["boundary_edit"]["edit_type"] == "original"


def test_boundary_refiner_never_drops_a_candidate() -> None:
    """The refiner must always return at least one span per input candidate."""
    text = "Strong leadership and excellent communication."
    candidate = SkillCandidate("Strong leadership", "SOFT_SKILL", 0, 18, "mistral", 0.9, "Strong leadership").to_dict()
    refined = BoundaryRefiner().refine(text, [candidate])
    assert len(refined) >= 1


def test_boundary_refiner_realigns_subword_to_word_boundary() -> None:
    """A subword-misaligned span (e.g. 'java' inside 'javascript') is realigned to its standalone word."""
    text = "javascript reactjs java"
    # Misaligned java at index 0..4 (the prefix of javascript)
    candidate = SkillCandidate("java", "HARD_SKILL", 0, 4, "mistral", 0.86, "java").to_dict()
    refined = BoundaryRefiner().refine(text, [candidate])
    assert len(refined) == 1
    assert refined[0]["surface_form"] == "java"
    assert refined[0]["start"] == 19
    assert refined[0]["end"] == 23
    assert refined[0]["metadata"]["boundary_edit"]["edit_type"] == "realign_word_boundary"
