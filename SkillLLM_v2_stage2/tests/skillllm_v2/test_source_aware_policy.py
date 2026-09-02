from skillllm_v2.source_aware_policy import (
    ANCHOR_SOURCES,
    candidate_sources,
    is_anchor_candidate,
    source_key,
)


def test_candidate_sources_falls_back_to_source_field() -> None:
    candidate = {"source": "gliner", "metadata": {}}
    assert candidate_sources(candidate) == {"gliner"}


def test_candidate_sources_uses_metadata_sources_when_present() -> None:
    candidate = {"source": "mistral", "metadata": {"sources": ["dictionary", "mistral"]}}
    assert candidate_sources(candidate) == {"dictionary", "mistral"}


def test_is_anchor_candidate_true_for_mistral_source() -> None:
    candidate = {"source": "mistral", "metadata": {}}
    assert is_anchor_candidate(candidate, ANCHOR_SOURCES) is True


def test_is_anchor_candidate_true_when_corroborated_by_mistral() -> None:
    candidate = {"source": "dictionary", "metadata": {"sources": ["dictionary", "mistral"]}}
    assert is_anchor_candidate(candidate) is True


def test_is_anchor_candidate_false_for_gliner_only() -> None:
    candidate = {"source": "gliner", "metadata": {}}
    assert is_anchor_candidate(candidate) is False


def test_source_key_is_sorted_and_order_independent() -> None:
    a = {"source": "gliner", "metadata": {"sources": ["gliner", "dictionary"]}}
    b = {"source": "dictionary", "metadata": {"sources": ["dictionary", "gliner"]}}
    assert source_key(a) == source_key(b) == "dictionary+gliner"


def test_source_key_single_source() -> None:
    candidate = {"source": "mistral", "metadata": {}}
    assert source_key(candidate) == "mistral"
