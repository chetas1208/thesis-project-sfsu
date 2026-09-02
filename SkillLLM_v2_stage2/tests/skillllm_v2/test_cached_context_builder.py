from skillllm_v2.context.cached_context_builder import build_cached_context_packet


def test_cached_context_builder_has_stable_hash() -> None:
    skill = {
        "surface_form": "AWS",
        "label": "HARD_SKILL",
        "start": 10,
        "end": 13,
        "source": "mistral",
        "confidence": 0.8,
        "evidence": "AWS",
    }
    first = build_cached_context_packet("Experience with AWS.", [skill])
    second = build_cached_context_packet("Experience with AWS.", [skill])
    assert first["packet_type"] == "skillllm_v2_cached_context"
    assert first["content_hash"] == second["content_hash"]
    assert first["cache_policy"]["rag"] == ["job_postings", "scraped_descriptions"]
