from skillllm_v2.context.context_compactor import compact_context


def test_context_compactor_separates_cag_rag_and_graph_policy() -> None:
    packet = compact_context(
        raw_profile_text="Python developer targeting data engineer jobs.",
        extracted_skills=[
            {
                "surface_form": "Python",
                "label": "HARD_SKILL",
                "start": 0,
                "end": 6,
                "source": "mistral",
                "confidence": 0.9,
                "evidence": "Python",
            }
        ],
        job_target="data engineer",
    )
    assert "stable_cag_context" in packet
    assert "job_postings" in packet["dynamic_rag_policy"]["rag_sources"]
    assert packet["graph_retrieval_policy"]["paths"]
