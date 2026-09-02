from skillllm_v2.context.profile_summarizer import summarize_profile


def test_profile_summarizer_preserves_evidence_and_offsets() -> None:
    text = "I use Python and communication in analytics projects."
    summary = summarize_profile(
        text,
        extracted_skills=[
            {
                "surface_form": "Python",
                "label": "HARD_SKILL",
                "start": 6,
                "end": 12,
                "source": "mistral",
                "confidence": 0.91,
                "evidence": "Python",
            }
        ],
        timestamp="2026-07-09T00:00:00+00:00",
    )
    assert summary["skills"][0]["surface_form"] == "Python"
    assert summary["skills"][0]["metadata"]["source_offsets"] == [[6, 12]]
    assert summary["skills"][0]["evidence"][0]["source"] == "mistral"
