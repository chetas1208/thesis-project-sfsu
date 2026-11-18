"""Tests for skillllm_v2.evals.research.compaction (generator truth integrity + metric helpers)."""

import pytest

from skillllm_v2.evals.research import compaction as C


@pytest.fixture(scope="module")
def histories():
    return C.generate_all(6, 7)


def test_generator_is_deterministic(histories):
    again = C.generate_all(6, 7)
    assert [h["text"] for h in histories] == [h["text"] for h in again]
    assert [h["truth"] for h in histories] == [h["truth"] for h in again]


def test_extraction_offsets_point_at_surfaces(histories):
    for h in histories:
        for e in h["extractions"]:
            assert h["text"][e["start"]:e["end"]] == e["surface_form"]


def test_truth_is_consistent(histories):
    for h in histories:
        t = h["truth"]
        held = set(t["held"])
        assert held, "every history holds at least one skill"
        for s in held:
            assert t["evidence"].get(s), f"held skill {s} has no supporting sentence"
            assert all(sent in h["text"] for sent in t["evidence"][s])
        assert not held & set(t["retracted"])
        assert t["goal"] == t["goal_history"][-1]
        assert set(t["gaps_required"]).isdisjoint(held) and set(t["gaps_preferred"]).isdisjoint(held)
        for c in t["contradictions"]:
            assert c["s1"] in h["text"] and c["s2"] in h["text"]
        lo, hi = C.LENGTHS[h["length_class"]]
        assert len(h["lines"]) >= lo


def test_confidence_arms_share_text_but_not_confidences():
    a, b = C.generate_all(3, 7, "informative"), C.generate_all(3, 7, "uninformative")
    assert [h["text"] for h in a] == [h["text"] for h in b]
    ca = [e["confidence"] for h in a for e in h["extractions"]]
    cb = [e["confidence"] for h in b for e in h["extractions"]]
    assert ca != cb
    # informative arm: non-held mentions are below held ones by construction
    for h in a:
        for e in h["extractions"]:
            if e["_kind"] in ("distractor", "aspiration"):
                assert e["confidence"] <= 0.65


def test_metric_helpers():
    assert C.jaccard(set(), set()) == 1.0
    assert C.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert C.top_overlap(["a", "b"], ["a", "b"]) == 1.0
    assert C.top_overlap([], []) == 1.0
    assert C.top_overlap(["a", "b", "c", "d", "e"], ["a", "x", "y", "z", "w"]) == pytest.approx(0.2)
    s = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert C.kendall(s, s) == 1.0
    assert C.kendall(s, {"a": 1.0, "b": 2.0, "c": 3.0}) == pytest.approx(-1.0)
    assert C.rank_ids({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]


def test_present_requires_long_enough_substring():
    sent = "2014: built an internal dashboard with Tableau at Cedar Logistics."
    rep_raw = {"evidence": ["intro\n" + sent + "\nmore"]}
    rep_snip = {"evidence": ["internal dashboard with Tableau at Cedar"]}
    rep_short = {"evidence": ["Tableau"]}
    assert C._present(sent, rep_raw) and C._present(sent, rep_snip)
    assert not C._present(sent, rep_short)


def test_mistral_token_count_is_not_truncated():
    if not C.TOKENIZER.exists():
        pytest.skip("anchor tokenizer not present")
    ntok = C.MistralTokens()
    assert ntok("word " * 2000) > 1500  # tokenizer.json ships with truncation at 512; the counter must disable it
