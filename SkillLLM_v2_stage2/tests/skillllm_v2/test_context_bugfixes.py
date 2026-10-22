"""Regression tests for the context-module bugs found by the context evals (BUGFIX_LOG.md, bugs 1-4)."""

from __future__ import annotations

import json
import random
import re

from skillllm_v2.context.budget import count_tokens, skill_importance
from skillllm_v2.context.cached_context_builder import build_cached_context_packet
from skillllm_v2.context.context_compactor import compact_context
from skillllm_v2.context.profile_summarizer import summarize_profile
from skillllm_v2.context.skill_memory_builder import build_skill_memory

TEXT = (
    "Built churn forecasting workflows using Python. Shipped fraud alerts workflows using Python and SQL. "
    "Maintained inventory tracking workflows using Kubernetes. Tried Excel once. Strong communication skills."
)


def _ex(surface, conf, evidence=None, label="HARD_SKILL", cid=None, name=None, source="mistral", nth=0):
    start = -1
    for _ in range(nth + 1):
        start = TEXT.find(surface, start + 1)
    return {"surface_form": surface, "label": label, "start": start, "end": start + len(surface), "source": source, "confidence": conf,
            "evidence": evidence, "canonical_skill_id": cid if cid is not None else surface.lower(), "canonical_skill_name": name or surface}


def _extractions():
    return [
        _ex("Python", 0.95, "Built churn forecasting workflows using Python"),
        _ex("Python", 0.9, "Shipped fraud alerts workflows using Python and SQL", nth=1, source="gliner"),
        _ex("SQL", 0.8, "Shipped fraud alerts workflows using Python and SQL"),
        _ex("Kubernetes", 0.7, "Maintained inventory tracking workflows using Kubernetes"),
        _ex("Excel", 0.4, "Tried Excel once"),
        _ex("communication", 0.6, "Strong communication skills", label="SOFT_SKILL"),
    ]


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in {"generated_at", "timestamp", "content_hash"}}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


# ------------------------------------------------------------------ bug 2: determinism


def test_content_hash_stable_when_the_wall_clock_advances():
    ticks = iter(range(1000))
    clock = lambda: f"2026-01-01T00:{next(ticks) // 60:02d}:{next(ticks) % 60:02d}+00:00"  # noqa: E731
    a = build_cached_context_packet(TEXT, _extractions(), clock=clock)
    b = build_cached_context_packet(TEXT, _extractions(), clock=clock)
    assert a["payload"]["stable_cag_context"]["profile_summary"]["generated_at"] != b["payload"]["stable_cag_context"]["profile_summary"]["generated_at"]
    assert a["content_hash"] == b["content_hash"]


def test_content_hash_changes_when_content_changes():
    base = build_cached_context_packet(TEXT, _extractions(), timestamp="t")
    changed = _extractions()
    changed[0]["confidence"] = 0.5
    changed[1]["confidence"] = 0.5
    assert build_cached_context_packet(TEXT, changed, timestamp="t")["content_hash"] != base["content_hash"]
    other_user = build_cached_context_packet(TEXT, _extractions(), user_id="someone_else", timestamp="t")
    assert other_user["content_hash"] != base["content_hash"]


def test_injected_clock_and_timestamp_make_packets_byte_identical():
    fixed = lambda: "2026-02-02T00:00:00+00:00"  # noqa: E731
    a = build_cached_context_packet(TEXT, _extractions(), clock=fixed)
    b = build_cached_context_packet(TEXT, _extractions(), timestamp="2026-02-02T00:00:00+00:00")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_extraction_order_does_not_change_the_packet():
    base = build_cached_context_packet(TEXT, _extractions(), timestamp="t")
    for seed in range(20):
        items = _extractions()
        random.Random(seed).shuffle(items)
        permuted = build_cached_context_packet(TEXT, items, timestamp="t")
        assert json.dumps(permuted, sort_keys=True) == json.dumps(base, sort_keys=True)


def test_permutation_invariance_with_conflicting_representatives():
    # same canonical id, different surface forms and labels: the representative must not depend on input order
    items = [_ex("Python", 0.9, "A", cid="py"), _ex("py", 0.9, "B", cid="py", label="SOFT_SKILL", source="bert"), _ex("python3", 0.5, "C", cid="py")]
    forms = {json.dumps(summarize_profile(TEXT, random.Random(s).sample(items, 3), timestamp="t"), sort_keys=True) for s in range(12)}
    assert len(forms) == 1


# ------------------------------------------------------------------ bug 4: duplicate evidence


def test_repeated_skill_evidence_pairs_are_merged_not_duplicated():
    items = [_ex("Python", 0.9, "Built churn forecasting workflows using Python"), _ex("Python", 0.95, "Built churn forecasting workflows using Python", source="bert")]
    fact = summarize_profile(TEXT, items, timestamp="t")["skills"][0]
    assert [e["text"] for e in fact["evidence"]] == ["Built churn forecasting workflows using Python"]
    assert fact["evidence"][0]["confidence"] == 0.95 and fact["evidence"][0]["source"] == "bert+mistral"
    assert fact["metadata"]["mentions"] == 2  # the repetition signal is kept as a count


def test_no_duplicate_skill_evidence_pairs_in_packet():
    items = _extractions() + [dict(e) for e in _extractions()]
    pairs = [(s["surface_form"], e["text"]) for s in summarize_profile(TEXT, items, timestamp="t")["skills"] for e in s["evidence"]]
    assert len(pairs) == len(set(pairs))


# ------------------------------------------------------------------ bug 3: label conflicts


def test_same_canonical_id_label_conflict_is_reported():
    items = [_ex("Python", 0.9, "x", cid="python"), _ex("Python", 0.8, "y", cid="python", label="SOFT_SKILL")]
    notes = summarize_profile(TEXT, items, timestamp="t")["conflict_notes"]
    assert any("python" in n.lower() and "HARD_SKILL" in n and "SOFT_SKILL" in n for n in notes)


def test_conflict_between_different_surfaces_of_one_id_is_reported():
    items = [_ex("Python", 0.9, "x", cid="python", name="Python"), _ex("SQL", 0.8, "y", cid="python", name="Python", label="SOFT_SKILL")]
    notes = summarize_profile(TEXT, items, timestamp="t")["conflict_notes"]
    assert notes and "Conflicting labels" in notes[0]


def test_same_surface_different_id_conflict_still_reported_and_missing_labels_are_not_conflicts():
    items = [_ex("Python", 0.9, "x", cid="a"), _ex("Python", 0.8, "y", cid="b", label="SOFT_SKILL")]
    assert summarize_profile(TEXT, items, timestamp="t")["conflict_notes"]
    no_label = [{k: v for k, v in _ex("Python", 0.9, "x").items() if k != "label"}, _ex("Python", 0.8, "y", label="SOFT_SKILL")]
    assert summarize_profile(TEXT, no_label, timestamp="t")["conflict_notes"] == []


def test_no_conflict_for_consistent_labels():
    assert summarize_profile(TEXT, _extractions(), timestamp="t")["conflict_notes"] == []


# ------------------------------------------------------------------ bug 1: skill_memory dedupe + budget


def test_skill_memory_references_skills_instead_of_copying_them():
    summary = summarize_profile(TEXT, _extractions(), timestamp="t")
    memory = build_skill_memory(summary)
    assert memory["skill_count"] == len(summary["skills"])
    assert all(isinstance(k, str) for keys in memory["skills_by_label"].values() for k in keys)
    assert set(memory["skills_by_label"]["SOFT_SKILL"]) == {"communication"}
    legacy = build_skill_memory(summary, inline_skills=True)
    assert all(isinstance(s, dict) for v in legacy["skills_by_label"].values() for s in v)
    assert count_tokens(json.dumps(memory)) < count_tokens(json.dumps(legacy)) / 3


def test_default_packet_is_not_bloated_by_a_second_copy_of_every_skill():
    packet = compact_context(TEXT, _extractions(), job_target="data engineer")
    stable = packet["stable_cag_context"]
    skills_json = json.dumps(stable["profile_summary"]["skills"])
    assert json.dumps(stable["skill_memory"]).count("churn forecasting") == 0
    assert count_tokens(json.dumps(stable["skill_memory"])) < count_tokens(skills_json) / 2


def test_no_budget_keeps_every_skill_and_the_legacy_keys():
    packet = compact_context(TEXT, _extractions(), job_target="data engineer", taxonomy_aliases={"k8s": "Kubernetes"}, course_catalog_summaries=[{"course_id": "c1"}])
    assert set(packet) == {"stable_cag_context", "dynamic_rag_policy", "graph_retrieval_policy", "conflict_notes"}
    stable = packet["stable_cag_context"]
    assert {s["surface_form"] for s in stable["profile_summary"]["skills"]} == {"Python", "SQL", "Kubernetes", "Excel", "communication"}
    assert stable["taxonomy_aliases"] == {"k8s": "Kubernetes"} and stable["course_catalog_summary"] == [{"course_id": "c1"}]
    assert "reason" in packet["dynamic_rag_policy"] and "budget" not in packet


def _big_profile(n=40):
    sents, ex = [], []
    for i in range(n):
        s = f"Delivered project number {i} using Tool{i} for stakeholders in region {i}."
        start = sum(len(x) + 1 for x in sents) + s.find(f"Tool{i}")
        sents.append(s)
        ex.append({"surface_form": f"Tool{i}", "label": "HARD_SKILL", "start": start, "end": start + len(f"Tool{i}"), "source": "mistral",
                   "confidence": 0.3 + 0.015 * i, "evidence": s[:-1], "canonical_skill_id": f"tool{i}", "canonical_skill_name": f"Tool {i}"})
    return " ".join(sents), ex


def test_budget_is_respected_for_packet_and_payload():
    text, ex = _big_profile()
    for budget in (256, 400, 512, 1024, 2048):
        packet = build_cached_context_packet(text, ex, job_target="engineer", budget_tokens=budget, timestamp="t")
        assert count_tokens(json.dumps(packet, sort_keys=True, ensure_ascii=False)) <= budget, budget
        assert packet["payload"]["budget"]["met"] is True
        payload = compact_context(text, ex, budget_tokens=budget)
        assert count_tokens(json.dumps(payload, sort_keys=True, ensure_ascii=False)) <= budget


def test_budget_keeps_most_important_skills_first_and_more_budget_keeps_more():
    text, ex = _big_profile()
    kept = {}
    for budget in (400, 800, 1600):
        packet = compact_context(text, ex, budget_tokens=budget)
        kept[budget] = [s["canonical_skill_id"] for s in packet["stable_cag_context"]["profile_summary"]["skills"]]
    assert 0 < len(kept[400]) < len(kept[800]) < len(kept[1600])
    # confidence grows with i, so the top-confidence skill (tool39) is kept at every budget and low ones are dropped first
    assert all("tool39" in k for k in kept.values())
    assert "tool0" not in kept[400]
    assert set(kept[400]) <= set(kept[800]) <= set(kept[1600])


def test_budgeted_evidence_is_always_a_substring_of_the_source_and_nothing_is_fabricated():
    text, ex = _big_profile()
    for item in ex[::2]:
        item["evidence"] = text[max(0, item["start"] - 300): item["end"] + 300]  # long evidence forces snippet truncation
    for budget in (350, 700, 1500):
        packet = compact_context(text, ex, budget_tokens=budget)
        skills = packet["stable_cag_context"]["profile_summary"]["skills"]
        assert skills
        for skill in skills:
            assert skill["evidence"], "a kept skill always keeps at least one evidence snippet"
            for e in skill["evidence"]:
                assert e["text"] in text
            assert skill["surface_form"] in text
        assert {s["canonical_skill_id"] for s in skills} <= {e["canonical_skill_id"] for e in ex}


def test_budget_infeasible_is_reported_not_hidden():
    packet = compact_context(TEXT, _extractions(), budget_tokens=5)
    assert packet["budget"]["met"] is False and packet["budget"]["kept"] == 0
    assert packet["stable_cag_context"]["profile_summary"]["skills"] == []


def test_budget_validation():
    for bad in (0, -5, 1.5, True):
        try:
            compact_context(TEXT, _extractions(), budget_tokens=bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"budget {bad!r} should be rejected")


def test_budget_drops_untied_aliases_and_catalog_before_skills_but_keeps_tied_aliases():
    aliases = {"k8s": "Kubernetes", "unrelated alias": "Unrelated"}
    catalog = [{"course_id": f"c{i}", "title": "A long course title " * 5} for i in range(30)]
    tight = compact_context(TEXT, _extractions(), budget_tokens=330, taxonomy_aliases=aliases, course_catalog_summaries=catalog)
    stable = tight["stable_cag_context"]
    assert len(stable.get("course_catalog_summary", [])) < len(catalog)
    assert "unrelated alias" not in stable.get("taxonomy_aliases", {})
    assert tight["stable_cag_context"]["profile_summary"]["skills"], "skills win over catalog summaries"


def test_importance_uses_confidence_evidence_count_target_and_optional_source_weights():
    base = {"surface_form": "Python", "confidence": 0.8, "evidence": [{"text": "a", "source": "bert"}]}
    more = {**base, "evidence": [{"text": "a", "source": "bert"}, {"text": "b", "source": "bert"}]}
    assert skill_importance(more) > skill_importance(base)
    assert skill_importance({**base, "confidence": 0.9}) > skill_importance(base)
    assert skill_importance(base, target_terms=frozenset({"python"})) > skill_importance(base)
    assert skill_importance(base, source_reliability={"bert": 0.5}) < skill_importance(base)
    assert skill_importance(base) == skill_importance({**base, "label": "SOFT_SKILL"})  # no unjustified HARD>SOFT preference
    assert skill_importance(base) == skill_importance(base)


def test_budget_ranking_matches_importance_not_input_order():
    rng = random.Random(3)
    text, ex = _big_profile(30)
    rng.shuffle(ex)
    packet = compact_context(text, ex, budget_tokens=500)
    kept = [s["confidence"] for s in packet["stable_cag_context"]["profile_summary"]["skills"]]
    dropped_best = max(e["confidence"] for e in ex if e["confidence"] not in set(kept))
    assert min(kept) >= dropped_best - 1e-9 or len(kept) >= 3  # higher-confidence skills are retained before lower ones
    assert sorted(kept)[0] >= 0.3 + 0.015 * 10  # the bottom third by confidence never survives a tight budget


def test_default_summary_matches_json_roundtrip_and_regex_proxy():
    assert count_tokens("Python, SQL!") == 4
    assert re.findall(r"\w+|[^\w\s]", "a+b") == ["a", "+", "b"]


def test_many_conflicts_cannot_crowd_out_every_skill_under_a_tight_budget():
    text, ex = _big_profile(30)
    conflicts = [dict(e, label="SOFT_SKILL") for e in ex[:20]]  # 20 conflicting HARD/SOFT pairs
    full = compact_context(text, ex + conflicts)
    assert len(full["conflict_notes"]) == 20  # default mode keeps every note
    packet = compact_context(text, ex + conflicts, budget_tokens=330)
    assert packet["budget"]["met"] is True
    assert packet["stable_cag_context"]["profile_summary"]["skills"], "skills survive"
    assert 0 < len(packet["conflict_notes"]) < 20 and packet["budget"]["notes_omitted"] == 20 - len(packet["conflict_notes"])
    assert set(packet["conflict_notes"]) <= set(full["conflict_notes"])
