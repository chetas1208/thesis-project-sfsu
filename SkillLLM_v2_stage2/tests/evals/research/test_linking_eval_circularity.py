"""B3 additions to L06: circularity diagnostics and L02-bucket tagging (tiny synthetic fixtures)."""

import json

from skillllm_v2.evals.research import linking_eval as L


def _alias(cid, alias, source, tax="CUSTOM"):
    return {"canonical_skill_id": cid, "canonical_skill_name": alias, "alias": alias, "normalized_alias": alias.lower(),
            "taxonomy": tax, "label_hint": "HARD_SKILL", "source": source}


def _span(doc_text, surface, linked, method, tax):
    start = doc_text.index(surface)
    sp = {"surface_form": surface, "start": start, "end": start + len(surface), "label": "HARD_SKILL"}
    if linked:
        sp["canonical_skill_name"] = surface
        sp["metadata"] = {"taxonomy_link": {"link_method": method, "taxonomy": tax}}
    return sp


def test_audit_circularity_reports_coverage_and_alias_provenance(tmp_path, monkeypatch):
    alias_path = tmp_path / "aliases.jsonl"
    rows = [_alias("python", "python", "skillspan_gold_train_dev", "SKILLSPAN_GOLD"),
            _alias("java", "java", "mistral_high_confidence"), _alias("sql", "sql", "manual_curated")]
    alias_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    text = "python java and kubernetes"
    linked = tmp_path / "linked.jsonl"
    linked.write_text(json.dumps({"id": "test-0", "text": text, "linked_skills": [
        _span(text, "python", True, "alias", "SKILLSPAN_GOLD"), _span(text, "java", True, "alias", "CUSTOM"),
        _span(text, "kubernetes", False, None, None)]}) + "\n")
    monkeypatch.setattr(L, "REPO_ALIAS_TABLE", alias_path)
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(L, "LEGACY_ARTIFACTS", {"toy": "linked.jsonl"})
    gold = [{"start": 0, "end": 6, "surface_form": "python", "label": "HARD_SKILL"},
            {"start": 7, "end": 11, "surface_form": "java", "label": "HARD_SKILL"}]
    docs = [{"id": "test-0", "text": text, "gold_spans": gold}]
    recs, out = L.audit_circularity(docs, [], seed=1, workers=1, n_null=5)
    a = out["artifacts"]["toy"]
    assert abs(a["legacy_coverage_reproduced"] - 2 / 3) < 1e-9
    assert a["n_gold_matching_spans"] == 2 and a["coverage_on_gold_matching_spans"] == 1.0
    assert a["coverage_on_non_gold_spans"] == 0.0
    assert out["alias_table"]["mined_rows_whose_alias_is_a_test_gold_surface"] == "1/1"
    assert out["alias_table"]["test_gold_mentions_with_surface_in_table"] == 1.0
    # every audit record is a labelled diagnostic, never an accuracy
    assert all(r["metric"].startswith("diagnostic_") for r in recs)


def test_assign_l02_buckets_uses_generalization_bucketer(monkeypatch):
    from skillllm_v2.evals.research import generalization as G
    train = [{"text": "python sql", "gold_spans": [{"start": 0, "end": 6, "label": "HARD_SKILL"},
                                                    {"start": 7, "end": 10, "label": "HARD_SKILL"}]}]
    monkeypatch.setattr(L, "load_skillspan", lambda split: train)
    monkeypatch.setattr(G, "load_alias_index", lambda root: ({}, {}))
    monkeypatch.setattr(G, "load_esco_index", lambda root: ({}, {"available": False}))
    ms = [{"span": "python"}, {"span": "Python"}, {"span": "rust"}]
    info = L.assign_l02_buckets(ms, 1)
    assert [m["l02_bucket"] for m in ms] == ["S1", "S2", "S4"] and info["bucket_counts"] == {"S1": 1, "S2": 1, "S4": 1}


def test_mark_span_highlights_the_right_occurrence():
    t = "develop a team and develop code"
    assert L.mark_span(t, 19, 26) == "develop a team and [[develop]] code"
    assert L.mark_span(t, None, None) == t and L.mark_span(t, 5, 5) == t


def test_cached_rank_computes_only_missing_and_invalidates_on_meta_change(tmp_path):
    calls = []

    def compute(miss):
        calls.append(list(miss))
        return {m: [("u:" + m, 1.0, "x")] for m in miss}
    path = tmp_path / "c.json"
    a = L._cached_rank(path, {"v": 1}, ["a", "b"], compute)
    b = L._cached_rank(path, {"v": 1}, ["a", "b", "c"], compute)
    assert calls == [["a", "b"], ["c"]] and b["a"] == a["a"] == [("u:a", 1.0, "x")]
    L._cached_rank(path, {"v": 2}, ["a"], compute)
    assert calls[-1] == ["a"]


def test_always_nil_baseline_abstains_and_scores_nil_only():
    ms = [{"mention_id": "a", "span": "x", "gold_uris": ["u:1"], "gold_labels": ["x"], "linkable": True, "nil": False, "cluster": "c"},
          {"mention_id": "b", "span": "y", "gold_uris": [], "gold_labels": [], "linkable": False, "nil": True, "cluster": "c"}]
    rows = {r["mention_id"]: r for r in L.score_mentions(ms, L.AlwaysNilLinker(), None)}
    assert rows["a"]["abstain"] and rows["a"]["hit@1"] == 0.0 and rows["a"]["e2e_correct"] == 0.0
    assert rows["b"]["abstain"] and rows["b"]["e2e_correct"] == 1.0
