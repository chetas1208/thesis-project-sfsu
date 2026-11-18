"""Tests for skillllm_v2.evals.research.linking_eval (tiny synthetic ESCO-like fixtures, no external data needed)."""

import numpy as np
import pytest

from skillllm_v2.evals.research import linking_eval as L

CONCEPTS = [
    {"uri": "u:python", "preferred_label": "Python (computer programming)", "alt_labels": ["Python", "python programming"]},
    {"uri": "u:sql", "preferred_label": "SQL", "alt_labels": ["structured query language"]},
    {"uri": "u:teamwork", "preferred_label": "work in teams", "alt_labels": ["teamwork", "team work"]},
    {"uri": "u:java", "preferred_label": "Java (computer programming)", "alt_labels": ["Java"]},
]
for c in CONCEPTS:
    c["labels"] = [c["preferred_label"]] + c["alt_labels"]


def _m(span, gold=(), mid=None, cluster="c1"):
    gold = list(gold)
    return {"mention_id": mid or span, "span": span, "gold_uris": gold, "gold_labels": gold, "linkable": bool(gold),
            "nil": not gold, "cluster": cluster}


def test_exact_linker_prefers_preferred_label_and_abstains_otherwise():
    ex = L.ExactLinker(CONCEPTS)
    ranked, s1, method = ex.rank(_m("SQL"))
    assert ranked[0][0] == "u:sql" and s1 == 1.0 and method == "pref"
    ranked, s1, method = ex.rank(_m("Teamwork"))
    assert ranked[0][0] == "u:teamwork" and method == "alt"
    ranked, s1, method = ex.rank(_m("kubernetes"))
    assert ranked == [] and s1 is None


def test_bm25_ranks_token_overlap_first():
    bm = L.BM25Linker(CONCEPTS)
    ranked, s1, _ = bm.rank(_m("python programming skills"))
    assert ranked[0][0] == "u:python" and s1 > 0
    ranked, s1, method = bm.rank(_m("zzzz"))
    assert ranked == [] and method == "none"


def test_dense_linker_uses_max_over_labels():
    labels, uris = [], []
    for c in CONCEPTS:
        for lab in c["labels"]:
            labels.append(lab)
            uris.append(c["uri"])
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(labels), 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q = vecs[labels.index("teamwork")].copy()  # identical to one alt label of u:teamwork
    dl = L.DenseLinker(CONCEPTS, {"label_vecs": vecs, "label_uris": uris, "query_vecs": {"team spirit": q}})
    ranked, s1, _ = dl.rank(_m("team spirit"))
    assert ranked[0][0] == "u:teamwork" and abs(s1 - 1.0) < 1e-5
    assert len({u for u, _ in ranked}) == len(ranked)  # one entry per concept


def test_score_mentions_metrics_and_abstention():
    ex = L.ExactLinker(CONCEPTS)
    ms = [_m("SQL", ["u:sql"]), _m("Java", ["u:python"]), _m("kubernetes", []), _m("teamwork", [])]
    rows = {r["mention_id"]: r for r in L.score_mentions(ms, ex, None)}
    assert rows["SQL"]["hit@1"] == 1.0 and rows["SQL"]["rr"] == 1.0 and rows["SQL"]["e2e_correct"] == 1.0
    assert rows["Java"]["hit@1"] == 0.0 and rows["Java"]["rr"] == 0.0
    assert rows["kubernetes"]["abstain"] and rows["kubernetes"]["e2e_correct"] == 1.0  # NIL correctly abstained
    assert not rows["teamwork"]["abstain"] and rows["teamwork"]["e2e_correct"] == 0.0  # NIL wrongly linked


def test_tau_tuning_picks_threshold_that_rejects_low_scores():
    class Fake:
        def rank(self, m):
            return [("u:sql", m["s"])], m["s"], "fake"
    ms = [dict(_m(f"a{i}", ["u:sql"]), s=0.9) for i in range(5)] + [dict(_m(f"n{i}", []), s=0.2) for i in range(5)]
    t = L.tune_tau(ms, Fake())
    assert t["tau"] == pytest.approx(0.9)
    assert t["val_nil_aware_acc"] == 1.0 and t["val_nil_aware_acc_never_abstain"] == 0.5


def test_cluster_and_nil_rows_aggregate_per_cluster():
    rows = [{"cluster": "a", "hit@1": 1.0, "linkable": True, "nil": False, "abstain": False},
            {"cluster": "a", "hit@1": 0.0, "linkable": True, "nil": False, "abstain": True},
            {"cluster": "b", "hit@1": 1.0, "linkable": True, "nil": False, "abstain": False},
            {"cluster": "b", "linkable": False, "nil": True, "abstain": True},
            {"cluster": "b", "linkable": False, "nil": True, "abstain": False}]
    arr, keys = L.cluster_rows(rows, "hit@1", lambda r: r["linkable"])
    assert keys == ["a", "b"] and arr.tolist() == [[1.0, 2.0], [1.0, 1.0]]
    nr, _ = L.nil_rows(rows)
    assert nr.sum(axis=0).tolist() == [1.0, 1.0, 1.0]  # tp, fp, fn for the NIL class


def test_conf_bins():
    assert L.conf_bin("alias", 1.0) == "exact_alias"
    assert L.conf_bin("bm25", 0.8) == "bm25_high" and L.conf_bin("bm25", 0.6) == "bm25_low"
    assert L.conf_bin("none", None) == "none"
