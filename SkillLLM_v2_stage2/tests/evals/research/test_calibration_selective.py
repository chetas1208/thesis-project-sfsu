"""Tests for research/calibration_selective.py (synthetic data, hand-checkable values)."""

import math

import numpy as np
import pytest

from skillllm_v2.evals.research import calibration_selective as K


def test_lp_logit_stable_and_exact():
    p = np.array([0.3, 0.9, 0.5])
    assert K.lp_logit(np.log(p)) == pytest.approx(np.log(p / (1 - p)))
    v = K.lp_logit(np.array([-1e-9, -1e-12, 0.0, -50.0]))
    assert np.all(np.isfinite(v)) and v[0] < v[1] and v[3] < -40


def test_selective_metrics_hand_example():
    # 4 spans, sorted by confidence: correct, correct, wrong, wrong ; G = 5 gold spans
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    m = K.metrics(conf, y, G=5.0)
    assert m["precision"] == 0.5
    assert m["sel_prec@100"] == 0.5 and m["sel_prec@50"] == 1.0
    assert m["sel_f1@100"] == pytest.approx(2 * 2 / (4 + 5))  # TP 2, k 4
    assert m["sel_f1@50"] == pytest.approx(2 * 2 / (2 + 5))  # abstaining removes 2 FP, keeps both TP
    assert m["aurc"] == pytest.approx(np.mean([0, 0, 1 / 3, 2 / 4]))
    assert m["auroc"] == 1.0
    assert m["brier"] == pytest.approx(np.mean((conf - y) ** 2))


def test_constant_confidence_has_no_selective_ability():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.7).astype(float)
    m = K.metrics(np.full(400, 0.86), y, G=500.0)
    assert m["aurc"] == pytest.approx(1 - y.mean())
    assert m["sel_prec@50"] == pytest.approx(y.mean())
    assert m["auroc"] == 0.5
    assert m["ece_w10"] == pytest.approx(abs(0.86 - y.mean()))


def test_calibrators_fit_and_bound():
    rng = np.random.default_rng(1)
    g = rng.normal(0, 2, 5000)
    y = (rng.random(5000) < K.sigmoid(g / 2.0)).astype(float)  # true temperature 2
    T = K.fit_calibrator("temperature", g, y)
    assert T.params["T"] == pytest.approx(2.0, rel=0.15)
    for kind in K.CALIBRATORS:
        p = K.fit_calibrator(kind, g, y)(g)
        assert np.all((p >= 0) & (p <= 1))
    iso = K.fit_calibrator("isotonic", g, y)
    z = np.sort(g)
    assert np.all(np.diff(iso(z)) >= -1e-12)


def test_crossfit_uses_only_other_folds():
    doc = np.repeat(np.arange(10), 3)
    X = doc.astype(float)
    y = np.zeros(30)
    seen = []

    def fit(Xtr, ytr):
        seen.append(set(Xtr.tolist()))
        return lambda Xte: np.array([float(x in seen[-1]) for x in Xte])
    out = K.crossfit(fit, X, y, doc, k=5, seed=0)
    assert np.all(out == 0.0)  # a test document never appears in its own training fold


def _gold(n_docs=6):
    gold = {}
    for d in range(n_docs):
        gold[f"test-{d}"] = {"gold_spans": [{"start": 0, "end": 4, "label": "HARD_SKILL"}, {"start": 5, "end": 9, "label": "SOFT_SKILL"}],
                             "text": "abcd efgh"}
    return gold


def test_table_dedup_and_cluster_bootstrap():
    gold = _gold()
    preds = {f"test-{d}": [{"start": 0, "end": 4, "label": "HARD_SKILL", "confidence": 0.9},
                           {"start": 0, "end": 4, "label": "HARD_SKILL", "confidence": 0.9},  # duplicate key: counted once
                           {"start": 1, "end": 4, "label": "HARD_SKILL", "confidence": 0.4}] for d in range(6)}
    t = K.Table("x", "test", gold, preds)
    assert t.n == 12 and t.G == 12.0 and t.y.sum() == 6
    res = K.evaluate(t, {"raw": t.score("conf")}, t.score("conf"), n_boot=50, seed=3, const_baseline=0.5)
    lo, hi = res["ci"]["raw"]["sel_prec@50"]
    assert lo <= res["point"]["raw"]["sel_prec@50"] <= hi
    assert res["point"]["raw"]["sel_prec@50"] == 1.0 and res["point"]["_constant"]["aurc"] == pytest.approx(0.5)


def test_anchor_features_shape():
    gold = _gold(2)
    lp = {"sum": -0.5, "mean": -0.25, "min": -0.4, "n_tokens": 2}
    preds = {f"test-{d}": [{"start": 0, "end": 4, "label": "HARD_SKILL", "confidence": 0.86, "surface_form": "abcd",
                            "metadata": {"logprob": lp}}] for d in range(2)}
    t = K.Table("a", "test", gold, preds, extra=lambda i, s, doc: {"agree": 1.0})
    X = K.anchor_features(t)
    assert X.shape == (2, len(K.ANCHOR_FEATS)) and np.all(np.isfinite(X))
