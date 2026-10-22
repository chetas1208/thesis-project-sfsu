"""Hand-verifiable toy cases + negative controls for calibration_evals and verifier_evals."""
from __future__ import annotations

import math

import numpy as np
import pytest

from skillllm_v2.evals import calibration_evals as ce
from skillllm_v2.evals import verifier_evals as ve
from skillllm_v2.evals.contract import EvalContext

# Toy: confidences / correctness used throughout.
CONF = [0.9, 0.8, 0.7, 0.6]
Y = [1, 1, 0, 1]


# ------------------------------------------------------------------ calibration primitives (hand-computed)

def test_ece_mce_equal_width_toy():
    # each value sits in its own 10-bin; gaps = |y - c| = .1, .2, .7, .4  -> ECE = 1.4/4 = 0.35, MCE = 0.7
    assert ce.ece_width(CONF, Y, 10) == pytest.approx(0.35)
    assert ce.mce(CONF, Y, 10, "width") == pytest.approx(0.7)
    # a single coarse bin: mean conf .75 == accuracy .75 -> ECE 0 (binning can hide miscalibration)
    assert ce.ece_width(CONF, Y, 1) == pytest.approx(0.0)


def test_ece_equal_mass_toy_and_bin_counts():
    bins = ce.reliability_bins(CONF, Y, 2, "mass")
    assert [b["count"] for b in bins] == [2, 2]
    assert bins[0]["mean_conf"] == pytest.approx(0.65) and bins[0]["accuracy"] == pytest.approx(0.5)
    assert bins[1]["mean_conf"] == pytest.approx(0.85) and bins[1]["accuracy"] == pytest.approx(1.0)
    assert ce.ece_mass(CONF, Y, 2) == pytest.approx(0.15)
    wbins = ce.reliability_bins(CONF, Y, 10, "width")
    assert sum(b["count"] for b in wbins) == 4 and wbins[9]["count"] == 1 and wbins[6]["count"] == 1


def test_conf_one_goes_to_last_bin_and_perfect_calibration_is_zero():
    bins = ce.reliability_bins([1.0, 1.0, 0.0], [1, 1, 0], 10, "width")
    assert bins[9]["count"] == 2 and bins[0]["count"] == 1
    conf = [0.7] * 10
    y = [1] * 7 + [0] * 3
    assert ce.ece_width(conf, y) == pytest.approx(0.0)


def test_equal_mass_never_splits_ties():
    conf = [0.5] * 6 + [0.9] * 2
    y = [1, 0, 1, 0, 1, 0, 1, 1]
    bins = ce.reliability_bins(conf, y, 4, "mass")
    assert sorted(b["count"] for b in bins) == [2, 6]  # the six 0.5s stay together


def test_brier_nll_toy():
    assert ce.brier(CONF, Y) == pytest.approx((0.01 + 0.04 + 0.49 + 0.16) / 4)
    expected = -(math.log(0.9) + math.log(0.8) + math.log(0.3) + math.log(0.6)) / 4
    assert ce.nll(CONF, Y) == pytest.approx(expected)
    assert math.isfinite(ce.nll([1.0, 0.0], [0, 1]))  # clipped, not inf


def test_aurc_selective_accuracy_toy():
    # errors by rank: 0,0,1,0 -> risk@k = 0, 0, 1/3, 1/4 -> AURC = (1/3 + 1/4)/4
    assert ce.aurc(CONF, Y) == pytest.approx((1 / 3 + 1 / 4) / 4)
    assert ce.aurc_oracle(Y) == pytest.approx(0.25 / 4)  # error ranked last: only risk@4 = 1/4
    assert ce.selective_accuracy(CONF, Y, 0.5) == pytest.approx(1.0)
    assert ce.selective_accuracy(CONF, Y, 0.75) == pytest.approx(2 / 3)
    assert ce.selective_accuracy(CONF, Y, 1.0) == pytest.approx(0.75)
    curve = ce.risk_coverage_curve(CONF, Y)
    assert [round(p["risk"], 4) for p in curve] == [0.0, 0.0, 0.3333, 0.25]


def test_aurc_tied_confidence_equals_base_risk():
    conf = [0.86] * 8
    y = [1, 1, 1, 0, 0, 1, 0, 1]
    assert ce.aurc(conf, y) == pytest.approx(3 / 8)  # no ranking ability -> AURC = base error rate
    assert ce.selective_accuracy(conf, y, 0.5) == pytest.approx(5 / 8)
    assert ce.auroc(conf, y) == pytest.approx(0.5)


def test_auroc_and_ap_toy():
    # positives .9,.8,.6 vs negative .7 -> 2 of 3 pairs ordered correctly
    assert ce.auroc(CONF, Y) == pytest.approx(2 / 3)
    assert ce.average_precision(CONF, Y) == pytest.approx((1 + 1 + 0.75) / 3)
    assert ce.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert math.isnan(ce.auroc([0.1, 0.2], [1, 1]))


def test_summary_flags_constant_confidence():
    s = ce.calibration_summary([0.86] * 20, [1] * 13 + [0] * 7)
    assert s["confidence_is_constant"] and s["n_distinct_confidences"] == 1
    assert s["ece_width"] == pytest.approx(abs(0.86 - 0.65))
    assert s["ece_mass"] == pytest.approx(s["ece_width"])


# ------------------------------------------------------------------ negative controls

def test_negative_control_random_confidence_ece_near_expectation():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.7).astype(int)
    conf = np.full(len(y), 0.9)  # real confidences are irrelevant to the random control
    nc = ce.negative_controls(conf, y, seed=1, n_rep=20)
    assert ce.expected_ece_random_uniform(0.7) == pytest.approx(0.29, abs=1e-9)  # mean |0.7 - bin midpoint| over 10 bins
    exp = ce.expected_ece_random_uniform(float(y.mean()))
    assert nc["random"]["ece_width_analytic_expectation"] == pytest.approx(exp)
    assert nc["random"]["ece_width_mean"] == pytest.approx(exp, abs=0.02)
    assert nc["random"]["auroc_mean"] == pytest.approx(0.5, abs=0.02)
    assert nc["random"]["aurc_mean"] == pytest.approx(1 - y.mean(), abs=0.02)


def test_negative_control_constant_base_rate_games_ece_but_not_aurc():
    # perfectly ranking but over-confident scorer: correct = .95, wrong = .90
    y = np.array([1] * 70 + [0] * 30)
    conf = np.where(y == 1, 0.95, 0.90)
    nc = ce.negative_controls(conf, y, seed=3, n_rep=10)
    const = nc["constant_base_rate"]
    assert const["ece_width"] == pytest.approx(0.0, abs=1e-9)
    assert const["aurc"] == pytest.approx(0.30)  # = base risk: zero selective ability
    assert const["brier"] == pytest.approx(0.7 * 0.3)
    assert nc["actual"]["aurc"] == pytest.approx(ce.aurc_oracle(y))  # the real scorer ranks perfectly (= oracle AURC)
    assert nc["actual"]["aurc"] < 0.06 < const["aurc"]
    assert nc["actual"]["ece_width"] > 0.2  # but is badly calibrated
    assert nc["ece_alone_is_gameable_here"] is True  # constant wins on ECE, loses on AURC


# ------------------------------------------------------------------ bootstraps

def test_cluster_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(5)
    n_docs = 60
    doc = np.repeat(np.arange(n_docs), 5)
    y = (rng.random(len(doc)) < 0.6).astype(int)
    conf = np.clip(0.6 + 0.1 * rng.standard_normal(len(doc)), 0.01, 0.99)
    a = ce.cluster_bootstrap(doc, n_docs, conf, y, n_boot=200, seed=7)
    b = ce.cluster_bootstrap(doc, n_docs, conf, y, n_boot=200, seed=7)
    assert a["brier"] == b["brier"]
    point = ce.brier(conf, y)
    assert a["brier"][0] <= point <= a["brier"][1]


def test_paired_cluster_bootstrap_identical_arms_and_worse_arm():
    rng = np.random.default_rng(9)
    n_docs = 80
    doc = np.repeat(np.arange(n_docs), 4)
    y = (rng.random(len(doc)) < 0.7).astype(int)
    good = np.where(y == 1, 0.8, 0.5) * 0.0 + 0.7  # perfectly calibrated constant
    bad = np.full(len(doc), 0.99)
    same = ce.paired_cluster_bootstrap((doc, good, y), (doc, good, y), n_docs, ("ece_width",), n_boot=100, seed=1)
    assert same["ece_width"]["delta"] == 0.0 and same["ece_width"]["p"] == 1.0
    diff = ce.paired_cluster_bootstrap((doc, good, y), (doc, bad, y), n_docs, ("ece_width", "brier"), n_boot=200, seed=1)
    assert diff["ece_width"]["delta"] < 0 and diff["ece_width"]["ci_high"] < 0  # `good` has lower ECE than `bad`
    assert diff["brier"]["delta"] < 0


def test_save_reliability_writes_json_and_markdown(tmp_path):
    s = ce.calibration_summary(CONF, Y)
    written = ce.save_reliability(tmp_path, {"toy": {"bins_width": s["bins_width"], "bins_mass": s["bins_mass"]}})
    assert (tmp_path / "calibration_reliability.json").exists()
    md = (tmp_path / "calibration_reliability.md").read_text()
    assert "Equal-width bins" in md and "Equal-mass bins" in md and "toy" in md
    assert len(written) >= 2


# ------------------------------------------------------------------ calibration run() on injected data

def _doc(i, text, spans):
    return {"id": f"test-{i}", "text": text, "gold_spans": [{"surface_form": text[s:e], "label": lab, "start": s, "end": e} for s, e, lab in spans]}


def _cand(text, s, e, label, conf, source="mistral", sources=None):
    md = {"sources": sources} if sources else {}
    return {"surface_form": text[s:e], "label": label, "start": s, "end": e, "source": source, "confidence": conf,
            "evidence": text[s:e], "canonical_skill_id": None, "canonical_skill_name": None, "metadata": md}


def _toy_world(n_docs=12):
    docs, rows_const, rows_merged = [], [], []
    for i in range(n_docs):
        text = f"python and java skills {i}"
        spans = [(0, 6, "HARD_SKILL"), (11, 15, "HARD_SKILL")]
        docs.append(_doc(i, text, spans))
        good1 = _cand(text, 0, 6, "HARD_SKILL", 0.86)
        good2 = _cand(text, 11, 15, "HARD_SKILL", 0.86)
        bad = _cand(text, 16, 22, "HARD_SKILL", 0.86)
        rows_const.append({"id": f"test-{i}", "text": text, "predictions": [good1, good2, bad]})
        m_good1 = _cand(text, 0, 6, "HARD_SKILL", 0.91, sources=["dictionary", "mistral"])
        m_bad = _cand(text, 16, 22, "HARD_SKILL", 0.30, source="dictionary", sources=["dictionary"])
        rows_merged.append({"id": f"test-{i}", "text": text, "predictions": [m_good1, good2, m_bad]})
    return docs, rows_const, rows_merged


def test_flatten_exact_match_and_surface_secondary():
    docs, rows, _ = _toy_world(2)
    gold = ce.load_gold(EvalContext(extra={"gold_docs": docs}))
    fl = ce.flatten_candidates(rows, gold)
    assert list(fl["y"]) == [1, 1, 0, 1, 1, 0] and fl["n_docs"] == 2
    # a candidate with a wrong offset but the right surface counts only for the secondary 'surface' match
    text = docs[0]["text"]
    off = [{"id": "test-0", "text": text, "predictions": [_cand(text, 1, 6, "HARD_SKILL", 0.5)]}]
    fl2 = ce.flatten_candidates(off, gold)
    assert fl2["y"][0] == 0 and fl2["y_surface"][0] == 0  # 'ython' != 'python'


def test_calibration_run_toy_world_and_calibrator_replay(tmp_path):
    docs, rows_const, rows_merged = _toy_world(12)
    ctx = EvalContext(root=tmp_path, n_boot=60, extra={
        "gold_docs": docs, "seen_texts": {docs[0]["text"]},
        "calibration_sources": {"raw_anchor": rows_const, "merged_candidates": rows_merged, "wrapper_final": rows_merged},
        "out_dir": tmp_path / "out"})
    res = {r.eval_name: r for r in ce.run(ctx)}
    ece = res["calibration.raw_anchor.ece_width"]
    assert ece.module == "calibrate" and ece.n == 36 and ece.status == "ok"
    assert ece.value == pytest.approx(abs(0.86 - 2 / 3), abs=1e-9)
    assert ece.ci is not None and ece.ci[0] <= ece.value <= ece.ci[1]
    assert any("CONSTANT" in c for c in ece.caveats)  # constant-confidence flag
    assert res["calibration.raw_anchor.aurc"].value == pytest.approx(1 / 3)
    assert res["calibration.raw_anchor.auroc"].value == pytest.approx(0.5)
    assert res["calibration.raw_anchor.control_constant_ece"].value == pytest.approx(0.0, abs=1e-9)
    assert res["calibration.raw_anchor.control_constant_ece"].details["ece_alone_is_gameable_here"] is True
    # merged world: dictionary-only low-confidence span is wrong -> informative confidence
    assert res["calibration.merged_candidates.auroc"].value > 0.5
    # leakage flag + text-disjoint view present because one doc text is in the 'seen' set
    assert any("LEAKAGE" in c for c in res["calibration.merged_candidates.ece_width"].caveats)
    assert "calibration.merged_candidates.ece_width_text_disjoint" in res
    # calibrator replay (library defaults, dictionary-only threshold .70) drops the 0.30 dictionary-only FP in every doc
    d = res["calibration.calibrator.delta_ece_width"]
    assert d.details["n_dropped"] == 12 and d.details["n_dropped_that_were_correct"] == 0
    assert res["calibration.calibrator.delta_brier"].value < 0  # removing wrong low-conf spans improves Brier
    assert (tmp_path / "out" / "calibration_reliability.json").exists()
    # small slices are skipped, never silently scored
    small = [r for r in res.values() if ".by_source." in r.eval_name and r.status == "skipped"]
    assert small and all("n=" in r.reason for r in small)


def test_calibration_run_skips_without_gold(tmp_path):
    res = ce.run(EvalContext(root=tmp_path, n_boot=10, extra={"gold_docs": []}))
    assert res and all(r.status == "skipped" for r in res)  # nothing to score offline -> explicit skips, no fabricated numbers
    assert all(r.reason for r in res)


# ------------------------------------------------------------------ verifier primitives

def test_binary_prf_false_drop_toy():
    m = ve.binary_prf([1, 1, 0, 0], [1, 0, 1, 0])  # tp=1 fp=1 fn=1 tn=1
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert m["false_drop_rate"] == 0.5 and m["false_keep_rate"] == 0.5
    keep_all = ve.binary_prf([1, 1, 1], [1, 1, 0])
    assert keep_all["false_drop_rate"] == 0.0 and keep_all["false_keep_rate"] == 1.0


def test_largest_threshold_within():
    rows = [{"threshold": 0.0, "recall_loss": 0.0, "f1": 0.7}, {"threshold": 0.2, "recall_loss": 0.005, "f1": 0.69},
            {"threshold": 0.4, "recall_loss": 0.03, "f1": 0.6}]
    assert ve.largest_threshold_within(rows, "recall_loss", 0.01, "max") == 0.2
    assert ve.largest_threshold_within(rows, "recall_loss", 0.0, "max") == 0.0
    assert ve.largest_threshold_within(rows, "f1", 0.665, "min") == 0.2
    assert ve.largest_threshold_within(rows, "f1", 0.9, "min") is None


def _verifier_world():
    # doc A: gold {python, java}; doc B: gold {c++}; candidates with model scores chosen by hand via text marker
    docs = [
        _doc(0, "python java x", [(0, 6, "HARD_SKILL"), (7, 11, "HARD_SKILL")]),
        _doc(1, "c++ and more", [(0, 3, "HARD_SKILL")]),
    ]
    rows = [
        {"id": "test-0", "text": docs[0]["text"], "predictions": [_cand(docs[0]["text"], 0, 6, "HARD_SKILL", .9), _cand(docs[0]["text"], 7, 11, "HARD_SKILL", .9)]},
        {"id": "test-1", "text": docs[1]["text"], "predictions": [_cand(docs[1]["text"], 0, 3, "HARD_SKILL", .9), _cand(docs[1]["text"], 4, 7, "HARD_SKILL", .9)]},
    ]
    return docs, rows


class FakeModel:
    """predict_proba keyed on the surface form: python .9, java .2, c++ .8, 'and' .1 (hand-chosen scores)."""
    table = {"python": 0.9, "java": 0.2, "c++": 0.8, "and": 0.1}

    def predict_proba(self, texts):
        p = np.array([self.table.get(t.split(" [SEP] ")[0], 0.5) for t in texts])
        return np.stack([1 - p, p], axis=1)


def test_real_set_counts_sweep_and_floor_views():
    docs, rows = _verifier_world()
    gold = ce.load_gold(EvalContext(extra={"gold_docs": docs}))
    cs = ve._real_candidate_set(rows, gold, None)
    none = ve.doc_counts(cs, np.ones(len(cs["cand"]), dtype=bool))
    assert none.tolist() == [[2, 0, 0], [1, 1, 0]]  # tp, fp, fn per doc
    score = ve.score_real_set(FakeModel(), cs)
    assert score.tolist() == [0.9, 0.2, 0.8, 0.1]
    tab = ve.sweep_table(cs, score, [0.0, 0.15, 0.25, 0.5], np.ones(4, dtype=bool))
    by_t = {r["threshold"]: r for r in tab}
    assert by_t[0.0]["recall_loss"] == 0.0 and by_t[0.0]["f1"] == pytest.approx(2 * 0.75 * 1.0 / 1.75)
    assert by_t[0.15]["n_dropped"] == 1 and by_t[0.15]["n_dropped_correct"] == 0 and by_t[0.15]["recall_loss"] == 0.0  # drops the FP 'and'
    assert by_t[0.15]["precision"] == pytest.approx(1.0)
    assert by_t[0.25]["n_dropped_correct"] == 1 and by_t[0.25]["recall_loss"] == pytest.approx(1 / 3)  # 'java' is a true skill
    assert by_t[0.25]["false_drop_rate"] == pytest.approx(1 / 3)
    assert ve.largest_threshold_within(tab, "recall_loss", 0.0, "max") == 0.15
    assert ve.largest_threshold_within(tab, "recall_loss", 0.4, "max") == 0.5
    # anchor-protected arm: when nothing is droppable the sweep is flat
    flat = ve.sweep_table(cs, score, [0.5], np.zeros(4, dtype=bool))
    assert flat[0]["n_dropped"] == 0 and flat[0]["recall_loss"] == 0.0


def test_paired_f1_bootstrap_identical_and_regression():
    ca = np.array([[2, 0, 0]] * 20, dtype=float)
    same = ve.paired_f1_bootstrap(ca, ca.copy(), 100, 1)
    assert same["f1"]["delta"] == 0.0 and same["f1"]["p"] == 1.0
    worse = np.array([[1, 0, 1]] * 20, dtype=float)  # same candidates minus a true skill each doc
    d = ve.paired_f1_bootstrap(worse, ca, 200, 1)
    assert d["recall"]["delta"] == pytest.approx(-0.5) and d["f1"]["delta"] < 0 and d["f1"]["ci_high"] < 0


def test_rule_verifier_drops_only_nonanchor_and_unprotected_false_drops():
    text = "knowledge of python"
    gold = ce.load_gold(EvalContext(extra={"gold_docs": [_doc(0, text, [(0, 9, "HARD_SKILL")])]}))  # 'knowledge' is a (weird) gold skill
    rows = [{"id": "test-0", "text": text, "predictions": [_cand(text, 0, 9, "HARD_SKILL", 0.86)]}]
    cs = ve._real_candidate_set(rows, gold, None)
    keep_shipped, _ = ve.rule_keep(cs, protect_anchor=True)
    keep_open, reasons = ve.rule_keep(cs, protect_anchor=False)
    assert keep_shipped.tolist() == [True]  # anchor never rejected on quality grounds
    assert keep_open.tolist() == [False] and reasons["generic_phrase"] == 1  # a true skill deleted: the critical failure mode


def test_verifier_run_end_to_end_with_fake_model(tmp_path):
    from skillllm_v2.train import build_verifier_dataset as bvd

    docs, rows = _verifier_world()
    for d in docs:  # build_verifier_dataset expects extra keys on docs
        d.setdefault("tokens", d["text"].split())

    def synth(ds):
        out = []
        for doc in ds:
            for span in doc["gold_spans"]:
                out.append(bvd._verifier_row(doc["text"], span, True, span["label"], span["surface_form"], "gold", {"reason": "gold_span"}))
                exp = bvd._expanded_negative(doc["text"], span)
                if exp:
                    out.append(bvd._verifier_row(doc["text"], exp, False, None, span["surface_form"], "boundary_noise", {"reason": "boundary_expanded"}))
        return out

    ctx = EvalContext(root=tmp_path, n_boot=50, extra={
        "verifier_model": FakeModel(), "gold_docs": docs, "test_docs": docs, "seen_texts": set(),
        "verifier_train_rows": synth(docs), "verifier_dev_rows": synth(docs[:1]),
        "real_candidate_sets": {"toy": rows}, "recall_tol": 0.0})
    res = {r.eval_name: r for r in ve.run(ctx)}
    # in-sample contrast is always flagged
    assert all(any("IN-SAMPLE" in c for c in r.caveats) for n, r in res.items() if "IN_SAMPLE" in n and r.status == "ok")
    assert res["verify.model.verifier_train_IN_SAMPLE.recall"].status == "ok"
    # e2e: unprotected model arm at 0.5 drops 'java' (0.2) and 'and' (0.1): recall 3/3 -> 2/3, precision 3/4 -> 2/2
    arm = res["verify.e2e.toy.model_t0.5_all.exact_f1_delta"]
    assert arm.details["n_true_skills_dropped"] == 1 and arm.details["recall_delta"] == pytest.approx(-1 / 3)
    f_without = res["verify.e2e.toy.exact_f1_without_verifier"].value
    assert f_without == pytest.approx(2 * 0.75 / 1.75)
    assert arm.value == pytest.approx(0.8 - f_without)  # F1 with = 2*1*(2/3)/(5/3) = 0.8
    sweep = res["verify.sweep.toy.all.max_threshold_recall_loss_le_tol"]
    assert sweep.value == pytest.approx(0.2)  # tol 0: largest grid threshold deleting no true skill (java scores exactly 0.2, kept at t=0.2)
    assert any("post hoc" in c or "same test candidates" in c for c in sweep.caveats)
    assert res["verify.real_dev_candidates"].status == "skipped"  # no dev candidates exist -> explicitly skipped
    assert res["verify.artifact_eval_predictions"].status == "skipped"


def test_verifier_run_skips_when_model_missing(tmp_path):
    docs, rows = _verifier_world()
    ctx = EvalContext(root=tmp_path, n_boot=20, extra={"gold_docs": docs, "test_docs": docs, "seen_texts": set(),
                                                       "verifier_train_rows": [], "verifier_dev_rows": [], "real_candidate_sets": {"toy": rows}})
    res = ve.run(ctx)
    skipped = [r for r in res if r.status == "skipped"]
    assert any("missing" in r.reason or "cannot load" in r.reason for r in skipped)
    # rule-based arms still run without the model
    assert any(r.eval_name == "verify.e2e.toy.rule_shipped_anchor_protected.exact_f1_delta" and r.status == "ok" for r in res)
