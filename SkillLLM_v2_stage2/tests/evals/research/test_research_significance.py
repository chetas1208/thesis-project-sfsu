"""Tests for research/significance.py (synthetic per_unit_scores files and predictions)."""

import json

import numpy as np
import pytest

from skillllm_v2.evals.research import significance as G
from skillllm_v2.evals.research import stats as S


def _sp(s, e, label="HARD_SKILL"):
    return {"start": s, "end": e, "label": label}


def test_vote_majority_union_unanimous():
    a = {"d": [_sp(0, 1), _sp(2, 3)]}
    b = {"d": [_sp(0, 1), _sp(4, 5), _sp(4, 5)]}  # duplicate counts once
    c = {"d": [_sp(0, 1), _sp(2, 3)]}
    keys = lambda out: sorted((x["start"], x["end"]) for x in out["d"])
    assert keys(G.vote([a, b, c], ["d"], 2)) == [(0, 1), (2, 3)]
    assert keys(G.vote([a, b, c], ["d"], 3)) == [(0, 1)]
    assert keys(G.vote([a, b, c], ["d"], 1)) == [(0, 1), (2, 3), (4, 5)]


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_ingest_counts_with_preference_reference(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for u in range(30):
        tp = int(rng.integers(0, 4))
        rows.append({"layer": "LX", "family": "F", "system": "mistral_lora", "metric": "exact", "label": "ALL", "unit_id": f"u{u}",
                     "tp": tp + 1, "fp": 1, "fn": 1})
        rows.append({"layer": "LX", "family": "F", "system": "other", "metric": "exact", "label": "ALL", "unit_id": f"u{u}",
                     "tp": tp, "fp": 2, "fn": 2})
    p = tmp_path / "per_unit_scores.jsonl"
    _write(p, rows)
    tests, meta = G.ingest(p, 200, 200, 1)
    assert meta["n_tests"] == 1
    t = tests[0]
    assert (t["system"], t["reference"]) == ("other", "mistral_lora")
    a = np.array([[r["tp"], r["fp"], r["fn"]] for r in rows if r["system"] == "other"], float)
    b = np.array([[r["tp"], r["fp"], r["fn"]] for r in rows if r["system"] == "mistral_lora"], float)
    assert t["delta"] == pytest.approx(float(S.micro_f1(a.sum(0)) - S.micro_f1(b.sum(0))))
    assert t["delta"] < 0 and t["ci_high"] < 0


def test_ingest_values_declared_reference_and_unpaired_units(tmp_path):
    rows = []
    for u in range(20):
        rows.append({"layer": "LY", "family": "H", "system": "full", "metric": "ndcg", "unit_id": u, "value": 0.5})
        rows.append({"layer": "LY", "family": "H", "system": "compact", "metric": "ndcg", "unit_id": u, "value": 0.6,
                     "reference": "full"})
    rows.append({"layer": "LY", "family": "H", "system": "compact", "metric": "ndcg", "unit_id": 99, "value": 1.0, "reference": "full"})
    p = tmp_path / "per_unit_scores.jsonl"
    _write(p, rows)
    tests, meta = G.ingest(p, 100, 100, 1)
    assert len(tests) == 1 and tests[0]["delta"] == pytest.approx(0.1) and tests[0]["n_units"] == 20
    assert tests[0]["units_dropped_unpaired"] == 1 and tests[0]["reference_rule"] == "declared"


def test_ingest_bucket_schema_and_bad_rows(tmp_path):
    rows = [{"layer": "L2", "family": "b", "system": s, "metric": "bucket_counts", "bucket": "S1", "unit_id": u,
             "gold_tp": g, "gold_n": 2, "pred_tp": g, "pred_n": 2} for u in range(10) for s, g in (("mistral_lora", 2), ("x", 1))]
    rows.append({"layer": "L2", "family": "b", "system": "x"})  # malformed
    p = tmp_path / "per_unit_scores.jsonl"
    _write(p, rows)
    tests, meta = G.ingest(p, 100, 100, 1)
    assert meta["n_unparsed_rows"] == 1
    assert tests[0]["delta"] == pytest.approx(-0.5) and "bucket=S1" in tests[0]["metric"]


def test_crosscheck_flags_mismatch():
    tests = [{"source": "central", "system": "qlora_v1", "reference": "anchor", "metric": "exact_span_f1", "delta": -0.0135,
              "ci_low": -0.034, "ci_high": 0.007, "p": 0.2, "p_ar": 0.2, "p_holm": 0.4}]
    legacy_ok = [{"file": "f", "system": "qlora_v1", "reference": "anchor", "metric": "exact_span_f1", "delta": -0.0135,
                  "ci_low": -0.034, "ci_high": 0.007, "p_boot": 0.21, "p_perm": 0.19, "p_holm_legacy": 0.8}]
    legacy_bad = [dict(legacy_ok[0], delta=-0.02, p_boot=0.01)]
    assert G.crosscheck(legacy_ok, tests)[0]["status"] == "agree"
    flags = G.crosscheck(legacy_bad, tests)[0]["flags"]
    assert any(f.startswith("DELTA_MISMATCH") for f in flags) and any(f.startswith("BOOT_DECISION") for f in flags)
    assert G.crosscheck([dict(legacy_ok[0], system="zzz")], tests)[0]["status"] == "not_recomputed"


def test_count_matrix_matches_repository_evaluator():
    ev = pytest.importorskip("skillllm_v2.evals.extraction_evals")
    from skillllm_v2.eval.extraction_eval import evaluate_extraction
    gold = {"d1": {"gold_spans": [_sp(0, 4), _sp(5, 9, "SOFT_SKILL")], "text": "abcd efgh"},
            "d2": {"gold_spans": [_sp(0, 2)], "text": "ab cd"}}
    preds = {"d1": [_sp(0, 4), _sp(5, 8, "SOFT_SKILL")], "d2": [_sp(0, 2), _sp(3, 5)]}
    M = G.count_matrix(gold, ["d1", "d2"], preds)
    ref = evaluate_extraction([gold[i]["gold_spans"] for i in ("d1", "d2")], [preds[i] for i in ("d1", "d2")])
    assert float(G.f1_on(G.METRIC_COLS["exact_span_f1"])(M.sum(0))) == pytest.approx(ref["exact_span_f1"])
    assert float(G.f1_on(G.METRIC_COLS["token_f1"])(M.sum(0))) == pytest.approx(ref["token_f1"])
    assert float(G.f1_on(G.METRIC_COLS["soft_skill_f1"])(M.sum(0))) == pytest.approx(ref["soft_skill_f1"])


def test_family_resamples_unblocks_holm_and_cross_family_reference(tmp_path):
    assert G.family_resamples(2000, 10) == 2000 and G.family_resamples(2000, 244) == 9761
    assert 2 / (G.family_resamples(2000, 300) + 1) * 300 < 0.05
    rows = []
    for u in range(25):
        rows.append({"layer": "L5", "family": "main", "system": "R0_raw_mistral", "metric": "m", "unit_id": u, "tp": 3, "fp": 0, "fn": 0})
        rows.append({"layer": "L5", "family": "other", "system": "variant", "metric": "m", "unit_id": u, "tp": 1, "fp": 2, "fn": 2,
                     "reference": "R0_raw_mistral"})
    p = tmp_path / "per_unit_scores.jsonl"
    _write(p, rows)
    tests, meta = G.ingest(p, 200, 200, 1)
    t = [x for x in tests if x["system"] == "variant"][0]
    assert t["reference"] == "R0_raw_mistral" and t["reference_rule"] == "declared" and t["delta"] < 0
