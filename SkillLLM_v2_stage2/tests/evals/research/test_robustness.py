"""Tests for L03 robustness statistics and L12 systems helpers (tiny synthetic fixtures)."""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from skillllm_v2.evals.extraction_evals import METRICS, N_COLS
from skillllm_v2.evals.research import robustness as R
from skillllm_v2.evals.research import systems as S


def span(s, e, label="HARD_SKILL", text=None):
    return {"surface_form": text[s:e] if text else "x", "label": label, "start": s, "end": e}


def test_count_rows_and_relabel():
    text = "Python and SQL"
    gold = {"d": [span(0, 6, "HARD_SKILL", text), span(11, 14, "SOFT_SKILL", text)]}
    preds = {"d": {"predictions": [span(0, 6, "SOFT_SKILL", text), span(11, 14, "SOFT_SKILL", text)]}}
    m = R.count_rows(["d"], gold, {"d": text}, preds)
    assert m[0][:3].tolist() == [1, 1, 1]  # label mismatch on Python
    m2 = R.count_rows(["d"], gold, {"d": text}, preds, relabel="SKILL")
    assert m2[0][:3].tolist() == [2, 0, 0]
    missing = R.count_rows(["d"], gold, {"d": text}, {})
    assert missing[0][:3].tolist() == [0, 0, 2]


def _rows(tp, fp, fn):
    r = np.zeros(N_COLS)
    r[0], r[1], r[2] = tp, fp, fn
    return r


def test_mean_degradation_zero_for_identical_and_known_value():
    base = ["a", "b", "c", "d"]
    fam = {"identity": {b: f"{b}::identity" for b in base}, "f1": {b: f"{b}::f1" for b in base},
           "f2": {b: f"{b}::f2" for b in base[:2]}}
    counts = {f"{b}::identity": _rows(2, 0, 0) for b in base}
    counts.update({f"{b}::f1": _rows(2, 0, 0) for b in base})
    counts.update({f"{b}::f2": _rows(2, 0, 0) for b in base[:2]})
    md = R.mean_degradation(base, fam, counts, ["f1", "f2"], "exact_span_f1", 200, 42)
    assert md["value"] == 0 and md["ci_low"] == 0 and md["ci_high"] == 0
    # f1 loses one TP per doc (tp=1, fn=1): F1 = 2/3 -> degradation 1/3; f2 unchanged -> mean 1/6
    counts.update({f"{b}::f1": _rows(1, 0, 1) for b in base})
    md = R.mean_degradation(base, fam, counts, ["f1", "f2"], "exact_span_f1", 200, 42)
    assert md["value"] == pytest.approx((1 - 2 / 3) / 2)
    assert md["ci_low"] <= md["value"] <= md["ci_high"]


def test_family_tables():
    rows = [{"id": "a::identity", "meta": {"base_id": "a", "family": "identity"}},
            {"id": "a::case_upper", "meta": {"base_id": "a", "family": "case_upper"}},
            {"id": "b::identity", "meta": {"base_id": "b", "family": "identity"}}]
    base, fam = R.family_tables(rows)
    assert base == ["a", "b"] and fam["case_upper"] == {"a": "a::case_upper"} and len(fam["identity"]) == 2


def test_slice_flags_and_boundary_subset():
    text = "Java / JavaScript and teamwork ( agile )"
    d = {"text": text, "gold_spans": [span(0, 4, "HARD_SKILL", text), span(7, 17, "HARD_SKILL", text),
                                      span(22, 30, "SOFT_SKILL", text), span(33, 38, "HARD_SKILL", text)]}
    f = R.slice_flags(d)
    assert f["coordination"] and f["parentheses"] and not f["nested_overlap"] and f["any_boundary_case"]
    sub = R.boundary_gold_subset(d)
    assert {s["surface_form"] for s in sub} >= {"Java", "JavaScript"}
    nested = {"text": "big data tools", "gold_spans": [span(0, 8, "HARD_SKILL", "big data tools"), span(0, 14, "SOFT_SKILL", "big data tools")]}
    assert R.slice_flags(nested)["nested_overlap"]
    plain = {"text": "Python skills", "gold_spans": [span(0, 6, "HARD_SKILL", "Python skills")]}
    assert not R.slice_flags(plain)["any_boundary_case"] and R.boundary_gold_subset(plain) == []


def test_doc_f1():
    assert R.doc_f1(_rows(0, 0, 0)) is None and R.doc_f1(_rows(1, 1, 0)) == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------- systems helpers
def test_legacy_offsets_reproduce_old_substring_behaviour():
    text = "javascript reactjs java"
    assert S.legacy_find_offsets(text, "java", set()) == (0, 4)  # old rule: first substring match (inside "javascript")
    preds = [{"surface_form": s, "label": "HARD_SKILL", "start": None, "end": None} for s in ("javascript", "reactjs", "java")]
    out = S.relocate_legacy(text, preds)
    assert [(p["start"], p["end"]) for p in out] == [(0, 10), (11, 18), (0, 4)]
    from skillllm_v2.schema import find_offsets  # current rule prefers word boundaries
    assert find_offsets(text, "java", set()) == (19, 23)


def test_output_signature_ignores_offsets():
    a = [{"surface_form": "SQL", "label": "HARD_SKILL", "start": 0, "end": 3, "metadata": {"raw_label": "HARD_SKILL"}}]
    b = [{"surface_form": "SQL", "label": "HARD_SKILL", "start": 5, "end": 8, "metadata": {"raw_label": "HARD_SKILL"}}]
    assert S.output_signature(a) == S.output_signature(b) and S.span_set(a) != S.span_set(b)


def test_count_params_from_safetensors_header(tmp_path):
    header = {"language_model.layers.0.w": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]},
              "vision_tower.x": {"dtype": "BF16", "shape": [3], "data_offsets": [64, 70]},
              "__metadata__": {"format": "pt"}}
    raw = json.dumps(header).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 70)
    c = S.count_params([p])
    assert c == {"language_model": 32, "vision_tower": 3, "total": 35}


def test_pct_and_boot_ci():
    v = list(range(1, 101))
    p = S.pct(v)
    assert p["p50"] == pytest.approx(50.5) and p["n"] == 100
    lo, hi = S.boot_pct_ci(v, 95, 200, 42)
    assert lo <= np.percentile(v, 95) <= hi
    assert S.pct([])["n"] == 0
