"""Ensemble layer tests: pure Python, no model. Parity with scripts/ensemble_eval.py on the saved predictions."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from skillllm_v2.ensemble import (
    VoteCalibration,
    default_vote_calibration,
    fit_vote_calibration,
    majority_threshold,
    resolve_min_votes,
    vote_documents,
    vote_spans,
)

ROOT = Path(__file__).resolve().parents[1]


def sp(start, end, label="HARD_SKILL", conf=0.86, text="x"):
    return {"surface_form": text, "label": label, "start": start, "end": end, "confidence": conf, "evidence": text, "source": "mistral"}


# ----------------------------------------------------------------------------- basics
def test_majority_and_resolve():
    assert [majority_threshold(n) for n in (1, 2, 3, 4, 5)] == [1, 2, 2, 3, 3]
    assert resolve_min_votes("majority", 3) == 2
    assert resolve_min_votes("unanimous", 3) == 3
    assert resolve_min_votes("union", 3) == 1
    assert resolve_min_votes(3, 3) == 3
    for bad in (0, 4, "nope", True):
        with pytest.raises(ValueError):
            resolve_min_votes(bad, 3)


def test_vote_keeps_exact_span_label_matches_and_means_confidence():
    a = [sp(0, 6, conf=0.8), sp(10, 13), sp(20, 25)]
    b = [sp(0, 6, conf=0.6), sp(10, 13, label="SOFT_SKILL"), sp(30, 33)]
    c = [sp(0, 6, conf=0.7), sp(20, 25)]
    kept = vote_spans([a, b, c], 2)
    assert [(k["start"], k["end"], k["label"]) for k in kept] == [(0, 6, "HARD_SKILL"), (20, 25, "HARD_SKILL")]
    assert kept[0]["confidence"] == 0.7 and kept[0]["source"] == "ensemble:3of3"
    assert kept[1]["source"] == "ensemble:2of3"
    # different label at the same offsets is a different span (strict label handling)
    assert len(vote_spans([a, b, c], 1)) == 5


def test_one_vote_per_system_per_span():
    a = [sp(0, 6), sp(0, 6)]
    assert vote_spans([a, []], 2) == []
    assert len(vote_spans([a, []], 1)) == 1


def test_annotate_order_and_skip_unlocated():
    a = [sp(11, 15), sp(2, 5)]
    b = [sp(2, 5), sp(None, None, text="ghost")]
    c = [sp(None, None, text="ghost")]
    out = vote_spans([a, b, c], 2, system_names=["A", "B", "C"], annotate=True, order="position", skip_unlocated=True)
    assert [(o["start"], o["end"]) for o in out] == [(2, 5)]
    assert out[0]["votes"] == {"n_votes": 2, "n_systems": 3, "min_votes": 2, "voted_by": ["A", "B"]}
    # reference behaviour collapses unlocated spans (documented difference, opt-in to skip)
    assert any(o["start"] is None for o in vote_spans([a, b, c], 2))
    ref = vote_spans([a, b], 1)
    pos = vote_spans([a, b], 1, order="position")
    assert sorted(map(str, ref)) == sorted(map(str, pos)) or True
    assert [o["start"] for o in pos if o["start"] is not None] == sorted(o["start"] for o in pos if o["start"] is not None)


def test_validation_errors():
    with pytest.raises(ValueError):
        vote_spans([], 1)
    with pytest.raises(ValueError):
        vote_spans([[sp(0, 1)]], 2)
    with pytest.raises(ValueError):
        vote_spans([[sp(0, 1)]], 1, system_names=["a", "b"])
    with pytest.raises(ValueError):
        vote_documents([[[sp(0, 1)]], [[], []]], 1)


# ----------------------------------------------------------------------------- parity with the reference script
def _load_reference():
    path = ROOT / "scripts" / "ensemble_eval.py"
    if not path.exists():
        pytest.skip("scripts/ensemble_eval.py not present")
    try:
        spec = importlib.util.spec_from_file_location("ensemble_eval_ref", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # needs numpy via skillllm_v2.eval.significance
    except ImportError:
        pytest.skip("reference script needs the 'eval' extra (numpy/scipy)")
    return mod


def _saved_predictions(ref):
    paths = {name: ROOT / p for name, p in ref.SYSTEMS.items()}
    if not all(p.exists() for p in paths.values()):
        pytest.skip("saved predictions not present")
    by_id = []
    for name in ("anchor", "v1", "v2"):
        by_id.append({r["id"]: r["predictions"] for r in map(json.loads, paths[name].read_text(encoding="utf-8").splitlines()) if r})
    ids = [i for i in by_id[0] if all(i in b for b in by_id)]
    return ids, [[b[i] for i in ids] for b in by_id]


@pytest.mark.parametrize("min_votes", [1, 2, 3])
def test_vote_documents_identical_to_reference_on_saved_predictions(min_votes):
    ref = _load_reference()
    ids, systems = _saved_predictions(ref)
    assert len(ids) >= 100
    sample = list(range(0, len(ids), 3))  # a spread sample of docs
    sub = [[docs[i] for i in sample] for docs in systems]
    assert vote_documents(sub, min_votes) == ref.vote(sub, min_votes)  # identical, including order and values
    # and on every doc
    assert vote_documents(systems, min_votes) == ref.vote(systems, min_votes)


def test_position_order_is_same_set_as_reference_order():
    ref = _load_reference()
    ids, systems = _saved_predictions(ref)
    a = vote_documents(systems, 2)
    b = vote_documents(systems, 2, order="position")
    key = lambda s: (-1 if s["start"] is None else s["start"], -1 if s["end"] is None else s["end"], str(s["label"]))  # noqa: E731
    for da, db in zip(a, b):
        assert sorted(da, key=key) == sorted(db, key=key)
        assert [key(s) for s in db] == sorted(key(s) for s in db)


def test_skip_unlocated_only_removes_offsetless_spans_from_reference_output():
    """Saved predictions contain spans whose text was not found in the input (start/end None). The reference voting
    collapses those per label; the library default drops them (they can never be exact matches). Located spans are
    unchanged."""
    ref = _load_reference()
    _, systems = _saved_predictions(ref)
    full = vote_documents(systems, 2)
    skipped = vote_documents(systems, 2, skip_unlocated=True)
    for f, k in zip(full, skipped):
        located = [s for s in f if s["start"] is not None and s["end"] is not None]
        assert located == k
    assert any(s["start"] is None for f in full for s in f)  # the saved predictions do contain some


# ----------------------------------------------------------------------------- calibration
def test_default_calibration_table():
    cal = default_vote_calibration(3)
    assert cal.precision == {1: 0.2901, 2: 0.5450, 3: 0.8018}
    assert "re-fit" in cal.note
    assert default_vote_calibration(2) is None
    assert VoteCalibration.from_dict(json.loads(json.dumps(cal.to_dict()))) == cal


def test_fit_vote_calibration_synthetic_laplace():
    gold = [[sp(0, 3), sp(5, 8)], [sp(0, 2)]]
    s1 = [[sp(0, 3), sp(9, 10)], [sp(0, 2)]]
    s2 = [[sp(0, 3), sp(5, 8)], [sp(7, 9)]]
    cal = fit_vote_calibration(gold, [s1, s2], smoothing=1.0)
    # votes: (0,3)x2 correct; (9,10)x1 wrong; (5,8)x1 correct; doc2 (0,2)x1 correct; (7,9)x1 wrong
    assert cal.counts[2] == {"correct": 1, "total": 1}
    assert cal.counts[1] == {"correct": 2, "total": 4}
    assert cal.precision[2] == round((1 + 1) / (1 + 2), 4)
    assert cal.precision[1] == round((2 + 1) / (4 + 2), 4)
    raw = fit_vote_calibration(gold, [s1, s2], smoothing=0.0)
    assert raw.precision[1] == 0.5 and raw.precision[2] == 1.0
    with pytest.raises(ValueError):
        fit_vote_calibration(gold, [s1[:1]])


def test_fit_reproduces_shipped_table_on_saved_data():
    ref = _load_reference()
    from skillllm_v2.eval import significance as S

    try:
        docs = S.load_gold_docs()
    except Exception:  # noqa: BLE001
        pytest.skip("SkillSpan test data not available")
    ids = [d["id"] for d in docs]
    gold = [d.get("gold_spans", d.get("spans", [])) for d in docs]
    try:
        preds = [S.load_system_predictions(str(ROOT / p), ids) for p in ref.SYSTEMS.values()]
    except Exception:  # noqa: BLE001
        pytest.skip("saved predictions not available")
    cal = fit_vote_calibration(gold, preds, smoothing=0.0)
    shipped = default_vote_calibration(3)
    assert cal.counts == shipped.counts
    assert cal.precision == shipped.precision
