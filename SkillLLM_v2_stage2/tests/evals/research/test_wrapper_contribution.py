"""Tests for research/wrapper_contribution.py (synthetic spans; real wrapper components where cheap)."""

import numpy as np
import pytest

from skillllm_v2.evals.research import wrapper_contribution as W
from skillllm_v2.evals.research.common import REPO_ROOT


def sp(s, e, label="HARD_SKILL", source="mistral", conf=0.86, text=None, **meta):
    return {"surface_form": text or f"x{s}", "label": label, "start": s, "end": e, "source": source, "confidence": conf,
            "evidence": text or f"x{s}", "canonical_skill_id": None, "canonical_skill_name": None, "metadata": dict(meta)}


def test_doc_row_accounting_and_apr_nwg():
    gold = [sp(0, 4), sp(5, 9), sp(10, 14, "SOFT_SKILL"), sp(20, 24)]
    anchor = [sp(0, 4), sp(5, 9), sp(10, 14, "SOFT_SKILL"), sp(30, 34)]  # TP x3, FP x1
    stage = [sp(0, 4), sp(20, 24), sp(30, 34), sp(40, 44)]  # keeps 1 anchor TP, gains 1, destroys 2 (one soft), adds 1 FP
    r = W.doc_row(gold, stage, anchor, " " * 50)
    C = W.C
    assert (r[C["anchor_tp"]], r[C["kept_tp"]], r[C["gained_tp"]], r[C["destroyed_tp"]]) == (3, 1, 1, 2)
    assert (r[C["anchor_fp"]], r[C["added_fp"]], r[C["removed_fp"]]) == (1, 1, 0)
    assert (r[C["soft_anchor_tp"]], r[C["soft_kept_tp"]], r[C["soft_gold"]]) == (1, 0, 1)
    assert r[C["changed"]] == 1
    s = r[None, :].sum(0)
    assert W.STATS["apr"](s) == pytest.approx(1 / 3)
    # NWG = 1000 * ((1 - 2) - (1 - 0)) / 4 gold = -500
    assert W.STATS["nwg_per_1000_gold"](s) == pytest.approx(-500.0)
    # identity stage -> APR 1, NWG 0, nothing changed
    r0 = W.doc_row(gold, anchor, anchor, " " * 50)
    assert W.STATS["apr"](r0) == 1.0 and W.STATS["nwg_per_1000_gold"](r0) == 0.0 and r0[C["changed"]] == 0


def test_gained_equals_recovered_fn():
    gold = [sp(0, 4), sp(5, 9)]
    anchor = [sp(0, 4)]
    stage = [sp(0, 4), sp(5, 9)]
    r = W.doc_row(gold, stage, anchor, " " * 20)
    assert r[W.C["gained_tp"]] == 1 and r[W.C["fn"]] == 0


@pytest.fixture()
def in_repo(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


def test_policy_toggles_verifier_and_merger(in_repo):
    text = "We need experience and Python and SQL"
    anchor = [sp(8, 18, text="experience")]  # generic phrase predicted by the anchor
    safe = W.build_components()
    off = W.build_components(policy_off=("verifier", "merger"))
    gk = {(8, 18, "HARD_SKILL")}
    out_safe, d_safe = W.run_chain(text, anchor, [], [], safe, {"verify"}, gk)
    out_off, d_off = W.run_chain(text, anchor, [], [], off, {"verify"}, gk)
    assert len(out_safe) == 1 and d_safe.get("verifier_rejected", 0) == 0  # anchor protected
    assert len(out_off) == 0 and d_off["verifier_rejected"] == 1 and d_off["verifier_false_rej"] == 1
    # a GLiNER candidate overlapping an anchor span: dropped by the add-only merger, kept without the policy
    anchor2 = [sp(23, 29, text="Python")]
    gl = [sp(23, 33, source="gliner", conf=0.99, text="Python and")]
    m_safe, _ = W.run_chain(text, anchor2, gl, [], safe, {"gliner"}, set())
    m_off, _ = W.run_chain(text, anchor2, gl, [], off, {"gliner"}, set())
    assert {(c["start"], c["end"]) for c in m_safe} == {(23, 29)}
    assert (23, 33) in {(c["start"], c["end"]) for c in m_off}


def test_linking_is_span_neutral(in_repo):
    text = "Python and SQL"
    anchor = [sp(0, 6, text="Python"), sp(11, 14, text="SQL")]
    out, _ = W.run_chain(text, anchor, [], [], W.build_components(), {"link"}, set())
    assert [(c["start"], c["end"], c["label"]) for c in out] == [(0, 6, "HARD_SKILL"), (11, 14, "HARD_SKILL")]


def test_destructive_selector_config():
    cfg = W.destructive_selector_config(W.PREFIX_LIKE_EDITS)
    assert set(cfg["allowed_edit_types"]) == set(W.PREFIX_LIKE_EDITS)
    assert cfg["edit_type_scores"]["original"] == 0.0 and cfg["edit_type_scores"]["split_conjunction"] == 1.0
    assert cfg["edit_type_scores"]["head_skill"] < 0
    full = W.destructive_selector_config()
    assert "head_skill" in full["allowed_edit_types"]


def test_ladder_definition_is_cumulative():
    prev = set()
    for _, enabled, _ in W.LADDER:
        assert prev <= enabled
        prev = enabled
    assert W.FULL == {"refine", "gliner", "dictionary", "link", "verify", "calibrate"}
    assert all("link" not in e for _, e, _ in W.LOO)
