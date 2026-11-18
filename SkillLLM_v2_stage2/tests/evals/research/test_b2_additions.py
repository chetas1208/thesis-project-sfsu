"""Tests for B2 additions: fair-baseline dedupe in significance.py and the L04 base-rate supplement helpers."""

import json

import numpy as np
import pytest

from skillllm_v2.evals.research import calibration_selective as K
from skillllm_v2.evals.research import significance as SG


def _mk(root, rel, text):
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "predictions.jsonl").write_text(text)
    (d / "DONE.json").write_text("{}")


def test_fair_baselines_counts_identical_copy_once(tmp_path):
    base = tmp_path / SG.FAIR_GLOB
    _mk(base, "runs/a_s42", '{"id": "x"}\n')
    _mk(base, "runs/a_s43", '{"id": "y"}\n')
    _mk(base, ".", '{"id": "x"}\n')  # top-level copy of the dev-selected run
    out = SG.fair_baselines(tmp_path)
    assert sorted(out) == ["fair_runs_a_s42", "fair_runs_a_s43"]


def test_fair_baselines_keeps_toplevel_if_unique(tmp_path):
    base = tmp_path / SG.FAIR_GLOB
    _mk(base, ".", '{"id": "z"}\n')
    assert list(SG.fair_baselines(tmp_path)) == ["fair_"]


def test_base_rate_constant_has_low_ece_but_no_ranking():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.675).astype(float)
    m = K.metrics(np.full(2000, y.mean()), y, G=2500.0)
    assert m["ece_w10"] == pytest.approx(0.0, abs=1e-9)
    assert m["auroc"] == 0.5 and m["aurc"] == pytest.approx(1 - y.mean())
    assert m["sel_prec@50"] == pytest.approx(y.mean())
