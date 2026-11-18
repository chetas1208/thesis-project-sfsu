"""Tests for skillllm_v2.evals.research.stats (tiny synthetic fixtures, hand-checkable values)."""

import itertools
import math

import numpy as np
import pytest

from skillllm_v2.evals.research import stats as S


def _counts(seed=0, n=60, shift=0):
    rng = np.random.default_rng(seed)
    tp = rng.integers(0, 5, n) + shift
    fp = rng.integers(0, 3, n)
    fn = rng.integers(0, 3, n)
    return np.stack([tp, fp, fn], axis=1).astype(float)


def test_micro_f1_matches_harmonic_mean():
    s = np.array([7.0, 3.0, 5.0])
    p, r = 7 / 10, 7 / 12
    assert S.micro_f1(s) == pytest.approx(2 * p * r / (p + r))
    assert S.micro_precision(s) == pytest.approx(p)
    assert S.micro_recall(s) == pytest.approx(r)
    assert S.micro_f1(np.zeros(3)) == 0.0


def test_bootstrap_ci_matches_legacy_boot_ci():
    ev = pytest.importorskip("skillllm_v2.evals.extraction_evals")
    c = _counts()
    legacy = ev.boot_ci(c, lambda a: S.micro_f1(a), 500, 7)
    ours = S.bootstrap_ci(c, S.micro_f1, 500, 7)
    assert ours["ci_low"] == pytest.approx(legacy[0], abs=1e-12)
    assert ours["ci_high"] == pytest.approx(legacy[1], abs=1e-12)
    assert ours["ci_low"] <= ours["value"] <= ours["ci_high"]


def test_paired_bootstrap_identical_systems_p_is_one():
    c = _counts()
    r = S.paired_bootstrap(c, c, n_boot=300, seed=1)
    assert r["delta"] == 0.0 and r["p"] == 1.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0
    assert r["n_units_differ"] == 0


def test_paired_bootstrap_detects_clear_difference_and_is_seeded():
    a, b = _counts(1, shift=3), _counts(1)
    r1 = S.paired_bootstrap(a, b, n_boot=400, seed=3)
    r2 = S.paired_bootstrap(a, b, n_boot=400, seed=3)
    assert r1 == r2
    assert r1["delta"] > 0 and r1["ci_low"] > 0
    assert r1["p"] == pytest.approx(2 / 401)  # every replicate > 0 -> smallest attainable p


def test_paired_bootstrap_mean_matches_simple_difference():
    x, y = [1.0, 2.0, 3.0, 4.0], [0.5, 2.5, 2.0, 4.0]
    r = S.paired_bootstrap_mean(x, y, n_boot=200, seed=0)
    assert r["delta"] == pytest.approx(np.mean(x) - np.mean(y))
    assert r["n_units_differ"] == 3


def test_ar_exact_enumeration_hand_checked():
    # 3 units differ; statistic = mean of a single per-unit value -> sign-flip test, exact 2^3 enumeration.
    x = np.array([1.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    r = S.sign_flip_test(x, y)
    # observed mean diff 0.75; only all-unflipped and all-flipped reach |0.75| -> p = 2/8
    assert r["n_shuffles"] == 8 and r["p"] == pytest.approx(0.25)


def test_ar_monte_carlo_matches_exact_and_is_seeded():
    a, b = _counts(5, n=40), _counts(6, n=40)
    exact = S.approximate_randomization(a[:12], b[:12], max_exact_units=16)
    mc = S.approximate_randomization(a[:12], b[:12], n_shuffles=20000, seed=9, max_exact_units=0)
    assert "exact" in exact["method"] and "Monte Carlo" in mc["method"]
    assert mc["p"] == pytest.approx(exact["p"], abs=0.02)
    assert S.approximate_randomization(a, b, n_shuffles=500, seed=4) == S.approximate_randomization(a, b, n_shuffles=500, seed=4)


def test_ar_identical_is_one():
    c = _counts()
    assert S.approximate_randomization(c, c)["p"] == 1.0


def _holm_bruteforce(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1, (m - k) * p[i]))
        adj[i] = run
    return adj


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert S.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert S.holm(p) == pytest.approx(_holm_bruteforce(p))
    # BH: sorted 0.005,0.01,0.03,0.04 -> 0.02, 0.02, 0.04, 0.04
    assert S.benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert S.holm([0.2, None]) == [pytest.approx(0.2), None]
    assert S.benjamini_hochberg([]) == []


def test_bh_matches_scipy_if_available():
    sp = pytest.importorskip("scipy.stats")
    if not hasattr(sp, "false_discovery_control"):
        pytest.skip("old scipy")
    rng = np.random.default_rng(0)
    p = rng.random(25) ** 2
    assert S.benjamini_hochberg(p) == pytest.approx(list(sp.false_discovery_control(p, method="bh")))


def test_adjust_families():
    rows = [{"family": "A", "p": 0.01}, {"family": "A", "p": 0.04}, {"family": "B", "p": 0.03}, {"family": "B", "p": None}]
    S.adjust_families(rows)
    assert rows[0]["p_holm"] == pytest.approx(0.02) and rows[1]["p_holm"] == pytest.approx(0.04)
    assert rows[2]["p_holm"] == pytest.approx(0.03) and rows[3]["p_holm"] is None
    assert rows[2]["family_size"] == 1
    assert rows[0]["p_bh"] == pytest.approx(0.03)


def test_non_inferiority_and_mde():
    assert S.non_inferiority(-0.004, -0.005)["non_inferior"] is True
    assert S.non_inferiority(-0.006, -0.005)["non_inferior"] is False
    assert S.non_inferiority(None, -0.005)["non_inferior"] is False
    ni = S.non_inferiority(0.0, -0.005, samples=np.array([0.01, -0.01, 0.0, 0.002]))
    assert ni["p_one_sided"] == pytest.approx(2 / 5)
    assert S.mde(0.01) == pytest.approx((1.959964 + 0.841621) * 0.01, rel=1e-4)
    assert S.ci_excludes_zero(0.01, 0.02) and not S.ci_excludes_zero(-0.01, 0.02)


def test_wilcoxon_runs_or_reports_unavailable():
    r = S.wilcoxon([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0])
    if r["available"]:
        assert r["p"] == pytest.approx(0.03125)  # exact two-sided, all 6 positive
        assert S.wilcoxon([1, 2], [1, 2])["p"] == 1.0


def test_compare_counts_bundle():
    a, b = _counts(1, shift=1), _counts(1)
    r = S.compare_counts(a, b, n_boot=300, n_shuffles=300, seed=2, margin=-0.005)
    assert {"delta", "ci_low", "ci_high", "p", "p_ar", "mde_80", "non_inferiority"} <= set(r)
    assert r["non_inferiority"]["non_inferior"] is True
    assert math.isclose(r["delta"], float(S.micro_f1(a.sum(0)) - S.micro_f1(b.sum(0))))


def test_ratio_stat_apr_like():
    rows = np.array([[1, 2], [3, 3], [0, 0]], dtype=float)  # (kept, anchor_tp)
    assert S.ratio_stat(0, 1)(rows.sum(0)) == pytest.approx(4 / 5)


def test_bootstrap_sums_shared_draws_and_summary():
    a, b = _counts(1, shift=2), _counts(1)
    sa, sb = S.bootstrap_sums(a, b, n_boot=300, seed=5)
    assert sa.shape == (300, 3) and sb.shape == (300, 3)
    d = S.micro_f1(sa) - S.micro_f1(sb)
    obs = float(S.micro_f1(a.sum(0)) - S.micro_f1(b.sum(0)))
    summ = S.summarize_samples(d, obs, paired=True)
    assert summ["ci_low"] > 0 and summ["p"] == pytest.approx(2 / 301)
    # identical inputs -> identical resampled sums (shared draws)
    s1, s2 = S.bootstrap_sums(a, a, n_boot=50, seed=1)
    assert np.array_equal(s1, s2)
