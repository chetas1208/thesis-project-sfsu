"""Agreement statistics against published worked examples and hand-computed values."""

import math

import numpy as np
import pytest

from skillllm_v2.evals.research import agreement as ag

N = None
# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability", section C: 4 observers x 12 units with missing values.
K2011 = [[1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
         [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
         [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
         [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N]]
# Reliability-data example used by the Wikipedia article and the `krippendorff` package docstring (3 coders x 15 units).
WIKI = [[N, N, N, N, N, 3, 4, 1, 2, 1, 1, 3, 3, N, 3],
        [1, N, 2, 1, 3, 3, 4, 3, N, N, N, N, N, N, N],
        [N, N, 2, 1, 3, 4, 4, N, 2, 1, 1, 3, 3, N, 4]]


@pytest.mark.parametrize("level,published", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849), ("ratio", 0.797)])
def test_krippendorff_2011_published_values(level, published):
    # published to 3 decimals; full-precision values confirmed with the `krippendorff` package 0.8.1 (A6 throwaway venv)
    assert round(ag.krippendorff_alpha(K2011, level), 3) == published


@pytest.mark.parametrize("level,expected", [("nominal", 0.743421), ("ordinal", 0.815388), ("interval", 0.849107),
                                            ("ratio", 0.797403)])
def test_krippendorff_2011_package_precision(level, expected):
    assert ag.krippendorff_alpha(K2011, level) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("level,expected", [("nominal", 0.691358), ("interval", 0.810845), ("ordinal", 0.806721),
                                            ("ratio", 0.808944)])
def test_krippendorff_wikipedia_example(level, expected):
    assert ag.krippendorff_alpha(WIKI, level) == pytest.approx(expected, abs=1e-6)


def test_krippendorff_optional_package_crosscheck():
    kr = pytest.importorskip("krippendorff")
    rng = np.random.default_rng(3)
    m = rng.integers(1, 6, size=(3, 30)).astype(float)
    m[rng.random(m.shape) < 0.2] = np.nan
    for lvl in ag.LEVELS:
        ours = ag.krippendorff_alpha([[None if math.isnan(v) else v for v in r] for r in m.tolist()], lvl)
        assert ours == pytest.approx(kr.alpha(reliability_data=m, level_of_measurement=lvl), abs=1e-9)


def test_alpha_undefined_and_perfect():
    assert math.isnan(ag.krippendorff_alpha([[1, 1, 1], [1, 1, 1]]))
    assert ag.krippendorff_alpha([[1, 2, 3, 1], [1, 2, 3, 1]]) == pytest.approx(1.0)
    # units with a single rating are not pairable and do not change alpha
    assert ag.krippendorff_alpha([[1, 2, 3, 1, 2], [1, 2, 3, 1, N]]) == pytest.approx(ag.krippendorff_alpha([[1, 2, 3, 1], [1, 2, 3, 1]]))


def test_cohen_kappa_textbook_example():
    # 50 items: yes/yes 20, A-yes/B-no 5, A-no/B-yes 10, no/no 15 -> p_o 0.70, p_e 0.50, kappa 0.40 (standard worked example)
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert ag.cohen_kappa(a, b) == pytest.approx(0.4)
    assert ag.percent_agreement(a, b) == pytest.approx(0.7)


def test_cohen_kappa_missing_pairwise_deletion():
    a = [1, 2, 3, None, 2]
    b = [1, 2, 3, 3, float("nan")]
    assert ag.cohen_kappa(a, b) == pytest.approx(1.0)


def test_weighted_kappa_against_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    a = rng.integers(0, 4, 80).tolist()
    b = [x if rng.random() < 0.55 else int(rng.integers(0, 4)) for x in a]
    for w in (None, "linear", "quadratic"):
        assert ag.cohen_kappa(a, b, [0, 1, 2, 3], w) == pytest.approx(skm.cohen_kappa_score(a, b, weights=w, labels=[0, 1, 2, 3]), abs=1e-12)


def test_weighted_kappa_rewards_near_misses():
    a = [0, 1, 2, 3, 0, 1, 2, 3]
    b = [1, 2, 3, 3, 0, 0, 1, 2]
    assert ag.cohen_kappa(a, b, [0, 1, 2, 3], "quadratic") > ag.cohen_kappa(a, b, [0, 1, 2, 3])


def test_gwet_ac1_hand_computed():
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    # pi_y = (25 + 30) / 100 = 0.55; p_e = 2 * 0.55 * 0.45 / 1 = 0.495; AC1 = (0.7 - 0.495) / 0.505
    assert ag.gwet_ac1(a, b) == pytest.approx((0.7 - 0.495) / 0.505)


def test_span_agreement_exact_and_overlap():
    a = {"d1": [(0, 6, "HARD"), (10, 20, "SOFT")], "d2": [(0, 4, "HARD")]}
    b = {"d1": [(0, 6, "HARD"), (12, 20, "SOFT")], "d2": [(0, 4, "SOFT")]}
    ex = ag.span_agreement(a, b, "exact")
    assert (ex["matches"], ex["n_a"], ex["n_b"]) == (1, 3, 3)
    assert ex["f1"] == pytest.approx(2 * 1 / 6)
    ov = ag.span_agreement(a, b, "overlap")
    assert ov["f1"] == pytest.approx(2 * 2 / 6)
    nolab = ag.span_agreement(a, b, "exact", use_label=False)
    assert nolab["f1"] == pytest.approx(2 * 2 / 6)
    # symmetric
    assert ag.span_agreement(b, a, "overlap")["f1"] == pytest.approx(ov["f1"])


def test_span_overlap_is_one_to_one():
    a = {"d": [(0, 10, "H")]}
    b = {"d": [(0, 4, "H"), (5, 10, "H")]}
    r = ag.span_agreement(a, b, "overlap")
    assert r["matches"] == 1 and r["f1"] == pytest.approx(2 / 3)


def test_span_token_kappa_identical_is_one():
    toks = {"d": [(0, 3), (4, 8), (9, 12)]}
    a = {"d": [(4, 8, "H")]}
    assert ag.span_token_kappa(toks, a, a) == pytest.approx(1.0)


def test_bootstrap_ci_contains_estimate_and_is_seeded():
    a = [0, 1, 1, 0, 1, 1, 0, 1, 0, 1] * 5
    b = [0, 1, 0, 0, 1, 1, 1, 1, 0, 1] * 5
    r1 = ag.kappa_with_ci(a, b, n_boot=500, seed=42)
    r2 = ag.kappa_with_ci(a, b, n_boot=500, seed=42)
    assert r1["ci"] == r2["ci"]
    assert r1["ci"][0] <= r1["value"] <= r1["ci"][1]
    al = ag.alpha_with_ci([a, b], "nominal", n_boot=300)
    assert al["ci"][0] <= al["value"] <= al["ci"][1]
    sp = ag.span_f1_with_ci({"x": [(0, 1, "H")], "y": [(2, 3, "H")]}, {"x": [(0, 1, "H")], "y": [(2, 4, "H")]}, n_boot=200)
    assert sp["value"] == pytest.approx(0.5)


def test_sample_size_helpers():
    assert ag.n_for_proportion(0.05) == 385
    assert ag.n_for_proportion(0.10) == 97
    assert ag.n_for_mean(0.15, 1.0) == 171
    po, pe = ag.expected_kappa_po_pe(0.6, 0.5)
    assert (po, pe) == pytest.approx((0.8, 0.5))
    assert ag.n_for_kappa(0.1, po, pe) == math.ceil(ag.Z95 ** 2 * 0.8 * 0.2 / (0.01 * 0.25))
    hw = ag.simulate_kappa_halfwidth(200, 0.6, 0.5, n_sim=400)
    assert 0.05 < hw < 0.2
