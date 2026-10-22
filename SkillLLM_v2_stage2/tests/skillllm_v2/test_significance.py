from __future__ import annotations

import numpy as np
import pytest

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.eval.significance import (
    METRIC_ORDER,
    count_matrix,
    doc_level_counts,
    holm,
    minimal_detectable_effect,
    paired_bootstrap,
    pooled_f1,
    render_markdown,
    sign_flip_permutation,
)


def _span(start, end, label="HARD_SKILL"):
    return {"start": start, "end": end, "label": label, "surface_form": "x"}


def _synthetic(n_docs=200, seed=0):
    rng = np.random.default_rng(seed)
    gold, base, better = [], [], []
    for _ in range(n_docs):
        g = [_span(0, 5), _span(10, 15, "SOFT_SKILL"), _span(20, 28)]
        gold.append(g)
        # baseline finds each gold span w.p. 0.5 and adds a spurious one sometimes
        b = [s for s in g if rng.random() < 0.5]
        if rng.random() < 0.3:
            b.append(_span(40, 45))
        base.append(b)
        # "better" system recovers every gold span baseline missed with prob 0.9
        better.append(b + [s for s in g if s not in b and rng.random() < 0.9])
    return gold, base, better


def test_counts_reproduce_extraction_eval_exactly():
    gold, base, _ = _synthetic()
    # add partial-overlap + token-only cases
    gold[0] = [_span(0, 10), _span(12, 20, "SOFT_SKILL")]
    base[0] = [_span(0, 8), _span(12, 20, "SOFT_SKILL"), _span(30, 33)]
    cm = count_matrix(gold, base)
    ref = evaluate_extraction(gold, base)
    for m in METRIC_ORDER:
        assert pooled_f1(cm, m) == pytest.approx(ref[m], abs=1e-12)


def test_identical_systems_p_near_one():
    gold, base, _ = _synthetic()
    cm = count_matrix(gold, base)
    boot = paired_bootstrap(cm, cm.copy(), METRIC_ORDER, 2000, seed=1)
    for m in METRIC_ORDER:
        assert boot[m]["observed_delta"] == 0.0
        assert boot[m]["p_two_sided"] == pytest.approx(1.0)
        assert boot[m]["ci95_low"] == 0.0 and boot[m]["ci95_high"] == 0.0
        perm = sign_flip_permutation(cm, cm.copy(), m, seed=1)
        assert perm["p_two_sided"] == 1.0
    d = doc_level_counts(cm, cm.copy())
    assert d["doc_f1_wins"] == d["doc_f1_losses"] == 0 and d["doc_f1_ties"] == len(gold)


def test_clearly_better_system_is_significant():
    gold, base, better = _synthetic()
    a, b = count_matrix(gold, base), count_matrix(gold, better)
    boot = paired_bootstrap(a, b, ["exact_span_f1", "token_f1"], 3000, seed=3)
    assert boot["exact_span_f1"]["observed_delta"] > 0.1
    assert boot["exact_span_f1"]["p_two_sided"] < 0.01
    assert boot["exact_span_f1"]["ci95_low"] > 0
    perm = sign_flip_permutation(a, b, "exact_span_f1", seed=3, mc_samples=20_000)
    assert perm["p_two_sided"] < 0.01
    d = doc_level_counts(a, b)
    assert d["doc_f1_wins"] > 10 * max(1, d["doc_f1_losses"])
    assert d["doc_f1_sign_test_p"] < 0.01


def test_bootstrap_is_seeded_and_reproducible():
    gold, base, better = _synthetic(60)
    a, b = count_matrix(gold, base), count_matrix(gold, better)
    r1 = paired_bootstrap(a, b, ["exact_span_f1"], 1000, seed=7)
    r2 = paired_bootstrap(a, b, ["exact_span_f1"], 1000, seed=7)
    assert r1 == r2


def test_exact_permutation_matches_hand_computation():
    # 3 docs, one gold span each. Variant gets doc0 right (anchor wrong); the rest identical.
    gold = [[_span(0, 5)], [_span(0, 5)], [_span(0, 5)]]
    anchor = [[_span(1, 5)], [_span(0, 5)], [_span(0, 5)]]
    variant = [[_span(0, 5)], [_span(0, 5)], [_span(0, 5)]]
    a, b = count_matrix(gold, anchor), count_matrix(gold, variant)
    perm = sign_flip_permutation(a, b, "exact_span_f1", seed=0)
    # one discordant doc -> 2 swaps, both give |delta| == |obs| -> p = 1
    assert perm["n_discordant_docs"] == 1 and perm["p_two_sided"] == 1.0
    # 8 docs all improved: p = 2 / 2^8
    gold = [[_span(0, 5)]] * 8
    a = count_matrix(gold, [[_span(1, 5)]] * 8)
    b = count_matrix(gold, [[_span(0, 5)]] * 8)
    perm = sign_flip_permutation(a, b, "exact_span_f1", seed=0)
    assert perm["method"] == "exact enumeration"
    assert perm["p_two_sided"] == pytest.approx(2 / 2 ** 8)


def test_holm_and_mde():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert minimal_detectable_effect(0.001) == pytest.approx(0.0028, abs=1e-4)


def test_markdown_renders_with_verdicts(tmp_path):
    from skillllm_v2.eval import significance as sig

    gold, base, better = _synthetic(80)
    a, b = count_matrix(gold, base), count_matrix(gold, better)
    boot = paired_bootstrap(a, b, METRIC_ORDER, 1000, seed=5)
    metrics = {}
    for m in METRIC_ORDER:
        perm = sign_flip_permutation(a, b, m, seed=5, mc_samples=2000)
        boot[m]["p_holm_all"] = boot[m]["p_two_sided"]
        perm["p_holm_all"] = perm["p_two_sided"]
        metrics[m] = {
            "anchor_f1": pooled_f1(a, m), "variant_f1": pooled_f1(b, m), "bootstrap": boot[m],
            "permutation": perm, "significant_after_holm": True, "bootstrap_permutation_agree": True,
            "mde_alpha05_power80": 0.001, "mde_holm_all_alpha_power80": 0.002,
        }
    res = {
        "config": {"n_docs": 80, "n_resamples": 1000, "seed": 5, "alpha": 0.05, "n_holm_tests": 5,
                   "reproduction_tolerance": 1e-6, "inputs": {}},
        "variants": {"full_cag_graph": {"metrics": metrics, "doc_level": doc_level_counts(a, b),
                                        "verdict": "ok"}},
    }
    text = sig.render_markdown(res)
    assert "Can the thesis claim" in text and "full_cag_graph" in text
