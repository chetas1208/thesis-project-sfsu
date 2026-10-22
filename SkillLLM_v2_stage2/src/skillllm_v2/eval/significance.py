"""Paired significance testing for extraction F1 deltas (anchor vs. wrapper variants).

Resampling unit is the DOCUMENT. For every system we first reduce each document to
additive integer counts (TP/FP/FN for exact span, token, partial and per-label F1)
using ``extraction_eval``'s own span matching helpers. Micro-F1 of any resample is
then recomputed from the POOLED counts (never an average of per-document F1).

Before any test runs, the pooled-count F1s are asserted to reproduce
``evaluate_extraction`` (and the F1s stored in each artifact's metrics.json) to 1e-6.

Tests per (variant, metric):
  * paired bootstrap over documents (seeded, >= 10,000 resamples): delta mean, 95%
    percentile CI, two-sided p = 2*min(P*(d<=0), P*(d>=0)) with +1 smoothing.
  * paired sign-flip permutation test (exact enumeration over discordant documents
    when feasible, otherwise seeded Monte Carlo) on the pooled-F1 difference.
  * doc-level win/loss/tie (doc exact-F1) with exact binomial sign test, and a
    McNemar test on "document perfectly extracted".
Holm correction is applied across all variants x metrics (and, separately, across
variants within each metric).

Usage:
    PYTHONPATH=src python -m skillllm_v2.eval.significance \
        --output-json reports/skillllm_v2/significance_f1.json \
        --output-md reports/skillllm_v2/significance_f1.md
"""
from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from skillllm_v2.eval.extraction_eval import (
    _key,
    _partial_matches,
    _prf,
    _tokens,
    evaluate_extraction,
)
from skillllm_v2.io_utils import read_json, read_jsonl, write_json

ANCHOR_PREDICTIONS = "artifacts/skillllm_v2/mistral_lora/predictions.jsonl"
VARIANT_PREDICTIONS = {
    "boundary_refiner": "artifacts/skillllm_v2/safe_wrapper/boundary_refiner/candidate_predictions.jsonl",
    "mistral_gliner_taxonomy": "artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/candidate_predictions.jsonl",
    "full_cag_graph": "artifacts/skillllm_v2/safe_wrapper/full_cag_graph/candidate_predictions.jsonl",
}
VARIANT_METRICS_JSON = {
    "boundary_refiner": "artifacts/skillllm_v2/safe_wrapper/boundary_refiner/metrics.json",
    "mistral_gliner_taxonomy": "artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/metrics.json",
    "full_cag_graph": "artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json",
}
ANCHOR_METRICS_JSON = "artifacts/skillllm_v2/mistral_lora/metrics.json"
DATASET_ROOT = "data/source/skillspan_hf"

# metric name -> (reported key in evaluate_extraction, column triple in the count matrix)
METRICS = {
    "exact_span_f1": "exact_span_f1",
    "token_f1": "token_f1",
    "partial_span_f1": "partial_span_f1",
    "hard_skill_f1": "hard_skill_f1",
    "soft_skill_f1": "soft_skill_f1",
}
METRIC_ORDER = list(METRICS)
COUNT_COLUMNS = {  # metric -> (tp_col, fp_col, fn_col) in the per-doc count matrix
    "exact_span_f1": (0, 1, 2),
    "token_f1": (3, 4, 5),
    "partial_span_f1": (6, 7, 8),
    "hard_skill_f1": (9, 10, 11),
    "soft_skill_f1": (12, 13, 14),
}
N_COLS = 15
PRIMARY_METRIC = "exact_span_f1"
REPRO_TOL = 1e-6


# --------------------------------------------------------------------------- counts
def doc_counts(gold_doc: list[dict[str, Any]], pred_doc: list[dict[str, Any]]) -> np.ndarray:
    """Additive per-document counts, using extraction_eval's own matching helpers.

    Mirrors ``evaluate_extraction`` line for line; the partial-F1 columns follow its
    ``_prf(partial_tp, fp + tp - partial_tp, fn + tp - partial_tp)`` convention.
    """
    gold_set = {_key(i) for i in gold_doc}
    pred_set = {_key(i) for i in pred_doc}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    ptp = _partial_matches(gold_doc, pred_doc)
    gt, pt = _tokens(gold_doc), _tokens(pred_doc)
    row = [tp, fp, fn, len(gt & pt), len(pt - gt), len(gt - pt), ptp, fp + tp - ptp, fn + tp - ptp]
    for label in ("HARD_SKILL", "SOFT_SKILL"):
        lg = {_key(i) for i in gold_doc if i.get("label") == label}
        lp = {_key(i) for i in pred_doc if i.get("label") == label}
        row += [len(lg & lp), len(lp - lg), len(lg - lp)]
    return np.array(row, dtype=np.int64)


def count_matrix(gold: list[list[dict]], pred: list[list[dict]]) -> np.ndarray:
    return np.stack([doc_counts(g, p) for g, p in zip(gold, pred)])


def f1_from_sums(sums: np.ndarray, metric: str) -> np.ndarray | float:
    """Micro-F1 from pooled counts. ``sums`` has shape (..., N_COLS). Equals ``_prf``."""
    tp_c, fp_c, fn_c = COUNT_COLUMNS[metric]
    tp, fp, fn = sums[..., tp_c], sums[..., fp_c], sums[..., fn_c]
    denom = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denom, out=np.zeros(np.shape(denom), dtype=float), where=denom > 0)


def pooled_f1(counts: np.ndarray, metric: str) -> float:
    s = counts.sum(axis=0)
    tp_c, fp_c, fn_c = COUNT_COLUMNS[metric]
    return _prf(int(s[tp_c]), int(s[fp_c]), int(s[fn_c]))[2]


def assert_reproduction(
    name: str,
    gold: list[list[dict]],
    pred: list[list[dict]],
    counts: np.ndarray,
    stored_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    """Pooled counts must reproduce evaluate_extraction (and stored metrics.json) to REPRO_TOL."""
    ref = evaluate_extraction(gold, pred)
    rows = {}
    for metric in METRIC_ORDER:
        mine = pooled_f1(counts, metric)
        diff_eval = abs(mine - ref[metric])
        if diff_eval > REPRO_TOL:
            raise AssertionError(f"{name}: {metric} pooled-count F1 {mine:.8f} != evaluate_extraction {ref[metric]:.8f}")
        row = {"recomputed": mine, "evaluate_extraction": ref[metric], "abs_diff_vs_eval": diff_eval}
        if stored_metrics is not None and metric in stored_metrics:
            diff_stored = abs(mine - float(stored_metrics[metric]))
            if diff_stored > REPRO_TOL:
                raise AssertionError(
                    f"{name}: {metric} recomputed {mine:.8f} does not reproduce stored artifact value "
                    f"{float(stored_metrics[metric]):.8f} (|diff|={diff_stored:.2e}); cannot test reported numbers"
                )
            row["stored"] = float(stored_metrics[metric])
            row["abs_diff_vs_stored"] = diff_stored
        rows[metric] = row
    return rows


# --------------------------------------------------------------------------- tests
def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, metrics: list[str], n_resamples: int, seed: int, chunk: int = 1000
) -> dict[str, dict[str, Any]]:
    """Paired document bootstrap of delta = F1(b) - F1(a) (b = variant, a = anchor).

    The same document resample is applied to both systems. Each resample's F1 is
    recomputed from the pooled TP/FP/FN of the resampled documents.
    """
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    deltas = {m: np.empty(n_resamples) for m in metrics}
    f1a_s = {m: np.empty(n_resamples) for m in metrics}
    f1b_s = {m: np.empty(n_resamples) for m in metrics}
    probs = np.full(n, 1.0 / n)
    done = 0
    while done < n_resamples:
        k = min(chunk, n_resamples - done)
        w = rng.multinomial(n, probs, size=k).astype(np.int64)  # (k, n) doc multiplicities
        sa, sb = w @ a, w @ b
        for m in metrics:
            fa, fb = f1_from_sums(sa, m), f1_from_sums(sb, m)
            f1a_s[m][done:done + k], f1b_s[m][done:done + k] = fa, fb
            deltas[m][done:done + k] = fb - fa
        done += k
    out = {}
    for m in metrics:
        d = deltas[m]
        obs = float(pooled_f1_from_matrix(b, m) - pooled_f1_from_matrix(a, m))
        le = int((d <= 0).sum())
        ge = int((d >= 0).sum())
        p = min(1.0, 2.0 * min((le + 1) / (n_resamples + 1), (ge + 1) / (n_resamples + 1)))
        out[m] = {
            "observed_delta": obs,
            "boot_delta_mean": float(d.mean()),
            "boot_delta_sd": float(d.std(ddof=1)),
            "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "p_two_sided": p,
            "n_resamples": n_resamples,
            "anchor_f1_boot_sd": float(f1a_s[m].std(ddof=1)),
        }
    return out


def pooled_f1_from_matrix(counts: np.ndarray, metric: str) -> float:
    return float(f1_from_sums(counts.sum(axis=0), metric))


def sign_flip_permutation(
    a: np.ndarray,
    b: np.ndarray,
    metric: str,
    seed: int,
    exact_max_discordant: int = 20,
    mc_samples: int = 200_000,
) -> dict[str, Any]:
    """Paired sign-flip permutation test of H0: systems exchangeable within each document.

    Statistic = pooled F1(b) - pooled F1(a). Under H0 each document's two count vectors
    may be swapped with prob 1/2. Documents with identical counts never change the
    statistic, so only the ``d`` discordant documents matter: enumerate all 2^d swaps
    exactly when d <= ``exact_max_discordant``, else seeded Monte Carlo.
    """
    tp_c, fp_c, fn_c = COUNT_COLUMNS[metric]
    cols = [tp_c, fp_c, fn_c]
    diff = (b - a)[:, cols]
    discordant = np.any(diff != 0, axis=1)
    d = int(discordant.sum())
    base_a = a[:, cols].sum(axis=0).astype(np.int64)
    base_b = b[:, cols].sum(axis=0).astype(np.int64)
    obs = float(_f1_triplet(base_b) - _f1_triplet(base_a))
    if d == 0:
        return {"observed_delta": obs, "p_two_sided": 1.0, "method": "degenerate (no discordant docs)",
                "n_discordant_docs": 0, "n_permutations": 0}
    dd = diff[discordant]  # (d, 3)
    tol = 1e-12
    extreme = 0
    total = 0
    if d <= exact_max_discordant:
        method = "exact enumeration"
        n_perm = 2 ** d
        # enumerate in blocks: bit i of the integer = swap doc i
        block = 1 << min(d, 16)
        hi_bits = d - min(d, 16)
        low = ((np.arange(block)[:, None] >> np.arange(min(d, 16))[None, :]) & 1).astype(np.int64)
        for hi in range(1 << hi_bits):
            hi_mask = np.array([(hi >> j) & 1 for j in range(hi_bits)], dtype=np.int64)
            swaps = np.concatenate([low, np.tile(hi_mask, (block, 1))], axis=1) if hi_bits else low
            extreme += _count_extreme(swaps, dd, base_a, base_b, obs, tol)
            total += swaps.shape[0]
        p = extreme / total
    else:
        method = f"seeded Monte Carlo ({mc_samples} sign-flips, +1 smoothed)"
        rng = np.random.default_rng(seed)
        remaining = mc_samples
        while remaining > 0:
            k = min(20_000, remaining)
            swaps = rng.integers(0, 2, size=(k, d), dtype=np.int64)
            extreme += _count_extreme(swaps, dd, base_a, base_b, obs, tol)
            total += k
            remaining -= k
        n_perm = mc_samples
        p = (extreme + 1) / (total + 1)
    return {"observed_delta": obs, "p_two_sided": float(min(1.0, p)), "method": method,
            "n_discordant_docs": d, "n_permutations": int(n_perm)}


def _f1_triplet(v: np.ndarray) -> np.ndarray:
    tp, fp, fn = v[..., 0], v[..., 1], v[..., 2]
    denom = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denom, out=np.zeros(np.shape(denom), dtype=float), where=denom > 0)


def _count_extreme(swaps: np.ndarray, dd: np.ndarray, base_a: np.ndarray, base_b: np.ndarray,
                   obs: float, tol: float) -> int:
    # swapping doc i moves dd[i] from "b" to "a": A' = A + dd... (b-a) -> -(b-a)
    shift = swaps @ dd  # (k, 3)  sum over swapped docs of (b - a)
    sum_a = base_a[None, :] + shift
    sum_b = base_b[None, :] - shift
    stat = _f1_triplet(sum_b) - _f1_triplet(sum_a)
    return int((np.abs(stat) >= abs(obs) - tol).sum())


def doc_level_counts(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """McNemar-style doc-level comparison on exact-span outcomes."""
    def doc_f1(c: np.ndarray) -> np.ndarray:
        return f1_from_sums(c, "exact_span_f1")

    fa, fb = doc_f1(a), doc_f1(b)
    wins = int((fb > fa + 1e-12).sum())
    losses = int((fb < fa - 1e-12).sum())
    ties = int(a.shape[0] - wins - losses)
    sign_p = _binom_two_sided(wins, wins + losses)
    perfect_a = (a[:, 1] == 0) & (a[:, 2] == 0)
    perfect_b = (b[:, 1] == 0) & (b[:, 2] == 0)
    b_only = int((perfect_b & ~perfect_a).sum())  # variant perfect, anchor not
    a_only = int((perfect_a & ~perfect_b).sum())
    return {
        "doc_f1_wins": wins, "doc_f1_losses": losses, "doc_f1_ties": ties,
        "doc_f1_sign_test_p": sign_p,
        "perfect_doc_variant_only": b_only, "perfect_doc_anchor_only": a_only,
        "perfect_doc_mcnemar_exact_p": _binom_two_sided(b_only, b_only + a_only),
        "n_docs": int(a.shape[0]),
    }


def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    return float(min(1.0, stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj


def minimal_detectable_effect(sd_delta: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Two-sided MDE ~= (z_{1-alpha/2} + z_{power}) * SE(delta); SE = paired bootstrap SD."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * sd_delta)


# --------------------------------------------------------------------------- data loading
def load_gold_docs(dataset_root: str = DATASET_ROOT) -> list[dict[str, Any]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

    docs = load_skillspan_split(Path(dataset_root), "test", limit=None)
    return [d for d in docs if d["gold_spans"]]  # same filter as run_safe_wrapper


def load_system_predictions(path: str, doc_ids: list[str]) -> list[list[dict]]:
    by_id = {row["id"]: row["predictions"] for row in read_jsonl(path)}
    missing = [i for i in doc_ids if i not in by_id]
    if missing:
        raise AssertionError(f"{path}: {len(missing)} gold docs missing predictions (e.g. {missing[:3]})")
    return [by_id[i] for i in doc_ids]


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- driver
def run_significance(
    n_resamples: int = 10_000,
    seed: int = 20260710,
    anchor_path: str = ANCHOR_PREDICTIONS,
    variant_paths: dict[str, str] | None = None,
    dataset_root: str = DATASET_ROOT,
    alpha: float = 0.05,
    check_stored_metrics: bool = True,
) -> dict[str, Any]:
    if n_resamples < 10_000:
        raise ValueError("n_resamples must be >= 10,000")
    variant_paths = variant_paths or VARIANT_PREDICTIONS
    t0 = time.time()
    docs = load_gold_docs(dataset_root)
    ids = [d["id"] for d in docs]
    gold = [d["gold_spans"] for d in docs]

    systems: dict[str, np.ndarray] = {}
    repro: dict[str, Any] = {}
    prov: dict[str, Any] = {}
    stored_paths = {"anchor": ANCHOR_METRICS_JSON, **VARIANT_METRICS_JSON}
    all_paths = {"anchor": anchor_path, **variant_paths}
    for name, path in all_paths.items():
        pred = load_system_predictions(path, ids)
        cm = count_matrix(gold, pred)
        stored = None
        if check_stored_metrics:
            payload = read_json(stored_paths[name], default=None)
            if payload is not None:
                stored = payload.get("candidate_metrics", payload) if name != "anchor" else payload
        repro[name] = assert_reproduction(name, gold, pred, cm, stored)
        systems[name] = cm
        prov[name] = {"path": path, "sha256_16": _sha(path)}

    a = systems["anchor"]
    results: dict[str, Any] = {}
    for vi, variant in enumerate(variant_paths):
        b = systems[variant]
        boot = paired_bootstrap(a, b, METRIC_ORDER, n_resamples, seed + vi)
        per_metric = {}
        for mi, m in enumerate(METRIC_ORDER):
            perm = sign_flip_permutation(a, b, m, seed + 1000 + 17 * vi + mi)
            per_metric[m] = {
                "anchor_f1": pooled_f1_from_matrix(a, m),
                "variant_f1": pooled_f1_from_matrix(b, m),
                "bootstrap": boot[m],
                "permutation": perm,
            }
        results[variant] = {"metrics": per_metric, "doc_level": doc_level_counts(a, b)}

    # Holm across variants x metrics (15 tests), for both the bootstrap and permutation p
    keys = [(v, m) for v in results for m in METRIC_ORDER]
    boot_adj = holm([results[v]["metrics"][m]["bootstrap"]["p_two_sided"] for v, m in keys])
    perm_adj = holm([results[v]["metrics"][m]["permutation"]["p_two_sided"] for v, m in keys])
    for (v, m), pb, pp in zip(keys, boot_adj, perm_adj):
        results[v]["metrics"][m]["bootstrap"]["p_holm_all"] = pb
        results[v]["metrics"][m]["permutation"]["p_holm_all"] = pp
    # Holm across the 3 variants within each metric (family = one metric)
    for m in METRIC_ORDER:
        vs = list(results)
        pb = holm([results[v]["metrics"][m]["bootstrap"]["p_two_sided"] for v in vs])
        pp = holm([results[v]["metrics"][m]["permutation"]["p_two_sided"] for v in vs])
        for v, x, y in zip(vs, pb, pp):
            results[v]["metrics"][m]["bootstrap"]["p_holm_within_metric"] = x
            results[v]["metrics"][m]["permutation"]["p_holm_within_metric"] = y

    n_tests = len(keys)
    for v, res in results.items():
        for m, r in res["metrics"].items():
            sig_boot = r["bootstrap"]["p_holm_all"] < alpha
            sig_perm = r["permutation"]["p_holm_all"] < alpha
            r["significant_after_holm"] = bool(sig_boot and sig_perm)
            r["bootstrap_permutation_agree"] = bool(sig_boot == sig_perm)
            sd = r["bootstrap"]["boot_delta_sd"]
            r["mde_alpha05_power80"] = minimal_detectable_effect(sd, alpha, 0.80)
            r["mde_holm_all_alpha_power80"] = minimal_detectable_effect(sd, alpha / n_tests, 0.80)
        ex = res["metrics"][PRIMARY_METRIC]
        res["verdict"] = _verdict(v, ex, alpha)

    return {
        "config": {
            "n_docs": len(docs), "n_resamples": n_resamples, "seed": seed, "alpha": alpha,
            "n_holm_tests": n_tests, "resampling_unit": "document",
            "variant_p_value_source": "candidate_predictions.jsonl (== gated predictions: all 3 variants were 'accept')",
            "reproduction_tolerance": REPRO_TOL, "runtime_seconds": round(time.time() - t0, 1),
            "split": "SkillSpan test, docs with >=1 gold span", "inputs": prov,
        },
        "reproduction": repro,
        "variants": results,
    }


def _verdict(variant: str, ex: dict[str, Any], alpha: float) -> str:
    d = ex["bootstrap"]["observed_delta"]
    lo, hi = ex["bootstrap"]["ci95_low"], ex["bootstrap"]["ci95_high"]
    status = "SIGNIFICANT after Holm" if ex["significant_after_holm"] else "NOT significant after Holm"
    return (
        f"{variant}: exact-span F1 delta {d:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]); "
        f"Holm-adjusted p bootstrap={ex['bootstrap']['p_holm_all']:.3f}, "
        f"permutation={ex['permutation']['p_holm_all']:.3f} -> {status}"
    )


# --------------------------------------------------------------------------- report
def render_markdown(res: dict[str, Any]) -> str:
    cfg = res["config"]
    lines = [
        "# Statistical significance of extraction F1 deltas (anchor vs safe-wrapper variants)",
        "",
        f"Paired document-level bootstrap ({cfg['n_resamples']:,} resamples, seed {cfg['seed']}) and paired "
        f"sign-flip permutation test on {cfg['n_docs']} SkillSpan test documents. Micro-F1 is recomputed from "
        f"pooled TP/FP/FN of every resample. Holm correction over {cfg['n_holm_tests']} tests "
        f"(3 variants x 5 metrics), alpha = {cfg['alpha']}. A delta is called significant only if BOTH the "
        "bootstrap and the permutation Holm-adjusted p-values are below alpha. Delta = variant minus Raw Mistral LoRA anchor.",
        "",
        "Reproduction check: pooled-count F1s match `evaluate_extraction` and the F1s stored in each artifact's "
        f"`metrics.json` to within {cfg['reproduction_tolerance']:g} for every system and metric (asserted before any test).",
        "",
        "Note: tests use the per-document predictions currently on disk (inputs and hashes at the bottom). Their "
        "exact-F1 values equal the current `safe_wrapper_summary.json` / `money_table.md`; if you quote older "
        "figures for the GLiNER+Taxonomy or Full CAG/Graph variants (e.g. 0.670222 / 0.672602), they come from an "
        "earlier artifact version that is not on disk and are NOT what was tested here.",
        "",
        "## Verdicts (primary metric: exact-span F1)",
        "",
    ]
    for v, r in res["variants"].items():
        lines.append(f"- {r['verdict']}")
    lines += ["", "## Exact-span F1 summary", "",
              "| Variant | Anchor F1 | Variant F1 | Delta | 95% CI | p boot (raw) | p perm (raw) | p boot (Holm) | p perm (Holm) | MDE (80% power) | Verdict |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in res["variants"].items():
        m = r["metrics"][PRIMARY_METRIC]
        b, p = m["bootstrap"], m["permutation"]
        lines.append(
            f"| {v} | {m['anchor_f1']:.6f} | {m['variant_f1']:.6f} | {b['observed_delta']:+.6f} | "
            f"[{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] | {b['p_two_sided']:.4f} | {p['p_two_sided']:.4f} | "
            f"{b['p_holm_all']:.4f} | {p['p_holm_all']:.4f} | {m['mde_alpha05_power80']:.4f} | "
            f"{'significant' if m['significant_after_holm'] else 'not significant'} |"
        )
    lines += ["", "## All metrics", "",
              "| Variant | Metric | Anchor | Variant | Delta (observed) | Boot mean | 95% CI | p boot | p perm | Holm boot | Holm perm | MDE | Sig. after Holm |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in res["variants"].items():
        for mname in METRIC_ORDER:
            m = r["metrics"][mname]
            b, p = m["bootstrap"], m["permutation"]
            lines.append(
                f"| {v} | {mname} | {m['anchor_f1']:.6f} | {m['variant_f1']:.6f} | {b['observed_delta']:+.6f} | "
                f"{b['boot_delta_mean']:+.6f} | [{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] | "
                f"{b['p_two_sided']:.4f} | {p['p_two_sided']:.4f} | {b['p_holm_all']:.4f} | {p['p_holm_all']:.4f} | "
                f"{m['mde_alpha05_power80']:.4f} | {'yes' if m['significant_after_holm'] else 'no'} |"
            )
    lines += ["", "## Permutation test details", "",
              "| Variant | Metric | Discordant docs | Method |", "|---|---|---|---|"]
    for v, r in res["variants"].items():
        for mname in METRIC_ORDER:
            p = r["metrics"][mname]["permutation"]
            lines.append(f"| {v} | {mname} | {p['n_discordant_docs']} | {p['method']} |")
    lines += ["", "## Document-level win / loss / tie (doc exact-span F1) and McNemar", "",
              "| Variant | Wins | Losses | Ties | Sign-test p | Perfect docs: variant-only | anchor-only | McNemar exact p |",
              "|---|---|---|---|---|---|---|---|"]
    for v, r in res["variants"].items():
        d = r["doc_level"]
        lines.append(
            f"| {v} | {d['doc_f1_wins']} | {d['doc_f1_losses']} | {d['doc_f1_ties']} | {d['doc_f1_sign_test_p']:.4f} | "
            f"{d['perfect_doc_variant_only']} | {d['perfect_doc_anchor_only']} | {d['perfect_doc_mcnemar_exact_p']:.4f} |"
        )
    lines += ["", "## Minimal detectable effect at this sample size", ""]
    for v, r in res["variants"].items():
        m = r["metrics"][PRIMARY_METRIC]
        lines.append(
            f"- {v}: paired bootstrap SD of the exact-F1 delta = {m['bootstrap']['boot_delta_sd']:.5f}; "
            f"MDE (two-sided alpha 0.05, 80% power) = {m['mde_alpha05_power80']:.4f}; "
            f"with Holm-level alpha 0.05/{cfg['n_holm_tests']} = {m['mde_holm_all_alpha_power80']:.4f}."
        )
    lines += _claims_section(res)
    lines += ["", "Inputs (sha256 prefix): " + "; ".join(
        f"{k}: `{v['path']}` ({v['sha256_16']})" for k, v in cfg["inputs"].items()), ""]
    return "\n".join(lines)


def _claims_section(res: dict[str, Any]) -> list[str]:
    """Data-driven plain-language guidance on what the thesis may claim."""
    cfg = res["config"]
    lines = ["", "## Can the thesis claim these deltas?", ""]
    for v, r in res["variants"].items():
        m = r["metrics"][PRIMARY_METRIC]
        b, d = m["bootstrap"], r["doc_level"]
        ci_excl0 = b["ci95_low"] > 0 or b["ci95_high"] < 0
        if m["significant_after_holm"]:
            text = (
                f"statistically reliable but tiny: {b['observed_delta']:+.4f} exact-F1, 95% CI "
                f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]. It rests on only "
                f"{m['permutation']['n_discordant_docs']} of {d['n_docs']} documents that differ at all "
                f"({d['doc_f1_wins']} better, {d['doc_f1_losses']} worse). Safe wording: \"a small, consistent "
                "improvement in span boundaries on the SkillSpan test set (a handful of documents, no regressions)\"; "
                "do not describe it as a meaningful accuracy gain."
            )
        elif ci_excl0:
            text = (
                f"nominal evidence only ({b['observed_delta']:+.4f}, CI excludes 0) but it does not survive Holm "
                f"correction over {cfg['n_holm_tests']} tests; report as exploratory."
            )
        else:
            text = (
                f"cannot be claimed. Delta {b['observed_delta']:+.4f}, 95% CI "
                f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] contains 0; "
                f"{d['doc_f1_wins']} documents better vs {d['doc_f1_losses']} worse. Treat as no measurable "
                "difference from the anchor (non-regression, not improvement)."
            )
        lines.append(f"- **{v}**: {text}")
    full = res["variants"].get("full_cag_graph")
    if full is not None:
        fm = full["metrics"][PRIMARY_METRIC]
        sd = fm["bootstrap"]["boot_delta_sd"]
        z = 0.0025 / sd if sd else float("inf")
        p_raw = 2 * (1 - stats.norm.cdf(z))
        lines += [
            "",
            f"**On a +0.0025 delta specifically.** At this sample size the paired standard error of an "
            f"exact-F1 delta for the full CAG/Graph variant is about {sd:.4f}, so a +0.0025 delta is roughly "
            f"{z:.1f} standard errors from zero (approximate raw two-sided p about {p_raw:.3f} under a normal "
            f"approximation; this is a back-of-envelope figure, not a rerun of the test on different predictions). "
            f"That would not survive Holm correction over {cfg['n_holm_tests']} tests (or over just the 3 variants), "
            f"and the MDE at 80% power is {fm['mde_alpha05_power80']:.4f}, i.e. larger than 0.0025. "
            "A +0.0025 gain for the full wrapper should therefore not be presented as a real improvement; report it "
            "with its confidence interval and say that the study is not powered to detect effects below that size.",
            "",
            "**Caveats that apply to every row.** (1) Resampling is over the 974 test documents of one split and one "
            "trained anchor model; it does not capture variance across training seeds, so even a significant delta is "
            "evidence about this model and test set only. (2) The variants are deterministic post-processing of the same "
            "Mistral predictions, so most documents are unchanged (ties dominate), which is why the tests are driven by "
            "a few documents and the exact permutation p-values are coarse (minimum attainable raw p is "
            "2/2^d for d discordant documents). (3) Holm over 15 correlated tests is conservative; the "
            "per-metric Holm columns (`p_holm_within_metric` in the JSON) are looser. (4) Effects of this "
            "magnitude (about 0.2 F1 points) are practically negligible even when statistically detectable. "
            "(5) If the wrapper code or artifacts are regenerated, rerun `scripts/run_significance_test.sh`: the "
            "numbers above are tied to the input hashes listed below.",
        ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired significance tests for extraction F1 deltas")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-json", default="reports/skillllm_v2/significance_f1.json")
    parser.add_argument("--output-md", default="reports/skillllm_v2/significance_f1.md")
    parser.add_argument("--no-stored-check", action="store_true")
    args = parser.parse_args()
    res = run_significance(args.n_resamples, args.seed, alpha=args.alpha, check_stored_metrics=not args.no_stored_check)
    write_json(args.output_json, res)
    Path(args.output_md).write_text(render_markdown(res), encoding="utf-8")
    for v, r in res["variants"].items():
        print(r["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
