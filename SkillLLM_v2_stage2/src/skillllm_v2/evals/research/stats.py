"""Shared statistics for the research evaluation layers (owner: A2; other layers import it).

Pure numpy (+ scipy only for the optional Wilcoxon test). Every resampling routine takes an explicit ``seed`` and uses
``numpy.random.default_rng(seed)`` so results are bit-for-bit reproducible.

Conventions (contract, "Statistics conventions"):

* The resampling unit is the independent unit (document / query / profile). Per-unit inputs are rows of additive
  sufficient statistics, e.g. ``[tp, fp, fn]`` per document for micro-F1. A resample sums the drawn rows and the
  statistic is recomputed from the pooled sums (never an average of per-unit F1).
* Paired comparisons draw ONE set of unit indices per replicate and apply it to both systems.
* Bootstrap CIs are 95 % percentile intervals.
* Bootstrap two-sided p (``p``) = ``min(1, 2 * min((1 + #{d* <= 0}) / (B + 1), (1 + #{d* >= 0}) / (B + 1)))``: the
  percentile-inversion p-value, consistent with the reported CI (p < 0.05 <=> the 95 % CI excludes 0, up to the +1
  smoothing) and identical in form to ``skillllm_v2.eval.significance``. Ties at 0 count on both sides, so two
  identical systems get p = 1 (the legacy ``extraction_evals.paired_boot`` returns p = 0 in that degenerate case).
  A null-shifted variant ``p_shift = (1 + #{|d* - d| >= |d|}) / (B + 1)`` is reported as a secondary.
* Approximate randomization (AR) = paired permutation: each unit's two outputs are swapped with probability 1/2,
  ``R`` = 10,000 shuffles, ``p = (1 + #{|stat_A' - stat_B'| >= |observed|}) / (R + 1)`` (Noreen 1989; Riezler & Maxwell
  2005). For per-unit means it is the paired sign-flip test.
* Multiplicity: Holm (step-down FWER) within a declared family; Benjamini-Hochberg (step-up FDR) across everything.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Stat = Callable[[np.ndarray], np.ndarray]  # maps summed rows (..., k) -> statistic (...)

_CHUNK_ELEMS = 400_000  # elements per resampling chunk; same chunking as extraction_evals.boot_ci so draws coincide


# ----------------------------------------------------------------------------- statistics on pooled sums
def _div(a: Any, b: Any) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def micro_f1(sums: np.ndarray, tp: int = 0, fp: int = 1, fn: int = 2) -> np.ndarray:
    """Micro-F1 from pooled counts = 2TP / (2TP + FP + FN) (0 when undefined); equals the harmonic mean of P and R."""
    s = np.asarray(sums, dtype=float)
    return _div(2 * s[..., tp], 2 * s[..., tp] + s[..., fp] + s[..., fn])


def micro_precision(sums: np.ndarray, tp: int = 0, fp: int = 1) -> np.ndarray:
    s = np.asarray(sums, dtype=float)
    return _div(s[..., tp], s[..., tp] + s[..., fp])


def micro_recall(sums: np.ndarray, tp: int = 0, fn: int = 2) -> np.ndarray:
    s = np.asarray(sums, dtype=float)
    return _div(s[..., tp], s[..., tp] + s[..., fn])


def ratio_stat(num: int, den: int) -> Stat:
    """Statistic sum(col num) / sum(col den), e.g. anchor preservation rate from per-doc (kept, anchor_tp) rows."""
    return lambda s: _div(np.asarray(s, dtype=float)[..., num], np.asarray(s, dtype=float)[..., den])


def mean_stat(n_units: int, col: int = 0) -> Stat:
    """Mean of a per-unit value given the pooled SUM of that value over ``n_units`` units."""
    return lambda s: np.asarray(s, dtype=float)[..., col] / float(n_units)


def _as_rows(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[:, None] if a.ndim == 1 else a


# ----------------------------------------------------------------------------- resampling core
def _boot_stats(rows: Sequence[np.ndarray], stats: Sequence[Stat], n_boot: int, seed: int) -> List[np.ndarray]:
    """For each replicate draw n unit indices with replacement (shared by all ``rows`` arrays) and return
    ``stats[j](rows[j][idx].sum(0))`` for every j. ``rows`` arrays must share the unit axis 0."""
    n = len(rows[0])
    rng = np.random.default_rng(seed)
    out = [np.empty(n_boot) for _ in rows]
    width = max(1, max(r.shape[1] for r in rows))
    chunk = max(1, _CHUNK_ELEMS // max(1, n * width))
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        idx = rng.integers(0, n, size=(b, n))
        for j, (r, st) in enumerate(zip(rows, stats)):
            out[j][lo:lo + b] = st(r[idx].sum(axis=1))
    return out


def bootstrap_sums(*rows_list: Any, n_boot: int = 2000, seed: int = 42) -> List[np.ndarray]:
    """Resampled column sums with SHARED unit draws for every array in ``rows_list`` (each n_units x k_j).
    Returns one (n_boot x k_j) array per input; any number of statistics (and paired deltas) can then be computed from
    the same replicates without re-drawing. Draws are identical to ``bootstrap_ci`` / ``paired_bootstrap`` only when
    the chunking coincides (same total width); use those for single statistics."""
    rows = [_as_rows(r) for r in rows_list]
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("all row arrays must share the unit axis")
    rng = np.random.default_rng(seed)
    out = [np.empty((n_boot, r.shape[1])) for r in rows]
    width = max(1, sum(r.shape[1] for r in rows))
    chunk = max(1, _CHUNK_ELEMS // max(1, n * width))
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        idx = rng.integers(0, n, size=(b, n))
        for j, r in enumerate(rows):
            out[j][lo:lo + b] = r[idx].sum(axis=1)
    return out


def summarize_samples(samples: np.ndarray, value: float, alpha: float = 0.05, paired: bool = False) -> Dict[str, Any]:
    """CI (+ SE; + two-sided p and p_shift when ``paired`` i.e. samples are bootstrap deltas) from replicates."""
    lo, hi = _pct_ci(np.asarray(samples, dtype=float), alpha)
    out = {"value": float(value), "ci_low": lo, "ci_high": hi,
           "se": float(np.nanstd(samples, ddof=1)) if len(samples) > 1 else None}
    if paired:
        out["p"], out["p_shift"] = _boot_p(np.asarray(samples, dtype=float), float(value))
    return out


def _pct_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    s = samples[np.isfinite(samples)]
    if len(s) == 0:
        return None, None
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def bootstrap_ci(rows: Any, stat: Stat = micro_f1, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> Dict[str, Any]:
    """Unit-level (cluster) percentile bootstrap of ``stat(sum of per-unit rows)``.

    Same draw scheme as ``skillllm_v2.evals.extraction_evals.boot_ci`` (rng.integers(0, n, (b, n)) per chunk), so for
    the same seed and chunking the intervals coincide; tests check agreement.
    """
    r = _as_rows(rows)
    n = len(r)
    if n == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "se": None, "n_units": 0, "n_boot": n_boot}
    value = float(stat(r.sum(axis=0)))
    (samples,) = _boot_stats([r], [stat], n_boot, seed)
    lo, hi = _pct_ci(samples, alpha)
    return {"value": value, "ci_low": lo, "ci_high": hi, "se": float(np.nanstd(samples, ddof=1)) if n_boot > 1 else None,
            "n_units": n, "n_boot": n_boot, "n_valid_boot": int(np.isfinite(samples).sum())}


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Dict[str, Any]:
    v = np.asarray(values, dtype=float)
    return bootstrap_ci(v[:, None], mean_stat(len(v)), n_boot, seed, alpha)


def _boot_p(diffs: np.ndarray, delta: float) -> Tuple[float, float]:
    d = diffs[np.isfinite(diffs)]
    b = len(d)
    if b == 0:
        return float("nan"), float("nan")
    below = (1 + int((d <= 0).sum())) / (b + 1)
    above = (1 + int((d >= 0).sum())) / (b + 1)
    p = min(1.0, 2 * min(below, above))
    p_shift = (1 + int((np.abs(d - delta) >= abs(delta) - 1e-15).sum())) / (b + 1)
    return float(p), float(min(1.0, p_shift))


def paired_bootstrap(a: Any, b: Any, stat: Stat = micro_f1, n_boot: int = 2000, seed: int = 42,
                     alpha: float = 0.05, keep_samples: bool = False) -> Dict[str, Any]:
    """Paired unit-level bootstrap of ``stat(A) - stat(B)`` with shared draws.

    ``a``/``b``: per-unit rows (n_units x k) in the SAME unit order (e.g. per-doc [tp, fp, fn]).
    Returns delta (observed), CI, two-sided p (percentile inversion, +1 smoothed), p_shift (null-shifted), SE, the
    values of both systems, and the number of units whose rows differ (when that number is tiny the CI is driven by a
    handful of units and should be read as such).
    """
    ra, rb = _as_rows(a), _as_rows(b)
    if ra.shape != rb.shape:
        raise ValueError(f"paired inputs need identical shapes, got {ra.shape} vs {rb.shape}")
    n = len(ra)
    if n == 0:
        raise ValueError("no units")
    va, vb = float(stat(ra.sum(axis=0))), float(stat(rb.sum(axis=0)))
    delta = va - vb
    sa, sb = _boot_stats([ra, rb], [stat, stat], n_boot, seed)
    diffs = sa - sb
    lo, hi = _pct_ci(diffs, alpha)
    p, p_shift = _boot_p(diffs, delta)
    out = {"value_a": va, "value_b": vb, "delta": delta, "ci_low": lo, "ci_high": hi, "p": p, "p_shift": p_shift,
           "se": float(np.nanstd(diffs, ddof=1)) if n_boot > 1 else None, "boot_mean": float(np.nanmean(diffs)),
           "n_units": n, "n_units_differ": int((np.abs(ra - rb).sum(axis=1) > 0).sum()), "n_boot": n_boot,
           "method": "paired percentile bootstrap over units (shared draws); p = 2*min tail share, +1 smoothed"}
    if keep_samples:
        out["samples"] = diffs
    return out


def paired_bootstrap_mean(x: Sequence[float], y: Sequence[float], n_boot: int = 2000, seed: int = 42,
                          alpha: float = 0.05) -> Dict[str, Any]:
    """Paired bootstrap of mean(x) - mean(y) over units (x[i], y[i] measured on the same unit i)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError("x and y must be equal-length 1-D sequences")
    n = len(xa)
    return paired_bootstrap(xa[:, None], ya[:, None], mean_stat(n), n_boot, seed, alpha)


def approximate_randomization(a: Any, b: Any, stat: Stat = micro_f1, n_shuffles: int = 10000, seed: int = 42,
                              max_exact_units: int = 16) -> Dict[str, Any]:
    """Two-sided paired approximate-randomization (permutation) test of H0: the two systems' per-unit outputs are
    exchangeable. Each shuffle swaps unit i's rows between A and B with probability 1/2 (independently per unit).

    Only units whose rows differ can change the statistic, so when <= ``max_exact_units`` units differ the full
    2^d swap set is enumerated (exact p, no Monte Carlo error; p = #{>=obs}/2^d). Otherwise ``n_shuffles`` seeded
    shuffles with p = (1 + #{|d'| >= |d_obs|}) / (R + 1).
    """
    ra, rb = _as_rows(a), _as_rows(b)
    if ra.shape != rb.shape:
        raise ValueError(f"paired inputs need identical shapes, got {ra.shape} vs {rb.shape}")
    sum_a, sum_b = ra.sum(axis=0), rb.sum(axis=0)
    obs = float(stat(sum_a) - stat(sum_b))
    diff_rows = rb - ra  # swapping unit i moves diff_rows[i] from B's sum into A's sum
    disc = np.flatnonzero(np.abs(diff_rows).sum(axis=1) > 0)
    dd = diff_rows[disc]
    d = len(disc)
    tol = 1e-12
    if d == 0:
        return {"observed_delta": obs, "p": 1.0, "n_units": len(ra), "n_units_differ": 0, "n_shuffles": 0,
                "method": "degenerate: no unit differs"}
    if d <= max_exact_units:
        masks = ((np.arange(2 ** d)[:, None] >> np.arange(d)[None, :]) & 1).astype(float)
        shift = masks @ dd
        perm = stat(sum_a + shift) - stat(sum_b - shift)
        p = float((np.abs(perm) >= abs(obs) - tol).mean())
        return {"observed_delta": obs, "p": min(1.0, p), "n_units": len(ra), "n_units_differ": d,
                "n_shuffles": int(2 ** d), "method": f"exact enumeration of 2^{d} swaps over differing units"}
    rng = np.random.default_rng(seed)
    count = 0
    chunk = max(1, _CHUNK_ELEMS // max(1, d))
    for lo in range(0, n_shuffles, chunk):
        m = min(chunk, n_shuffles - lo)
        masks = (rng.random((m, d)) < 0.5).astype(float)
        shift = masks @ dd
        perm = stat(sum_a + shift) - stat(sum_b - shift)
        count += int((np.abs(perm) >= abs(obs) - tol).sum())
    p = (1 + count) / (n_shuffles + 1)
    return {"observed_delta": obs, "p": float(min(1.0, p)), "n_units": len(ra), "n_units_differ": d,
            "n_shuffles": n_shuffles, "method": f"Monte Carlo approximate randomization ({n_shuffles} shuffles, +1 smoothed)"}


def sign_flip_test(x: Sequence[float], y: Sequence[float], n_shuffles: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Paired sign-flip permutation test on mean(x - y) (AR for per-unit scores)."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return approximate_randomization(xa[:, None], ya[:, None], mean_stat(len(xa)), n_shuffles, seed)


# ----------------------------------------------------------------------------- multiplicity
def _clean_p(pvalues: Sequence[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.array([np.nan if v is None else float(v) for v in pvalues], dtype=float)
    return p, np.isfinite(p)


def holm(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1). None/NaN entries stay None and do not
    count towards the family size."""
    p, ok = _clean_p(pvalues)
    out: List[Optional[float]] = [None] * len(p)
    idx = np.flatnonzero(ok)
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = float(running)
    return out


def benjamini_hochberg(pvalues: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg step-up adjusted p-values (q-values; monotone, capped at 1). None/NaN stay None."""
    p, ok = _clean_p(pvalues)
    out: List[Optional[float]] = [None] * len(p)
    idx = np.flatnonzero(ok)
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        out[i] = float(min(1.0, running))
    return out


def adjust_families(rows: List[Dict[str, Any]], p_key: str = "p", family_key: str = "family",
                    holm_key: str = "p_holm", bh_key: str = "p_bh") -> List[Dict[str, Any]]:
    """In place: Holm within each ``row[family_key]`` and BH across all rows (secondary). Returns ``rows``."""
    fams: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        fams.setdefault(str(r.get(family_key)), []).append(i)
    for idxs in fams.values():
        for i, adj in zip(idxs, holm([rows[i].get(p_key) for i in idxs])):
            rows[i][holm_key] = adj
            rows[i]["family_size"] = sum(1 for j in idxs if rows[j].get(p_key) is not None)
    for r, q in zip(rows, benjamini_hochberg([r.get(p_key) for r in rows])):
        r[bh_key] = q
    return rows


# ----------------------------------------------------------------------------- decision helpers
def non_inferiority(ci_low: Optional[float], margin: float, delta: Optional[float] = None,
                    samples: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Non-inferiority of A vs B with a NEGATIVE margin (e.g. -0.005): non-inferior iff the lower bound of the
    two-sided 95 % CI of delta = A - B is above ``margin`` (= one-sided test at alpha 0.025). If bootstrap ``samples``
    are given, also returns the one-sided bootstrap p for H0: delta <= margin."""
    out: Dict[str, Any] = {"margin": float(margin), "ci_low": ci_low, "delta": delta,
                           "non_inferior": bool(ci_low is not None and ci_low > margin),
                           "rule": f"lower 95% CI bound of delta > {margin}"}
    if samples is not None:
        s = np.asarray(samples, dtype=float)
        s = s[np.isfinite(s)]
        out["p_one_sided"] = float((1 + int((s <= margin).sum())) / (len(s) + 1)) if len(s) else None
    return out


def ci_excludes_zero(ci_low: Optional[float], ci_high: Optional[float]) -> bool:
    return ci_low is not None and ci_high is not None and (ci_low > 0 or ci_high < 0)


def mde(se: Optional[float], alpha: float = 0.05, power: float = 0.80) -> Optional[float]:
    """Minimum detectable effect (two-sided normal approximation) for a paired delta with bootstrap SE ``se``."""
    if se is None or not math.isfinite(se):
        return None
    from statistics import NormalDist
    z = NormalDist()
    return float((z.inv_cdf(1 - alpha / 2) + z.inv_cdf(power)) * se)


def wilcoxon(x: Sequence[float], y: Optional[Sequence[float]] = None, zero_method: str = "wilcox") -> Dict[str, Any]:
    """Wilcoxon signed-rank test (scipy) on per-unit paired values; ``available=False`` if scipy is missing."""
    try:
        from scipy.stats import wilcoxon as _w
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"scipy unavailable: {exc}"}
    xa = np.asarray(x, dtype=float)
    d = xa - np.asarray(y, dtype=float) if y is not None else xa
    nz = int((d != 0).sum())
    if nz == 0:
        return {"available": True, "statistic": None, "p": 1.0, "n_nonzero": 0, "note": "all differences are zero"}
    res = _w(d, zero_method=zero_method)
    return {"available": True, "statistic": float(res.statistic), "p": float(res.pvalue), "n_nonzero": nz,
            "zero_method": zero_method}


def compare_counts(a: Any, b: Any, stat: Stat = micro_f1, n_boot: int = 2000, n_shuffles: int = 10000,
                   seed: int = 42, margin: Optional[float] = None) -> Dict[str, Any]:
    """One-stop paired comparison of two systems from per-unit count rows: bootstrap (delta, CI, p, SE, MDE) +
    approximate randomization (+ non-inferiority when ``margin`` is given)."""
    boot = paired_bootstrap(a, b, stat, n_boot, seed, keep_samples=margin is not None)
    ar = approximate_randomization(a, b, stat, n_shuffles, seed + 1)
    out = {k: v for k, v in boot.items() if k != "samples"}
    out.update({"p_ar": ar["p"], "ar_method": ar["method"], "mde_80": mde(boot["se"])})
    if margin is not None:
        out["non_inferiority"] = non_inferiority(boot["ci_low"], margin, boot["delta"], boot.get("samples"))
    return out
