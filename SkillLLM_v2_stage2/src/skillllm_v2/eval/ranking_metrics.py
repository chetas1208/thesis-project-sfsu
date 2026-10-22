"""Graded-relevance ranking metrics and bootstrap statistics for the recommender benchmark.

All ranking functions take

    ranked_ids : list[str]            course ids in rank order (best first); duplicates are ignored after first
    grades     : dict[str, float]     graded relevance 0..3 for EVERY judged course (missing id => 0)

Conventions (stated explicitly because they differ between papers):
  * nDCG@k uses exponential gain (2^rel - 1) and log2(rank+1) discount; the ideal ranking is taken over
    ALL judged courses of the scenario (not only the retrieved ones).  A scenario with no relevant
    course (ideal DCG = 0) returns None and must be excluded from averages.
  * Precision@k / Recall@k / AP@k binarise relevance at ``rel_threshold`` (default grade >= 2).
    Precision@k divides by k (a short list is penalised).  Recall@k divides by the total number of
    relevant courses in the catalogue for the scenario; None if there are none.
  * AP@k = (1 / min(R, k)) * sum_{i<=k} P@i * rel_i  with R the total number of relevant courses
    (the common "truncated AP" used by e.g. Kaggle / RecSys evaluations).  MAP is the mean of AP@k over
    scenarios.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def _dedupe(ranked_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for r in ranked_ids:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def dcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    total = 0.0
    for i, cid in enumerate(_dedupe(ranked_ids)[:k], start=1):
        rel = float(grades.get(cid, 0.0))
        total += (2.0 ** rel - 1.0) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int = 10) -> float | None:
    ideal = sorted((float(g) for g in grades.values()), reverse=True)[:k]
    idcg = sum((2.0 ** g - 1.0) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    if idcg <= 0.0:
        return None
    return dcg_at_k(ranked_ids, grades, k) / idcg


def _is_rel(grades: Mapping[str, float], cid: str, thr: float) -> bool:
    return float(grades.get(cid, 0.0)) >= thr


def n_relevant(grades: Mapping[str, float], rel_threshold: float = 2.0) -> int:
    return sum(1 for g in grades.values() if float(g) >= rel_threshold)


def precision_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int = 5, rel_threshold: float = 2.0) -> float:
    top = _dedupe(ranked_ids)[:k]
    hits = sum(1 for c in top if _is_rel(grades, c, rel_threshold))
    return hits / k


def recall_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int = 10, rel_threshold: float = 2.0) -> float | None:
    total = n_relevant(grades, rel_threshold)
    if total == 0:
        return None
    top = _dedupe(ranked_ids)[:k]
    hits = sum(1 for c in top if _is_rel(grades, c, rel_threshold))
    return hits / total


def average_precision_at_k(ranked_ids: Sequence[str], grades: Mapping[str, float], k: int = 10, rel_threshold: float = 2.0) -> float | None:
    total = n_relevant(grades, rel_threshold)
    if total == 0:
        return None
    hits = 0
    s = 0.0
    for i, cid in enumerate(_dedupe(ranked_ids)[:k], start=1):
        if _is_rel(grades, cid, rel_threshold):
            hits += 1
            s += hits / i
    return s / min(total, k)


# ----------------------------------------------------------------------------- statistics
def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean over scenarios.  Returns (mean, lo, hi)."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired bootstrap of mean(a - b) over scenarios.

    p-value: two-sided bootstrap p = 2 * min(P*(d <= 0), P*(d >= 0)) computed on the resampled mean
    differences, floored at 1/(n_boot+1) and capped at 1.  (This is the usual "difference-in-means"
    bootstrap; it is a descriptive test of whether the paired advantage is robust to resampling of
    scenarios, and it ignores the uncertainty of the label procedure itself.)
    """
    a_ = np.asarray(a, dtype=float)
    b_ = np.asarray(b, dtype=float)
    d = a_ - b_
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p_le = (np.sum(means <= 0) + 1) / (n_boot + 1)
    p_ge = (np.sum(means >= 0) + 1) / (n_boot + 1)
    p = float(min(1.0, 2.0 * min(p_le, p_ge)))
    return {
        "mean_diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": p,
        "n": int(n),
        "wins": int(np.sum(d > 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
    }


def holm_adjust(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj
