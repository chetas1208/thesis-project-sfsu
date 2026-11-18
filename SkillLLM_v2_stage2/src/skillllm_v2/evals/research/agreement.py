"""Inter-annotator agreement for the L11 human evaluation (pure numpy; no third-party agreement package needed).

Implemented
  * ``cohen_kappa``          Cohen's kappa, optionally linear / quadratic weighted (ordinal scales), pairwise deletion of
                             missing ratings.  Weighted form: kappa_w = 1 - sum(w*O) / sum(w*E) with disagreement weights
                             w_ij = |i-j|/(q-1) (linear) or ((i-j)/(q-1))^2 (quadratic) over the ordered category list.
  * ``gwet_ac1``             Gwet's AC1 (robust to the "kappa paradox" under skewed prevalence); reported next to kappa.
  * ``krippendorff_alpha``   Krippendorff's alpha for nominal / ordinal / interval / ratio data, any number of coders,
                             missing values allowed (coincidence-matrix form of Krippendorff 2011, "Computing Krippendorff's
                             Alpha-Reliability"): alpha = 1 - (n-1) * sum_ck o_ck d2_ck / sum_ck n_c n_k d2_ck.
  * ``span_agreement``       span-level inter-annotator F1 (Hripcsak & Rothschild 2005: for span annotation without a
                             fixed item set, pairwise F1 is the appropriate agreement measure; it is symmetric in A and B).
                             Modes: exact (start, end[, label]) and overlap (one-to-one, largest overlap first).
  * ``span_token_kappa``     token-level Cohen's kappa over BIO-free per-token labels (O / label) as a chance-corrected
                             complement to span F1.
  * ``bootstrap_ci``         percentile bootstrap over independent units (items / documents), seed-fixed.
  * sample-size helpers      ``n_for_proportion``, ``n_for_mean``, ``n_for_kappa`` (Cohen 1960 large-sample SE) and
                             ``simulate_kappa_halfwidth`` (Monte-Carlo CI half-width for a binary kappa).

Missing ratings are ``None`` or ``float('nan')``.  Every function returns ``nan`` (never raises) when the statistic is
undefined, e.g. a single category used by everyone (expected disagreement 0).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LEVELS = ("nominal", "ordinal", "interval", "ratio")
Z95 = 1.959963984540054


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() == "")


# ---------------------------------------------------------------------------------------------------------------- kappa
def _pairs(a: Sequence[Any], b: Sequence[Any]) -> List[Tuple[Any, Any]]:
    if len(a) != len(b):
        raise ValueError("rating vectors must have equal length (one entry per item)")
    return [(x, y) for x, y in zip(a, b) if not _missing(x) and not _missing(y)]


def confusion(a: Sequence[Any], b: Sequence[Any], categories: Optional[Sequence[Any]] = None) -> Tuple[np.ndarray, List[Any]]:
    pairs = _pairs(a, b)
    cats = list(categories) if categories is not None else sorted({x for p in pairs for x in p}, key=lambda v: (str(type(v)), v))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)), dtype=float)
    for x, y in pairs:
        if x not in idx or y not in idx:
            raise ValueError(f"rating {x!r}/{y!r} not in categories {cats}")
        m[idx[x], idx[y]] += 1
    return m, cats


def cohen_kappa(a: Sequence[Any], b: Sequence[Any], categories: Optional[Sequence[Any]] = None,
                weights: Optional[str] = None) -> float:
    """Cohen's kappa between two raters; ``weights`` in (None, 'linear', 'quadratic'); ``categories`` fixes the order."""
    m, cats = confusion(a, b, categories)
    n = m.sum()
    q = len(cats)
    if n == 0 or q < 2:
        return float("nan")
    i, j = np.indices((q, q))
    if weights is None:
        w = (i != j).astype(float)
    elif weights == "linear":
        w = np.abs(i - j) / (q - 1)
    elif weights == "quadratic":
        w = ((i - j) / (q - 1)) ** 2
    else:
        raise ValueError(f"weights must be None, 'linear' or 'quadratic', not {weights!r}")
    obs = m / n
    exp = np.outer(m.sum(1), m.sum(0)) / n ** 2
    den = float((w * exp).sum())
    if den == 0:
        return float("nan")
    return float(1.0 - (w * obs).sum() / den)


def percent_agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    pairs = _pairs(a, b)
    return float("nan") if not pairs else sum(x == y for x, y in pairs) / len(pairs)


def gwet_ac1(a: Sequence[Any], b: Sequence[Any], categories: Optional[Sequence[Any]] = None) -> float:
    """Gwet (2008) AC1 for two raters, nominal categories."""
    m, cats = confusion(a, b, categories)
    n, q = m.sum(), len(cats)
    if n == 0 or q < 2:
        return float("nan")
    po = float(np.trace(m) / n)
    pi = (m.sum(1) + m.sum(0)) / (2 * n)
    pe = float((pi * (1 - pi)).sum() / (q - 1))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


# --------------------------------------------------------------------------------------------------- Krippendorff alpha
def _as_matrix(data: Sequence[Sequence[Any]]) -> List[List[Any]]:
    rows = [list(r) for r in data]
    if not rows or len({len(r) for r in rows}) != 1:
        raise ValueError("reliability data must be a non-empty coders x units rectangle")
    return rows


def coincidence_matrix(data: Sequence[Sequence[Any]], values: Optional[Sequence[Any]] = None) -> Tuple[np.ndarray, List[Any]]:
    """Coders x units -> (o_ck, ordered value list).  Units with < 2 pairable values are ignored."""
    rows = _as_matrix(data)
    n_units = len(rows[0])
    present = sorted({v for r in rows for v in r if not _missing(v)}, key=lambda v: (str(type(v)), v))
    vals = list(values) if values is not None else present
    idx = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)), dtype=float)
    for u in range(n_units):
        col = [r[u] for r in rows if not _missing(r[u])]
        mu = len(col)
        if mu < 2:
            continue
        counts = np.zeros(len(vals))
        for v in col:
            if v not in idx:
                raise ValueError(f"value {v!r} not in value domain")
            counts[idx[v]] += 1
        # ordered pairs from different coders: n_c*n_k for c != k, n_c*(n_c-1) for c == k
        pair = np.outer(counts, counts) - np.diag(counts)
        o += pair / (mu - 1)
    return o, vals


def _delta2(vals: Sequence[Any], n_c: np.ndarray, level: str) -> np.ndarray:
    q = len(vals)
    if level == "nominal":
        return 1.0 - np.eye(q)
    x = np.asarray([float(v) for v in vals])
    if level == "interval":
        return (x[:, None] - x[None, :]) ** 2
    if level == "ratio":
        s = x[:, None] + x[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(s == 0, 0.0, ((x[:, None] - x[None, :]) / s) ** 2)
        return d
    if level == "ordinal":
        cum = np.concatenate([[0.0], np.cumsum(n_c)])
        d = np.zeros((q, q))
        for c in range(q):
            for k in range(q):
                lo, hi = min(c, k), max(c, k)
                d[c, k] = (cum[hi + 1] - cum[lo] - (n_c[c] + n_c[k]) / 2.0) ** 2
        return d
    raise ValueError(f"level must be one of {LEVELS}")


def krippendorff_alpha(data: Sequence[Sequence[Any]], level: str = "nominal",
                       values: Optional[Sequence[Any]] = None) -> float:
    """Krippendorff's alpha.  ``data`` = coders x units (missing = None/nan).  ``values`` fixes the value domain / order
    (needed for ordinal data whose categories are not naturally sortable)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    o, vals = coincidence_matrix(data, values)
    n_c = o.sum(1)
    n = n_c.sum()
    if n <= 1 or len(vals) < 2:
        return float("nan")
    d2 = _delta2(vals, n_c, level)
    d_o = float((o * d2).sum())
    d_e = float((np.outer(n_c, n_c) * d2).sum())
    if d_e == 0:
        return float("nan")
    return 1.0 - (n - 1) * d_o / d_e


# ----------------------------------------------------------------------------------------------------- span agreement
Span = Tuple[int, int, Optional[str]]


def _norm_spans(spans: Iterable[Any], use_label: bool) -> List[Span]:
    out = set()
    for s in spans:
        if isinstance(s, Mapping):
            st, en, lab = int(s["start"]), int(s["end"]), s.get("label")
        else:
            st, en = int(s[0]), int(s[1])
            lab = s[2] if len(s) > 2 else None
        if en <= st:
            continue
        out.add((st, en, lab if use_label else None))
    return sorted(out, key=lambda t: (t[0], t[1], str(t[2])))


def _overlap_matches(a: List[Span], b: List[Span]) -> int:
    cand = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x[2] != y[2]:
                continue
            ov = min(x[1], y[1]) - max(x[0], y[0])
            if ov > 0:
                cand.append((-ov, i, j))
    used_a, used_b, m = set(), set(), 0
    for _, i, j in sorted(cand):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            m += 1
    return m


def span_unit_counts(a: Iterable[Any], b: Iterable[Any], mode: str = "exact", use_label: bool = True) -> Tuple[int, int, int]:
    """(matches, |A|, |B|) for one unit (sentence/document)."""
    sa, sb = _norm_spans(a, use_label), _norm_spans(b, use_label)
    if mode == "exact":
        m = len(set(sa) & set(sb))
    elif mode == "overlap":
        m = _overlap_matches(sa, sb)
    else:
        raise ValueError("mode must be 'exact' or 'overlap'")
    return m, len(sa), len(sb)


def f1_from_counts(counts: np.ndarray) -> float:
    m, na, nb = counts.sum(0) if counts.ndim == 2 else counts
    return float("nan") if na + nb == 0 else 2.0 * m / (na + nb)


def span_agreement(ann_a: Mapping[Hashable, Iterable[Any]], ann_b: Mapping[Hashable, Iterable[Any]], mode: str = "exact",
                   use_label: bool = True, units: Optional[Sequence[Hashable]] = None) -> Dict[str, Any]:
    """Pairwise span F1 between annotators A and B over the units both annotated (pooled/micro).

    F1 = 2*matches / (|A| + |B|): treating either annotator as the reference gives the same value.
    Returns {"f1", "matches", "n_a", "n_b", "n_units", "per_unit": np.ndarray[(m, na, nb)], "units"}."""
    ids = list(units) if units is not None else sorted(set(ann_a) & set(ann_b), key=str)
    rows = np.array([span_unit_counts(ann_a.get(u, []), ann_b.get(u, []), mode, use_label) for u in ids], dtype=float)
    rows = rows.reshape(-1, 3)
    return {"f1": f1_from_counts(rows) if len(rows) else float("nan"), "matches": int(rows[:, 0].sum()) if len(rows) else 0,
            "n_a": int(rows[:, 1].sum()) if len(rows) else 0, "n_b": int(rows[:, 2].sum()) if len(rows) else 0,
            "n_units": len(ids), "per_unit": rows, "units": ids}


def span_token_kappa(token_offsets: Mapping[Hashable, Sequence[Tuple[int, int]]], ann_a: Mapping[Hashable, Iterable[Any]],
                     ann_b: Mapping[Hashable, Iterable[Any]], use_label: bool = True) -> float:
    """Cohen's kappa over tokens: each token gets the label of the span covering it (or 'O')."""
    la: List[str] = []
    lb: List[str] = []
    for u, toks in token_offsets.items():
        for side, out in ((ann_a.get(u, []), la), (ann_b.get(u, []), lb)):
            spans = _norm_spans(side, use_label)
            for ts, te in toks:
                lab = "O"
                for s in spans:
                    if max(ts, s[0]) < min(te, s[1]):
                        lab = str(s[2]) if use_label else "SPAN"
                        break
                out.append(lab)
    return cohen_kappa(la, lb)


# ---------------------------------------------------------------------------------------------------------- bootstrap
def bootstrap_ci(n_units: int, stat: Callable[[np.ndarray], float], n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> Dict[str, Any]:
    """Percentile bootstrap over unit indices 0..n_units-1; ``stat(idx)`` computes the statistic on a resample.
    Replicates where the statistic is undefined (nan) are dropped and counted."""
    if n_units < 2:
        return {"ci": None, "n_boot": n_boot, "n_nan": n_boot, "method": "percentile bootstrap (insufficient units)"}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = stat(rng.integers(0, n_units, n_units))
    ok = vals[~np.isnan(vals)]
    if len(ok) < max(10, n_boot // 2):
        return {"ci": None, "n_boot": n_boot, "n_nan": int(n_boot - len(ok)), "method": "percentile bootstrap (too many undefined replicates)"}
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2])
    return {"ci": (float(lo), float(hi)), "n_boot": n_boot, "n_nan": int(n_boot - len(ok)),
            "method": f"{100 * (1 - alpha):.0f}% percentile bootstrap over units, seed {seed}, {n_boot} resamples"}


def kappa_with_ci(a: Sequence[Any], b: Sequence[Any], categories: Optional[Sequence[Any]] = None, weights: Optional[str] = None,
                  n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    a, b = list(a), list(b)
    keep = [i for i in range(len(a)) if not _missing(a[i]) and not _missing(b[i])]
    a2, b2 = [a[i] for i in keep], [b[i] for i in keep]
    cats = list(categories) if categories is not None else None
    est = cohen_kappa(a2, b2, cats, weights)
    ci = bootstrap_ci(len(a2), lambda idx: cohen_kappa([a2[i] for i in idx], [b2[i] for i in idx], cats, weights), n_boot, seed)
    return {"value": est, "n_items": len(a2), **ci}


def alpha_with_ci(data: Sequence[Sequence[Any]], level: str = "nominal", values: Optional[Sequence[Any]] = None,
                  n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    rows = _as_matrix(data)
    est = krippendorff_alpha(rows, level, values)
    n_units = len(rows[0])
    vals = values
    if vals is None:  # fix the value domain so resamples missing a category keep the ordinal metric well defined
        vals = sorted({v for r in rows for v in r if not _missing(v)}, key=lambda v: (str(type(v)), v))
    ci = bootstrap_ci(n_units, lambda idx: krippendorff_alpha([[r[i] for i in idx] for r in rows], level, vals), n_boot, seed)
    pairable = sum(1 for u in range(n_units) if sum(not _missing(r[u]) for r in rows) >= 2)
    return {"value": est, "n_items": pairable, **ci}


def span_f1_with_ci(ann_a: Mapping[Hashable, Iterable[Any]], ann_b: Mapping[Hashable, Iterable[Any]], mode: str = "exact",
                    use_label: bool = True, n_boot: int = 2000, seed: int = 42,
                    units: Optional[Sequence[Hashable]] = None) -> Dict[str, Any]:
    res = span_agreement(ann_a, ann_b, mode, use_label, units)
    rows = res["per_unit"]
    ci = bootstrap_ci(len(rows), lambda idx: f1_from_counts(rows[idx]), n_boot, seed)
    return {"value": res["f1"], "n_items": res["n_units"], "matches": res["matches"], "n_a": res["n_a"], "n_b": res["n_b"], **ci}


# -------------------------------------------------------------------------------------------------------- sample size
def _z(conf: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(0.5 + conf / 2)


def n_for_proportion(halfwidth: float, p: float = 0.5, conf: float = 0.95) -> int:
    """Items needed for a Wald CI of a proportion with the given half-width (p = 0.5 is the worst case)."""
    return int(math.ceil(_z(conf) ** 2 * p * (1 - p) / halfwidth ** 2))


def n_for_mean(halfwidth: float, sd: float, conf: float = 0.95) -> int:
    return int(math.ceil((_z(conf) * sd / halfwidth) ** 2))


def n_for_kappa(halfwidth: float, p_o: float, p_e: float, conf: float = 0.95) -> int:
    """Cohen (1960) large-sample SE: se(kappa) ~= sqrt(p_o (1 - p_o) / (n (1 - p_e)^2)); solve for n.  Approximate."""
    return int(math.ceil(_z(conf) ** 2 * p_o * (1 - p_o) / (halfwidth ** 2 * (1 - p_e) ** 2)))


def simulate_kappa_halfwidth(n: int, kappa: float, prevalence: float, n_sim: int = 2000, seed: int = 42) -> float:
    """Monte-Carlo half-width of the central 95% interval of kappa-hat for two binary raters with equal marginals."""
    p = prevalence
    cells = np.array([p * p + kappa * p * (1 - p), p * (1 - p) * (1 - kappa), p * (1 - p) * (1 - kappa),
                      (1 - p) ** 2 + kappa * p * (1 - p)])
    if (cells < -1e-12).any():
        raise ValueError("kappa not attainable for this prevalence")
    cells = np.clip(cells, 0, None) / cells.clip(0, None).sum()
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_sim):
        c = rng.multinomial(n, cells)
        m = np.array([[c[0], c[1]], [c[2], c[3]]], dtype=float)
        po = np.trace(m) / n
        pe = (m.sum(1) * m.sum(0)).sum() / n ** 2
        if pe < 1:
            ks.append((po - pe) / (1 - pe))
    lo, hi = np.quantile(ks, [0.025, 0.975])
    return float((hi - lo) / 2)


def expected_kappa_po_pe(kappa: float, prevalence: float) -> Tuple[float, float]:
    """(p_o, p_e) implied by a target binary kappa with equal marginals ``prevalence`` (for n_for_kappa)."""
    pe = prevalence ** 2 + (1 - prevalence) ** 2
    return pe + kappa * (1 - pe), pe
