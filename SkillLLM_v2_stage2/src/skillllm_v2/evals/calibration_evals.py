"""Calibration evals for every confidence-bearing candidate source.

Unit of analysis: a predicted span (candidate). It is *correct* when its (start, end, label) exactly matches a gold
span of its document (the project's Exact Span metric). A secondary "surface" correctness (same label + same
lower-cased surface form as some gold span of the doc) is reported in ``details`` to separate boundary/offset errors
from "is this a real skill" errors.

Confidence intervals are cluster (document-level) bootstraps: documents are the independent units, candidates inside a
document are resampled together.

Public, hand-verifiable primitives (pure functions over arrays): ``reliability_bins``, ``ece_width``, ``ece_mass``,
``mce``, ``brier``, ``nll``, ``aurc``, ``selective_accuracy``, ``auroc``, ``average_precision``,
``calibration_summary``, ``negative_controls``.

Honest-reporting notes baked in:
  * The shipped extractors emit (near-)constant, rule-assigned confidences (raw anchor = 0.86 for every span). For
    those sources ECE degenerates to |c - accuracy| and carries no ranking information (AUROC 0.5, AURC = base risk).
    The result details say so (``confidence_is_constant``).
  * ``ConfidenceCalibrator`` is a hard-threshold FILTER, not a re-mapper: it never changes a confidence value, so
    "before vs after" only measures the effect of dropping candidates.
  * The calibrator's own input is not persisted (rejected candidates are removed), so the calibrator is replayed here
    on the persisted merged candidates; the whole persisted chain (merge -> refine -> verify -> calibrate) is reported
    separately.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.contract import EvalContext, EvalResult

MODULE = "calibrate"
NLL_EPS = 1e-6
COVERAGES = (0.5, 0.7, 0.9)
MIN_SLICE_N = 30
SKILLSPAN_ROOT = Path("data/source/skillspan_hf")
DEFAULT_OUT_DIR = Path("reports/skillllm_v2/evals")

# name -> (relative path, stage description). All are test-split predictions.
DEFAULT_SOURCES: Dict[str, Tuple[str, str]] = {
    "raw_anchor": ("artifacts/skillllm_v2/mistral_lora/predictions.jsonl", "Raw Mistral LoRA anchor (constant rule-assigned confidence)"),
    "merged_candidates": ("artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/candidate_predictions.jsonl", "CandidateMerger output (mistral+gliner+dictionary, pre refine/verify/calibrate)"),
    "boundary_refiner": ("artifacts/skillllm_v2/safe_wrapper/boundary_refiner/candidate_predictions.jsonl", "anchor after BoundaryRefiner+verifier+calibrator"),
    "wrapper_final": ("artifacts/skillllm_v2/safe_wrapper/predictions.jsonl", "final safe-wrapper predictions (post calibrator)"),
    "gliner_baseline": ("artifacts/skillllm_v2/gliner_baseline/predictions.jsonl", "GLiNER baseline (continuous model scores)"),
    "bert_baseline": ("artifacts/skillllm_v2/bert_baseline/predictions.jsonl", "BERT baseline (constant rule-assigned confidence)"),
}


# --------------------------------------------------------------------------------------- primitives

def _arr(x: Sequence[float], dtype=float) -> np.ndarray:
    return np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=dtype)


def _clean(x: Optional[float]) -> Optional[float]:
    """NaN/inf -> None so results stay JSON-safe."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def reliability_bins(conf: Sequence[float], correct: Sequence[int], n_bins: int = 10, strategy: str = "width") -> List[Dict[str, Any]]:
    """Reliability-diagram data. ``strategy``: 'width' (equal-width on [0,1], last bin closed) or 'mass' (equal-count;
    ties are never split across bins, so identical confidences always share one bin and empty bins are dropped)."""
    c, y = _arr(conf), _arr(correct)
    n = len(c)
    if n == 0:
        return []
    bins: List[Dict[str, Any]] = []
    if strategy == "width":
        idx = np.minimum(np.floor(c * n_bins + 1e-12).astype(int), n_bins - 1)
        idx = np.maximum(idx, 0)
        for b in range(n_bins):
            m = idx == b
            cnt = int(m.sum())
            lo, hi = b / n_bins, (b + 1) / n_bins
            if cnt == 0:
                bins.append({"bin": b, "lo": lo, "hi": hi, "count": 0, "mean_conf": None, "accuracy": None, "gap": None})
            else:
                mc, acc = float(c[m].mean()), float(y[m].mean())
                bins.append({"bin": b, "lo": lo, "hi": hi, "count": cnt, "mean_conf": mc, "accuracy": acc, "gap": abs(acc - mc)})
    elif strategy == "mass":
        order = np.argsort(c, kind="stable")
        cs, ys = c[order], y[order]
        cuts = {0, n}
        for b in range(1, n_bins):
            pos = int(round(b * n / n_bins))
            if 0 < pos < n:
                pos = int(np.searchsorted(cs, cs[pos - 1], side="right"))  # snap to end of tie run
            cuts.add(pos)
        edges = sorted(cuts)
        for b, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
            if e <= s:
                continue
            mc, acc = float(cs[s:e].mean()), float(ys[s:e].mean())
            bins.append({"bin": b, "lo": float(cs[s]), "hi": float(cs[e - 1]), "count": e - s, "mean_conf": mc, "accuracy": acc, "gap": abs(acc - mc)})
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return bins


def _ece_from_bins(bins: List[Dict[str, Any]]) -> float:
    n = sum(b["count"] for b in bins)
    if n == 0:
        return float("nan")
    return sum(b["count"] / n * b["gap"] for b in bins if b["count"])


def _mce_from_bins(bins: List[Dict[str, Any]]) -> float:
    gaps = [b["gap"] for b in bins if b["count"]]
    return max(gaps) if gaps else float("nan")


def ece_width(conf, correct, n_bins: int = 10) -> float:
    return _ece_from_bins(reliability_bins(conf, correct, n_bins, "width"))


def ece_mass(conf, correct, n_bins: int = 10) -> float:
    return _ece_from_bins(reliability_bins(conf, correct, n_bins, "mass"))


def mce(conf, correct, n_bins: int = 10, strategy: str = "width") -> float:
    return _mce_from_bins(reliability_bins(conf, correct, n_bins, strategy))


def brier(conf, correct) -> float:
    c, y = _arr(conf), _arr(correct)
    return float(np.mean((c - y) ** 2)) if len(c) else float("nan")


def nll(conf, correct, eps: float = NLL_EPS) -> float:
    """Binary negative log-likelihood (natural log); confidences clipped to [eps, 1-eps]."""
    c, y = np.clip(_arr(conf), eps, 1 - eps), _arr(correct)
    return float(np.mean(-(y * np.log(c) + (1 - y) * np.log(1 - c)))) if len(c) else float("nan")


def _expected_errors(conf: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Expected cumulative error count after k=1..N most-confident items, averaging over random tie-breaking
    (items with identical confidence are exchangeable). Returns array of length N."""
    order = np.argsort(-conf, kind="stable")
    cs, es = conf[order], err[order].astype(float)
    n = len(cs)
    new_group = np.concatenate([[True], cs[1:] != cs[:-1]])
    starts = np.flatnonzero(new_group)
    ends = np.concatenate([starts[1:], [n]])
    cum_before = np.concatenate([[0.0], np.cumsum(es)])
    out = np.empty(n)
    for s, e in zip(starts, ends):
        rate = (cum_before[e] - cum_before[s]) / (e - s)
        out[s:e] = cum_before[s] + rate * np.arange(1, e - s + 1)
    return out


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve = mean over k=1..N of (expected error rate among the top-k most confident).
    Lower is better. Constant confidence -> exactly the base error rate (no selective ability)."""
    c, y = _arr(conf), _arr(correct)
    if len(c) == 0:
        return float("nan")
    exp_err = _expected_errors(c, 1 - y)
    return float(np.mean(exp_err / np.arange(1, len(c) + 1)))


def aurc_oracle(correct) -> float:
    """AURC of a perfect ranker (all correct first) -- the lower bound for the same accuracy."""
    y = _arr(correct)
    if len(y) == 0:
        return float("nan")
    err_sorted = np.sort(1 - y)  # errors last
    return float(np.mean(np.cumsum(err_sorted) / np.arange(1, len(y) + 1)))


def selective_accuracy(conf, correct, coverage: float) -> float:
    """Accuracy on the ceil(coverage*N) most-confident candidates (tie-aware expectation)."""
    c, y = _arr(conf), _arr(correct)
    n = len(c)
    if n == 0:
        return float("nan")
    k = min(n, max(1, int(math.ceil(coverage * n - 1e-9))))
    exp_err = _expected_errors(c, 1 - y)
    return float(1 - exp_err[k - 1] / k)


def risk_coverage_curve(conf, correct) -> List[Dict[str, float]]:
    """Risk-coverage points, one per distinct confidence value (tie-group boundary), most confident first."""
    c, y = _arr(conf), _arr(correct)
    n = len(c)
    if n == 0:
        return []
    exp_err = _expected_errors(c, 1 - y)
    cs = np.sort(c)[::-1]
    ends = np.flatnonzero(np.concatenate([cs[1:] != cs[:-1], [True]]))
    return [{"coverage": float((e + 1) / n), "risk": float(exp_err[e] / (e + 1)), "threshold": float(cs[e])} for e in ends]


def auroc(score, positive) -> float:
    """Rank-based AUROC with average ranks for ties (0.5 for a constant score)."""
    s, y = _arr(score), _arr(positive).astype(int)
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks_u = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = ranks_u[inv]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(score, positive) -> float:
    """Step-wise average precision (sklearn definition; tied scores form one threshold)."""
    s, y = _arr(score), _arr(positive).astype(int)
    total_pos = int(y.sum())
    if total_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    ss, ys = s[order], y[order]
    ends = np.flatnonzero(np.concatenate([ss[1:] != ss[:-1], [True]]))
    tp = np.cumsum(ys)[ends]
    k = ends + 1
    precision = tp / k
    recall = tp / total_pos
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def calibration_summary(conf, correct, n_bins: int = 10) -> Dict[str, Any]:
    c, y = _arr(conf), _arr(correct)
    n = len(c)
    bw = reliability_bins(c, y, n_bins, "width")
    bm = reliability_bins(c, y, n_bins, "mass")
    base = float(y.mean()) if n else float("nan")
    out: Dict[str, Any] = {
        "n": n,
        "base_rate": base,
        "mean_confidence": float(c.mean()) if n else float("nan"),
        "ece_width": _ece_from_bins(bw),
        "ece_mass": _ece_from_bins(bm),
        "mce_width": _mce_from_bins(bw),
        "mce_mass": _mce_from_bins(bm),
        "brier": brier(c, y),
        "nll": nll(c, y),
        "aurc": aurc(c, y),
        "aurc_oracle": aurc_oracle(y),
        "aurc_random": 1 - base if n else float("nan"),
        "auroc": auroc(c, y),
        "n_distinct_confidences": int(len(np.unique(c))) if n else 0,
        "confidence_is_constant": bool(n and len(np.unique(c)) == 1),
        "bins_width": bw,
        "bins_mass": bm,
    }
    out["e_aurc"] = out["aurc"] - out["aurc_oracle"] if n else float("nan")
    for cov in COVERAGES:
        out[f"sel_acc@{int(cov * 100)}"] = selective_accuracy(c, y, cov)
    return out


_SCALAR_KEYS = ("ece_width", "ece_mass", "mce_width", "brier", "nll", "aurc", "auroc", "sel_acc@50", "sel_acc@70", "sel_acc@90", "base_rate")


def _scalars(conf: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    bw = reliability_bins(conf, y, n_bins, "width")
    bm = reliability_bins(conf, y, n_bins, "mass")
    d = {
        "ece_width": _ece_from_bins(bw), "ece_mass": _ece_from_bins(bm), "mce_width": _mce_from_bins(bw),
        "brier": brier(conf, y), "nll": nll(conf, y), "aurc": aurc(conf, y), "auroc": auroc(conf, y),
        "base_rate": float(y.mean()),
    }
    exp_err = _expected_errors(conf, 1 - y)
    n = len(conf)
    for cov in COVERAGES:
        k = min(n, max(1, int(math.ceil(cov * n - 1e-9))))
        d[f"sel_acc@{int(cov * 100)}"] = float(1 - exp_err[k - 1] / k)
    return d


# --------------------------------------------------------------------------------------- bootstraps

def _boot_idx(doc_idx: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Candidate indices of a document-level resample: doc d is repeated counts[d] times."""
    return np.repeat(np.arange(len(doc_idx)), counts[doc_idx])


def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    s = samples[np.isfinite(samples)]
    if len(s) < 10:
        return None
    return (float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2))))


def cluster_bootstrap(doc_idx: Sequence[int], n_docs: int, conf: Sequence[float], y: Sequence[int],
                      fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]] = _scalars,
                      n_boot: int = 1000, seed: int = 42) -> Dict[str, Optional[Tuple[float, float]]]:
    """95% percentile CIs of every metric in ``fn(conf, y)`` under a document-level bootstrap."""
    d, c, yy = _arr(doc_idx, int), _arr(conf), _arr(y)
    if len(c) == 0 or n_docs == 0:
        return {}
    rng = np.random.default_rng(seed)
    draws: Dict[str, List[float]] = {}
    p = np.full(n_docs, 1.0 / n_docs)
    for _ in range(n_boot):
        counts = rng.multinomial(n_docs, p)
        idx = _boot_idx(d, counts)
        if len(idx) == 0:
            continue
        for k, v in fn(c[idx], yy[idx]).items():
            draws.setdefault(k, []).append(v)
    return {k: _percentile_ci(np.asarray(v)) for k, v in draws.items()}


def paired_cluster_bootstrap(arm_a: Tuple[Sequence[int], Sequence[float], Sequence[int]],
                             arm_b: Tuple[Sequence[int], Sequence[float], Sequence[int]], n_docs: int,
                             metrics: Sequence[str] = ("ece_width", "brier"), n_boot: int = 1000, seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Paired document-level bootstrap of metric(a) - metric(b). Both arms are (doc_idx, conf, correct) over the
    same document index space [0, n_docs); one set of document draws is shared by both arms."""
    da, ca, ya = (_arr(arm_a[0], int), _arr(arm_a[1]), _arr(arm_a[2]))
    db, cb, yb = (_arr(arm_b[0], int), _arr(arm_b[1]), _arr(arm_b[2]))
    rng = np.random.default_rng(seed)
    p = np.full(n_docs, 1.0 / n_docs)
    point_a, point_b = _scalars(ca, ya), _scalars(cb, yb)
    deltas: Dict[str, List[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        counts = rng.multinomial(n_docs, p)
        ia, ib = _boot_idx(da, counts), _boot_idx(db, counts)
        if len(ia) == 0 or len(ib) == 0:
            continue
        sa, sb = _scalars(ca[ia], ya[ia]), _scalars(cb[ib], yb[ib])
        for m in metrics:
            deltas[m].append(sa[m] - sb[m])
    out: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        arr = np.asarray(deltas[m])
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10:
            out[m] = {"delta": point_a[m] - point_b[m], "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan")}
            continue
        below = float((arr <= 0).mean())
        above = float((arr >= 0).mean())
        out[m] = {
            "delta": float(point_a[m] - point_b[m]),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
            "p": max(min(1.0, 2 * min(below, above)), 1.0 / len(arr)),  # floored at 1/n_boot (resolution limit)
        }
    return out


# --------------------------------------------------------------------------------------- controls

def expected_ece_random_uniform(base_rate: float, n_bins: int = 10) -> float:
    """Large-sample ECE of confidences ~ U(0,1) independent of correctness (Bernoulli(base_rate)): the mean over
    equal-width bins of |base_rate - bin midpoint|."""
    mids = (np.arange(n_bins) + 0.5) / n_bins
    return float(np.mean(np.abs(base_rate - mids)))


def negative_controls(conf: Sequence[float], correct: Sequence[int], seed: int = 42, n_rep: int = 200, n_bins: int = 10) -> Dict[str, Any]:
    """Controls that expose what the metrics can and cannot detect on this label set.

    random:   uniform random confidences -> ECE should sit near its analytic expectation, AUROC ~ 0.5, AURC ~ base risk.
    constant: constant confidence = base rate -> ECE ~ 0 (perfect "calibration") while AURC = base risk (zero
              selective ability): ECE alone is gameable.
    permuted: the real confidences with labels shuffled -> finite-sample null for the real ECE / AUROC.
    """
    c, y = _arr(conf), _arr(correct)
    n = len(c)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    p = float(y.mean())
    actual = calibration_summary(c, y, n_bins)
    rand_ece, rand_aurc, rand_auroc = [], [], []
    for _ in range(n_rep):
        r = rng.random(n)
        rand_ece.append(ece_width(r, y, n_bins))
        rand_aurc.append(aurc(r, y))
        rand_auroc.append(auroc(r, y))
    const = np.full(n, p)
    perm_ece, perm_auroc = [], []
    for _ in range(n_rep):
        yp = rng.permutation(y)
        perm_ece.append(ece_width(c, yp, n_bins))
        perm_auroc.append(auroc(c, yp))
    exp_ece = expected_ece_random_uniform(p, n_bins)
    const_stats = {
        "ece_width": ece_width(const, y, n_bins), "ece_mass": ece_mass(const, y, n_bins),
        "brier": brier(const, y), "aurc": aurc(const, y), "auroc": auroc(const, y),
        "brier_expected_p(1-p)": p * (1 - p), "aurc_expected_1-p": 1 - p,
    }
    gameable = bool(const_stats["ece_width"] <= actual["ece_width"] + 1e-12 and const_stats["aurc"] >= actual["aurc"] - 1e-12)
    return {
        "base_rate": p,
        "random": {
            "ece_width_mean": float(np.mean(rand_ece)), "ece_width_sd": float(np.std(rand_ece)),
            "ece_width_analytic_expectation": exp_ece,
            "ece_gap_vs_analytic": float(np.mean(rand_ece) - exp_ece),
            "aurc_mean": float(np.mean(rand_aurc)), "aurc_expected_1-p": 1 - p,
            "auroc_mean": float(np.nanmean(rand_auroc)), "n_rep": n_rep,
        },
        "constant_base_rate": const_stats,
        "permuted_labels_null": {
            "ece_width_mean": float(np.mean(perm_ece)),
            "ece_width_p95": float(np.percentile(perm_ece, 95)),
            "auroc_mean": float(np.nanmean(perm_auroc)),
            "auroc_p95": float(np.nanpercentile(perm_auroc, 95)),
            "actual_ece_above_null_p95": bool(actual["ece_width"] > np.percentile(perm_ece, 95)),
            "actual_auroc_above_null_p95": bool(actual["auroc"] > np.nanpercentile(perm_auroc, 95)) if np.isfinite(actual["auroc"]) else None,
        },
        "ece_alone_is_gameable_here": gameable,
        "actual": {k: actual[k] for k in ("ece_width", "ece_mass", "brier", "aurc", "auroc")},
    }


# --------------------------------------------------------------------------------------- rendering

def reliability_markdown(name: str, bins_width: List[Dict[str, Any]], bins_mass: List[Dict[str, Any]], bar_width: int = 20) -> str:
    def table(title: str, bins: List[Dict[str, Any]]) -> List[str]:
        lines = [f"**{title}**", "", "| bin | range | n | mean conf | accuracy | gap | acc bar (# = accuracy, + = mean conf) |", "|---|---|---|---|---|---|---|"]
        for b in bins:
            if not b["count"]:
                lines.append(f"| {b['bin']} | [{b['lo']:.2f}, {b['hi']:.2f}] | 0 | - | - | - | |")
                continue
            acc_pos = int(round(b["accuracy"] * bar_width))
            conf_pos = min(bar_width - 1, int(round(b["mean_conf"] * bar_width)))
            bar = ["#" if i < acc_pos else "." for i in range(bar_width)]
            bar[conf_pos] = "+"
            lines.append(f"| {b['bin']} | [{b['lo']:.2f}, {b['hi']:.2f}] | {b['count']} | {b['mean_conf']:.3f} | {b['accuracy']:.3f} | {b['gap']:.3f} | `{''.join(bar)}` |")
        return lines + [""]

    return "\n".join([f"### {name}", ""] + table("Equal-width bins", bins_width) + table("Equal-mass bins", bins_mass))


def save_reliability(out_dir: Path, reliability: Dict[str, Dict[str, Any]], tag: str = "calibration") -> List[str]:
    """Write reliability JSON + markdown (+ PNG when matplotlib is importable). Returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    jp = out_dir / f"{tag}_reliability.json"
    jp.write_text(json.dumps(reliability, indent=2), encoding="utf-8")
    written.append(str(jp))
    md = ["# Reliability diagrams (exact-span correctness)", ""]
    for name, d in reliability.items():
        md.append(reliability_markdown(name, d["bins_width"], d["bins_mass"]))
    mp = out_dir / f"{tag}_reliability.md"
    mp.write_text("\n".join(md), encoding="utf-8")
    written.append(str(mp))
    try:  # matplotlib only if already importable
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(reliability)
        fig, axes = plt.subplots(1, max(1, len(names)), figsize=(3.2 * max(1, len(names)), 3.4), squeeze=False)
        for ax, name in zip(axes[0], names):
            bw = [b for b in reliability[name]["bins_width"] if b["count"]]
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.plot([b["mean_conf"] for b in bw], [b["accuracy"] for b in bw], "o-")
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("mean confidence")
            ax.set_ylabel("accuracy")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        fig.tight_layout()
        pp = out_dir / f"{tag}_reliability.png"
        fig.savefig(pp, dpi=120)
        plt.close(fig)
        written.append(str(pp))
    except Exception:
        pass
    return written


# --------------------------------------------------------------------------------------- data

def _norm_surface(s: str) -> str:
    return " ".join(s.lower().split())


def load_gold(ctx: EvalContext) -> Optional[Dict[str, Dict[str, Any]]]:
    """Gold per document id ('test-N'): exact spans, surface set and text. None when the dataset is unavailable."""
    if "gold_docs" in ctx.extra:
        docs = ctx.extra["gold_docs"]
    else:
        try:
            from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

            docs = load_skillspan_split(ctx.path(*SKILLSPAN_ROOT.parts), "test")
        except Exception:
            return None
    return {
        d["id"]: {
            "spans": {(g["start"], g["end"], g["label"]) for g in d["gold_spans"]},
            "surfaces": {(_norm_surface(g["surface_form"]), g["label"]) for g in d["gold_spans"]},
            "text": d.get("text", ""),
            "n_gold": len(d["gold_spans"]),
        }
        for d in docs
    }


def load_seen_texts(ctx: EvalContext) -> Optional[set]:
    """Texts of train+dev SkillSpan docs (the extractor / verifier were fit or early-stopped on these)."""
    if "seen_texts" in ctx.extra:
        return set(ctx.extra["seen_texts"])
    try:
        from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

        seen: set = set()
        for split in ("train", "dev"):
            seen |= {d["text"] for d in load_skillspan_split(ctx.path(*SKILLSPAN_ROOT.parts), split)}
        return seen
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_source_key(c: Dict[str, Any]) -> str:
    srcs = (c.get("metadata") or {}).get("sources") or [c.get("source", "?")]
    return "+".join(sorted(srcs))


def flatten_candidates(rows: List[Dict[str, Any]], gold: Dict[str, Dict[str, Any]], seen_texts: Optional[set] = None,
                       drop_docs_without_gold: bool = True) -> Dict[str, Any]:
    """Flatten per-document predictions into parallel arrays. Document index space = documents that have >= 1
    candidate (documents without candidates carry no calibration information)."""
    doc_of: Dict[str, int] = {}
    cols: Dict[str, List[Any]] = {k: [] for k in ("doc", "conf", "y", "y_surface", "label", "source_key", "n_words", "seen", "anchor", "cand")}
    skipped_no_gold = 0
    for r in rows:
        g = gold.get(r["id"])
        if g is None:
            skipped_no_gold += len(r.get("predictions", []))
            continue
        for c in r.get("predictions", []):
            d = doc_of.setdefault(r["id"], len(doc_of))
            key = (c.get("start"), c.get("end"), c.get("label"))
            cols["doc"].append(d)
            cols["conf"].append(float(c["confidence"]))
            cols["y"].append(int(key in g["spans"]))
            cols["y_surface"].append(int((_norm_surface(c["surface_form"]), c["label"]) in g["surfaces"]))
            cols["label"].append(c["label"])
            sk = candidate_source_key(c)
            cols["source_key"].append(sk)
            cols["n_words"].append(len(c["surface_form"].split()))
            cols["seen"].append(bool(seen_texts is not None and (g["text"] in seen_texts)))
            cols["anchor"].append("mistral" in sk.split("+"))
            cols["cand"].append(c)
    out = {k: (np.asarray(v) if k not in ("label", "source_key", "cand") else v) for k, v in cols.items()}
    out["n_docs"] = len(doc_of)
    out["doc_ids"] = list(doc_of)
    out["skipped_no_gold"] = skipped_no_gold
    return out


def _subset(fl: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    """Subset of a flattened source; document indices are re-compacted."""
    m = np.asarray(mask, dtype=bool)
    docs = fl["doc"][m]
    uniq, new_doc = np.unique(docs, return_inverse=True)
    sub = {k: (v[m] if isinstance(v, np.ndarray) else [x for x, keep in zip(v, m) if keep]) for k, v in fl.items() if k in ("doc", "conf", "y", "y_surface", "label", "source_key", "n_words", "seen", "anchor", "cand")}
    sub["doc"] = new_doc
    sub["n_docs"] = int(len(uniq))
    return sub


# --------------------------------------------------------------------------------------- run

def _res(name: str, metric: str, value: Optional[float], n: int, dataset: str, ci=None, details=None, caveats=None,
         status: str = "ok", reason: str = "") -> EvalResult:
    return EvalResult(eval_name=name, module=MODULE, metric=metric, value=_clean(value), n=int(n), dataset=dataset,
                      ci=ci if ci is not None and None not in ci else None, details=details or {}, caveats=list(caveats or []),
                      status=status, reason=reason)


def _ci_pair(cis: Dict[str, Any], key: str) -> Optional[Tuple[float, float]]:
    v = cis.get(key)
    return v if v else None


def _source_caveats(name: str, summ: Dict[str, Any], fl: Dict[str, Any], note: str) -> List[str]:
    cav = [f"source: {note}", "correctness = exact (start,end,label) match to SkillSpan test gold"]
    if summ["confidence_is_constant"]:
        cav.append(f"confidence is CONSTANT ({summ['mean_confidence']:.3f}) for every span: it is a rule-assigned score, not a probability; "
                   "ECE equals |c - accuracy| and there is no ranking information (AUROC 0.5, AURC = base risk)")
    seen_frac = float(np.mean(fl["seen"])) if len(fl["seen"]) else 0.0
    if seen_frac > 0:
        cav.append(f"LEAKAGE FLAG: {seen_frac:.1%} of candidates come from test docs whose exact text also occurs in the train/dev split "
                   "(duplicate sentences across SkillSpan splits); see *_text_disjoint results")
    return cav


def run(ctx: EvalContext) -> List[EvalResult]:
    results: List[EvalResult] = []
    gold = load_gold(ctx)
    if gold is None:
        return [_res("calibration.all", MODULE, None, 0, "skillspan_test", status="skipped",
                     reason="SkillSpan test gold (data/source/skillspan_hf/test.json) not available offline")]
    seen_texts = load_seen_texts(ctx)
    sources: Dict[str, Tuple[List[Dict[str, Any]], str]] = {}
    if "calibration_sources" in ctx.extra:
        for k, v in ctx.extra["calibration_sources"].items():
            sources[k] = (v, "injected")
    else:
        for name, (rel, note) in DEFAULT_SOURCES.items():
            p = ctx.path(rel)
            if not p.exists():
                results.append(_res(f"calibration.{name}", MODULE, None, 0, "skillspan_test", status="skipped", reason=f"missing artifact {rel}"))
                continue
            rows = _read_jsonl(p)
            if ctx.smoke:
                rows = rows[:40]
            sources[name] = (rows, note)

    n_boot = ctx.n_boot
    flat: Dict[str, Dict[str, Any]] = {}
    reliability: Dict[str, Dict[str, Any]] = {}
    for name, (rows, note) in sources.items():
        fl = flatten_candidates(rows, gold, seen_texts)
        if len(fl["conf"]) == 0:
            results.append(_res(f"calibration.{name}", MODULE, None, 0, "skillspan_test", status="skipped", reason="no candidates matched gold documents"))
            continue
        flat[name] = fl
        n = len(fl["conf"])
        dataset = f"skillspan_test ({fl['n_docs']} docs with candidates)"
        boot = min(n_boot, 1000 if n > 10000 else n_boot)
        summ = calibration_summary(fl["conf"], fl["y"])
        cis = cluster_bootstrap(fl["doc"], fl["n_docs"], fl["conf"], fl["y"], n_boot=boot, seed=ctx.seed)
        surf = {k: v for k, v in calibration_summary(fl["conf"], fl["y_surface"]).items() if k in ("base_rate", "ece_width", "ece_mass", "brier", "aurc")}
        cav = _source_caveats(name, summ, fl, note)
        reliability[name] = {"bins_width": summ["bins_width"], "bins_mass": summ["bins_mass"], "n": n}
        base_details = {
            "n_candidates": n, "n_docs": fl["n_docs"], "n_bootstrap": boot, "ci_method": "document-cluster percentile bootstrap",
            "surface_match_secondary": surf, "n_distinct_confidences": summ["n_distinct_confidences"],
            "confidence_is_constant": summ["confidence_is_constant"], "skipped_candidates_no_gold_doc": fl["skipped_no_gold"],
        }
        for key, label in (("ece_width", "ECE (10 equal-width bins)"), ("ece_mass", "ECE (10 equal-mass bins)"), ("mce_width", "MCE (equal-width)"),
                           ("brier", "Brier"), ("nll", "NLL"), ("aurc", "AURC (lower=better)"), ("auroc", "AUROC conf->correct"),
                           ("sel_acc@50", "selective accuracy @50% coverage"), ("sel_acc@70", "selective accuracy @70% coverage"),
                           ("sel_acc@90", "selective accuracy @90% coverage"), ("base_rate", "span precision (base rate)")):
            det = dict(base_details)
            if key == "ece_width":
                det["bins"] = summ["bins_width"]
            if key == "ece_mass":
                det["bins"] = summ["bins_mass"]
            if key == "aurc":
                det.update({"aurc_oracle": summ["aurc_oracle"], "aurc_random": summ["aurc_random"], "e_aurc": summ["e_aurc"],
                            "risk_coverage_curve": risk_coverage_curve(fl["conf"], fl["y"])})
            results.append(_res(f"calibration.{name}.{key}", f"{key}", summ[key], n, dataset, _ci_pair(cis, key), det, cav))

        # leakage-safe view: only docs whose text is NOT in train/dev
        if seen_texts is not None:
            keep = ~fl["seen"]
            if keep.sum() >= MIN_SLICE_N and keep.sum() < len(keep):
                sub = _subset(fl, keep)
                ss = calibration_summary(sub["conf"], sub["y"])
                sci = cluster_bootstrap(sub["doc"], sub["n_docs"], sub["conf"], sub["y"], n_boot=boot, seed=ctx.seed)
                results.append(_res(f"calibration.{name}.ece_width_text_disjoint", "ece_width", ss["ece_width"], len(sub["conf"]),
                                    f"skillspan_test text-disjoint from train/dev ({sub['n_docs']} docs)", _ci_pair(sci, "ece_width"),
                                    {"brier": ss["brier"], "base_rate": ss["base_rate"], "n_removed": int((~keep).sum())},
                                    ["candidates from docs whose text occurs in train/dev removed"]))

        # per-label and per-source-key breakdown
        for dim, col in (("label", fl["label"]), ("source", fl["source_key"])):
            col_arr = np.asarray(col)
            for val in sorted(set(col_arr.tolist())):
                m = col_arr == val
                nm = int(m.sum())
                en = f"calibration.{name}.by_{dim}.{val}.ece_width"
                if nm < MIN_SLICE_N:
                    results.append(_res(en, "ece_width", None, nm, dataset, status="skipped", reason=f"slice n={nm} < {MIN_SLICE_N}"))
                    continue
                sub = _subset(fl, m)
                ss = calibration_summary(sub["conf"], sub["y"])
                sci = cluster_bootstrap(sub["doc"], sub["n_docs"], sub["conf"], sub["y"], n_boot=min(boot, 500), seed=ctx.seed)
                results.append(_res(en, "ece_width", ss["ece_width"], nm, dataset, _ci_pair(sci, "ece_width"),
                                    {"brier": ss["brier"], "base_rate": ss["base_rate"], "mean_confidence": ss["mean_confidence"],
                                     "aurc": ss["aurc"], "auroc": ss["auroc"], "ece_mass": ss["ece_mass"]},
                                    ["breakdown slice; CI from document-cluster bootstrap inside the slice"]))

        # negative controls on this source's real labels
        if n >= MIN_SLICE_N:
            nc = negative_controls(fl["conf"], fl["y"], seed=ctx.seed, n_rep=50 if ctx.smoke else 200)
            results.append(_res(f"calibration.{name}.control_random_ece", "ece_width_random_conf", nc["random"]["ece_width_mean"], n, dataset,
                                None, nc["random"], ["NEGATIVE CONTROL: uniform random confidences on the real labels; should be near the analytic expectation"]))
            results.append(_res(f"calibration.{name}.control_constant_ece", "ece_width_constant_base_rate", nc["constant_base_rate"]["ece_width"], n, dataset,
                                None, {**nc["constant_base_rate"], "actual": nc["actual"], "ece_alone_is_gameable_here": nc["ece_alone_is_gameable_here"],
                                       "permuted_labels_null": nc["permuted_labels_null"]},
                                ["NEGATIVE CONTROL: constant confidence = base rate gives ECE~0 but AURC = base risk; ECE alone is gameable"]))

    # ---- calibrator before / after (paired) ----
    results.extend(_calibrator_before_after(ctx, flat, gold, seen_texts, n_boot))

    # ---- artifacts ----
    out_dir = ctx.extra.get("out_dir")
    if out_dir is None and ctx.extra.get("write_artifacts", False):
        out_dir = ctx.path(*DEFAULT_OUT_DIR.parts)
    if out_dir is not None and reliability:
        written = save_reliability(Path(out_dir), reliability)
        results.append(_res("calibration.artifacts", "files_written", float(len(written)), len(reliability), "n/a", None, {"paths": written}))
    return results


def _calibrator_before_after(ctx: EvalContext, flat: Dict[str, Dict[str, Any]], gold: Dict[str, Dict[str, Any]],
                             seen_texts: Optional[set], n_boot: int) -> List[EvalResult]:
    out: List[EvalResult] = []
    name = "calibration.calibrator"
    if "merged_candidates" not in flat:
        return [_res(f"{name}.before_after", MODULE, None, 0, "skillspan_test", status="skipped", reason="merged candidates (calibrator input) not available")]
    try:
        from skillllm_v2.verify.confidence_calibrator import DEFAULT_TUNED_CONFIG_PATH, ConfidenceCalibrator

        tuned_path = ctx.path(DEFAULT_TUNED_CONFIG_PATH)
        cal = ConfidenceCalibrator(tuned_config_path=str(tuned_path))
        tuned_present = tuned_path.exists()
    except Exception as exc:  # pragma: no cover
        return [_res(f"{name}.before_after", MODULE, None, 0, "skillspan_test", status="failed", reason=f"cannot load ConfidenceCalibrator: {exc}")]

    fl = flat["merged_candidates"]
    keep = np.array([c["confidence"] >= cal.threshold_for(c) for c in fl["cand"]], dtype=bool)
    dropped_correct = int(((~keep) & (fl["y"] == 1)).sum())
    before = (fl["doc"], fl["conf"], fl["y"])
    after = (fl["doc"][keep], fl["conf"][keep], fl["y"][keep])
    dataset = f"skillspan_test ({fl['n_docs']} docs with candidates); identical merged candidates, calibrator replayed"
    cav = [
        "ConfidenceCalibrator is a hard-threshold filter and never changes a confidence value; the delta measures only the effect of dropping candidates",
        f"calibrator config: {'tuned_config.yaml present' if tuned_present else 'tuned_config.yaml ABSENT -> library defaults'} (thresholds hard={cal.hard_skill_threshold}, soft={cal.soft_skill_threshold}, dict_only={cal.dictionary_only_threshold}, multi={cal.multi_source_threshold}, protected={sorted(cal.protect_anchor_sources)})",
        "anchor-derived candidates are never rejected by design, so the calibrator can only touch the few non-anchor additions",
    ]
    pb = paired_cluster_bootstrap(after, before, fl["n_docs"], metrics=("ece_width", "brier", "aurc"), n_boot=n_boot, seed=ctx.seed)
    sb, sa = _scalars(before[1], before[2]), _scalars(after[1], after[2])
    n_dropped = int((~keep).sum())
    common = {"n_before": int(len(keep)), "n_after": int(keep.sum()), "n_dropped": n_dropped, "n_dropped_that_were_correct": dropped_correct,
              "tuned_config_present": tuned_present, "before": sb, "after": sa}
    for m in ("ece_width", "brier", "aurc"):
        d = pb[m]
        ci = (d["ci_low"], d["ci_high"]) if math.isfinite(d["ci_low"]) else None
        out.append(_res(f"{name}.delta_{m}", f"delta_{m}(after-before)", d["delta"], fl["n_docs"], dataset, ci,
                        {**common, "p_value": _clean(d["p"]), "paired": "document-cluster bootstrap, shared draws"}, cav))
    # whole persisted chain: merged -> wrapper_final
    if "wrapper_final" in flat:
        wf = flat["wrapper_final"]
        all_docs = {i: k for k, i in enumerate(sorted(set(fl["doc_ids"]) | set(wf["doc_ids"])))}
        da = np.array([all_docs[wf["doc_ids"][i]] for i in wf["doc"]], dtype=int)
        db = np.array([all_docs[fl["doc_ids"][i]] for i in fl["doc"]], dtype=int)
        pc = paired_cluster_bootstrap((da, wf["conf"], wf["y"]), (db, fl["conf"], fl["y"]), len(all_docs), metrics=("ece_width", "brier"), n_boot=n_boot, seed=ctx.seed)
        for m in ("ece_width", "brier"):
            d = pc[m]
            ci = (d["ci_low"], d["ci_high"]) if math.isfinite(d["ci_low"]) else None
            out.append(_res(f"{name}.chain_delta_{m}", f"delta_{m}(wrapper_final-merged)", d["delta"], len(all_docs),
                            "skillspan_test (union of docs)", ci, {"p_value": _clean(d["p"]), "n_before": int(len(fl["conf"])), "n_after": int(len(wf["conf"]))},
                            ["whole persisted chain (merge -> boundary refine/offset realign -> verify -> calibrate), not calibrator-only; "
                             "offset realignment changes which candidates are exact-correct, so this mixes boundary repair with calibration"]))
    return out
