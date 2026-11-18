"""L04_calibration — calibration and selective prediction (abstention) for every confidence-bearing span source.

Sources (test split; dev used ONLY to fit calibrators / select the score, Decisions D-A2-03):
  * anchor_logprob   Raw Mistral LoRA anchor re-run by A3 with token log-probs (artifacts/skillllm_v2/evals_gpu/
                     anchor_{test,dev}_logprobs, DONE.json); pending_gpu until both are complete.
  * anchor_stored    stored anchor predictions, constant confidence 0.86 (the constant-confidence baseline).
  * gliner           zero-shot GLiNER scores (A2 CPU run on dev + test, same model/threshold as the wrapper).
  * dictionary       DictionaryExtractor candidates (0.82 exact alias, <=0.72 fuzzy), dev + test.
  * wrapper          final safe-wrapper confidences (test: stored deployed output; dev: wrapper replayed on A3's dev
                     anchor predictions + dev GLiNER + dictionary).
  * v3 / v4          QLoRA loop adapters with test-only log-probs -> uncalibrated metrics + 5-fold cross-fitted (on TEST)
                     calibrators, labelled SECONDARY.

Metrics: ECE (equal-width 10/15), adaptive ECE (equal-mass 10/15), MCE, Brier, NLL, AURC, e-AURC, AUROC, reliability
diagrams (PNG + CSV), risk-coverage curves with selective precision and resulting micro-F1 (abstained = missed) at
100/95/90/80/70/50 % coverage; document-cluster bootstrap CIs. H4 verdict.

    python -m skillllm_v2.evals.research.calibration_selective [--smoke] [--seed 42] [--n-boot 2000]
    python -m skillllm_v2.evals.research.calibration_selective --build-gliner-cache   # GPU venv (gliner installed), CPU ok
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.research import stats as S
from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, md_table, record, rel, write_layer

LAYER = "L04_calibration"
DERIVED = REPO_ROOT / "inputs" / "evals" / "research" / LAYER
P = {
    "data_root": "data/source/skillspan_hf",
    "gold_test": "data/source/skillspan_hf/test.json",
    "gold_dev": "data/source/skillspan_hf/dev.json",
    "anchor_test_lp": "artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs",
    "anchor_dev_lp": "artifacts/skillllm_v2/evals_gpu/anchor_dev_logprobs",
    "anchor_stored": "artifacts/skillllm_v2/mistral_lora/predictions.jsonl",
    "wrapper_test": "artifacts/skillllm_v2/safe_wrapper/predictions.jsonl",
    "v3": "artifacts/skillllm_v2/qlora_loop_v3_eval/predictions.jsonl",
    "v4": "artifacts/skillllm_v2/qlora_loop_v4_eval/predictions.jsonl",
    "v3_report": "reports/skillllm_v2/logprob_confidence_v3.json",
    "v4_report": "reports/skillllm_v2/logprob_confidence_v4.json",
    "gliner_stored_test": "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl",
    "gliner_dev_cache": "inputs/evals/research/L04_calibration/gliner_dev.jsonl",
    "gliner_test_cache": "inputs/evals/research/L04_calibration/gliner_test.jsonl",
}
COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
EPS = 1e-6
NLL_EPS = 1e-6


# ============================================================================ data
def load_gold(split: str) -> Dict[str, Dict[str, Any]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    docs = load_skillspan_split(REPO_ROOT / P["data_root"], split)
    return {d["id"]: d for d in docs if d["gold_spans"]}


def read_rows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r.get("predictions", [])
    return out


def done(job_dir: Path) -> bool:
    return (job_dir / "DONE.json").exists() and (job_dir / "predictions.jsonl").exists()


def key(sp: Dict[str, Any]) -> Tuple[Any, Any, str]:
    return (sp.get("start"), sp.get("end"), sp.get("label"))


def lp_logit(lp: np.ndarray) -> np.ndarray:
    """logit(exp(lp)) computed stably: lp - log(-expm1(lp)); lp clipped to <= -1e-12."""
    lp = np.minimum(np.asarray(lp, dtype=float), -1e-12)
    return lp - np.log(-np.expm1(lp))


def prob_logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


class Table:
    """Flattened spans of one source on one split: doc index, correctness, features; plus per-doc gold counts."""

    def __init__(self, name: str, split: str, gold: Dict[str, Dict[str, Any]], preds: Dict[str, List[Dict[str, Any]]],
                 extra: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], Dict[str, float]]] = None,
                 ids: Optional[Sequence[str]] = None) -> None:
        self.name, self.split = name, split
        self.doc_ids = list(ids) if ids is not None else [i for i in gold]
        self.n_gold_doc = np.array([len({key(g) for g in gold[i]["gold_spans"]}) for i in self.doc_ids], dtype=float)
        cols: Dict[str, List[Any]] = defaultdict(list)
        for d, i in enumerate(self.doc_ids):
            gk = {key(g) for g in gold[i]["gold_spans"]}
            seen = set()
            for sp in preds.get(i, []):
                k = key(sp)
                if k in seen:  # evaluator de-duplicates per document
                    continue
                seen.add(k)
                cols["doc"].append(d)
                cols["y"].append(int(k in gk))
                cols["conf"].append(float(sp.get("confidence", np.nan)))
                cols["label_soft"].append(int(sp.get("label") == "SOFT_SKILL"))
                cols["n_words"].append(max(1, len(str(sp.get("surface_form", "")).split())))
                lp = (sp.get("metadata") or {}).get("logprob") or {}
                for f in ("sum", "mean", "min"):
                    cols[f"lp_{f}"].append(float(lp[f]) if lp.get(f) is not None else np.nan)
                cols["lp_ntok"].append(float(lp.get("n_tokens") or np.nan))
                srcs = (sp.get("metadata") or {}).get("sources") or [sp.get("source", "?")]
                cols["n_sources"].append(len(set(srcs)))
                if extra is not None:
                    for k2, v in extra(i, sp, gold[i]).items():
                        cols[k2].append(v)
        self.c = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
        self.n = len(cols["y"])
        if self.n == 0:
            self.c = {k: np.zeros(0) for k in ("doc", "y", "conf")}
        self.doc = self.c["doc"].astype(int)
        self.y = self.c["y"]
        self.n_docs = len(self.doc_ids)
        self.G = float(self.n_gold_doc.sum())

    def score(self, name: str) -> np.ndarray:
        """Raw confidence-like scores in [0,1] (for log-probs: exp(lp))."""
        if name == "conf":
            return self.c["conf"]
        if name.startswith("lp_"):
            return np.exp(self.c[name])
        raise KeyError(name)

    def logit(self, name: str) -> np.ndarray:
        if name.startswith("lp_"):
            return lp_logit(self.c[name])
        return prob_logit(self.score(name))


# ============================================================================ metrics
def _exp_err(conf: np.ndarray, err: np.ndarray) -> np.ndarray:
    from skillllm_v2.evals.calibration_evals import _expected_errors
    return _expected_errors(conf, err)


def ece_bins(conf: np.ndarray, y: np.ndarray, n_bins: int, strategy: str) -> Tuple[float, float, List[Dict[str, Any]]]:
    from skillllm_v2.evals.calibration_evals import reliability_bins
    bins = reliability_bins(conf, y, n_bins, strategy)
    n = sum(b["count"] for b in bins)
    ece = sum(b["count"] / n * b["gap"] for b in bins if b["count"]) if n else float("nan")
    mce = max((b["gap"] for b in bins if b["count"]), default=float("nan"))
    return ece, mce, bins


def nll(conf: np.ndarray, y: np.ndarray) -> float:
    c = np.clip(conf, NLL_EPS, 1 - NLL_EPS)
    return float(np.mean(-(y * np.log(c) + (1 - y) * np.log(1 - c)))) if len(c) else float("nan")


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    from skillllm_v2.evals.calibration_evals import auroc as _a
    return _a(score, y)


def metrics(conf: np.ndarray, y: np.ndarray, G: float, rank_score: Optional[np.ndarray] = None,
            full: bool = True) -> Dict[str, float]:
    """All scalar metrics for probabilities ``conf`` (ranking by ``rank_score`` if given, else ``conf``).
    G = number of gold spans in the document universe (for selective F1)."""
    n = len(conf)
    out: Dict[str, float] = {"n": float(n)}
    if n == 0:
        return out
    r = conf if rank_score is None else rank_score
    ee = _exp_err(np.asarray(r, dtype=float), 1 - y)
    out["precision"] = float(y.mean())
    out["aurc"] = float(np.mean(ee / np.arange(1, n + 1)))
    ys = np.sort(1 - y)
    out["aurc_oracle"] = float(np.mean(np.cumsum(ys) / np.arange(1, n + 1)))
    out["e_aurc"] = out["aurc"] - out["aurc_oracle"]
    out["auroc"] = auroc(r, y)
    for cov in COVERAGES:
        k = min(n, max(1, int(math.ceil(cov * n - 1e-9))))
        tp = k - ee[k - 1]
        out[f"sel_prec@{int(cov * 100)}"] = float(tp / k)
        out[f"sel_risk@{int(cov * 100)}"] = float(1 - tp / k)
        out[f"sel_f1@{int(cov * 100)}"] = float(2 * tp / (k + G)) if (k + G) > 0 else 0.0
    if full:
        for nb in (10, 15):
            out[f"ece_w{nb}"], m, _ = ece_bins(conf, y, nb, "width")
            out[f"ece_m{nb}"], _, _ = ece_bins(conf, y, nb, "mass")
            if nb == 10:
                out["mce_w10"] = m
        out["brier"] = float(np.mean((conf - y) ** 2))
        out["nll"] = nll(conf, y)
        out["mean_conf"] = float(conf.mean())
    return out


def cluster_boot(tab: "Table", fns: Dict[str, Callable[[np.ndarray, np.ndarray, float], Dict[str, float]]], n_boot: int,
                 seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Document-cluster bootstrap: each replicate redraws the document universe with replacement; every candidate of a
    drawn document is repeated as often as its document; G = sum of drawn documents' gold counts.
    ``fns[name](idx, y_idx, G)`` returns a metric dict; result[name][metric] = array of replicate values."""
    rng = np.random.default_rng(seed)
    nd = tab.n_docs
    cand = np.arange(tab.n)
    out: Dict[str, Dict[str, List[float]]] = {k: defaultdict(list) for k in fns}
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, nd, nd), minlength=nd)
        idx = np.repeat(cand, cnt[tab.doc])  # every candidate repeated as often as its document was drawn
        G = float((cnt * tab.n_gold_doc).sum())
        yb = tab.y[idx]
        for name, fn in fns.items():
            for m, v in fn(idx, yb, G).items():
                out[name][m].append(v)
    return {k: {m: np.asarray(v, dtype=float) for m, v in d.items()} for k, d in out.items()}


# ============================================================================ calibrators (fit on dev only)
def _fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1e4):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=5000)
    m.fit(X, y.astype(int))
    return m


def fit_calibrator(kind: str, g: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """g = logit-scale score. Returns a function g -> calibrated probability."""
    if kind == "none":
        return lambda z: sigmoid(z)
    if kind == "temperature":
        from scipy.optimize import minimize_scalar
        def loss(logT):
            p = np.clip(sigmoid(g / np.exp(logT)), NLL_EPS, 1 - NLL_EPS)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        T = float(np.exp(minimize_scalar(loss, bounds=(-6, 6), method="bounded").x))
        f = lambda z, T=T: sigmoid(z / T)
        f.params = {"T": T}  # type: ignore[attr-defined]
        return f
    if kind == "platt":
        m = _fit_logistic(g[:, None], y)
        a, b = float(m.coef_[0, 0]), float(m.intercept_[0])
        f = lambda z, a=a, b=b: sigmoid(a * z + b)
        f.params = {"a": a, "b": b}  # type: ignore[attr-defined]
        return f
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        ir.fit(g, y)
        return lambda z, ir=ir: ir.predict(z)
    raise ValueError(kind)


CALIBRATORS = ("none", "temperature", "platt", "isotonic")


def group_folds(doc: np.ndarray, k: int, seed: int) -> np.ndarray:
    docs = np.unique(doc)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(docs)
    fold_of = {d: i % k for i, d in enumerate(perm)}
    return np.array([fold_of[d] for d in doc])


def crossfit(fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]], X: np.ndarray, y: np.ndarray,
             doc: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    folds = group_folds(doc, k, seed)
    out = np.full(len(y), np.nan)
    for f in range(k):
        tr, te = folds != f, folds == f
        if te.sum() == 0:
            continue
        out[te] = fit(X[tr], y[tr])(X[te])
    return out


ANCHOR_FEATS = ("g_mean", "g_sum", "g_min", "log_ntok", "log_nwords", "label_soft", "agree")


def anchor_features(t: "Table") -> np.ndarray:
    return np.column_stack([t.logit("lp_mean"), t.logit("lp_sum"), t.logit("lp_min"), np.log(np.maximum(1, t.c["lp_ntok"])),
                            np.log(t.c["n_words"]), t.c["label_soft"], t.c.get("agree", np.zeros(t.n))])


def fit_multi(X: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = _fit_logistic((X - mu) / sd, y, C=1.0)
    return lambda Z, m=m, mu=mu, sd=sd: m.predict_proba((Z - mu) / sd)[:, 1]


# ============================================================================ GLiNER / dictionary / wrapper candidates
def build_gliner_cache(splits: Sequence[str] = ("dev", "test"), threshold: float = 0.35) -> None:
    """Run the wrapper's GLiNERExtractor (zero-shot urchade/gliner_medium-v2.1, labels hard/soft skill, threshold 0.35)
    on the gold-bearing docs of each split. Needs the `gliner` package (GPU venv). Device from SKILLLM_V2_GLINER_DEVICE."""
    from skillllm_v2.extract.gliner_extractor import GLiNERExtractor
    ex = GLiNERExtractor(threshold=threshold, allow_fallback=False)
    ex.load()
    if not ex.is_model_loaded():
        raise SystemExit("GLiNER model could not be loaded (gliner package / model download missing)")
    DERIVED.mkdir(parents=True, exist_ok=True)
    for split in splits:
        gold = load_gold(split)
        out = REPO_ROOT / P[f"gliner_{split}_cache"]
        t0 = time.time()
        with open(out, "w") as fh:
            for i, doc in gold.items():
                fh.write(json.dumps({"id": i, "predictions": ex.extract(doc["text"])}) + "\n")
        meta = {"split": split, "n_docs": len(gold), "seconds": time.time() - t0, "model": ex.model_name, "threshold": threshold,
                "device": ex.device, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta), flush=True)


def dictionary_preds(gold: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    from skillllm_v2.extract import DictionaryExtractor
    ex = DictionaryExtractor()
    return {i: ex.extract(d["text"]) for i, d in gold.items()}


def wrapper_replay(gold: Dict[str, Dict[str, Any]], anchor: Dict[str, List[Dict[str, Any]]], gl: Dict[str, List[Dict[str, Any]]],
                   dic: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Safe wrapper full chain (merge -> refine -> verify -> calibrate) on given anchor predictions (CPU)."""
    from skillllm_v2.extract import CandidateMerger
    from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier
    from skillllm_v2.verify.boundary_selector import BoundarySelector
    mg, rf, sel, vf, cal = CandidateMerger(), BoundaryRefiner(), BoundarySelector(), SpanVerifier(), ConfidenceCalibrator()
    out = {}
    for i, d in gold.items():
        merged = mg.merge(list(anchor.get(i, [])) + list(gl.get(i, [])) + list(dic.get(i, [])))
        out[i] = cal.calibrate(vf.filter(rf.refine(d["text"], merged, selector=sel)))
    return out


def agreement_extra(gl: Dict[str, List[Dict[str, Any]]], dic: Dict[str, List[Dict[str, Any]]]):
    gk = {i: {key(x) for x in v} for i, v in gl.items()}
    dk = {i: {key(x) for x in v} for i, v in dic.items()}
    return lambda i, sp, doc: {"agree": float(key(sp) in gk.get(i, set())) + float(key(sp) in dk.get(i, set()))}


# ============================================================================ evaluation of one (source, score) on test
def evaluate(tab: "Table", probs: Dict[str, np.ndarray], rank: np.ndarray, n_boot: int, seed: int,
             const_baseline: Optional[float] = None) -> Dict[str, Any]:
    """probs: calibrator name -> test probabilities. rank = ranking score (shared by all calibrators for AURC etc.)."""
    res: Dict[str, Any] = {"point": {}, "ci": {}}
    for name, pr in probs.items():
        res["point"][name] = metrics(pr, tab.y, tab.G, rank_score=pr if name != "_rank" else rank)
    fns = {name: (lambda pr: lambda idx, yb, G: metrics(pr[idx], yb, G))(pr) for name, pr in probs.items()}
    if const_baseline is not None:
        cb = np.full(tab.n, const_baseline)
        res["point"]["_constant"] = metrics(cb, tab.y, tab.G)
        fns["_constant"] = lambda idx, yb, G: metrics(cb[idx], yb, G)
    boots = cluster_boot(tab, fns, n_boot, seed) if tab.n else {}
    res["boot"] = boots
    for name, d in boots.items():
        res["ci"][name] = {m: S._pct_ci(v) for m, v in d.items()}
    return res


def paired_deltas(res: Dict[str, Any], a: str, b: str, mets: Sequence[str] = ("ece_w10", "ece_m10", "brier", "nll", "aurc", "auroc")
                  ) -> Dict[str, Dict[str, Any]]:
    """a − b for each metric using the SHARED document replicates of ``evaluate`` (paired cluster bootstrap)."""
    out = {}
    for m in mets:
        if m not in res["boot"].get(a, {}) or m not in res["boot"].get(b, {}):
            continue
        obs = res["point"][a][m] - res["point"][b][m]
        out[m] = S.summarize_samples(res["boot"][a][m] - res["boot"][b][m], obs, paired=True)
    return out


def reliability_csv(path: Path, entries: List[Tuple[str, np.ndarray, np.ndarray]]) -> None:
    from skillllm_v2.evals.calibration_evals import reliability_bins
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "strategy", "n_bins", "bin", "lo", "hi", "count", "mean_conf", "accuracy", "gap"])
        for series, conf, y in entries:
            for strat in ("width", "mass"):
                for nb in (10, 15):
                    for b in reliability_bins(conf, y, nb, strat):
                        w.writerow([series, strat, nb, b["bin"], b["lo"], b["hi"], b["count"], b["mean_conf"], b["accuracy"], b["gap"]])


def plot_reliability(path: Path, entries: List[Tuple[str, np.ndarray, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skillllm_v2.evals.calibration_evals import reliability_bins
    k = len(entries)
    cols = min(4, k)
    rows = int(math.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.3 * rows), squeeze=False)
    for ax, (series, conf, y) in zip(axes.ravel(), entries):
        bw = [b for b in reliability_bins(conf, y, 10, "width") if b["count"]]
        bm = [b for b in reliability_bins(conf, y, 10, "mass") if b["count"]]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot([b["mean_conf"] for b in bw], [b["accuracy"] for b in bw], "o-", ms=3, label="equal-width")
        ax.plot([b["mean_conf"] for b in bm], [b["accuracy"] for b in bm], "s-", ms=3, alpha=0.7, label="equal-mass")
        ax.set_title(series, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6)
    fig.supxlabel("mean confidence")
    fig.supylabel("exact-span precision")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def risk_coverage_rows(series: str, rank: np.ndarray, y: np.ndarray, G: float) -> List[List[Any]]:
    n = len(rank)
    if n == 0:
        return []
    ee = _exp_err(np.asarray(rank, dtype=float), 1 - y)
    ks = sorted(set([max(1, int(round(n * c / 100))) for c in range(1, 101)]))
    return [[series, k / n, 1 - (k - ee[k - 1]) / k, (k - ee[k - 1]) / k, 2 * (k - ee[k - 1]) / (k + G)] for k in ks]


def plot_risk_coverage(path: Path, rows: List[List[Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for s, rr in by.items():
        rr = sorted(rr, key=lambda r: r[1])
        axes[0].plot([r[1] for r in rr], [r[3] for r in rr], label=s, lw=1)
        axes[1].plot([r[1] for r in rr], [r[4] for r in rr], label=s, lw=1)
    axes[0].set_xlabel("coverage (share of predicted spans kept)")
    axes[0].set_ylabel("selective precision")
    axes[1].set_xlabel("coverage")
    axes[1].set_ylabel("micro-F1 (abstained spans = missed)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================ main analysis
def run(smoke: bool, seed: int, n_boot: int, n_boot_head: int, log=print) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    t0 = time.time()
    gold_test, gold_dev = load_gold("test"), load_gold("dev")
    if smoke:
        gold_test = dict(list(gold_test.items())[:120])
        gold_dev = dict(list(gold_dev.items())[:120])
    records: List[Dict[str, Any]] = []
    rep: Dict[str, Any] = {"sources": {}, "notes": []}
    ds_test = f"SkillSpan test, {len(gold_test)} docs with >=1 gold span"
    ds_dev = f"SkillSpan dev, {len(gold_dev)} docs with >=1 gold span"
    out_dir = EVALS_ROOT / LAYER / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_entries: List[Tuple[str, np.ndarray, np.ndarray]] = []
    rc_rows: List[List[Any]] = []

    def emit(source: str, tab: Table, res: Dict[str, Any], prov: List[str], split_note: str, cav: List[str],
             series_names: Dict[str, str], headline_series: Optional[str] = None, secondary: bool = False):
        for name, pt in res["point"].items():
            sysname = f"{source}:{series_names.get(name, name)}"
            ci = res["ci"].get(name, {})
            for m, v in pt.items():
                if m == "n" or v is None or (isinstance(v, float) and not math.isfinite(v)):
                    continue
                c = ci.get(m)
                direction = "higher" if (m.startswith("sel_prec") or m.startswith("sel_f1") or m in ("auroc", "precision")) else (
                    "none" if m in ("mean_conf",) else "lower")
                det = {"split_fit": split_note, "n_spans": int(tab.n), "n_docs": tab.n_docs, "n_gold": int(tab.G),
                       "secondary": secondary,
                       "headline": bool(headline_series == name and m in ("ece_w10", "sel_risk@80"))}
                nbt = len(res["boot"].get(name, {}).get("precision", []))
                records.append(record(LAYER, sysname, m, v, n_units=tab.n_docs, unit="document (cluster of spans)", split="test",
                                      dataset=ds_test, direction=direction, ci=c if c and c[0] is not None else None,
                                      ci_method=f"document-cluster percentile bootstrap ({nbt})" if nbt else "",
                                      provenance=prov, caveats=cav, details=det))

    # ---------------------------------------------------------------- anchor stored (constant)
    anchor_stored = read_rows(REPO_ROOT / P["anchor_stored"])
    t_const = Table("anchor_stored", "test", gold_test, anchor_stored)
    res = evaluate(t_const, {"raw": t_const.score("conf")}, t_const.score("conf"), n_boot, seed)
    emit("anchor_stored", t_const, res, [P["anchor_stored"], P["gold_test"]], "no fitting",
         ["Confidence is the constant 0.86 for every span: ECE = |0.86 - precision|, AUROC 0.5, AURC = error rate "
          "(no ranking information)."], {"raw": "constant_0.86"})
    rel_entries.append(("anchor stored (const 0.86)", t_const.score("conf"), t_const.y))
    rc_rows += risk_coverage_rows("anchor_stored_const", t_const.score("conf"), t_const.y, t_const.G)
    rep["sources"]["anchor_stored"] = {"n_spans": t_const.n, "point": res["point"]}
    log(f"anchor_stored done {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- GLiNER + dictionary (dev-fit)
    dic_dev, dic_test = dictionary_preds(gold_dev), dictionary_preds(gold_test)
    gl_dev = read_rows(REPO_ROOT / P["gliner_dev_cache"]) if (REPO_ROOT / P["gliner_dev_cache"]).exists() else None
    gl_test = read_rows(REPO_ROOT / P["gliner_test_cache"]) if (REPO_ROOT / P["gliner_test_cache"]).exists() else None
    gl_stored = read_rows(REPO_ROOT / P["gliner_stored_test"])
    for src, dev_preds, test_preds, prov in (
            ("gliner", gl_dev, gl_test, [P["gliner_dev_cache"], P["gliner_test_cache"], P["gold_dev"], P["gold_test"]]),
            ("dictionary", dic_dev, dic_test, [P["gold_dev"], P["gold_test"], "src/skillllm_v2/extract/dictionary_extractor.py"])):
        if dev_preds is None or test_preds is None:
            records.append(record(LAYER, f"{src}:all", "ece_w10", None, n_units=0, unit="document", split="test", dataset=ds_test,
                                  direction="lower", status="not_available",
                                  reason=f"{src} dev/test candidate cache missing ({P['gliner_dev_cache']}); run --build-gliner-cache"))
            continue
        td, tt = Table(src, "dev", gold_dev, dev_preds), Table(src, "test", gold_test, test_preds)
        probs = {}
        cal_params = {}
        if td.n >= 20 and len(np.unique(td.y)) == 2:
            for kind in CALIBRATORS:
                f = fit_calibrator(kind, td.logit("conf"), td.y)
                probs[kind] = f(tt.logit("conf"))
                cal_params[kind] = getattr(f, "params", None)
            cv_nll = {kind: nll(crossfit(lambda X, y, kind=kind: fit_calibrator(kind, X, y), td.logit("conf"), td.y, td.c["doc"], 5, seed),
                                td.y) for kind in CALIBRATORS}
        else:
            probs["none"] = tt.score("conf")
            cv_nll = {}
        res = evaluate(tt, probs, tt.score("conf"), n_boot, seed + 11, const_baseline=float(td.y.mean()) if td.n else None)
        best = min(cv_nll, key=cv_nll.get) if cv_nll else "none"
        emit(src, tt, res, prov, "calibrators fit on dev; reported calibrator chosen by dev 5-fold CV NLL",
             [f"{src} confidences on test; calibrators fit on SkillSpan dev ({td.n} spans, {td.n_docs} docs).",
              "Ranking metrics (AURC/AUROC/selective) use the raw score; monotone calibration does not change the ranking "
              "(isotonic can merge ties)."] + (["GLiNER candidates from A2's CPU run of the wrapper's GLiNERExtractor "
                                                "(zero-shot urchade/gliner_medium-v2.1, thr 0.35)."] if src == "gliner" else
                                               ["Dictionary confidences are rule-assigned (0.82 exact alias / <=0.72 fuzzy): "
                                                "almost no ranking information."]),
             {k: (k if k != "none" else "raw") for k in probs} | {"_constant": "dev_base_rate_constant"})
        for kind, pr in probs.items():
            rel_entries.append((f"{src} {kind if kind != 'none' else 'raw'}", pr, tt.y))
        rc_rows += risk_coverage_rows(f"{src}_raw", tt.score("conf"), tt.y, tt.G)
        rep["sources"][src] = {"n_spans_test": tt.n, "n_spans_dev": td.n, "cv_nll_dev": cv_nll, "best_by_dev_cv": best,
                               "params": cal_params, "point": res["point"]}
        if src == "gliner":  # reproduction of the stored GLiNER output
            ts = Table("gliner_stored", "test", gold_test, gl_stored)
            same = sum(1 for i in gold_test if sorted(map(str, (key(x) for x in gl_stored.get(i, [])))) ==
                       sorted(map(str, (key(x) for x in test_preds.get(i, [])))))
            rep["sources"][src]["stored_vs_rerun_docs_identical"] = [same, len(gold_test)]
            rep["sources"][src]["stored_n_spans"] = ts.n
        log(f"{src} done {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- v3 / v4: test-only log-probs (secondary)
    for tag in ("v3", "v4"):
        preds = read_rows(REPO_ROOT / P[tag])
        tt = Table(tag, "test", gold_test, preds)
        if tt.n == 0 or np.isnan(tt.c["lp_mean"]).all():
            continue
        ok = ~np.isnan(tt.c["lp_mean"])
        probs = {"raw_exp_mean_lp": np.where(ok, tt.score("lp_mean"), 0.0), "raw_exp_sum_lp": np.where(ok, tt.score("lp_sum"), 0.0),
                 "raw_exp_min_lp": np.where(ok, tt.score("lp_min"), 0.0)}
        for sc in ("lp_mean", "lp_sum"):
            g = np.where(ok, tt.logit(sc), -20.0)
            for kind in ("temperature", "platt", "isotonic"):
                probs[f"xfit_{kind}_{sc}"] = crossfit(lambda X, y, kind=kind: fit_calibrator(kind, X, y), g, tt.y, tt.c["doc"], 5, seed)
        res = evaluate(tt, {"raw_exp_mean_lp": probs["raw_exp_mean_lp"]}, probs["raw_exp_mean_lp"], n_boot, seed + 21,
                       const_baseline=0.86)
        res_all = {"point": {k: metrics(v, tt.y, tt.G) for k, v in probs.items()}, "ci": {}, "boot": {}}
        res_all["point"]["_constant"] = res["point"]["_constant"]
        res_all["ci"] = {"raw_exp_mean_lp": res["ci"]["raw_exp_mean_lp"], "_constant": res["ci"]["_constant"]}
        res_all["boot"] = res["boot"]
        emit(tag, tt, res_all, [P[tag], P["gold_test"]], "uncalibrated (raw_*) or 5-fold document-grouped cross-fitting ON TEST (xfit_*)",
             ["SECONDARY: no dev predictions exist for this adapter, so calibrators are cross-fitted on the test documents; "
              "not a held-out calibration result."], {"_constant": "constant_0.86"}, secondary=True)
        rel_entries.append((f"{tag} exp(mean lp) raw", probs["raw_exp_mean_lp"], tt.y))
        rc_rows += risk_coverage_rows(f"{tag}_mean_lp", probs["raw_exp_mean_lp"], tt.y, tt.G)
        # reproduction of the legacy report
        legacy = json.loads((REPO_ROOT / P[f"{tag}_report"]).read_text()) if (REPO_ROOT / P[f"{tag}_report"]).exists() else None
        repro = {}
        if legacy and not smoke:
            lc = legacy["confidences"].get("p_mean = exp(mean lp)", {})
            pm = res_all["point"]["raw_exp_mean_lp"]
            repro = {"auroc_legacy": lc.get("auroc"), "auroc_now": pm["auroc"], "aurc_legacy": lc.get("aurc"), "aurc_now": pm["aurc"],
                     "ece_legacy": lc.get("ece_width"), "ece_now": pm["ece_w10"], "n_legacy": legacy.get("n_spans"), "n_now": tt.n}
        rep["sources"][tag] = {"n_spans": tt.n, "point": res_all["point"], "legacy_reproduction": repro}
        log(f"{tag} done {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- anchor log-probs (H4) + wrapper
    jt, jd = REPO_ROOT / P["anchor_test_lp"], REPO_ROOT / P["anchor_dev_lp"]
    h4: Dict[str, Any] = {}
    have = done(jt) and done(jd)
    if smoke and not have and (jt / "predictions.jsonl").exists() and (jd / "predictions.jsonl").exists():
        have = "partial"  # smoke: validate the code path on whatever A3 has written so far
    if not have:
        missing = [str(rel(j)) for j in (jt, jd) if not done(j)]
        for metric in ("H4_verdict", "ece_w10", "sel_risk@80", "aurc"):
            records.append(record(LAYER, "anchor_logprob", metric, None, n_units=0, unit="document", split="test", dataset=ds_test,
                                  direction="lower" if metric != "H4_verdict" else "higher", status="pending_gpu",
                                  reason=f"A3 anchor log-prob jobs not complete (missing DONE.json: {missing}); "
                                         "rerun: python -m skillllm_v2.evals.research.calibration_selective",
                                  details={"verdict": "NOT TESTABLE (pending_gpu)"} if metric == "H4_verdict" else {}))
        for metric in ("ece_w10", "sel_risk@80"):
            records.append(record(LAYER, "wrapper", metric, None, n_units=0, unit="document", split="test", dataset=ds_test,
                                  direction="lower", status="pending_gpu",
                                  reason="wrapper dev replay needs A3's anchor_dev_logprobs predictions (J2)"))
        h4 = {"verdict": "NOT TESTABLE (pending_gpu)", "missing": missing}
    else:
        h4 = anchor_and_wrapper(gold_test, gold_dev, gl_dev, gl_test, dic_dev, dic_test, n_boot, n_boot_head, seed, records, emit,
                                rel_entries, rc_rows, rep, ds_test, partial=(have == "partial"))
    rep["h4"] = h4
    log(f"anchor done {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- figures + csv
    try:
        reliability_csv(out_dir / "reliability_bins.csv", rel_entries)
        plot_reliability(out_dir / "reliability_diagrams.png", rel_entries)
        with open(out_dir / "risk_coverage.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["series", "coverage", "selective_risk", "selective_precision", "micro_f1_abstained_as_missed"])
            w.writerows(rc_rows)
        plot_risk_coverage(out_dir / "risk_coverage.png", rc_rows)
    except Exception as exc:  # noqa: BLE001
        rep["notes"].append(f"figure error: {exc}")
    rep["elapsed_s"] = time.time() - t0
    return records, report(rep, len(gold_test), len(gold_dev), n_boot, n_boot_head, seed), rep


def anchor_and_wrapper(gold_test, gold_dev, gl_dev, gl_test, dic_dev, dic_test, n_boot, n_boot_head, seed, records, emit,
                       rel_entries, rc_rows, rep, ds_test, partial=False) -> Dict[str, Any]:
    jt, jd = REPO_ROOT / P["anchor_test_lp"], REPO_ROOT / P["anchor_dev_lp"]
    at, ad = read_rows(jt / "predictions.jsonl"), read_rows(jd / "predictions.jsonl")
    ids_t = [i for i in gold_test if i in at]
    ids_d = [i for i in gold_dev if i in ad]
    ex_t = agreement_extra(gl_test or {}, dic_test)
    ex_d = agreement_extra(gl_dev or {}, dic_dev)
    tt = Table("anchor_logprob", "test", gold_test, at, ex_t, ids=ids_t)
    td = Table("anchor_logprob", "dev", gold_dev, ad, ex_d, ids=ids_d)
    miss_t = int(np.isnan(tt.c["lp_mean"]).sum())
    miss_d = int(np.isnan(td.c["lp_mean"]).sum())
    fill = lambda t, a: np.where(np.isnan(a), -50.0, a)
    for t in (tt, td):
        for f in ("lp_sum", "lp_mean", "lp_min"):
            t.c[f] = fill(t, t.c[f])
        t.c["lp_ntok"] = np.where(np.isnan(t.c["lp_ntok"]), 1.0, t.c["lp_ntok"])
    # ---- dev: score selection (lowest dev AURC; logistic by grouped CV) and calibrator selection (dev CV NLL)
    cand_scores = {"mean_lp": td.logit("lp_mean"), "sum_lp": td.logit("lp_sum"), "min_lp": td.logit("lp_min")}
    dev_aurc = {k: metrics(sigmoid(v), td.y, td.G, full=False)["aurc"] for k, v in cand_scores.items()}
    Xd, Xt = anchor_features(td), anchor_features(tt)
    multi_cv = crossfit(fit_multi, Xd, td.y, td.c["doc"], 5, seed)
    dev_aurc["logistic_multi"] = metrics(multi_cv, td.y, td.G, full=False)["aurc"]
    primary = min(dev_aurc, key=dev_aurc.get)
    if primary == "logistic_multi":
        fm = fit_multi(Xd, td.y)
        g_dev, g_test = prob_logit(fm(Xd)), prob_logit(fm(Xt))
        rank_test = fm(Xt)
    else:
        col = {"mean_lp": "lp_mean", "sum_lp": "lp_sum", "min_lp": "lp_min"}[primary]
        g_dev, g_test = td.logit(col), tt.logit(col)
        rank_test = g_test
    cv_nll = {k: nll(crossfit(lambda X, y, k=k: fit_calibrator(k, X, y), g_dev, td.y, td.c["doc"], 5, seed), td.y) for k in CALIBRATORS}
    best_cal = min(cv_nll, key=cv_nll.get)
    probs = {}
    params = {}
    for k in CALIBRATORS:
        f = fit_calibrator(k, g_dev, td.y)
        probs[k] = f(g_test)
        params[k] = getattr(f, "params", None)
    # also: every raw score uncalibrated, and the multi-feature logistic (fit on dev)
    probs["raw_exp_mean_lp"], probs["raw_exp_sum_lp"], probs["raw_exp_min_lp"] = tt.score("lp_mean"), tt.score("lp_sum"), tt.score("lp_min")
    if primary != "logistic_multi":
        probs["logistic_multi_devfit"] = fit_multi(Xd, td.y)(Xt)
    nb = n_boot_head
    res = evaluate(tt, probs, rank_test, nb, seed + 31, const_baseline=0.86)
    # ranking metrics of every series must use the primary rank for calibrators; recompute point for raw series w/ own ranking
    names = {k: f"primary[{primary}]_{k if k != 'none' else 'uncalibrated_sigmoid'}" for k in CALIBRATORS}
    names.update({"_constant": "constant_0.86"})
    emit("anchor_logprob", tt, res, [rel(jt / "predictions.jsonl"), rel(jd / "predictions.jsonl"), P["gold_test"], P["gold_dev"]],
         f"score selected on dev (dev AURC: {json.dumps({k: round(v, 5) for k, v in dev_aurc.items()})}); calibrators fit on dev",
         ["Anchor re-run by A3 with log-probs (greedy decoding; current offset code — see A3 integrity flag on offset drift); "
          "its span set can differ slightly from the stored anchor predictions."] + (["PARTIAL A3 OUTPUT (smoke)"] if partial else []),
         names, headline_series=best_cal)
    # ---- H4 contrasts (paired, same replicates)
    b = res["boot"]
    prim = b[best_cal]
    const = b["_constant"]
    d80 = prim["sel_prec@80"] - prim["sel_prec@100"]
    d50 = prim["sel_prec@50"] - prim["sel_prec@100"]
    daurc = prim["aurc"] - const["aurc"]
    pt = res["point"][best_cal]
    pc = res["point"]["_constant"]
    obs = {"d80": pt["sel_prec@80"] - pt["sel_prec@100"], "d50": pt["sel_prec@50"] - pt["sel_prec@100"], "daurc": pt["aurc"] - pc["aurc"]}
    sm = {k: S.summarize_samples(v, obs[k], paired=True) for k, v in (("d80", d80), ("d50", d50), ("daurc", daurc))}
    cond_a = bool(sm["d80"]["ci_low"] is not None and sm["d80"]["ci_low"] > 0 and sm["d50"]["ci_low"] > 0)
    cond_b = bool(sm["daurc"]["ci_high"] is not None and sm["daurc"]["ci_high"] < 0)
    verdict = "SUPPORTED" if (cond_a and cond_b) else "NOT SUPPORTED"
    if partial:
        verdict = "NOT TESTABLE (partial A3 output, smoke)"
    h4 = {"verdict": verdict, "primary_score": primary, "dev_aurc": dev_aurc, "calibrator": best_cal, "dev_cv_nll": cv_nll,
          "params": params, "contrasts": {k: {kk: vv for kk, vv in v.items()} for k, v in sm.items()}, "cond_a": cond_a, "cond_b": cond_b,
          "n_spans_test": tt.n, "n_spans_dev": td.n, "n_docs_test": tt.n_docs, "n_docs_dev": td.n_docs,
          "missing_logprob_test": miss_t, "missing_logprob_dev": miss_d, "partial": bool(partial),
          "point_primary": pt, "point_constant": pc}
    p_holm = S.holm([sm["d80"]["p"], sm["d50"]["p"], sm["daurc"]["p"]])
    rule = ("H4: selective precision@80% and @50% minus precision@100% > 0 (CI excl. 0) AND AURC(primary) - AURC(constant 0.86) "
            "< 0 (CI excl. 0); score + calibrator selected/fit on dev only (D-A2-03).")
    records.append(record(LAYER, "anchor_logprob", "H4_verdict", sm["d80"]["value"], n_units=tt.n_docs, unit="document",
                          split="test", dataset=ds_test, direction="higher", ci=(sm["d80"]["ci_low"], sm["d80"]["ci_high"]),
                          ci_method=f"paired document-cluster bootstrap ({nb})",
                          provenance=[rel(jt / "predictions.jsonl"), rel(jd / "predictions.jsonl"), P["gold_test"], P["gold_dev"]],
                          caveats=["Primary effect = selective precision@80% minus precision@100% under the dev-selected score."],
                          details={"verdict": verdict, "rule": rule, "p_raw": sm["d80"]["p"], "p_holm": p_holm[0], "family": "H4",
                                   "p_holm_all3": p_holm, **{k: v for k, v in h4.items() if k not in ("point_primary", "point_constant")}}))
    # ---- paired calibration deltas: dev-fit calibrated vs constant 0.86 and vs uncalibrated
    h4["paired_vs_constant"] = paired_deltas(res, best_cal, "_constant")
    h4["paired_vs_uncalibrated"] = paired_deltas(res, best_cal, "none") if best_cal != "none" else {}
    for tag, dd in (("vs_constant_0.86", h4["paired_vs_constant"]), ("vs_uncalibrated", h4["paired_vs_uncalibrated"])):
        for m, v in dd.items():
            records.append(record(LAYER, f"anchor_logprob:{best_cal}", f"delta_{m}_{tag}", v["value"], n_units=tt.n_docs, unit="document",
                                  split="test", dataset=ds_test, direction="higher" if m == "auroc" else "lower",
                                  ci=(v["ci_low"], v["ci_high"]), ci_method=f"paired document-cluster bootstrap ({nb})",
                                  provenance=[rel(jt / "predictions.jsonl"), rel(jd / "predictions.jsonl")],
                                  details={"p": v["p"], "comparison": f"{best_cal} (dev-fit) minus {tag}"}))
    # ---- deployable operating point: abstention threshold chosen on DEV (max dev micro-F1), applied once to test
    f_best = fit_calibrator(best_cal, g_dev, td.y)
    p_dev, p_test = f_best(g_dev), probs[best_cal]
    grid = np.unique(np.quantile(p_dev, np.linspace(0, 0.9, 91)))
    dev_f1 = [(float(2 * td.y[p_dev >= t].sum() / ((p_dev >= t).sum() + td.G)), float(t)) for t in grid]
    best_f1_dev, tau = max(dev_f1)

    def op(idx, yb, G, p=p_test, tau=tau):
        keep = p[idx] >= tau
        tp_all, tp_k = yb.sum(), yb[keep].sum()
        f_all, f_k = 2 * tp_all / (len(yb) + G), 2 * tp_k / (keep.sum() + G)
        return {"f1_full": f_all, "f1_abstain": f_k, "delta": f_k - f_all, "coverage": keep.mean() if len(keep) else np.nan,
                "precision_abstain": yb[keep].mean() if keep.any() else np.nan}
    allidx = np.arange(tt.n)
    pt_op = op(allidx, tt.y, tt.G)
    bo = cluster_boot(tt, {"op": op}, nb, seed + 51)["op"]
    s_op = S.summarize_samples(bo["delta"], pt_op["delta"], paired=True)
    h4["dev_tuned_abstention"] = {"tau": tau, "dev_f1_at_tau": best_f1_dev, "dev_f1_full": dev_f1[0][0] if dev_f1 else None,
                                  **{k: float(v) for k, v in pt_op.items()}, "delta_ci": [s_op["ci_low"], s_op["ci_high"]], "p": s_op["p"]}
    records.append(record(LAYER, "anchor_logprob:dev_tuned_abstention", "delta_f1_abstain_minus_full", pt_op["delta"], n_units=tt.n_docs,
                          unit="document", split="test", dataset=ds_test, direction="higher", ci=(s_op["ci_low"], s_op["ci_high"]),
                          ci_method=f"paired document-cluster bootstrap ({nb})",
                          provenance=[rel(jt / "predictions.jsonl"), rel(jd / "predictions.jsonl")],
                          caveats=["Threshold tau chosen on dev to maximise dev micro-F1 (abstained = missed); applied once to test."],
                          details=h4["dev_tuned_abstention"]))
    rel_entries.append((f"anchor {primary} {best_cal} (dev-fit)", probs[best_cal], tt.y))
    rel_entries.append(("anchor exp(mean lp) raw", probs["raw_exp_mean_lp"], tt.y))
    rc_rows += risk_coverage_rows(f"anchor_{primary}", rank_test, tt.y, tt.G)
    rep["sources"]["anchor_logprob"] = {"point": res["point"], "h4": h4}
    # ---- wrapper confidences: test = stored deployed output; dev = replay on A3 dev anchor
    if gl_dev is not None:
        wd = wrapper_replay({i: gold_dev[i] for i in ids_d}, ad, gl_dev, dic_dev)
        wt = read_rows(REPO_ROOT / P["wrapper_test"])
        twd, twt = Table("wrapper", "dev", gold_dev, wd, ids=ids_d), Table("wrapper", "test", gold_test, wt)
        wprobs = {}
        for k in CALIBRATORS:
            wprobs[k] = fit_calibrator(k, twd.logit("conf"), twd.y)(twt.logit("conf"))
        wres = evaluate(twt, wprobs, twt.score("conf"), n_boot, seed + 41, const_baseline=float(twd.y.mean()))
        emit("wrapper", twt, wres, [P["wrapper_test"], rel(jd / "predictions.jsonl"), P["gliner_dev_cache"], P["gold_dev"], P["gold_test"]],
             "calibrators fit on the wrapper replayed on dev", ["Wrapper confidences are rule-assigned (0.86 + 0.05 per extra source, "
                                                               "± boundary-edit deltas): little ranking information.",
                                                               "Test = stored deployed safe-wrapper output; dev = replay on A3's dev anchor run."],
             {k: (k if k != "none" else "raw") for k in CALIBRATORS} | {"_constant": "dev_base_rate_constant"})
        rel_entries.append(("wrapper raw", twt.score("conf"), twt.y))
        rc_rows += risk_coverage_rows("wrapper_raw", twt.score("conf"), twt.y, twt.G)
        rep["sources"]["wrapper"] = {"point": wres["point"], "n_dev": twd.n, "n_test": twt.n,
                                     "distinct_conf_test": int(len(np.unique(twt.score("conf"))))}
    return h4


def _f(v, d=4):
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def report(rep: Dict[str, Any], n_test: int, n_dev: int, n_boot: int, n_boot_head: int, seed: int) -> str:
    L = ["# L04_calibration — calibration and selective prediction per confidence source", "",
         f"Test: {n_test} SkillSpan test docs with ≥1 gold span; dev: {n_dev} dev docs (calibrators / score selection only). "
         f"Seed {seed}; document-cluster bootstrap {n_boot} ({n_boot_head} for H4). Design: Decisions.md D-A2-03.", "",
         "## H4 verdict (anchor log-prob abstention)", ""]
    h4 = rep.get("h4", {})
    L.append(f"**{h4.get('verdict')}**")
    if "contrasts" in h4:
        c = h4["contrasts"]
        L += ["", f"- Primary ranking score selected on dev by AURC: `{h4['primary_score']}` (dev AURC {json.dumps({k: round(v, 4) for k, v in h4['dev_aurc'].items()})}); "
              f"calibrator by dev 5-fold CV NLL: `{h4['calibrator']}` ({json.dumps({k: round(v, 4) for k, v in h4['dev_cv_nll'].items()})}).",
              f"- Selective precision@80% − @100%: {c['d80']['value']:+.4f} [{_f(c['d80']['ci_low'])}, {_f(c['d80']['ci_high'])}]; "
              f"@50% − @100%: {c['d50']['value']:+.4f} [{_f(c['d50']['ci_low'])}, {_f(c['d50']['ci_high'])}] → (a) {h4['cond_a']}.",
              f"- AURC(primary) − AURC(constant 0.86): {c['daurc']['value']:+.4f} [{_f(c['daurc']['ci_low'])}, {_f(c['daurc']['ci_high'])}] → (b) {h4['cond_b']}.",
              f"- n = {h4['n_spans_test']} test spans / {h4['n_docs_test']} docs; dev {h4['n_spans_dev']} spans / {h4['n_docs_dev']} docs; "
              f"spans without log-prob: test {h4['missing_logprob_test']}, dev {h4['missing_logprob_dev']}."]
        for tag in ("paired_vs_constant", "paired_vs_uncalibrated"):
            if h4.get(tag):
                L.append(f"- Calibrated − {'constant 0.86' if tag.endswith('constant') else 'uncalibrated'}: " + "; ".join(
                    f"{m} {v['value']:+.4f} [{_f(v['ci_low'])}, {_f(v['ci_high'])}]" for m, v in h4[tag].items()) + ".")
        if "dev_tuned_abstention" in h4:
            o = h4["dev_tuned_abstention"]
            L.append(f"- Deployable operating point (threshold τ={o['tau']:.4f} chosen on dev by max dev micro-F1): test coverage "
                     f"{o['coverage']:.3f}, precision {o['precision_abstain']:.4f}, micro-F1 {o['f1_abstain']:.4f} vs {o['f1_full']:.4f} without "
                     f"abstention (Δ {o['delta']:+.4f} [{_f(o['delta_ci'][0])}, {_f(o['delta_ci'][1])}], p {o['p']:.4f}).")
        pp, pc = h4["point_primary"], h4["point_constant"]
        L += ["", md_table([{"series": "primary (dev-fit)", **{k: pp.get(k) for k in ("precision", "ece_w10", "ece_m10", "brier", "nll", "aurc", "auroc",
                                                                                      "sel_prec@80", "sel_prec@50", "sel_f1@100", "sel_f1@80")}},
                            {"series": "constant 0.86", **{k: pc.get(k) for k in ("precision", "ece_w10", "ece_m10", "brier", "nll", "aurc", "auroc",
                                                                                   "sel_prec@80", "sel_prec@50", "sel_f1@100", "sel_f1@80")}}],
                           ["series", "precision", "ece_w10", "ece_m10", "brier", "nll", "aurc", "auroc", "sel_prec@80", "sel_prec@50",
                            "sel_f1@100", "sel_f1@80"])]
    else:
        L.append(f"\nReason: {h4.get('missing')}. The anchor rows are `pending_gpu`; rerun this module once both DONE.json files exist.")
    L += ["", "## Per-source summary (test; point estimates — CIs in results.json)", ""]
    cols = ["source/series", "n", "precision", "ece_w10", "ece_w15", "ece_m10", "ece_m15", "brier", "nll", "aurc", "auroc",
            "sel_prec@90", "sel_prec@80", "sel_prec@50", "sel_f1@100", "sel_f1@80", "sel_f1@50"]
    rows = []
    for src, d in rep["sources"].items():
        for name, pt in d.get("point", {}).items():
            rows.append({"source/series": f"{src}:{name}", **{k: pt.get(k) for k in cols[1:]}})
    L.append(md_table(rows, cols))
    L += ["", "Naming: for `anchor_logprob`, `none/temperature/platt/isotonic` are calibrations of the dev-selected primary score "
          "(`none` = its raw probability); `raw_exp_*` are the uncalibrated log-prob probabilities; `_constant` is the stored 0.86 "
          "(anchor, v3, v4) or the dev base rate (gliner, dictionary, wrapper). Ranking metrics of each series use that series' own values "
          "(isotonic merges ties, which can change AURC slightly).",
          "", "Selective F1 counts abstained spans as missed: F1_c = 2·TP_c / (k_c + G), G = gold spans of the document universe; "
          "it can only fall below the 100 % value unless abstention removes more FP weight than TP weight.", ""]
    for src in ("gliner", "dictionary"):
        d = rep["sources"].get(src)
        if d:
            L.append(f"- {src}: dev spans {d.get('n_spans_dev')}, test spans {d.get('n_spans_test')}; dev CV NLL "
                     f"{json.dumps({k: round(v, 4) for k, v in d.get('cv_nll_dev', {}).items()})}; best {d.get('best_by_dev_cv')}"
                     + (f"; re-run vs stored GLiNER test output identical docs {d['stored_vs_rerun_docs_identical']}" if "stored_vs_rerun_docs_identical" in d else ""))
    for tag in ("v3", "v4"):
        d = rep["sources"].get(tag)
        if d and d.get("legacy_reproduction"):
            r = d["legacy_reproduction"]
            L.append(f"- {tag} legacy reproduction (exp(mean lp)): AUROC {_f(r['auroc_legacy'])} → {_f(r['auroc_now'])}, AURC "
                     f"{_f(r['aurc_legacy'])} → {_f(r['aurc_now'])}, ECE {_f(r['ece_legacy'])} → {_f(r['ece_now'])}, n {r['n_legacy']} → {r['n_now']}.")
    if "wrapper" in rep["sources"]:
        L.append(f"- wrapper: {rep['sources']['wrapper']['distinct_conf_test']} distinct confidence values on test.")
    L += ["", "## Caveats", "",
          "- Unit = predicted span, correctness = exact (start, end, label); CIs resample documents.",
          "- v3/v4 rows are SECONDARY (no dev predictions; cross-fitted on test).",
          "- ECE is gameable (a constant equal to the base rate has ECE≈0 and zero selective ability); always read it with AURC/AUROC.",
          "- Only the 974/914 gold-bearing docs: abstention on skill-free sentences is not measured here.",
          "", "Files: `results.json`, `reliability_bins.csv`, `reliability_diagrams.png`, `risk_coverage.{csv,png}`."]
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-head", type=int, default=10000)
    ap.add_argument("--build-gliner-cache", action="store_true")
    args = ap.parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.build_gliner_cache:
        build_gliner_cache()
        return 0
    n_boot = min(args.n_boot, 100) if args.smoke else args.n_boot
    n_head = min(args.n_boot_head, 200) if args.smoke else args.n_boot_head
    records, md, rep = run(args.smoke, args.seed, n_boot, n_head)
    out = write_layer(LAYER, records, md, seed=args.seed, smoke=args.smoke,
                      extra_meta={"n_boot": n_boot, "n_boot_headline": n_head, "h4": rep.get("h4", {}).get("verdict")})
    (out / "summary.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"wrote {out}: {len(records)} records; H4 {rep.get('h4', {}).get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
