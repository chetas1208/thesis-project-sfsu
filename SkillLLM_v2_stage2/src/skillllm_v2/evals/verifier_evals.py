"""Span-verifier evals: keep/drop quality, end-to-end effect on span F1, false drops on gold spans, threshold sweep.

What is evaluated (and what is NOT wired):
  * ``SpanVerifier`` (rule-based, shipped in the wrapper) -- its anchor protection means it can only ever drop
    non-anchor additions (and spans with missing offsets).
  * The TF-IDF + logistic-regression verifier artifact (``artifacts/skillllm_v2/span_verifier/model``). Nothing in
    ``src/`` loads it at inference time (only ``train_span_verifier`` writes it), so it is evaluated here as a
    *hypothetical* keep/drop classifier, clearly labelled as not part of the shipped pipeline.

Datasets and their leakage status:
  * verifier_dev        (data/skillllm_v2/verifier_dev.jsonl): not used to fit the model -> held-out for the model, but
                        synthetic negatives and (some) duplicated sentences across splits (flagged, text-disjoint view given).
  * verifier_test_synth (rebuilt from SkillSpan test with the same generator as the training data): held-out for the model.
  * verifier_train      scored only as an IN-SAMPLE contrast, always caveated (never a headline).
  * real candidates     persisted extractor outputs on SkillSpan test (merged candidates, GLiNER baseline): the only
                        real-distribution data; the model never saw this distribution (domain shift), and no dev
                        candidate dump is persisted, so any threshold chosen on them is post hoc (cross-fit view given).

Negatives in the verifier datasets are synthetic (+-5-character boundary expansions and a fixed list of generic
words). The rule verifier's reject list is the same list the generator draws from, so its synthetic score is circular.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.calibration_evals import (
    MIN_SLICE_N,
    _clean,
    _percentile_ci,
    _res as _cal_res,
    average_precision,
    auroc,
    cluster_bootstrap,
    flatten_candidates,
    load_gold,
    load_seen_texts,
)
from skillllm_v2.evals.contract import EvalContext, EvalResult, bootstrap_ci

MODULE = "verify"
MODEL_PATH = "artifacts/skillllm_v2/span_verifier/model/verifier_model.joblib"
GUARDRAILS_PATH = "configs/skillllm_v2_wrapper_guardrails.yaml"
DEFAULT_RECALL_TOL = 0.01  # absolute drop in exact-span recall vs no verifier
DEFAULT_F1_FLOOR = 0.665  # anchor 0.675034 - max_drop 0.010034 (configs/skillllm_v2_wrapper_guardrails.yaml)
THRESHOLDS = [round(0.05 * i, 2) for i in range(0, 20)]
REAL_SETS = {
    "merged_candidates": "artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/candidate_predictions.jsonl",
    "gliner_baseline": "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl",
}


def _res(name: str, metric: str, value: Optional[float], n: int, dataset: str, ci=None, details=None, caveats=None,
         status: str = "ok", reason: str = "") -> EvalResult:
    r = _cal_res(name, metric, value, n, dataset, ci, details, caveats, status, reason)
    r.module = MODULE
    return r


# --------------------------------------------------------------------------------------- pure helpers

def binary_prf(keep: Sequence[bool], y: Sequence[int]) -> Dict[str, float]:
    """Keep/drop classification where positive = 'is a real skill' (should be kept)."""
    k, t = np.asarray(keep, dtype=bool), np.asarray(y, dtype=int)
    tp = int((k & (t == 1)).sum())
    fp = int((k & (t == 0)).sum())
    fn = int((~k & (t == 1)).sum())
    tn = int((~k & (t == 0)).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) and math.isfinite(p) and math.isfinite(r) else (0.0 if tp + fp + fn else float("nan"))
    return {
        "precision": p, "recall": r, "f1": f1, "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "false_drop_rate": fn / (tp + fn) if tp + fn else float("nan"),  # true skills deleted
        "false_keep_rate": fp / (fp + tn) if fp + tn else float("nan"),  # non-skills let through
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def micro_prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def largest_threshold_within(rows: List[Dict[str, Any]], key: str, bound: float, mode: str = "max") -> Optional[float]:
    """Largest threshold whose ``key`` is <= bound (mode='max') or >= bound (mode='min'). Thresholds where the
    verifier drops nothing satisfy the 'no loss' constraints, so the answer is >= 0 whenever the sweep starts at 0."""
    ok = [r["threshold"] for r in rows if (r[key] <= bound + 1e-12 if mode == "max" else r[key] >= bound - 1e-12)]
    return max(ok) if ok else None


# --------------------------------------------------------------------------------------- model + datasets

def load_model(ctx: EvalContext):
    if "verifier_model" in ctx.extra:
        return ctx.extra["verifier_model"], ""
    p = ctx.path(MODEL_PATH)
    if not p.exists():
        return None, f"missing {MODEL_PATH}"
    try:
        import joblib

        return joblib.load(p), ""
    except Exception as exc:
        return None, f"cannot load verifier model (joblib/scikit-learn): {exc}"


def model_text(surface: str, text: str) -> str:
    """Exactly the feature string used at training time (train_span_verifier._load_examples)."""
    return f"{surface} [SEP] {text}"


def score_model(model, items: Sequence[Tuple[str, str]], batch: int = 4000) -> np.ndarray:
    out: List[np.ndarray] = []
    for i in range(0, len(items), batch):
        out.append(model.predict_proba([model_text(s, t) for s, t in items[i:i + batch]])[:, 1])
    return np.concatenate(out) if out else np.zeros(0)


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _rows_to_items(rows: List[Dict[str, Any]], pos_keys: Optional[set] = None) -> Dict[str, Any]:
    texts = [r["text"] for r in rows]
    doc_of: Dict[str, int] = {}
    doc = np.array([doc_of.setdefault(t, len(doc_of)) for t in texts])
    cand = [r["candidate"] for r in rows]
    y = np.array([int(bool(r["is_valid_skill"])) for r in rows])
    keys = [(r["text"], r["start"], r["end"]) for r in rows]
    if pos_keys is None:
        pos_keys = {k for k, yy in zip(keys, y) if yy == 1}
    return {
        "rows": rows, "text": texts, "doc": doc, "n_docs": len(doc_of), "y": y, "cand": cand,
        "surface": [c["surface_form"] for c in cand], "label": [c["label"] for c in cand],
        "n_words": np.array([len(c["surface_form"].split()) for c in cand]),
        "kind": [r.get("source", "?") for r in rows],
        "noisy_negative": np.array([int(yy == 0 and k in pos_keys) for k, yy in zip(keys, y)]),
    }


def build_test_synth_rows(ctx: EvalContext) -> Optional[List[Dict[str, Any]]]:
    """Rebuild verifier rows from SkillSpan test with the exact generator used for train/dev
    (build_verifier_dataset): gold positives, +-5-char boundary negatives, generic-word negatives."""
    try:
        from skillllm_v2.train import build_verifier_dataset as bvd

        if "test_docs" in ctx.extra:
            docs = ctx.extra["test_docs"]
        else:
            docs = bvd.load_skillspan_split(ctx.path("data/source/skillspan_hf"), "test")
    except Exception:
        return None
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        for span in doc["gold_spans"]:
            rows.append(bvd._verifier_row(doc["text"], span, True, span["label"], span["surface_form"], "gold", {"reason": "gold_span"}))
            exp = bvd._expanded_negative(doc["text"], span)
            if exp:
                rows.append(bvd._verifier_row(doc["text"], exp, False, None, span["surface_form"], "boundary_noise", {"reason": "boundary_expanded"}))
        for neg in bvd._generic_negatives(doc["text"]):
            rows.append(bvd._verifier_row(doc["text"], neg, False, None, None, "synthetic_negative", {"reason": "generic_phrase"}))
    return rows


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------------------- classifier eval

def _clf_fn(thr: float):
    def fn(score: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        m = binary_prf(score >= thr, y)
        return {"precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "false_drop_rate": m["false_drop_rate"],
                "false_keep_rate": m["false_keep_rate"], "auroc": auroc(score, y), "average_precision": average_precision(score, y)}
    return fn


def _rule_fn(score: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    m = binary_prf(score >= 0.5, y)
    return {k: m[k] for k in ("precision", "recall", "f1", "false_drop_rate", "false_keep_rate")}


def _classifier_results(prefix: str, label: str, items: Dict[str, Any], scores: np.ndarray, thr: float, dataset: str,
                        caveats: List[str], n_boot: int, seed: int, details_extra: Dict[str, Any]) -> List[EvalResult]:
    y = items["y"]
    point = {**{k: v for k, v in binary_prf(scores >= thr, y).items()}, "auroc": auroc(scores, y), "average_precision": average_precision(scores, y)}
    cis = cluster_bootstrap(items["doc"], items["n_docs"], scores, y, fn=_clf_fn(thr), n_boot=n_boot, seed=seed)
    base = float(y.mean())
    keep_all = binary_prf(np.ones(len(y), dtype=bool), y)
    det = {"threshold": thr, "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()), "base_rate_positive": base,
           "keep_all_baseline_f1": keep_all["f1"], "confusion": {k: point[k] for k in ("tp", "fp", "fn", "tn")},
           "n_bootstrap": n_boot, "ci_method": "document-cluster percentile bootstrap (unit = source sentence)", **details_extra}
    out = []
    for k in ("precision", "recall", "f1", "auroc", "average_precision", "false_drop_rate", "false_keep_rate"):
        out.append(_res(f"{prefix}.{k}", k, point[k], items["n_docs"], dataset, cis.get(k), det, caveats))
    return out


def _real_candidate_set(rows: List[Dict[str, Any]], gold: Dict[str, Dict[str, Any]], seen: Optional[set]) -> Dict[str, Any]:
    fl = flatten_candidates(rows, gold, seen)
    all_ids = [r["id"] for r in rows if r["id"] in gold]
    gidx = {i: k for k, i in enumerate(all_ids)}
    fl["gdoc"] = np.array([gidx[fl["doc_ids"][d]] for d in fl["doc"]], dtype=int)
    fl["all_ids"] = all_ids
    fl["gold_sets"] = [gold[i]["spans"] for i in all_ids]
    fl["keys"] = [(c.get("start"), c.get("end"), c.get("label")) for c in fl["cand"]]
    fl["text_of"] = [gold[fl["doc_ids"][d]]["text"] for d in fl["doc"]]
    return fl


def doc_counts(cs: Dict[str, Any], keep: np.ndarray) -> np.ndarray:
    """Per-document (tp, fp, fn) of exact spans for the kept candidates (set semantics, like evaluate_extraction)."""
    d_n = len(cs["all_ids"])
    pred: List[set] = [set() for _ in range(d_n)]
    for i in np.flatnonzero(keep):
        pred[cs["gdoc"][i]].add(cs["keys"][i])
    out = np.zeros((d_n, 3))
    for d in range(d_n):
        g, p = cs["gold_sets"][d], pred[d]
        out[d] = (len(g & p), len(p - g), len(g - p))
    return out


def paired_f1_bootstrap(ca: np.ndarray, cb: np.ndarray, n_boot: int, seed: int) -> Dict[str, Dict[str, float]]:
    """Paired document-level bootstrap of micro precision/recall/F1 of arm a minus arm b (per-doc tp/fp/fn arrays)."""
    d_n = len(ca)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(d_n, np.full(d_n, 1.0 / d_n), size=n_boot).astype(float)
    sa, sb = w @ ca, w @ cb

    def prf(s: np.ndarray):
        p = np.divide(s[:, 0], s[:, 0] + s[:, 1], out=np.zeros(len(s)), where=(s[:, 0] + s[:, 1]) > 0)
        r = np.divide(s[:, 0], s[:, 0] + s[:, 2], out=np.zeros(len(s)), where=(s[:, 0] + s[:, 2]) > 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros(len(s)), where=(p + r) > 0)
        return {"precision": p, "recall": r, "f1": f}

    ba, bb = prf(sa), prf(sb)
    pa, pb = micro_prf(*ca.sum(axis=0)), micro_prf(*cb.sum(axis=0))
    out = {}
    for i, k in enumerate(("precision", "recall", "f1")):
        diff = ba[k] - bb[k]
        ci = _percentile_ci(diff)
        below, above = float((diff <= 0).mean()), float((diff >= 0).mean())
        out[k] = {"a": pa[i], "b": pb[i], "delta": pa[i] - pb[i], "ci_low": ci[0] if ci else float("nan"), "ci_high": ci[1] if ci else float("nan"),
                  "p": max(min(1.0, 2 * min(below, above)), 1.0 / len(diff))}  # floored at 1/n_boot
    return out


def score_real_set(model, cs: Dict[str, Any]) -> np.ndarray:
    return score_model(model, list(zip([c["surface_form"] for c in cs["cand"]], cs["text_of"])))


def rule_keep(cs: Dict[str, Any], protect_anchor: bool) -> Tuple[np.ndarray, Counter]:
    from skillllm_v2.verify.span_verifier import SpanVerifier

    ver = SpanVerifier() if protect_anchor else SpanVerifier(protect_anchor_sources=frozenset())
    keep = np.ones(len(cs["cand"]), dtype=bool)
    reasons: Counter = Counter()
    for i, c in enumerate(cs["cand"]):
        try:
            d = ver.verify(c)
        except Exception as exc:  # malformed candidate -> keep (verifier would have raised in the pipeline)
            reasons[f"error:{type(exc).__name__}"] += 1
            continue
        keep[i] = bool(d["is_valid_skill"])
        if not keep[i]:
            reasons[d["reason"]] += 1
    return keep, reasons


def sweep_table(cs: Dict[str, Any], score: np.ndarray, thresholds: Sequence[float], apply_mask: np.ndarray,
                doc_sel: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Corpus-level exact-span metrics when candidates in ``apply_mask`` with score < t are dropped. ``doc_sel``
    restricts the corpus to a document subset (for cross-fitting). Recall loss is vs. keeping everything."""
    y = cs["y"]
    d_n = len(cs["all_ids"])
    sel = np.ones(d_n, dtype=bool) if doc_sel is None else doc_sel
    cand_in = sel[cs["gdoc"]]
    base_counts = doc_counts(cs, cand_in)[sel].sum(axis=0)
    _, base_r, base_f1 = micro_prf(*base_counts)
    rows = []
    for t in thresholds:
        keep = cand_in & ~(apply_mask & (score < t))
        c = doc_counts(cs, keep)[sel].sum(axis=0)
        p, r, f1 = micro_prf(*c)
        n_correct = int((y[cand_in] == 1).sum())
        dropped_correct = int(((cand_in & ~keep) & (y == 1)).sum())
        rows.append({
            "threshold": t, "n_dropped": int((cand_in & ~keep).sum()), "n_dropped_correct": dropped_correct,
            "false_drop_rate": dropped_correct / n_correct if n_correct else float("nan"),
            "precision": p, "recall": r, "f1": f1, "recall_loss": base_r - r, "f1_delta": f1 - base_f1,
        })
    return rows


def _floor_from_guardrails(ctx: EvalContext) -> Tuple[float, str]:
    try:
        from skillllm_v2.config import load_yaml

        p = ctx.path(GUARDRAILS_PATH)
        if p.exists():
            g = load_yaml(p)
            return float(g["exact_span_f1"]["target"]) - float(g["exact_span_f1"]["max_drop"]), GUARDRAILS_PATH
    except Exception:
        pass
    return DEFAULT_F1_FLOOR, "built-in default 0.665"


# --------------------------------------------------------------------------------------- slices

def _length_bucket(n: int) -> str:
    return "1w" if n <= 1 else "2w" if n == 2 else "3w" if n == 3 else "4w+"


def slice_results(prefix: str, dataset: str, dims: Dict[str, Sequence[Any]], y: np.ndarray, ml_keep: Optional[np.ndarray],
                  rule_keep_arr: Optional[np.ndarray], seed: int, n_boot: int, caveats: List[str]) -> List[EvalResult]:
    """Per-slice false-drop rate on true skills (critical) and false-keep rate on non-skills, for model and rules."""
    out: List[EvalResult] = []
    for dim, col in dims.items():
        col_arr = np.asarray(col)
        for val in sorted(set(col_arr.tolist()), key=str):
            m = col_arr == val
            ypos, yneg = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
            base = f"{prefix}.slice.{dim}.{val}"
            primary = ml_keep if ml_keep is not None else rule_keep_arr
            if primary is None:
                continue
            det: Dict[str, Any] = {"n": int(m.sum()), "n_true_skills": ypos, "n_non_skills": yneg,
                                   "base_rate_positive": ypos / max(1, int(m.sum())),
                                   "ci_method": "bootstrap over candidates in the slice (not document-clustered)"}
            for name, keep in (("model", ml_keep), ("rule", rule_keep_arr)):
                if keep is None:
                    continue
                mk = binary_prf(keep[m], y[m])
                det[name] = {k: _clean(mk[k]) for k in ("precision", "recall", "f1", "false_drop_rate", "false_keep_rate")}
            if ypos >= MIN_SLICE_N:
                dropped_among_pos = (~primary[m & (y == 1)]).astype(float).tolist()
                out.append(_res(f"{base}.false_drop_rate", "false_drop_rate", float(np.mean(dropped_among_pos)), ypos, dataset,
                                bootstrap_ci(dropped_among_pos, n_boot=n_boot, seed=seed), det, caveats))
            elif ypos > 0:
                out.append(_res(f"{base}.false_drop_rate", "false_drop_rate", None, int(m.sum()), dataset, status="skipped",
                                reason=f"only {ypos} true skills in slice (< {MIN_SLICE_N}); false-drop rate not estimable"))
            if yneg >= MIN_SLICE_N:
                kept_neg = (primary[m & (y == 0)]).astype(float).tolist()
                out.append(_res(f"{base}.false_keep_rate", "false_keep_rate", float(np.mean(kept_neg)), yneg, dataset,
                                bootstrap_ci(kept_neg, n_boot=n_boot, seed=seed), det, caveats))
            elif yneg > 0:
                out.append(_res(f"{base}.false_keep_rate", "false_keep_rate", None, int(m.sum()), dataset, status="skipped",
                                reason=f"only {yneg} non-skills in slice (< {MIN_SLICE_N}); false-keep rate not estimable"))
    return out


# --------------------------------------------------------------------------------------- run

def run(ctx: EvalContext) -> List[EvalResult]:
    results: List[EvalResult] = []
    model, why = load_model(ctx)
    n_boot = min(ctx.n_boot, 200) if ctx.smoke else ctx.n_boot
    thr = float(ctx.extra.get("model_threshold", 0.5))

    # ---------------- synthetic verifier datasets: keep/drop classifier ----------------
    train_rows = dev_rows = None
    tp, dp = ctx.path("data/skillllm_v2/verifier_train.jsonl"), ctx.path("data/skillllm_v2/verifier_dev.jsonl")
    if "verifier_train_rows" in ctx.extra:
        train_rows, dev_rows = ctx.extra["verifier_train_rows"], ctx.extra["verifier_dev_rows"]
    else:
        if tp.exists():
            train_rows = _read_jsonl(tp)
        if dp.exists():
            dev_rows = _read_jsonl(dp)
    if ctx.smoke and "verifier_train_rows" not in ctx.extra:
        train_rows = train_rows[:300] if train_rows else train_rows
        dev_rows = dev_rows[:300] if dev_rows else dev_rows
    train_texts = {r["text"] for r in train_rows} if train_rows else set()
    test_rows = build_test_synth_rows(ctx)
    if ctx.smoke and test_rows:
        test_rows = test_rows[:300]

    datasets: List[Tuple[str, Optional[List[Dict[str, Any]]], List[str]]] = [
        ("verifier_dev", dev_rows, ["held-out for the model (not used for fitting); synthetic negatives (+-5-char boundary expansion / generic words), so this is not real-extractor precision"]),
        ("verifier_test_synth", test_rows, ["held-out for the model: rebuilt from SkillSpan test with the training-data generator; synthetic negatives"]),
        ("verifier_train_IN_SAMPLE", train_rows, ["LEAKAGE: IN-SAMPLE -- the model was fit on this split; shown only to quantify the generalisation gap, never cite as performance"]),
    ]
    item_cache: Dict[str, Dict[str, Any]] = {}
    score_cache: Dict[str, np.ndarray] = {}
    for name, rows, cav in datasets:
        if not rows:
            results.append(_res(f"verify.model.{name}", MODULE, None, 0, name, status="skipped", reason="dataset not available offline"))
            continue
        items = _rows_to_items(rows)
        item_cache[name] = items
        seen_frac = float(np.mean([t in train_texts for t in items["text"]])) if train_texts and name != "verifier_train_IN_SAMPLE" else None
        extra = {"n_rows": len(rows), "n_docs": items["n_docs"], "frac_rows_text_in_verifier_train": seen_frac,
                 "n_negatives_that_are_also_gold_spans": int(items["noisy_negative"].sum()),
                 "kind_counts": dict(Counter(items["kind"]))}
        cavs = list(cav)
        if seen_frac:
            cavs.append(f"LEAKAGE FLAG: {seen_frac:.1%} of rows come from a sentence whose exact text also occurs in verifier_train (duplicate sentences across splits); see text-disjoint view")
        if items["noisy_negative"].sum():
            cavs.append(f"label noise: {int(items['noisy_negative'].sum())} 'negative' rows are exactly a gold span of the same sentence (generic-word negatives that are real skills)")
        dataset = f"{name} ({len(rows)} rows)"
        # rule-based SpanVerifier as a keep/drop classifier
        rk = _rule_keep_rows(items)
        results.extend(_rule_clf_results(f"verify.rule.{name}", dataset, items, rk, cavs + ["rule verifier's reject list is the generator's own generic-word list: circular on synthetic negatives"], n_boot, ctx.seed, extra))
        if model is None:
            results.append(_res(f"verify.model.{name}", MODULE, None, len(rows), dataset, status="skipped", reason=why))
            continue
        sc = score_model(model, list(zip(items["surface"], items["text"])))
        score_cache[name] = sc
        boot = min(n_boot, 200) if name == "verifier_train_IN_SAMPLE" else n_boot
        results.extend(_classifier_results(f"verify.model.{name}", name, items, sc, thr, dataset,
                                           cavs + ["model is NOT wired into the shipped pipeline (nothing loads verifier_model.joblib at inference)"], boot, ctx.seed, extra))
        if seen_frac and name != "verifier_train_IN_SAMPLE":
            keep = np.array([t not in train_texts for t in items["text"]])
            if keep.sum() >= MIN_SLICE_N:
                sub = _rows_to_items([r for r, k in zip(rows, keep) if k], pos_keys=None)
                ssc = sc[keep]
                results.extend(_classifier_results(f"verify.model.{name}_text_disjoint", name, sub, ssc, thr, f"{name} text-disjoint from verifier_train ({len(sub['y'])} rows)",
                                                   ["rows from sentences that also appear in verifier_train removed (leakage-safe view)"], boot, ctx.seed, {"n_rows": len(sub["y"])}))
        # slices (model): kind / label / length
        dims = {"kind": items["kind"], "label": items["label"], "length": [_length_bucket(n) for n in items["n_words"]]}
        results.extend(slice_results(f"verify.model.{name}", dataset, dims, items["y"], sc >= thr, rk, ctx.seed, min(n_boot, 500),
                                     cavs + ["synthetic slices"]))

    # ---------------- real candidates: end-to-end effect ----------------
    gold = load_gold(ctx)
    if gold is None:
        results.append(_res("verify.e2e", MODULE, None, 0, "skillspan_test", status="skipped", reason="SkillSpan test gold not available"))
        return results
    seen = load_seen_texts(ctx)
    sets: Dict[str, Dict[str, Any]] = {}
    if "real_candidate_sets" in ctx.extra:
        raw_sets = ctx.extra["real_candidate_sets"]
    else:
        raw_sets = {}
        for k, rel in REAL_SETS.items():
            p = ctx.path(rel)
            if p.exists():
                rows = _read_jsonl(p)
                raw_sets[k] = rows[:40] if ctx.smoke else rows
            else:
                results.append(_res(f"verify.e2e.{k}", MODULE, None, 0, "skillspan_test", status="skipped", reason=f"missing artifact {rel}"))
    floor, floor_src = _floor_from_guardrails(ctx)
    tol = float(ctx.extra.get("recall_tol", DEFAULT_RECALL_TOL))
    for sname, rows in raw_sets.items():
        cs = _real_candidate_set(rows, gold, seen)
        if len(cs["cand"]) == 0:
            results.append(_res(f"verify.e2e.{sname}", MODULE, None, 0, "skillspan_test", status="skipped", reason="no candidates"))
            continue
        sets[sname] = cs
        results.extend(_e2e_for_set(ctx, sname, cs, model, why, thr, tol, floor, floor_src, n_boot))

    # dev candidates do not exist
    results.append(_res("verify.real_dev_candidates", MODULE, None, 0, "skillspan_dev", status="skipped",
                        reason="no persisted dev candidate dump (dev_candidates_cache.jsonl absent): thresholds for real candidates cannot be tuned on held-out dev; "
                               "test-set sweeps are post hoc and a split-half cross-fit is reported instead"))
    results.append(_res("verify.artifact_eval_predictions", MODULE, None, 20, "span_verifier/eval_predictions.jsonl", status="skipped",
                        reason="persisted eval_predictions.jsonl is a 20-row smoke sample of the rule verifier (metrics.json precision 0.526 / recall 1.0 on n=20); too small to use"))
    return results


def _rule_keep_rows(items: Dict[str, Any]) -> np.ndarray:
    from skillllm_v2.verify.span_verifier import SpanVerifier

    ver = SpanVerifier()
    keep = np.ones(len(items["cand"]), dtype=bool)
    for i, c in enumerate(items["cand"]):
        try:
            keep[i] = bool(ver.verify(c)["is_valid_skill"])
        except Exception:
            keep[i] = True
    return keep


def _rule_clf_results(prefix: str, dataset: str, items: Dict[str, Any], keep: np.ndarray, caveats: List[str], n_boot: int, seed: int,
                      extra: Dict[str, Any]) -> List[EvalResult]:
    y = items["y"]
    sc = keep.astype(float)
    cis = cluster_bootstrap(items["doc"], items["n_docs"], sc, y, fn=_rule_fn, n_boot=n_boot, seed=seed)
    m = binary_prf(keep, y)
    det = {"confusion": {k: m[k] for k in ("tp", "fp", "fn", "tn")}, "n_bootstrap": n_boot, "decision_rule": "SpanVerifier().verify (candidate.source='dictionary', i.e. not anchor-protected)", **extra}
    return [_res(f"{prefix}.{k}", k, m[k], items["n_docs"], dataset, cis.get(k), det, caveats)
            for k in ("precision", "recall", "f1", "false_drop_rate", "false_keep_rate")]


def _e2e_for_set(ctx: EvalContext, sname: str, cs: Dict[str, Any], model, why: str, thr: float, tol: float, floor: float,
                 floor_src: str, n_boot: int) -> List[EvalResult]:
    out: List[EvalResult] = []
    n_c, n_d = len(cs["cand"]), len(cs["all_ids"])
    dataset = f"skillspan_test candidate set '{sname}' ({n_d} docs, {n_c} candidates)"
    anchor = np.asarray(cs["anchor"], dtype=bool)
    y = cs["y"]
    seen_frac = float(np.mean(cs["seen"])) if len(cs["seen"]) else 0.0
    base_cav = ["real extractor candidates on SkillSpan test; identical candidate set in every arm; exact (start,end,label) micro-F1 with set semantics as in evaluate_extraction",
                "paired document-level bootstrap (shared draws) over documents"]
    if seen_frac:
        base_cav.append(f"LEAKAGE FLAG: {seen_frac:.1%} of candidates come from test sentences whose text also occurs in train/dev (affects extractor and verifier alike)")
    none_counts = doc_counts(cs, np.ones(n_c, dtype=bool))
    p0, r0, f0 = micro_prf(*none_counts.sum(axis=0))

    arms: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    rk_s, reasons_s = rule_keep(cs, protect_anchor=True)
    arms["rule_shipped_anchor_protected"] = (rk_s, {"drop_reasons": dict(reasons_s), "policy": "SpanVerifier() as shipped (anchor-derived candidates protected)"})
    rk_u, reasons_u = rule_keep(cs, protect_anchor=False)
    arms["rule_unprotected"] = (rk_u, {"drop_reasons": dict(reasons_u), "policy": "SpanVerifier with no anchor protection"})
    score = None
    if model is not None:
        score = score_real_set(model, cs)
        arms[f"model_t{thr:g}_all"] = (score >= thr, {"policy": "TF-IDF/LR verifier applied to every candidate (hypothetical; not shipped)"})
        if anchor.any():
            arms[f"model_t{thr:g}_nonanchor_only"] = (~(~anchor & (score < thr)), {"policy": "TF-IDF/LR applied only to non-anchor candidates (anchor protected, like the shipped policy)"})
    else:
        out.append(_res(f"verify.e2e.{sname}.model", MODULE, None, n_c, dataset, status="skipped", reason=why))

    out.append(_res(f"verify.e2e.{sname}.exact_f1_without_verifier", "exact_span_f1", f0, n_d, dataset, None,
                    {"precision": p0, "recall": r0, "n_candidates": n_c, "n_correct_candidates": int(y.sum()),
                     "n_gold_spans": int(sum(len(g) for g in cs["gold_sets"]))}, base_cav))
    for aname, (keep, extra) in arms.items():
        counts = doc_counts(cs, keep)
        pb = paired_f1_bootstrap(counts, none_counts, n_boot, ctx.seed)
        m = binary_prf(keep, y)
        det = {**extra, "n_dropped": int((~keep).sum()), "n_true_skills_dropped": int((~keep & (y == 1)).sum()),
               "candidate_false_drop_rate": _clean(m["false_drop_rate"]), "candidate_false_keep_rate": _clean(m["false_keep_rate"]),
               "f1_with": pb["f1"]["a"], "f1_without": pb["f1"]["b"], "precision_delta": pb["precision"]["delta"],
               "recall_delta": pb["recall"]["delta"], "recall_delta_ci": [pb["recall"]["ci_low"], pb["recall"]["ci_high"]],
               "precision_delta_ci": [pb["precision"]["ci_low"], pb["precision"]["ci_high"]], "p_value_f1": pb["f1"]["p"],
               "n_bootstrap": n_boot}
        caveats = list(base_cav)
        if aname.startswith("model"):
            caveats.append("domain shift: model fit on gold + synthetic-corruption verifier data, applied to real extractor output; it is not part of the shipped wrapper")
        d = pb["f1"]
        out.append(_res(f"verify.e2e.{sname}.{aname}.exact_f1_delta", "delta_exact_span_f1(with-without)", d["delta"], n_d, dataset,
                        (d["ci_low"], d["ci_high"]), det, caveats))
        r = pb["recall"]
        out.append(_res(f"verify.e2e.{sname}.{aname}.recall_delta", "delta_exact_span_recall(with-without)", r["delta"], n_d, dataset,
                        (r["ci_low"], r["ci_high"]), det, caveats))
        out.append(_res(f"verify.e2e.{sname}.{aname}.candidate_false_drop_rate", "false_drop_rate_on_correct_candidates", m["false_drop_rate"],
                        int(y.sum()), dataset, bootstrap_ci((~keep[y == 1]).astype(float).tolist(), n_boot=n_boot, seed=ctx.seed) if y.sum() else None,
                        det, caveats + ["CI: bootstrap over correct candidates (not document-clustered)"]))

    if score is not None:
        out.extend(_real_clf_and_sweeps(ctx, sname, cs, score, thr, tol, floor, floor_src, n_boot, dataset, base_cav, arms))
    return out


def _real_clf_and_sweeps(ctx: EvalContext, sname: str, cs: Dict[str, Any], score: np.ndarray, thr: float, tol: float, floor: float,
                         floor_src: str, n_boot: int, dataset: str, base_cav: List[str], arms) -> List[EvalResult]:
    out: List[EvalResult] = []
    y = cs["y"]
    anchor = np.asarray(cs["anchor"], dtype=bool)
    items = {"doc": cs["doc"], "n_docs": cs["n_docs"], "y": y}
    if len(set(y.tolist())) == 2:
        out.extend(_classifier_results(f"verify.model.real_{sname}", sname, items, score, thr, dataset,
                                       base_cav + ["keep/drop label = candidate is an exact gold match; model scored out-of-distribution (trained on synthetic negatives)"],
                                       n_boot, ctx.seed, {"n_candidates": len(y)}))
    # sweeps
    for pname, mask in (("all", np.ones(len(y), dtype=bool)), ("nonanchor_only", ~anchor)):
        if pname == "nonanchor_only" and not anchor.any():
            continue
        tab = sweep_table(cs, score, THRESHOLDS, mask)
        t_tol = largest_threshold_within(tab, "recall_loss", tol, "max")
        t_floor = largest_threshold_within(tab, "f1", floor, "min")
        d_n = len(cs["all_ids"])
        fold = np.arange(d_n) % 2 == 0
        cross = []
        for tr, te in ((fold, ~fold), (~fold, fold)):
            tt = largest_threshold_within(sweep_table(cs, score, THRESHOLDS, mask, doc_sel=tr), "recall_loss", tol, "max")
            row = sweep_table(cs, score, [tt if tt is not None else 0.0], mask, doc_sel=te)[0]
            cross.append({"selected_threshold": tt, "heldout_recall_loss": row["recall_loss"], "heldout_f1_delta": row["f1_delta"], "heldout_false_drop_rate": row["false_drop_rate"]})
        cav = base_cav + [
            "threshold chosen on the same test candidates it is reported on (no persisted dev candidates) -> operating curve, not held-out performance; see split-half cross-fit in details",
            f"recall tolerance = {tol} absolute exact-span recall vs keeping every candidate; F1 floor = {floor:.6f} from {floor_src}",
        ]
        det = {"policy": pname, "sweep": tab, "recall_tolerance": tol, "f1_floor": floor, "f1_floor_source": floor_src,
               "largest_threshold_f1_at_least_floor": t_floor, "split_half_crossfit": cross,
               "interpretation": ("0.0 means any dropping at all already exceeds the tolerance; the model is unusable as a hard filter here" if t_tol == 0.0 else
                          "f1 floor not met even at threshold 0 (no-verifier F1 is already below the floor)" if t_floor is None else "")}
        out.append(_res(f"verify.sweep.{sname}.{pname}.max_threshold_recall_loss_le_tol", "largest_threshold_with_recall_loss<=tol", t_tol, len(cs["all_ids"]), dataset, None, det, cav))
        out.append(_res(f"verify.sweep.{sname}.{pname}.max_threshold_f1_ge_floor", "largest_threshold_with_f1>=floor", t_floor, len(cs["all_ids"]), dataset, None, det, cav))
    # slices on real candidates (model @thr for all candidates + shipped rule)
    rk_s = arms["rule_shipped_anchor_protected"][0]
    dims = {"source_type": cs["source_key"], "label": cs["label"], "length": [_length_bucket(n) for n in cs["n_words"]]}
    out.extend(slice_results(f"verify.real_{sname}", dataset, dims, y, score >= thr, rk_s, ctx.seed, min(n_boot, 500),
                             base_cav + ["slice: model arm applied to every candidate; rule arm = shipped SpanVerifier; CIs over candidates"]))
    return out
