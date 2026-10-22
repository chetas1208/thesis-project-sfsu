"""Extraction evals (module=extract).

Everything here is recomputed offline from saved per-document predictions
(``artifacts/skillllm_v2/*/predictions.jsonl``) against the SkillSpan HF test gold
(``data/source/skillspan_hf/test.json``).  The span scorer is the repository's own
``skillllm_v2.eval.extraction_eval`` (its private helpers are reused so the numbers
are identical, and every headline metric is asserted to reproduce the stored metric
files within 1e-6).  On top of that:

* doc-level cluster-bootstrap CIs (documents are the independent unit; spans inside
  a document are correlated),
* per-label breakdown, slices (source/domain, doc length, skills per doc, gold and
  predicted span length), an error taxonomy with example ids,
* nervaluate-style strict / exact / partial / ent_type modes (the tests cross-check
  them against an independent re-implementation),
* a shuffle-across-documents negative control plus positive / empty controls.

Caveats that apply to every result here are repeated in ``EvalResult.caveats``.
``token_f1`` in the repository evaluator is *character*-offset overlap, not word-token
overlap; a word-level variant is reported separately as ``word_token_f1``.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.eval.extraction_eval import _key, _partial_matches, _prf, _tokens, evaluate_extraction
from skillllm_v2.evals.contract import EvalContext, EvalResult
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

MODULE = "extract"
DATASET_ROOT = "data/source/skillspan_hf"
DATASET = "SkillSpan(HF) test, 974-doc subset that has cached Mistral predictions (tech+house job ads)"
LABELS = ("HARD_SKILL", "SOFT_SKILL")

# system -> (predictions path, [(stored metrics file, key inside file or None)])
SYSTEMS: Dict[str, Dict[str, Any]] = {
    "mistral_lora": {
        "pred": "artifacts/skillllm_v2/mistral_lora/predictions.jsonl",
        "stored": [
            ("artifacts/skillllm_v2/mistral_lora/metrics.json", None),
            ("reports/skillllm_v2/raw_mistral_lora_metrics.json", None),
        ],
    },
    "mistral_boundary": {
        "pred": "artifacts/skillllm_v2/mistral_boundary/predictions.jsonl",
        "stored": [("artifacts/skillllm_v2/mistral_boundary/metrics.json", None)],
    },
    "safe_wrapper": {
        "pred": "artifacts/skillllm_v2/safe_wrapper/predictions.jsonl",
        "stored": [("artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json", "candidate_metrics")],
    },
    "full_cag_graph": {
        "pred": "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl",
        "stored": [("artifacts/skillllm_v2/full_cag_graph/metrics.json", None)],
    },
}
DETAIL_SYSTEMS = ("mistral_lora", "safe_wrapper")  # slices / error taxonomy / controls run on these
REPRO_TOL = 1e-6

BASE_CAVEATS = [
    "Metric definitions are repository-specific (see details.definition); only strict exact-span equals nervaluate 'strict'; "
    "partial/token metrics are not SkillSpan-official.",
    "Single SkillSpan test split, one training run: CIs capture document-sampling noise only, not seed/training variance.",
    "Spans are scored on character offsets with label equality; the repository 'token_f1' is character-overlap, not word-token overlap.",
    "The evaluator de-duplicates predictions per document (set semantics), so duplicate spans never cost precision.",
]

METRIC_DEFINITIONS: Dict[str, str] = {
    "exact_span": "strict match on (start, end, label) after per-document set de-duplication; equals nervaluate 'strict' P/R/F1.",
    "partial_span": ("repository definition, NOT nervaluate 'partial': a prediction counts fully (weight 1, not 0.5) if it overlaps "
                     "an unmatched gold span of the same label (greedy first-fit one-to-one, prediction order); "
                     "P = matched/|pred set|, R = matched/|gold set|. Closest to nervaluate 'ent_type' but greedy."),
    "token": ("repository definition: overlap of (character offset, label) pairs, i.e. character-level, not word-level; "
              "not a SkillSpan-official metric (SkillSpan publishes no official partial-span/token-F1 definition)."),
    "word_token": "supplementary: whitespace-word level; a word is positive for a label if it overlaps a span of that label.",
    "per_label": "exact_span definition restricted to one label; macro_f1 = mean of HARD_SKILL and SOFT_SKILL F1.",
}
DEFINITION_CAVEAT = ("partial_span_f1 and token_f1 use repository-specific definitions (see details.definition); they are "
                     "not SkillSpan-official and partial_span_f1 is not comparable to nervaluate 'partial'.")


def definition_for(metric: str) -> str:
    for key in ("exact_span", "partial_span", "word_token", "token"):
        if metric.startswith(key):
            return METRIC_DEFINITIONS[key]
    return METRIC_DEFINITIONS["per_label"] if metric in ("hard_skill_f1", "soft_skill_f1", "macro_f1", "micro_f1") else ""


# column layout of the per-document count matrix
C_TP, C_FP, C_FN, C_PTP = 0, 1, 2, 3
C_TTP, C_TFP, C_TFN = 4, 5, 6
C_HTP, C_HFP, C_HFN = 7, 8, 9
C_STP, C_SFP, C_SFN = 10, 11, 12
C_WTP, C_WFP, C_WFN = 13, 14, 15
N_COLS = 16


# ----------------------------------------------------------------------------- loading
def load_gold(ctx: EvalContext) -> Dict[str, Dict[str, Any]]:
    docs = load_skillspan_split(ctx.path(DATASET_ROOT), "test", limit=None)
    return {d["id"]: d for d in docs}


def load_predictions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["id"]] = row["predictions"]
    return out


# ----------------------------------------------------------------------------- counts
def _word_sets(text: str, spans: Sequence[Dict[str, Any]]) -> set:
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out = set()
    for sp in spans:
        if sp.get("start") is None or sp.get("end") is None:
            continue
        for wi, (ws, we) in enumerate(words):
            if max(ws, sp["start"]) < min(we, sp["end"]):
                out.add((wi, sp.get("label", "HARD_SKILL")))
    return out


def doc_counts(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], text: str = "") -> List[int]:
    """Per-document sufficient statistics; summing them and applying ``_prf`` reproduces evaluate_extraction."""
    gk, pk = {_key(x) for x in gold}, {_key(x) for x in pred}
    row = [0] * N_COLS
    row[C_TP], row[C_FP], row[C_FN] = len(gk & pk), len(pk - gk), len(gk - pk)
    row[C_PTP] = _partial_matches(list(gold), list(pred))
    gt, pt = _tokens(list(gold)), _tokens(list(pred))
    row[C_TTP], row[C_TFP], row[C_TFN] = len(gt & pt), len(pt - gt), len(gt - pt)
    for label, cols in (("HARD_SKILL", (C_HTP, C_HFP, C_HFN)), ("SOFT_SKILL", (C_STP, C_SFP, C_SFN))):
        lg = {_key(x) for x in gold if x.get("label") == label}
        lp = {_key(x) for x in pred if x.get("label") == label}
        row[cols[0]], row[cols[1]], row[cols[2]] = len(lg & lp), len(lp - lg), len(lg - lp)
    if text:
        gw, pw = _word_sets(text, gold), _word_sets(text, pred)
        row[C_WTP], row[C_WFP], row[C_WFN] = len(gw & pw), len(pw - gw), len(gw - pw)
    return row


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def _f1(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    return _div(2 * p * r, p + r)


def _prf_cols(s: np.ndarray, tp: int, fp: int, fn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, r = _div(s[..., tp], s[..., tp] + s[..., fp]), _div(s[..., tp], s[..., tp] + s[..., fn])
    return p, r, _f1(p, r)


METRICS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "exact_span_precision": lambda s: _prf_cols(s, C_TP, C_FP, C_FN)[0],
    "exact_span_recall": lambda s: _prf_cols(s, C_TP, C_FP, C_FN)[1],
    "exact_span_f1": lambda s: _prf_cols(s, C_TP, C_FP, C_FN)[2],
    "partial_span_precision": lambda s: _div(s[..., C_PTP], s[..., C_TP] + s[..., C_FP]),
    "partial_span_recall": lambda s: _div(s[..., C_PTP], s[..., C_TP] + s[..., C_FN]),
    "partial_span_f1": lambda s: _f1(
        _div(s[..., C_PTP], s[..., C_TP] + s[..., C_FP]), _div(s[..., C_PTP], s[..., C_TP] + s[..., C_FN])
    ),
    "token_precision": lambda s: _prf_cols(s, C_TTP, C_TFP, C_TFN)[0],
    "token_recall": lambda s: _prf_cols(s, C_TTP, C_TFP, C_TFN)[1],
    "token_f1": lambda s: _prf_cols(s, C_TTP, C_TFP, C_TFN)[2],
    "hard_skill_f1": lambda s: _prf_cols(s, C_HTP, C_HFP, C_HFN)[2],
    "soft_skill_f1": lambda s: _prf_cols(s, C_STP, C_SFP, C_SFN)[2],
    "word_token_f1": lambda s: _prf_cols(s, C_WTP, C_WFP, C_WFN)[2],
}
METRICS["macro_f1"] = lambda s: (METRICS["hard_skill_f1"](s) + METRICS["soft_skill_f1"](s)) / 2
METRICS["micro_f1"] = METRICS["exact_span_f1"]
HEADLINE = ("exact_span_f1", "partial_span_f1", "token_f1", "word_token_f1", "exact_span_precision",
            "exact_span_recall", "hard_skill_f1", "soft_skill_f1", "macro_f1")


def aggregate(counts: np.ndarray) -> Dict[str, float]:
    s = np.asarray(counts, dtype=float).sum(axis=0)
    return {name: float(fn(s)) for name, fn in METRICS.items()}


def boot_ci(counts: np.ndarray, stat: Callable[[np.ndarray], np.ndarray], n_boot: int, seed: int,
            alpha: float = 0.05) -> Optional[Tuple[float, float]]:
    """Document-level cluster bootstrap of ``stat(sum of per-document count rows)``."""
    arr = np.asarray(counts, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    chunk = max(1, 400_000 // (n * max(1, arr.shape[1])))
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        idx = rng.integers(0, n, size=(b, n))
        out[lo:lo + b] = stat(arr[idx].sum(axis=1))
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def paired_boot(a: np.ndarray, b: np.ndarray, stat: Callable[[np.ndarray], np.ndarray], n_boot: int,
                seed: int) -> Dict[str, float]:
    """Paired doc-level bootstrap of stat(a) - stat(b) (same resampled documents for both systems)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    chunk = max(1, 400_000 // (n * a.shape[1]))
    for lo in range(0, n_boot, chunk):
        m = min(chunk, n_boot - lo)
        idx = rng.integers(0, n, size=(m, n))
        diffs[lo:lo + m] = stat(a[idx].sum(axis=1)) - stat(b[idx].sum(axis=1))
    delta = float(stat(a.sum(axis=0)) - stat(b.sum(axis=0)))
    below = float((diffs <= 0).mean())
    return {"delta": delta, "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "p": min(1.0, 2 * min(below, 1 - below))}


# ----------------------------------------------------------------------------- error taxonomy
def _overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    if a.get("start") is None or b.get("start") is None:  # unlocated span (offset search failed): overlaps nothing
        return 0
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def classify_errors(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Error events for one document.

    Exact (start, end, label) matches are correct.  Remaining items are classified:

    * duplicate        - a prediction repeating an earlier prediction's exact (start, end, label)
    * boundary_shift   - unmatched prediction overlapping an unclaimed gold span of the SAME label
    * label_confusion  - unmatched prediction overlapping an unclaimed gold span of ANOTHER label
                         (``subtype`` says whether the boundaries were identical)
    * spurious         - unmatched prediction overlapping no unclaimed gold span (``subtype='unlocated'`` when the
                         prediction has no offsets: the evaluator counts those as false positives too)
    * missed           - gold span claimed by no prediction

    Passes are ordered (boundary-shift pairs first) so results do not depend on prediction order.
    Invariants (checked by tests): FP = boundary + label + spurious, FN = missed + boundary + label.
    """
    gold = [g for g in gold if g.get("start") is not None]
    pred = list(pred)
    pred_keys = {_key(p) for p in pred}
    claimed = {i for i, g in enumerate(gold) if _key(g) in pred_keys}
    gold_keys = {_key(g) for g in gold}
    events: List[Dict[str, Any]] = []
    seen: set = set()
    remaining: List[Dict[str, Any]] = []
    for p in pred:
        k = _key(p)
        if k in seen:
            events.append({"kind": "duplicate", "pred": p, "gold": None})
            continue
        seen.add(k)
        if k not in gold_keys:
            remaining.append(p)

    def best_gold(p: Dict[str, Any], same_label: bool) -> Optional[int]:
        cands = [(i, _overlap(p, g)) for i, g in enumerate(gold)
                 if i not in claimed and (g["label"] == p["label"]) == same_label and _overlap(p, g) > 0]
        return max(cands, key=lambda t: (t[1], -t[0]))[0] if cands else None

    leftover: List[Dict[str, Any]] = []
    for p in remaining:  # pass 1: same-label overlap
        gi = best_gold(p, True)
        if gi is None:
            leftover.append(p)
        else:
            claimed.add(gi)
            events.append({"kind": "boundary_shift", "pred": p, "gold": gold[gi]})
    for p in leftover:  # pass 2: other-label overlap, else spurious
        gi = best_gold(p, False)
        if gi is None:
            events.append({"kind": "spurious", "pred": p, "gold": None,
                           "subtype": "unlocated" if p.get("start") is None else "no_overlap"})
        else:
            claimed.add(gi)
            g = gold[gi]
            sub = "same_boundaries" if (g["start"], g["end"]) == (p["start"], p["end"]) else "boundaries_and_label"
            events.append({"kind": "label_confusion", "pred": p, "gold": g, "subtype": sub})
    for i, g in enumerate(gold):
        if i not in claimed:
            events.append({"kind": "missed", "pred": None, "gold": g})
    return events


ERROR_KINDS = ("missed", "spurious", "boundary_shift", "label_confusion", "duplicate")


def error_taxonomy(ids: Sequence[str], gold_docs: Sequence[Sequence[Dict[str, Any]]],
                   pred_docs: Sequence[Sequence[Dict[str, Any]]], texts: Sequence[str],
                   max_examples: int = 5) -> Dict[str, Any]:
    counts: Counter = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    docs_with: Dict[str, set] = defaultdict(set)
    confusion: Counter = Counter()
    sub: Counter = Counter()
    unlocated = 0
    for did, g, p, text in zip(ids, gold_docs, pred_docs, texts):
        for ev in classify_errors(g, p):
            kind = ev["kind"]
            counts[kind] += 1
            docs_with[kind].add(did)
            if kind == "spurious" and ev.get("subtype") == "unlocated":
                unlocated += 1
            if kind == "label_confusion":
                confusion[f"{ev['gold']['label']}->{ev['pred']['label']}"] += 1
                sub[ev["subtype"]] += 1
            if len(examples[kind]) < max_examples:
                def show(sp):
                    if sp is None:
                        return None
                    shown = sp["surface_form"] if sp.get("start") is None else text[sp["start"]:sp["end"]]
                    return {"span": [sp.get("start"), sp.get("end")], "label": sp["label"], "text": shown}
                examples[kind].append({"doc_id": did, "pred": show(ev["pred"]), "gold": show(ev["gold"])})
    return {"counts": {k: counts.get(k, 0) for k in ERROR_KINDS}, "examples": dict(examples),
            "n_docs_affected": {k: len(docs_with.get(k, ())) for k in ERROR_KINDS},
            "label_confusion_directions": dict(confusion), "label_confusion_subtypes": dict(sub),
            "spurious_unlocated": unlocated}


# ----------------------------------------------------------------------------- nervaluate-style modes
def _max_matching(adj: List[List[int]], n_right: int) -> List[int]:
    """Kuhn's algorithm; returns match_right[j] = left index or -1."""
    match = [-1] * n_right

    def try_left(u: int, seen: set) -> bool:
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if match[v] == -1 or try_left(match[v], seen):
                match[v] = u
                return True
        return False

    for u in range(len(adj)):
        try_left(u, set())
    return match


def nervaluate_doc(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], mode: str) -> Tuple[int, int, int, int]:
    """(COR, PAR, ACT, POS) for one document under strict / exact / partial / ent_type (nervaluate semantics,
    one-to-one: each gold and each prediction is used at most once; predictions are counted as a list)."""
    gold = [g for g in gold if g.get("start") is not None]
    pred = list(pred)  # unlocated predictions stay in ACT but can never match
    same = lambda p, g: p.get("start") is not None and p["start"] == g["start"] and p["end"] == g["end"]
    ov = lambda p, g: _overlap(p, g) > 0
    if mode == "strict":
        compat1 = lambda p, g: same(p, g) and p["label"] == g["label"]
        compat2 = None
    elif mode == "exact":
        compat1, compat2 = same, None
    elif mode == "ent_type":
        compat1 = lambda p, g: ov(p, g) and p["label"] == g["label"]
        compat2 = None
    elif mode == "partial":
        compat1, compat2 = same, ov
    else:
        raise ValueError(mode)
    adj = [[j for j, g in enumerate(gold) if compat1(p, g)] for p in pred]
    m1 = _max_matching(adj, len(gold))
    cor = sum(1 for x in m1 if x != -1)
    par = 0
    if compat2 is not None:
        used_p = {x for x in m1 if x != -1}
        free_p = [i for i in range(len(pred)) if i not in used_p]
        free_g = [j for j in range(len(gold)) if m1[j] == -1]
        adj2 = [[k for k, j in enumerate(free_g) if compat2(pred[i], gold[j])] for i in free_p]
        par = sum(1 for x in _max_matching(adj2, len(free_g)) if x != -1)
    return cor, par, len(pred), len(gold)


NER_MODES = ("strict", "exact", "partial", "ent_type")


def nervaluate_counts(gold_docs, pred_docs) -> Dict[str, np.ndarray]:
    return {m: np.array([nervaluate_doc(g, p, m) for g, p in zip(gold_docs, pred_docs)], dtype=float)
            for m in NER_MODES}


def ner_prf(sums: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num = sums[..., 0] + 0.5 * sums[..., 1]
    p, r = _div(num, sums[..., 2]), _div(num, sums[..., 3])
    return p, r, _f1(p, r)


# ----------------------------------------------------------------------------- controls
def shuffled_predictions(pred_docs: Sequence[Any], rng: random.Random) -> List[Any]:
    """Cyclic derangement: every document receives another document's predictions."""
    n = len(pred_docs)
    order = list(range(n))
    rng.shuffle(order)
    out: List[Any] = [None] * n
    for i in range(n):
        out[order[i]] = pred_docs[order[(i + 1) % n]]
    return out


def negative_control(gold_docs, pred_docs, n_shuffles: int = 20, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    runs = [evaluate_extraction(list(gold_docs), shuffled_predictions(pred_docs, rng)) for _ in range(n_shuffles)]
    keys = ("exact_span_f1", "partial_span_f1", "token_f1")
    return {
        "n_shuffles": n_shuffles,
        "mean": {k: sum(r[k] for r in runs) / len(runs) for k in keys},
        "max": {k: max(r[k] for r in runs) for k in keys},
        "positive_control_gold_as_pred": {k: evaluate_extraction(list(gold_docs), list(gold_docs))[k] for k in keys},
        "empty_predictions": {k: evaluate_extraction(list(gold_docs), [[] for _ in gold_docs])[k] for k in keys},
    }


# ----------------------------------------------------------------------------- reproduction
def reproduction_check(system: str, recomputed: Dict[str, float], ctx: EvalContext) -> Dict[str, Any]:
    """Compare recomputed metrics with every stored metrics file for the system."""
    out: Dict[str, Any] = {"max_abs_diff": 0.0, "files": {}, "missing": []}
    for rel, key in SYSTEMS[system]["stored"]:
        path = ctx.path(rel)
        if not path.exists():
            out["missing"].append(rel)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if key:
            data = data[key]
        diffs = {k: abs(float(v) - recomputed[k]) for k, v in data.items()
                 if isinstance(v, (int, float)) and k in recomputed}
        worst = max(diffs.values()) if diffs else None
        out["files"][rel] = {"n_metrics_compared": len(diffs), "max_abs_diff": worst}
        if worst is not None:
            out["max_abs_diff"] = max(out["max_abs_diff"], worst)
    out["reproduced"] = bool(out["files"]) and out["max_abs_diff"] <= REPRO_TOL
    return out


# ----------------------------------------------------------------------------- slices
def _bucket(value: int, edges: Sequence[Tuple[int, Optional[int], str]]) -> str:
    for lo, hi, name in edges:
        if value >= lo and (hi is None or value <= hi):
            return name
    return "other"


DOC_LEN_EDGES = [(0, 5, "1-5 words"), (6, 12, "6-12 words"), (13, 24, "13-24 words"), (25, None, "25+ words")]
SKILLS_EDGES = [(0, 1, "1 gold skill"), (2, 2, "2 gold skills"), (3, 4, "3-4 gold skills"), (5, None, "5+ gold skills")]
SPAN_LEN_EDGES = [(1, 1, "1 word"), (2, 2, "2 words"), (3, 3, "3 words"), (4, None, "4+ words")]


def doc_slices(ids, gold_docs, doc_meta) -> Dict[str, Dict[str, List[int]]]:
    out: Dict[str, Dict[str, List[int]]] = {"source_domain": defaultdict(list), "doc_length": defaultdict(list),
                                            "skills_per_doc": defaultdict(list)}
    for i, (g, meta) in enumerate(zip(gold_docs, doc_meta)):
        out["source_domain"][f"source={meta['source']}"].append(i)
        out["doc_length"][_bucket(meta["n_tokens"], DOC_LEN_EDGES)].append(i)
        out["skills_per_doc"][_bucket(len(g), SKILLS_EDGES)].append(i)
    return out


def span_length_pairs(gold_docs, pred_docs, texts, side: str) -> Dict[str, np.ndarray]:
    """Per-document (hits, total) arrays by span word-length.

    side='gold': recall slice (hit = exact (start,end,label) match; also partial hit = same-label overlap)
    side='pred': precision slice (hit = exact match).  Columns: exact_hits, partial_hits, total.
    """
    res = {name: np.zeros((len(gold_docs), 3)) for _, _, name in SPAN_LEN_EDGES}
    for i, (g, p, text) in enumerate(zip(gold_docs, pred_docs, texts)):
        gk, pk = {_key(x) for x in g}, {_key(x) for x in p}
        items, other, otherkeys = (g, p, pk) if side == "gold" else (p, g, gk)
        seen = set()
        for sp in items:
            if sp.get("start") is None or _key(sp) in seen:
                continue
            seen.add(_key(sp))
            nwords = max(1, len(text[sp["start"]:sp["end"]].split()))
            name = _bucket(nwords, SPAN_LEN_EDGES)
            res[name][i, 2] += 1
            if _key(sp) in otherkeys:
                res[name][i, 0] += 1
            if any(o["label"] == sp["label"] and _overlap(o, sp) > 0 for o in other if o.get("start") is not None):
                res[name][i, 1] += 1
    return res


# ----------------------------------------------------------------------------- run
def _result(name: str, metric: str, value: Optional[float], n: int, ci, details: Dict[str, Any],
            extra_caveats: Sequence[str] = (), status: str = "ok", reason: str = "") -> EvalResult:
    return EvalResult(eval_name=name, module=MODULE, metric=metric, value=value, n=n, dataset=DATASET,
                      ci=tuple(ci) if ci else None, details=details, caveats=list(BASE_CAVEATS) + list(extra_caveats),
                      status=status, reason=reason)


def _skipped(name: str, metric: str, reason: str) -> EvalResult:
    return _result(name, metric, None, 0, None, {}, status="skipped", reason=reason)


def evaluate_system(system: str, gold: Dict[str, Dict[str, Any]], preds: Dict[str, List[Dict[str, Any]]],
                    ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    ids = list(ids) if ids is not None else [i for i in preds if i in gold]
    gold_docs = [gold[i]["gold_spans"] for i in ids]
    pred_docs = [preds[i] for i in ids]
    texts = [gold[i]["text"] for i in ids]
    counts = np.array([doc_counts(g, p, t) for g, p, t in zip(gold_docs, pred_docs, texts)], dtype=float)
    return {"system": system, "ids": ids, "gold_docs": gold_docs, "pred_docs": pred_docs, "texts": texts,
            "counts": counts, "metrics": aggregate(counts)}


def run(ctx: EvalContext) -> List[EvalResult]:
    results: List[EvalResult] = []
    try:
        gold = load_gold(ctx)
    except FileNotFoundError as exc:
        return [_skipped("extract.all", "all", f"gold data unavailable: {exc}")]
    wanted = ctx.extra.get("systems") or list(SYSTEMS)
    n_boot = min(ctx.n_boot, 200) if ctx.smoke else ctx.n_boot
    evals: Dict[str, Dict[str, Any]] = {}
    anchor_ids: Optional[List[str]] = None
    for system in wanted:
        path = ctx.path(SYSTEMS[system]["pred"])
        if not path.exists():
            results.append(_skipped(f"extract.{system}", "all", f"predictions missing: {SYSTEMS[system]['pred']}"))
            continue
        preds = load_predictions(path)
        ids = [i for i in preds if i in gold]
        if anchor_ids is None:
            anchor_ids = ids
        ids = [i for i in anchor_ids if i in preds]
        if ctx.smoke:
            ids = ids[: int(ctx.extra.get("smoke_docs", 60))]
        evals[system] = evaluate_system(system, gold, preds, ids)

    for system, ev in evals.items():
        counts, ids = ev["counts"], ev["ids"]
        n = len(ids)
        # reproduction: (a) our aggregation == repository evaluator, (b) == stored metric files
        ref = evaluate_extraction(ev["gold_docs"], ev["pred_docs"])
        agg_diff = max(abs(ref[k] - ev["metrics"][k]) for k in ref)
        if ctx.smoke:
            rep = {"max_abs_diff": None, "files": {}, "reproduced": None}
            status, reason = "skipped", "smoke subset; stored metrics are over the full 974-doc set"
        else:
            rep = reproduction_check(system, ev["metrics"], ctx)
            status = "ok" if rep["reproduced"] and agg_diff <= REPRO_TOL else "failed"
            reason = "" if status == "ok" else (
                f"recomputed metrics differ from stored (max |diff|={rep['max_abs_diff']}, evaluator agreement diff={agg_diff})")
        results.append(_result(f"extract.{system}.reproduction", "max_abs_diff_vs_stored", rep["max_abs_diff"], n, None,
                               {"system": system, "tolerance": REPRO_TOL, "evaluator_agreement_max_diff": agg_diff,
                                "stored_files": rep["files"], "missing_files": rep.get("missing", [])},
                               status=status, reason=reason))
        # headline metrics with doc-bootstrap CIs
        for k, name in enumerate(HEADLINE):
            ci = boot_ci(counts, METRICS[name], n_boot, ctx.seed + k)
            results.append(_result(f"extract.{system}.{name}", name, ev["metrics"][name], n, ci,
                                   {"system": system, "definition": definition_for(name),
                                    "n_gold_spans": int(counts[:, C_TP].sum() + counts[:, C_FN].sum()),
                                    "n_pred_spans": int(counts[:, C_TP].sum() + counts[:, C_FP].sum()), "bootstrap": "docs"}))
        # per-label
        for label, cols in (("HARD_SKILL", (C_HTP, C_HFP, C_HFN)), ("SOFT_SKILL", (C_STP, C_SFP, C_SFN))):
            s = counts.sum(axis=0)
            p, r, f = (float(x) for x in _prf_cols(s, *cols))
            stat = lambda a, cols=cols: _prf_cols(a, *cols)[2]
            results.append(_result(f"extract.{system}.label.{label}.f1", "exact_span_f1", f, n,
                                   boot_ci(counts, stat, n_boot, ctx.seed + 100),
                                   {"system": system, "definition": METRIC_DEFINITIONS["per_label"], "label": label, "precision": p, "recall": r,
                                    "support_gold_spans": int(s[cols[0]] + s[cols[2]])}))

    # nervaluate modes + paired delta + detail analyses
    for system in [s for s in DETAIL_SYSTEMS if s in evals]:
        ev = evals[system]
        n = len(ev["ids"])
        nc = nervaluate_counts(ev["gold_docs"], ev["pred_docs"])
        for mode, arr in nc.items():
            s = arr.sum(axis=0)
            p, r, f = (float(x) for x in ner_prf(s))
            ci = boot_ci(arr, lambda a: ner_prf(a)[2], n_boot, ctx.seed + 7)
            results.append(_result(f"extract.{system}.nervaluate.{mode}.f1", f"nervaluate_{mode}_f1", f, n, ci,
                                   {"system": system, "precision": p, "recall": r, "mode": mode,
                                    "definition": "nervaluate semantics (end exclusive here), one-to-one, predictions counted as a list; "
                                                  "differs from repository partial_span_f1 (0.5 credit for partial overlap)."},
                                   ["nervaluate semantics: strict=exact boundary+type; exact=boundary only; "
                                    "partial=boundary exact counts 1, any overlap counts 0.5; ent_type=overlap+same type. "
                                    "Predictions are counted as a list (duplicates cost precision), unlike the repository scorer."]))
        # error taxonomy
        tax = error_taxonomy(ev["ids"], ev["gold_docs"], ev["pred_docs"], ev["texts"])
        tot_err = sum(tax["counts"].values())
        fp_expected = int(ev["counts"][:, C_FP].sum())
        fn_expected = int(ev["counts"][:, C_FN].sum())
        c = tax["counts"]
        reconciles = (c["spurious"] + c["boundary_shift"] + c["label_confusion"] == fp_expected
                      and c["missed"] + c["boundary_shift"] + c["label_confusion"] == fn_expected)
        for kind in ERROR_KINDS:
            results.append(_result(f"extract.{system}.errors.{kind}", f"error_count_{kind}", float(c[kind]), n, None,
                                   {"system": system, "share_of_error_events": (c[kind] / tot_err if tot_err else 0.0),
                                    "n_docs_affected": tax["n_docs_affected"][kind], "examples": tax["examples"].get(kind, []),
                                    "reconciles_with_evaluator_fp_fn": reconciles,
                                    "label_confusion_directions": tax["label_confusion_directions"] if kind == "label_confusion" else None,
                                    "label_confusion_subtypes": tax["label_confusion_subtypes"] if kind == "label_confusion" else None,
                                    "spurious_unlocated": tax["spurious_unlocated"] if kind == "spurious" else None},
                                   ["Counts are error events, not spans: a boundary shift is one FP and one FN."],
                                   status="ok" if reconciles else "failed",
                                   reason="" if reconciles else "error taxonomy does not reconcile with evaluator FP/FN"))
        # slices
        docs_meta = [{"source": gold[i]["source"], "n_tokens": len(gold[i]["tokens"])} for i in ev["ids"]]
        for family, buckets in doc_slices(ev["ids"], ev["gold_docs"], docs_meta).items():
            for bname, idxs in sorted(buckets.items()):
                sub = ev["counts"][idxs]
                ci = boot_ci(sub, METRICS["exact_span_f1"], n_boot, ctx.seed + 11)
                m = aggregate(sub)
                results.append(_result(f"extract.{system}.slice.{family}.{bname}", "exact_span_f1", m["exact_span_f1"],
                                       len(idxs), ci,
                                       {"system": system, "definition": METRIC_DEFINITIONS["exact_span"] + " Other metrics in this dict follow the repository definitions.",
                                        "slice_family": family, "bucket": bname, "partial_span_f1": m["partial_span_f1"],
                                        "token_f1": m["token_f1"], "precision": m["exact_span_precision"],
                                        "recall": m["exact_span_recall"]},
                                       ["Slice CIs are wide for small buckets; slice = descriptive, not a causal explanation."]))
        for side in ("gold", "pred"):
            pairs = span_length_pairs(ev["gold_docs"], ev["pred_docs"], ev["texts"], side)
            metric = "exact_recall" if side == "gold" else "exact_precision"
            for bname, arr in pairs.items():
                if arr[:, 2].sum() == 0:
                    continue
                stat = lambda a: _div(a[..., 0], a[..., 2])
                results.append(_result(f"extract.{system}.slice.{side}_span_length.{bname}", metric,
                                       float(stat(arr.sum(axis=0))), int((arr[:, 2] > 0).sum()),
                                       boot_ci(arr, stat, n_boot, ctx.seed + 13),
                                       {"system": system, "slice_family": f"{side}_span_length", "bucket": bname,
                                        "n_spans": int(arr[:, 2].sum()),
                                        "same_label_overlap_rate": float(arr[:, 1].sum() / arr[:, 2].sum())},
                                       ["n counts documents containing such a span (cluster bootstrap); n_spans in details."]))
        # negative / positive / empty controls
        ctrl = negative_control(ev["gold_docs"], ev["pred_docs"], n_shuffles=3 if ctx.smoke else 20, seed=ctx.seed)
        ok = ctrl["mean"]["exact_span_f1"] < 0.05 and ctrl["positive_control_gold_as_pred"]["exact_span_f1"] == 1.0 \
            and ctrl["empty_predictions"]["exact_span_f1"] == 0.0
        results.append(_result(f"extract.{system}.negative_control", "shuffled_exact_span_f1", ctrl["mean"]["exact_span_f1"], n,
                               None, {"system": system, **ctrl},
                               ["Predictions cyclically shifted across documents; residual F1 comes from short generic documents "
                                "sharing offsets+label by chance. Pass if mean exact F1 < 0.05. Overlap-based partial/token F1 do NOT collapse to ~0 "
                                "under shuffling (see details.mean/max): they have a chance floor and should be read relative to it."],
                               status="ok" if ok else "failed", reason="" if ok else "control did not collapse / positive control != 1"))
    if "safe_wrapper" in evals and "mistral_lora" in evals:
        a, b = evals["safe_wrapper"], evals["mistral_lora"]
        if a["ids"] == b["ids"]:
            for k, name in enumerate(("exact_span_f1", "partial_span_f1", "token_f1")):
                d = paired_boot(a["counts"], b["counts"], METRICS[name], n_boot, ctx.seed + 50 + k)
                results.append(_result(f"extract.delta.safe_wrapper_vs_mistral_lora.{name}", f"delta_{name}", d["delta"],
                                       len(a["ids"]), (d["ci_low"], d["ci_high"]),
                                       {"p": d["p"], "comparison": "safe_wrapper - mistral_lora", "bootstrap": "paired, docs", "definition": definition_for(name)},
                                       ["A sub-point gain whose CI spans zero is not evidence of improvement."]))
    return results
