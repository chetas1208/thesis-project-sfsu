"""L01_extraction - thesis-grade extraction benchmark recomputed from stored predictions (owner: A1).

Run (CPU, repo root)::

    PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.extraction_full [--smoke] [--seed 42] [--n-boot 2000]

What it does (decisions D-A1-01..D-A1-05, D-A1-08 in ``Decisions.md``):

1. Reproduces every stored metrics file from the stored predictions (tolerance 1e-6) and reconciles the F1 values claimed
   in README / MODEL_CARD / reports / docs / thesis .tex against the recomputed values (``claims_reconciliation.csv``).
2. Scores every system on the SAME 974 SkillSpan test sentences that carry gold spans: exact span, repo partial span,
   repo char-level token and word-level token P/R/F1, char-IoU and word-IoU (Hungarian-matched mean IoU, IoU>=0.5 F1),
   HARD/SOFT/macro/micro F1, SemEval-2013 / nervaluate Strict/Exact/Partial/Type with COR/INC/PAR/MIS/SPU (oracle-checked
   against the ``nervaluate`` package run in a throwaway venv), and a component-based boundary-error taxonomy.
3. Sentence-level cluster-bootstrap CIs (10,000 resamples for the anchor and the safe wrapper, 2,000 otherwise) plus a
   posting-level (65 job postings) sensitivity CI for headline metrics; ``per_unit_scores.jsonl`` for L10.
4. Skill-free-sentence metrics (spurious predictions on the 2,595 test sentences without gold spans; full 3,569-sentence
   P/R/F1) for every system that has predictions there; the anchor's come from A3's ``anchor_test_nospan`` GPU job and are
   ``pending_gpu`` until its ``DONE.json`` exists (re-running this module is the only step needed).
5. H1 (pre-registered): anchor vs every baseline on exact-span F1, paired bootstrap + approximate randomization + Holm,
   with a baseline-fairness section quoting the published SkillSpan results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.eval.extraction_eval import _key, evaluate_extraction
from skillllm_v2.evals import extraction_evals as EE
from skillllm_v2.evals.contract import EvalContext
from skillllm_v2.evals.research import common as C
from skillllm_v2.evals.research import stats as ST

LAYER = "L01_extraction"
DATASET = "SkillSpan (HF jjzha/skillspan; HOUSE+TECH sources) test split"
SPLIT_SUB = "test[974 sentences with >=1 gold span]"
SPLIT_ALL = "test[all 3,569 sentences]"
SPLIT_FREE = "test[2,595 sentences without gold spans]"
GOLD_REL = "data/source/skillspan_hf/test.json"
EVALS_GPU = "artifacts/skillllm_v2/evals_gpu"
NOSPAN_JOB = "anchor_test_nospan"
ANCHOR = "mistral_lora"
HEADLINE_SYSTEMS = ("mistral_lora", "safe_wrapper")
LABELS = ("HARD_SKILL", "SOFT_SKILL")
DEFAULT_NERVALUATE_PY = "/usr/data/models/skillllm-v2-stage2/a1_venvs/nervaluate_oracle/bin/python"
REPRO_TOL = 1e-6
H1_MIN_DELTA = 0.10
H1_ALPHA = 0.05
H1_RULE = ("H1 SUPPORTED iff anchor exact-span F1 minus baseline >= 0.10 AND Holm-adjusted p < 0.05 against EVERY "
           "baseline in family H1_anchor_vs_baselines (paired, 974 test sentences; p = approximate randomization, "
           "10,000 shuffles; CI = paired bootstrap, 10,000 resamples).")

BASE_CAVEATS = [
    "Single SkillSpan test split and one training run per system: CIs reflect sentence sampling only, not seed/training variance.",
    "Primary population = the 974 test sentences that contain gold spans; false positives on the 2,595 skill-free test "
    "sentences are only counted in the explicitly labelled full-test records.",
    "Resampling unit = sentence (repo 'document'); sentences come from 65 job postings, so posting-level CIs (details.ci_posting) are wider.",
]

# --------------------------------------------------------------------------------------------- system registry
# stored: (metrics file, key inside the file or None, population the stored file was computed on)
SYSTEMS: List[Dict[str, Any]] = [
    {"id": "mistral_lora", "role": "anchor", "name": "Mistral-24B LoRA (anchor)",
     "pred": "artifacts/skillllm_v2/mistral_lora/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/mistral_lora/metrics.json", None, "subset"),
                ("reports/skillllm_v2/raw_mistral_lora_metrics.json", None, "subset"),
                ("artifacts/skillllm_v2/safe_wrapper/metrics.json", "anchor_metrics", "subset")]},
    {"id": "bert_baseline", "role": "baseline", "name": "token-lookup baseline (repo name 'BERT baseline'; not a neural model)",
     "pred": "artifacts/skillllm_v2/bert_baseline/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/bert_baseline/metrics.json", None, "all"),
                ("reports/skillllm_v2/bert_baseline_metrics.json", None, "all")]},
    {"id": "gliner_baseline", "role": "baseline", "name": "GLiNER medium-v2.1 zero-shot (thr 0.35)",
     "pred": "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/gliner_baseline/metrics.json", None, "all"),
                ("reports/skillllm_v2/gliner_baseline_metrics.json", None, "all")]},
    {"id": "mistral_boundary", "role": "legacy_wrapper", "name": "legacy Boundary Refiner (ungated)",
     "pred": "artifacts/skillllm_v2/mistral_boundary/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/mistral_boundary/metrics.json", None, "subset"),
                ("reports/skillllm_v2/mistral_boundary_metrics.json", None, "subset")]},
    {"id": "mistral_gliner_taxonomy", "role": "legacy_wrapper", "name": "legacy Mistral+GLiNER+Taxonomy (ungated)",
     "pred": "artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json", None, "subset"),
                ("reports/skillllm_v2/mistral_gliner_taxonomy_metrics.json", None, "subset")]},
    {"id": "full_cag_graph", "role": "legacy_wrapper", "name": "legacy Full CAG/Graph (ungated)",
     "pred": "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/full_cag_graph/metrics.json", None, "subset"),
                ("reports/skillllm_v2/full_cag_graph_metrics.json", None, "subset")]},
    {"id": "safe_wrapper", "role": "safe_wrapper", "name": "safe wrapper, selected output (= full_cag_graph variant)",
     "pred": "artifacts/skillllm_v2/safe_wrapper/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json", "selected_metrics", "subset")]},
    {"id": "safe_wrapper_boundary_refiner", "role": "safe_wrapper", "name": "safe wrapper: Boundary Refiner variant",
     "pred": "artifacts/skillllm_v2/safe_wrapper/boundary_refiner/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/safe_wrapper/boundary_refiner/metrics.json", "candidate_metrics", "subset"),
                ("artifacts/skillllm_v2/safe_wrapper/boundary_refiner/metrics.json", "selected_metrics", "subset")]},
    {"id": "safe_wrapper_mistral_gliner_taxonomy", "role": "safe_wrapper", "name": "safe wrapper: Mistral+GLiNER+Taxonomy variant",
     "pred": "artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/metrics.json", "candidate_metrics", "subset"),
                ("artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/metrics.json", "selected_metrics", "subset")]},
    {"id": "safe_wrapper_full_cag_graph", "role": "safe_wrapper", "name": "safe wrapper: Full CAG/Graph variant",
     "pred": "artifacts/skillllm_v2/safe_wrapper/full_cag_graph/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json", "candidate_metrics", "subset"),
                ("artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json", "selected_metrics", "subset")]},
    {"id": "qlora_loop_v1", "role": "qlora_loop", "name": "QLoRA loop v1",
     "pred": "artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/qlora_loop_eval/metrics.json", None, "subset"),
                ("reports/skillllm_v2/qlora_loop_test_metrics.json", None, "subset")]},
    {"id": "qlora_loop_v2", "role": "qlora_loop", "name": "QLoRA loop v2",
     "pred": "artifacts/skillllm_v2/qlora_loop_v2_eval/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/qlora_loop_v2_eval/metrics.json", None, "subset"),
                ("reports/skillllm_v2/qlora_loop_v2_test_metrics.json", None, "subset")]},
    {"id": "qlora_loop_v3", "role": "qlora_loop", "name": "QLoRA loop v3 (short prompt)",
     "pred": "artifacts/skillllm_v2/qlora_loop_v3_eval/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/qlora_loop_v3_eval/metrics.json", None, "subset"),
                ("reports/skillllm_v2/qlora_loop_v3_test_metrics.json", None, "subset")]},
    {"id": "qlora_loop_v4", "role": "qlora_loop", "name": "QLoRA loop v4 (ensemble distillation)",
     "pred": "artifacts/skillllm_v2/qlora_loop_v4_eval/predictions.jsonl",
     "stored": [("artifacts/skillllm_v2/qlora_loop_v4_eval/metrics.json", None, "subset"),
                ("reports/skillllm_v2/qlora_loop_v4_test_metrics.json", None, "subset")]},
]

# Published SkillSpan results, read from the ACL Anthology PDF (Table 5, TEST rows, span-F1 mean +- sd over 5 seeds).
PUBLISHED_SKILLSPAN: Dict[str, Any] = {
    "citation": "Zhang, Jensen, Sonniks, Plank (2022). SkillSpan: Hard and Soft Skill Extraction from English Job "
                "Postings. NAACL-HLT 2022, pp. 4962-4984. doi:10.18653/v1/2022.naacl-main.366",
    "url": "https://aclanthology.org/2022.naacl-main.366.pdf",
    "pdf_sha256": "d8eddf4c63b96c76fa2d9f9740a7030085e46edccc5fb68a38c052cee77ee34f",
    "table": "Table 5 (Appendix E), TEST rows; columns SKILL / KNOWLEDGE / COMBINED x STL / MTL; span-F1 x100",
    "test_rows": {
        "BERTbase": {"SKILL_STL": "54.34±0.74", "SKILL_MTL": "54.20±0.68", "KNOWLEDGE_STL": "62.43±0.41",
                     "KNOWLEDGE_MTL": "61.66±0.83", "COMBINED_STL": "58.16±0.47", "COMBINED_MTL": "57.73±0.66"},
        "JobBERT": {"SKILL_STL": "56.11±0.49", "SKILL_MTL": "55.46±0.75", "KNOWLEDGE_STL": "63.88±0.28",
                    "KNOWLEDGE_MTL": "63.35±0.30", "COMBINED_STL": "59.73±0.38", "COMBINED_MTL": "59.18±0.37"},
        "JobSpanBERT": {"SKILL_STL": "56.64±0.83", "SKILL_MTL": "56.27±0.55", "KNOWLEDGE_STL": "61.06±0.99",
                        "KNOWLEDGE_MTL": "61.87±0.55", "COMBINED_STL": "58.72±0.69", "COMBINED_MTL": "58.90±0.48"},
    },
    "comparability": [
        "The paper's TEST split has 4,680 sentences from BIG+HOUSE+TECH (Table 2); the HF release used here has only "
        "HOUSE+TECH (3,569 sentences; the paper lists 1,216+2,352=3,568), because BIG was not released.",
        "The paper scores ALL test sentences, including those without skills; the repo's anchor 0.670 is on the "
        "974-sentence subset with gold spans, which cannot contain false positives on skill-free sentences.",
        "Label mapping: SkillSpan SKILL = repo SOFT_SKILL, KNOWLEDGE = repo HARD_SKILL; COMBINED (STL) = the two "
        "single-task models' spans merged, closest to the repo's labelled micro exact-span F1.",
    ],
}

# --------------------------------------------------------------------------------------------- matrices
IOU_COLS = ("char_iou_sum", "char_iou_pairs", "char_iou50_tp", "word_iou_sum", "word_iou_pairs", "word_iou50_tp",
            "n_pred", "n_gold")
SEM_MODES = ("strict", "exact", "partial", "ent_type")
SEM_COLS = ("COR", "INC", "PAR", "MIS", "SPU")
O2O_COLS = ("COR", "PAR", "ACT", "POS")
OFF_BASE, OFF_IOU = 0, EE.N_COLS
OFF_SEM = OFF_IOU + len(IOU_COLS)
OFF_O2O = OFF_SEM + len(SEM_MODES) * len(SEM_COLS)
N_ALL = OFF_O2O + len(SEM_MODES) * len(O2O_COLS)


def _div(a: Any, b: Any) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.zeros(np.broadcast(a, b).shape), where=b > 0)


def _f1(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    return _div(2 * p * r, p + r)


def _prf(s: np.ndarray, tp: int, fp: int, fn: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, r = _div(s[..., tp], s[..., tp] + s[..., fp]), _div(s[..., tp], s[..., tp] + s[..., fn])
    return p, r, _f1(p, r)


def sem_prf(s: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nervaluate 1.2.1 formulas from summed (COR, INC, PAR, MIS, SPU): partial/ent_type give PAR half credit."""
    cor, inc, par, mis, spu = (s[..., i] for i in range(5))
    act, pos = cor + inc + par + spu, cor + inc + par + mis
    num = cor + 0.5 * par if mode in ("partial", "ent_type") else cor
    p, r = _div(num, act), _div(num, pos)
    return p, r, _f1(p, r)


def _sem_off(mode: str) -> int:
    return OFF_SEM + SEM_MODES.index(mode) * len(SEM_COLS)


def _o2o_off(mode: str) -> int:
    return OFF_O2O + SEM_MODES.index(mode) * len(O2O_COLS)


def _o2o_prf(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num = s[..., 0] + 0.5 * s[..., 1]
    p, r = _div(num, s[..., 2]), _div(num, s[..., 3])
    return p, r, _f1(p, r)


def _metric_table() -> Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], str]]:
    """metric name -> (stat over the summed N_ALL-column row, definition)."""
    E = EE
    t: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], str]] = {}
    for name in ("exact_span_precision", "exact_span_recall", "exact_span_f1", "partial_span_precision",
                 "partial_span_recall", "partial_span_f1", "token_precision", "token_recall", "token_f1",
                 "hard_skill_f1", "soft_skill_f1", "macro_f1", "micro_f1", "word_token_f1"):
        t[name] = (E.METRICS[name], E.definition_for(name) or "")
    t["word_token_precision"] = (lambda s: _prf(s, E.C_WTP, E.C_WFP, E.C_WFN)[0], E.METRIC_DEFINITIONS["word_token"])
    t["word_token_recall"] = (lambda s: _prf(s, E.C_WTP, E.C_WFP, E.C_WFN)[1], E.METRIC_DEFINITIONS["word_token"])
    for lab, cols in (("hard_skill", (E.C_HTP, E.C_HFP, E.C_HFN)), ("soft_skill", (E.C_STP, E.C_SFP, E.C_SFN))):
        t[f"{lab}_precision"] = (lambda s, c=cols: _prf(s, *c)[0], E.METRIC_DEFINITIONS["per_label"])
        t[f"{lab}_recall"] = (lambda s, c=cols: _prf(s, *c)[1], E.METRIC_DEFINITIONS["per_label"])
    o = OFF_IOU
    for kind, a in (("char", 0), ("word", 3)):
        t[f"{kind}_iou_mean_matched"] = (
            lambda s, a=a: _div(s[..., o + a], s[..., o + a + 1]),
            f"mean {kind}-level IoU over same-label (pred, gold) pairs matched by the Hungarian algorithm maximising "
            f"total IoU (pairs with IoU>0); exact matches contribute IoU 1.")
        defn = (f"one-to-one matching maximising the number of same-label pairs with {kind}-IoU >= 0.5; P over "
                f"de-duplicated predictions (unlocated included), R over gold spans.")
        t[f"{kind}_iou50_precision"] = (lambda s, a=a: _div(s[..., o + a + 2], s[..., o + 6]), defn)
        t[f"{kind}_iou50_recall"] = (lambda s, a=a: _div(s[..., o + a + 2], s[..., o + 7]), defn)
        t[f"{kind}_iou50_f1"] = (lambda s, a=a: _f1(_div(s[..., o + a + 2], s[..., o + 6]),
                                                    _div(s[..., o + a + 2], s[..., o + 7])), defn)
    for mode in SEM_MODES:
        so = _sem_off(mode)
        defn = (f"SemEval-2013 / nervaluate 1.2.1 '{mode}' scheme (greedy; predictions de-duplicated and sorted by "
                "(start, end, label); inclusive end offsets; min overlap 1% of the gold span; unlocated predictions = SPU)")
        for k, nm in enumerate(("precision", "recall", "f1")):
            t[f"semeval_{mode}_{nm}"] = (lambda s, so=so, mode=mode, k=k: sem_prf(s[..., so:so + 5], mode)[k], defn)
        oo = _o2o_off(mode)
        t[f"semeval_{mode}_one_to_one_f1"] = (
            lambda s, oo=oo: _o2o_prf(s[..., oo:oo + 4])[2],
            f"'{mode}' scheme with maximum bipartite (one-to-one) matching instead of greedy order "
            "(extraction_evals.nervaluate_doc); partial credit 0.5.")
    return t


METRIC_TABLE = _metric_table()
HEADLINE_METRICS = ("exact_span_f1", "token_f1")
DIRECTION_LOWER = set()


# --------------------------------------------------------------------------------------------- per-document scorers
def dedup(pred: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for p in pred:
        k = _key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def canonical_order(pred: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """De-duplicated predictions sorted by (start, end, label); unlocated last. Makes greedy nervaluate scoring
    independent of the order in which a pipeline happened to emit its spans (decision D-A1-05)."""
    return sorted(dedup(pred), key=lambda x: (x.get("start") is None, x.get("start") or 0, x.get("end") or 0,
                                              str(x.get("label"))))


def _located(sp: Dict[str, Any]) -> bool:
    return sp.get("start") is not None and sp.get("end") is not None


def _word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _word_ids(sp: Dict[str, Any], words: Sequence[Tuple[int, int]]) -> frozenset:
    return frozenset(i for i, (ws, we) in enumerate(words) if max(ws, sp["start"]) < min(we, sp["end"]))


def _char_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def _set_iou(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def iou_doc(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], text: str) -> List[float]:
    """Per-sentence IoU sufficient statistics in IOU_COLS order."""
    from scipy.optimize import linear_sum_assignment

    preds = dedup(pred)
    row = [0.0] * len(IOU_COLS)
    row[6], row[7] = float(len(preds)), float(len(gold))
    words = _word_spans(text)
    for label in LABELS:
        G = [g for g in gold if g.get("label") == label and _located(g)]
        P = [p for p in preds if p.get("label") == label and _located(p)]
        if not G or not P:
            continue
        mc = np.array([[_char_iou(p, g) for g in G] for p in P])
        gw, pw = [_word_ids(g, words) for g in G], [_word_ids(p, words) for p in P]
        mw = np.array([[_set_iou(a, b) for b in gw] for a in pw])
        for M, base in ((mc, 0), (mw, 3)):
            r, c = linear_sum_assignment(-M)
            v = M[r, c]
            v = v[v > 0]
            row[base] += float(v.sum())
            row[base + 1] += float(len(v))
            elig = M >= 0.5 - 1e-12
            W = np.where(elig, 1000.0 + M, 0.0)
            r, c = linear_sum_assignment(-W)
            row[base + 2] += float(elig[r, c].sum())
    return row


def _ov_pct(ps: int, pe: int, ts: int, te: int) -> float:
    """nervaluate 1.2.1 EvaluationStrategy._calculate_overlap_percentage (inclusive ends; % of the gold span)."""
    if ps > te or pe < ts:
        return 0.0
    return (min(pe, te) - max(ps, ts) + 1) / (te - ts + 1) * 100.0


def semeval_doc(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], mode: str,
                tags: Sequence[str] = LABELS, min_overlap: float = 1.0) -> Tuple[int, int, int, int, int]:
    """(COR, INC, PAR, MIS, SPU) for one sentence; exact re-implementation of nervaluate 1.2.1's greedy strategies.

    Offsets are converted from end-exclusive to nervaluate's inclusive ends. Only located spans with a label in ``tags``
    are evaluated (the caller adds unlocated predictions as SPU).
    """
    T = [(g["label"], g["start"], g["end"] - 1) for g in gold if _located(g) and g.get("label") in tags]
    P = [(p["label"], p["start"], p["end"] - 1) for p in pred if _located(p) and p.get("label") in tags]
    cor = inc = par = spu = 0
    matched: set = set()
    for pl, ps, pe in P:
        if mode in ("strict", "exact"):
            found, inc_idx = False, None
            for ti, (tl, ts, te) in enumerate(T):
                if ti in matched:
                    continue
                if (mode == "exact" or pl == tl) and ps == ts and pe == te:
                    cor += 1
                    matched.add(ti)
                    found = True
                    break
                if inc_idx is None and _ov_pct(ps, pe, ts, te) >= min_overlap:
                    inc_idx = ti
            if not found:
                if inc_idx is not None:
                    inc += 1
                    matched.add(inc_idx)
                else:
                    spu += 1
        elif mode == "partial":
            found, par_idx = False, None
            for ti, (tl, ts, te) in enumerate(T):
                if ti in matched:
                    continue
                if _ov_pct(ps, pe, ts, te) >= min_overlap:
                    if ps == ts and pe == te:
                        cor += 1
                        matched.add(ti)
                        found = True
                        break
                    if par_idx is None:
                        par_idx = ti
            if not found:
                if par_idx is not None:
                    par += 1
                    matched.add(par_idx)
                else:
                    spu += 1
        elif mode == "ent_type":
            best, best_d, inc_idx = None, None, None
            for ti, (tl, ts, te) in enumerate(T):
                if ti in matched:
                    continue
                if _ov_pct(ps, pe, ts, te) >= min_overlap:
                    d = abs(ps - ts) + abs(pe - te)
                    if pl == tl:
                        if best_d is None or d < best_d:
                            best, best_d = ti, d
                    elif inc_idx is None:
                        inc_idx = ti
            if best is not None:
                cor += 1
                matched.add(best)
            elif inc_idx is not None:
                inc += 1
                matched.add(inc_idx)
            else:
                spu += 1
        else:
            raise ValueError(mode)
    return cor, inc, par, len(T) - len(matched), spu


# --------------------------------------------------------------------------------------------- boundary taxonomy
TAX_KINDS = ("left_extension", "right_extension", "left_truncation", "right_truncation", "both_side_shift", "split",
             "merge", "many_to_many", "label_confusion_exact_boundary", "overlap_wrong_label", "spurious",
             "spurious_unlocated", "missed")


def _overlaps(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _located(a) and _located(b) and max(a["start"], b["start"]) < min(a["end"], b["end"])


def boundary_events(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]],
                    text: str = "") -> List[Dict[str, Any]]:
    """Component-based error events for one sentence (decision D-A1-04). Each event has kind, subtype, fp, fn."""
    preds = dedup(pred)
    gkeys, pkeys = {_key(g) for g in gold}, {_key(p) for p in preds}
    tp_gold = [g for g in gold if _key(g) in pkeys]
    G = [g for g in gold if _key(g) not in pkeys]
    P = [p for p in preds if _key(p) not in gkeys and _located(p)]
    events: List[Dict[str, Any]] = [{"kind": "spurious_unlocated", "subtype": "no_offsets", "fp": 1, "fn": 0}
                                    for p in preds if not _located(p) and _key(p) not in gkeys]
    parent = list(range(len(P) + len(G)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(P):
        for j, g in enumerate(G):
            if _overlaps(p, g):
                a, b = find(i), find(len(P) + j)
                if a != b:
                    parent[a] = b
    comps: Dict[int, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for i in range(len(P)):
        comps[find(i)][0].append(i)
    for j in range(len(G)):
        comps[find(len(P) + j)][1].append(j)
    for ps_idx, gs_idx in comps.values():
        ps, gs = [P[i] for i in ps_idx], [G[j] for j in gs_idx]
        if not gs:
            for p in ps:
                sub = "overlaps_correct_gold" if any(_overlaps(p, g) for g in tp_gold) else "no_overlap"
                ev = {"kind": "spurious", "subtype": sub, "fp": 1, "fn": 0}
                if text and any(g["label"] == p["label"] and text[g["start"]:g["end"]] == text[p["start"]:p["end"]]
                                and (g["start"], g["end"]) != (p["start"], p["end"]) for g in gold):
                    ev["mislocated_occurrence"] = True
                events.append(ev)
            continue
        if not ps:
            events.extend({"kind": "missed", "subtype": "", "fp": 0, "fn": 1} for _ in gs)
            continue
        fp, fn = len(ps), len(gs)
        if fn == 1 and fp == 1:
            p, g = ps[0], gs[0]
            if p["label"] != g["label"]:
                kind = ("label_confusion_exact_boundary" if (p["start"], p["end"]) == (g["start"], g["end"])
                        else "overlap_wrong_label")
                sub = f"{g['label']}->{p['label']}"
            else:
                ds, de = p["start"] - g["start"], p["end"] - g["end"]
                if de == 0:
                    kind, sub = ("left_extension" if ds < 0 else "left_truncation"), ""
                elif ds == 0:
                    kind, sub = ("right_extension" if de > 0 else "right_truncation"), ""
                else:
                    kind = "both_side_shift"
                    sub = "superset" if (ds < 0 < de) else "subset" if (de < 0 < ds) else "crossing"
            ev = {"kind": kind, "subtype": sub, "fp": 1, "fn": 1}
        elif fn == 1:
            same = all(p["label"] == gs[0]["label"] for p in ps)
            ev = {"kind": "split", "subtype": "same_label" if same else "mixed_label", "fp": fp, "fn": 1}
        elif fp == 1:
            same = all(g["label"] == ps[0]["label"] for g in gs)
            ev = {"kind": "merge", "subtype": "same_label" if same else "mixed_label", "fp": 1, "fn": fn}
        else:
            ev = {"kind": "many_to_many", "subtype": f"{fp}p:{fn}g", "fp": fp, "fn": fn}
        if text:
            ev["mislocated_occurrence"] = any(
                g["label"] == p["label"] and text[g["start"]:g["end"]] == text[p["start"]:p["end"]]
                and (g["start"], g["end"]) != (p["start"], p["end"]) for p in ps for g in gold)
        events.append(ev)
    return events


def taxonomy(ids: Sequence[str], gold_docs: Sequence[Sequence[Dict[str, Any]]],
             pred_docs: Sequence[Sequence[Dict[str, Any]]], texts: Sequence[str], max_examples: int = 5) -> Dict[str, Any]:
    counts, fp, fn, docs = Counter(), Counter(), Counter(), defaultdict(set)
    subtypes: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[str]] = defaultdict(list)
    mislocated = 0
    n_dupes = 0
    for did, g, p, t in zip(ids, gold_docs, pred_docs, texts):
        n_dupes += len(p) - len(dedup(p))
        for ev in boundary_events(g, p, t):
            k = ev["kind"]
            counts[k] += 1
            fp[k] += ev["fp"]
            fn[k] += ev["fn"]
            docs[k].add(did)
            if ev.get("subtype"):
                subtypes[k][ev["subtype"]] += 1
            if ev.get("mislocated_occurrence"):
                mislocated += ev["fp"]
            if did not in examples[k] and len(examples[k]) < max_examples:
                examples[k].append(did)
    return {"events": {k: counts.get(k, 0) for k in TAX_KINDS}, "fp": {k: fp.get(k, 0) for k in TAX_KINDS},
            "fn": {k: fn.get(k, 0) for k in TAX_KINDS}, "n_sentences": {k: len(docs.get(k, ())) for k in TAX_KINDS},
            "subtypes": {k: dict(v) for k, v in subtypes.items()}, "example_ids": dict(examples),
            "fp_total": sum(fp.values()), "fn_total": sum(fn.values()), "mislocated_occurrence_fp": mislocated,
            "duplicate_predictions_removed": n_dupes}


# --------------------------------------------------------------------------------------------- loading
def _sent_index(doc_id: str) -> int:
    return int(doc_id.rsplit("-", 1)[1])


def load_gold_all(root: Path = C.REPO_ROOT) -> Dict[str, Dict[str, Any]]:
    return EE.load_gold(EvalContext(root=root))


def posting_id(doc: Dict[str, Any]) -> str:
    return f"{doc.get('source')}:{doc.get('document_id')}"


def discover_gpu_systems(root: Path = C.REPO_ROOT) -> Tuple[List[Dict[str, Any]], List[str]]:
    """GPU outputs published by A3 under artifacts/skillllm_v2/evals_gpu/.

    * BERT-family baselines (dir name contains "bert"): the dev-selected model (``SELECTED.json`` or ``DONE.json`` +
      ``predictions.jsonl`` in the dir) enters H1 as role ``baseline``; every finished seed run
      (``runs/<run>/DONE.json``) is reported descriptively as ``baseline_seed_run`` (never used for selection or H1).
    * Anchor re-inference on the 974 test sentences (``anchor_test_logprobs`` with DONE.json) as ``anchor_rerun``.
    Returns (specs, names of BERT-family jobs announced but not finished).
    """
    found, pending = [], []
    base = root / EVALS_GPU
    if not base.exists():
        return found, pending
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        name = d.name
        if name in ("smoke",) or name.endswith("_smoke") or name == NOSPAN_JOB:
            continue
        if name == "anchor_test_logprobs":
            if (d / "DONE.json").exists() and (d / "predictions.jsonl").exists():
                found.append({"id": "gpu_anchor_test_rerun", "role": "anchor_rerun",
                              "name": "anchor re-inference by A3 (J1; current offset code)",
                              "pred": str((d / "predictions.jsonl").relative_to(root)), "stored": []})
            continue
        if "bert" not in name.lower():
            continue
        final = ((d / "SELECTED.json").exists() or (d / "DONE.json").exists()) and (d / "predictions.jsonl").exists()
        if final:
            found.append({"id": f"gpu_{name}", "role": "baseline", "name": f"A3 fair BERT-family baseline ({name}, dev-selected)",
                          "pred": str((d / "predictions.jsonl").relative_to(root)), "stored": []})
        else:
            pending.append(name)
        for run in sorted((d / "runs").glob("*/DONE.json")) if (d / "runs").exists() else []:
            if (run.parent / "predictions.jsonl").exists():
                found.append({"id": f"gpu_{name}_run_{run.parent.name}", "role": "baseline_seed_run",
                              "name": f"A3 {name} seed run {run.parent.name} (descriptive only)",
                              "pred": str((run.parent / "predictions.jsonl").relative_to(root)), "stored": []})
    return found, pending


# --------------------------------------------------------------------------------------------- bootstrap
def boot_multi(rows: np.ndarray, stats: Dict[str, Callable[[np.ndarray], np.ndarray]], n_boot: int, seed: int,
               alpha: float = 0.05, return_samples: bool = False) -> Any:
    """Cluster bootstrap with one set of draws shared by all statistics: stat(sum of resampled unit rows).
    Returns {stat: (lo, hi)} (and the bootstrap samples when ``return_samples``)."""
    r = np.asarray(rows, dtype=float)
    n = len(r)
    if n == 0:
        empty = {k: (None, None) for k in stats}
        return (empty, {k: np.array([]) for k in stats}) if return_samples else empty
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in stats}
    chunk = 250
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        idx = rng.integers(0, n, size=(b, n))
        W = np.bincount((idx + n * np.arange(b)[:, None]).ravel(), minlength=b * n).reshape(b, n).astype(float)
        S = W @ r
        for k, fn in stats.items():
            out[k][lo:lo + b] = fn(S)
    cis = {k: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) for k, v in out.items()}
    return (cis, out) if return_samples else cis


def cluster_rows(rows: np.ndarray, clusters: Sequence[str]) -> np.ndarray:
    keys = sorted(set(clusters))
    pos = {k: i for i, k in enumerate(keys)}
    out = np.zeros((len(keys), rows.shape[1]))
    for i, c in enumerate(clusters):
        out[pos[c]] += rows[i]
    return out


# --------------------------------------------------------------------------------------------- system evaluation
def score_documents(gold_docs, pred_docs, texts) -> np.ndarray:
    rows = np.zeros((len(gold_docs), N_ALL))
    for i, (g, p, t) in enumerate(zip(gold_docs, pred_docs, texts)):
        rows[i, OFF_BASE:OFF_BASE + EE.N_COLS] = EE.doc_counts(g, p, t)
        rows[i, OFF_IOU:OFF_IOU + len(IOU_COLS)] = iou_doc(g, p, t)
        pd = canonical_order(p)
        n_unloc = sum(1 for x in pd if not _located(x))
        for mode in SEM_MODES:
            c = list(semeval_doc(g, pd, mode))
            c[4] += n_unloc
            rows[i, _sem_off(mode):_sem_off(mode) + 5] = c
            rows[i, _o2o_off(mode):_o2o_off(mode) + 4] = EE.nervaluate_doc(g, pd, mode)
    return rows


def evaluate_system(spec: Dict[str, Any], gold: Dict[str, Dict[str, Any]], ids: Sequence[str],
                    root: Path = C.REPO_ROOT) -> Dict[str, Any]:
    preds = EE.load_predictions(root / spec["pred"])
    missing = [i for i in ids if i not in preds]
    gold_docs = [gold[i]["gold_spans"] for i in ids]
    pred_docs = [preds.get(i, []) for i in ids]
    texts = [gold[i]["text"] for i in ids]
    rows = score_documents(gold_docs, pred_docs, texts)
    return {"spec": spec, "ids": list(ids), "gold_docs": gold_docs, "pred_docs": pred_docs, "texts": texts,
            "rows": rows, "preds_all": preds, "missing": missing,
            "metrics": {k: float(fn(rows.sum(axis=0))) for k, (fn, _) in METRIC_TABLE.items()}}


# --------------------------------------------------------------------------------------------- reproduction
def reproduce(spec: Dict[str, Any], gold: Dict[str, Dict[str, Any]], preds: Dict[str, List[Dict[str, Any]]],
              subset_ids: Sequence[str], all_ids: Sequence[str], root: Path = C.REPO_ROOT) -> Dict[str, Any]:
    cache: Dict[str, Dict[str, float]] = {}

    def recomputed(pop: str) -> Dict[str, float]:
        if pop not in cache:
            ids = subset_ids if pop == "subset" else all_ids
            cache[pop] = evaluate_extraction([gold[i]["gold_spans"] for i in ids], [preds.get(i, []) for i in ids])
        return cache[pop]

    out = {"files": [], "max_abs_diff": 0.0, "reproduced": True}
    for rel_path, key, pop in spec.get("stored", []):
        p = root / rel_path
        if not p.exists():
            out["files"].append({"file": rel_path, "key": key, "missing": True})
            out["reproduced"] = False
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if key:
            data = data.get(key, {})
        rec = recomputed(pop)
        diffs = {k: abs(float(v) - rec[k]) for k, v in data.items() if isinstance(v, (int, float)) and k in rec}
        worst = max(diffs.values()) if diffs else None
        out["files"].append({"file": rel_path, "key": key, "population": pop, "n_metrics_compared": len(diffs),
                             "max_abs_diff": worst, "worst_metric": max(diffs, key=diffs.get) if diffs else None})
        if worst is None or worst > REPRO_TOL:
            out["reproduced"] = False
        if worst is not None:
            out["max_abs_diff"] = max(out["max_abs_diff"], worst)
    out["recomputed"] = cache
    return out


# --------------------------------------------------------------------------------------------- claims reconciliation
CLAIM_GLOBS = ("README.md", "MODEL_CARD.md", "reports/skillllm_v2/*.md", "docs/**/*.md", "docs/thesis/*.tex")
CLAIM_EXCLUDE = ("docs/thesis/evaluation/",)  # written concurrently by A6 from these very layers
NUM_RE = re.compile(r"(?<![\w.])(0\.\d{3,})(?!\d)")
CI_RE = re.compile(r"\[\s*[-+−]?\d*\.\d+\s*,\s*[-+−]?\d*\.\d+\s*\]")
SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
METRIC_CTX = re.compile(r"f1|f-1|f\s*score|exact|partial|token|precision|recall|\bspan\b", re.I)
NON_EXTRACTION_CTX = re.compile(
    r"ndcg|mrr|p@|precision@|recall@|hit@|map@|\bece\b|brier|aurc|auroc|coverage|\brisk\b|latency|jaccard|linking|"
    r"accuracy|top-?1|wsgc|course|job match|kappa|alpha|spearman|pearson|correlat|threshold|logprob|calibrat|"
    r"similarity|retriev|bm25|dense|overlap \(user|prereq|violation|cosine|\bauc\b|nll|selective|abstain|"
    r"agreement|faithful", re.I)
PVAL_HEADER_CTX = re.compile(r"(^|\W)p(\W|$)|p-value|p_holm|holm|mcnemar|sign-test|permutation|\bperm\b|\bboot\b|"
                             r"wins|losses|ties|\bse\b|std|delta|\u0394", re.I)
PVAL_PREFIX = re.compile(r"(\bp|p_holm|bootstrap|permutation|holm|perm|boot|sign-test|mcnemar|\bq|\bse|sd)\s*"
                         r"(about|of|=|<|>|:|\u2248|~)?\s*$", re.I)
DEV_CTX = re.compile(r"\bdev\b|development|validation|\bval\b", re.I)
NOTREC_CTX = re.compile(r"ensemble|vote|majority|run 1|80-epoch run 1|earlier artifact", re.I)
THRESH_CTX = re.compile(r"target|floor|threshold|margin|guardrail|>=|≥", re.I)

SYS_RULES: List[Tuple[str, str]] = [
    ("safe_wrapper_boundary_refiner", r"(safe.?wrapper|gated|\(safe).{0,40}refiner|refiner.{0,40}(safe.?wrapper|\(safe|gated)"),
    ("safe_wrapper_mistral_gliner_taxonomy", r"(safe.?wrapper|\(safe|gated).{0,40}(gliner|taxonomy)|(gliner|taxonomy).{0,40}(safe.?wrapper|\(safe|gated)"),
    ("safe_wrapper_full_cag_graph", r"(safe.?wrapper|\(safe|gated).{0,40}(cag|graph)|(cag|graph).{0,40}(safe.?wrapper|\(safe|gated)"),
    ("mistral_boundary", r"(old|legacy|pre-gate|ungated).{0,30}refiner"),
    ("mistral_gliner_taxonomy", r"(old|legacy|pre-gate|ungated).{0,30}(gliner|taxonomy)"),
    ("full_cag_graph", r"(old|legacy|pre-gate|ungated).{0,30}(cag|graph)"),
    ("mistral_lora", r"anchor|raw mistral|mistral lora|mistral_lora|raw_mistral|mistral-24b lora|mistral.?only"),
    ("bert_baseline", r"\bbert\b|bert_baseline"),
    ("gliner_baseline", r"gliner baseline|gliner_baseline|zero-shot gliner|^\W*gliner\W*$"),
    ("qlora_loop_v4", r"loop.?v4|\bv4\b|distill"),
    ("qlora_loop_v3", r"loop.?v3|\bv3\b|short.?prompt"),
    ("qlora_loop_v2", r"loop.?v2|\bv2\b"),
    ("qlora_loop_v1", r"loop.?v1|\bv1\b|qlora_loop_eval|qlora loop(?!.?v\d)"),
]
GENERIC_SYS: List[Tuple[Sequence[str], str]] = [
    (("mistral_boundary", "safe_wrapper_boundary_refiner"), r"refiner"),
    (("mistral_gliner_taxonomy", "safe_wrapper_mistral_gliner_taxonomy"), r"gliner.{0,10}taxonomy"),
    (("full_cag_graph", "safe_wrapper_full_cag_graph", "safe_wrapper"), r"cag|full.{0,3}graph"),
    (("safe_wrapper", "safe_wrapper_full_cag_graph", "safe_wrapper_boundary_refiner",
      "safe_wrapper_mistral_gliner_taxonomy"), r"safe.?wrapper|wrapper"),
]
METRIC_RULES: List[Tuple[str, str]] = [
    ("hard_skill", r"hard"), ("soft_skill", r"soft"), ("macro", r"macro"), ("partial_span", r"partial"),
    ("word_token", r"word"), ("token", r"token"), ("exact_span", r"exact|span|micro|\bf1\b"),
]
WRAPPER_OR_ANCHOR = {"mistral_lora", "mistral_boundary", "mistral_gliner_taxonomy", "full_cag_graph", "safe_wrapper",
                     "safe_wrapper_boundary_refiner", "safe_wrapper_mistral_gliner_taxonomy",
                     "safe_wrapper_full_cag_graph"}
KNOWN_CAUSES = {
    "reports/skillllm_v2/extraction_ablation.md": "stub-extractor ablation on 20 DEV documents (already flagged as unusable "
                                                  "in docs/INFERENCE_AND_EVALS.md, items E13/I7); not a test result",
    "reports/skillllm_v2/wrapper_error_analysis.md": "June-2026 report computed from the earlier run-1 anchor (exact F1 "
                                                     "0.675034) and pre-gate wrapper stages; no stored predictions reproduce it",
}
KNOWN_VALUES = ("0.670069", "0.6701", "0.675034", "0.675", "0.669478", "0.670222", "0.672602", "0.672444",
                "0.671854", "0.671102", "0.670062")


def _attribute_systems(ctx: str) -> List[str]:
    c = ctx.lower()
    out = [sid for sid, pat in SYS_RULES if re.search(pat, c)]
    if not out:
        for sids, pat in GENERIC_SYS:
            if re.search(pat, c):
                out.extend(s for s in sids if s not in out)
    return out


def _attribute_metric(ctx: str) -> Optional[str]:
    c = ctx.lower()
    for name, pat in METRIC_RULES:
        if re.search(pat, c):
            return name
    return None


def _metric_ok(metric_key: str, family: Optional[str]) -> bool:
    if family is None:
        return True
    if family == "token":
        return metric_key.startswith("token_")
    if family == "exact_span":
        return metric_key.startswith("exact_span") or metric_key == "micro_f1"
    return metric_key.startswith(family)


def iter_claim_lines(root: Path = C.REPO_ROOT) -> List[Tuple[str, int, str, Optional[List[str]], Optional[List[str]]]]:
    """(file, line no, line text, table header cells or None, row cells or None)."""
    files: List[Path] = []
    for pat in CLAIM_GLOBS:
        files.extend(Path(p) for p in glob.glob(str(root / pat), recursive=True))
    out = []
    for f in sorted(set(files)):
        relp = C.rel(f)
        if any(relp.startswith(x) for x in CLAIM_EXCLUDE) or not f.is_file():
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        header: Optional[List[str]] = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("|"):
                cells = [x.strip() for x in s.strip("|").split("|")]
                if i + 1 < len(lines) and SEP_RE.match(lines[i + 1].strip()):
                    header = cells
                    continue
                if SEP_RE.match(s):
                    continue
                out.append((relp, i + 1, line, header, cells))
            else:
                header = None
                out.append((relp, i + 1, line, None, None))
    return out


def comparator_values(evals_subset: Dict[str, Dict[str, Any]], extra: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
    vals = []
    for sid, ev in evals_subset.items():
        rec = evaluate_extraction(ev["gold_docs"], ev["pred_docs"])
        for k, v in rec.items():
            vals.append((sid, k, float(v)))
        for k in ("hard_skill_precision", "hard_skill_recall", "soft_skill_precision", "soft_skill_recall",
                  "word_token_f1", "word_token_precision", "word_token_recall"):
            vals.append((sid, k, ev["metrics"][k]))
    for sid, rec in extra.items():
        for k, v in rec.items():
            vals.append((sid, k, float(v)))
    return vals


def _rounds_to(claim: str, value: float) -> bool:
    d = len(claim.split(".")[1])
    return abs(float(claim) - value) <= 0.5 * 10 ** (-d) + 1e-12


def reconcile_claims(comparators: List[Tuple[str, str, float]], root: Path = C.REPO_ROOT) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    known: Dict[str, Dict[str, Any]] = {v: {"occurrences": 0, "files": Counter()} for v in KNOWN_VALUES}
    for relp, ln, line, header, cells in iter_claim_lines(root):
        for v in KNOWN_VALUES:
            n = len(re.findall(r"(?<![\w.])" + re.escape(v) + r"(?!\d)", line))
            if n:
                known[v]["occurrences"] += n
                known[v]["files"][relp] += n
        ci_spans = [m.span() for m in CI_RE.finditer(line)]
        items: List[Tuple[str, str, str]] = []  # (number, system context, metric context)
        if cells is not None and header is not None and len(header) == len(cells):
            for col, cell in zip(header, cells):
                m = NUM_RE.search(cell)
                if not m or PVAL_HEADER_CTX.search(col):
                    continue
                pre = cell[: m.start()]
                if pre.strip().endswith(("+", "-", "\u2212", "\u00b1")) or PVAL_PREFIX.search(pre):
                    continue
                items.append((m.group(1), cells[0] + " " + col, col))
        else:
            for m in NUM_RE.finditer(line):
                if any(a <= m.start() < b for a, b in ci_spans):
                    continue
                pre = line[max(0, m.start() - 24): m.start()]
                if pre.rstrip().endswith(("+", "-", "\u2212", "\u00b1", "\u0394")) or PVAL_PREFIX.search(pre) \
                        or re.search(r"p-value|p_holm", pre):
                    continue
                items.append((m.group(1), line, line))
        for num, sys_ctx, met_ctx in items:
            val = float(num)
            if not (0.03 <= val <= 0.95):
                continue
            full_ctx = (" ".join(header) + " " + line) if header else line
            if not METRIC_CTX.search(met_ctx if header else full_ctx) or NON_EXTRACTION_CTX.search(met_ctx):
                continue
            if header is None and NON_EXTRACTION_CTX.search(full_ctx):
                continue
            systems = _attribute_systems(sys_ctx)
            fam = _attribute_metric(met_ctx)
            hits = [(s, k, x) for s, k, x in comparators if _rounds_to(num, x)]
            hit_attr = [(s, k, x) for s, k, x in hits if (not systems or s in systems) and _metric_ok(k, fam)]
            if DEV_CTX.search(sys_ctx if header else full_ctx) and not hit_attr:
                status = "not_recomputable_dev_or_other_split"
            elif NOTREC_CTX.search(sys_ctx if header else full_ctx) and not hit_attr:
                status = "not_recomputable_ensemble_or_old_run"
            elif hit_attr:
                status = "match"
            elif hits:
                status = "match_other_system_or_metric"
            elif THRESH_CTX.search(full_ctx):
                status = "threshold_or_target_value"
            else:
                status = "mismatch"
            near = None
            cand = [(s, k, x) for s, k, x in comparators if (not systems or s in systems) and _metric_ok(k, fam)]
            if cand:
                s, k, x = min(cand, key=lambda t: abs(t[2] - val))
                near = {"system": s, "metric": k, "value": x, "abs_diff": abs(x - val)}
            rows.append({"file": relp, "line": ln, "claimed": num, "attributed_systems": systems, "metric_family": fam,
                         "status": status, "nearest_attributed": near,
                         "matches": [f"{s}:{k}" for s, k, _ in hits[:6]],
                         "anchor_or_wrapper": bool(systems) and set(systems) <= WRAPPER_OR_ANCHOR})
    known_out = {}
    for v, info in known.items():
        hits = sorted({f"{s}:{k}" for s, k, x in comparators if _rounds_to(v, x)})
        known_out[v] = {"occurrences": info["occurrences"], "n_files": len(info["files"]),
                        "files": dict(info["files"].most_common(8)), "recomputed_matches": hits}
    return {"rows": rows, "known_values": known_out}


# --------------------------------------------------------------------------------------------- nervaluate oracle
ORACLE_SNIPPET = r'''
import json, sys
from nervaluate import Evaluator
import importlib.metadata as md
data = json.load(open(sys.argv[1]))
out = {"_version": md.version("nervaluate")}
for name, pair in data.items():
    ev = Evaluator(pair["true"], pair["pred"], tags=pair["tags"], loader="dict")
    res = ev.evaluate()["overall"]
    out[name] = {m: {k: getattr(r, k) for k in ("correct", "incorrect", "partial", "missed", "spurious",
                                                 "precision", "recall", "f1")} for m, r in res.items()}
json.dump(out, open(sys.argv[2], "w"))
'''


def nervaluate_oracle(evals: Dict[str, Dict[str, Any]], work_dir: Path) -> Dict[str, Any]:
    """Run the real nervaluate package (throwaway venv, python -I) on the same located, de-duplicated spans."""
    py = os.environ.get("A1_NERVALUATE_PYTHON", DEFAULT_NERVALUATE_PY)
    if not Path(py).exists():
        return {"available": False, "reason": f"nervaluate interpreter not found: {py}"}
    payload = {}
    for sid, ev in evals.items():
        true = [[{"label": g["label"], "start": g["start"], "end": g["end"] - 1} for g in gd if _located(g)]
                for gd in ev["gold_docs"]]
        pred = [[{"label": p["label"], "start": p["start"], "end": p["end"] - 1} for p in canonical_order(pd) if _located(p)]
                for pd in ev["pred_docs"]]
        payload[sid] = {"true": true, "pred": pred, "tags": list(LABELS)}
    work_dir.mkdir(parents=True, exist_ok=True)
    inp, outp = work_dir / "nervaluate_oracle_input.json", work_dir / "nervaluate_oracle_output.json"
    inp.write_text(json.dumps(payload))
    proc = subprocess.run([py, "-I", "-c", ORACLE_SNIPPET, str(inp), str(outp)], capture_output=True, text=True,
                          timeout=1800, cwd=str(work_dir))
    if proc.returncode != 0:
        return {"available": False, "reason": f"oracle failed: {proc.stderr[-800:]}"}
    res = json.loads(outp.read_text())
    version = res.pop("_version", None)
    comp = {}
    all_ok = True
    for sid, ev in evals.items():
        per = {}
        for mode in SEM_MODES:
            mine = ev["rows"][:, _sem_off(mode):_sem_off(mode) + 5].sum(axis=0).astype(int).tolist()
            n_unloc = sum(1 for pd in ev["pred_docs"] for p in dedup(pd) if not _located(p))
            mine[4] -= n_unloc  # oracle saw located spans only
            o = res[sid][mode]
            theirs = [o["correct"], o["incorrect"], o["partial"], o["missed"], o["spurious"]]
            ok = mine == theirs
            all_ok &= ok
            per[mode] = {"mine": mine, "nervaluate": theirs, "match": ok, "nervaluate_f1_located_only": o["f1"]}
        comp[sid] = per
    try:
        inp.unlink()  # derived offsets only; regenerated on every run
    except OSError:
        pass
    return {"available": True, "nervaluate_version": version, "all_match": all_ok, "per_system": comp,
            "interpreter": py, "output_file": C.rel(outp)}


# --------------------------------------------------------------------------------------------- skill-free sentences
def skillfree_rows(pred_docs: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
    """Per skill-free sentence: [n de-duplicated predicted spans, has >=1 prediction, n unlocated]."""
    out = np.zeros((len(pred_docs), 3))
    for i, p in enumerate(pred_docs):
        d = dedup(p)
        out[i] = [len(d), 1.0 if d else 0.0, sum(1 for x in d if not _located(x))]
    return out


def anchor_full_predictions(root: Path, ids_free: Sequence[str]) -> Tuple[Optional[Dict[str, List]], str, Optional[Path]]:
    d = root / EVALS_GPU / NOSPAN_JOB
    if not (d / "DONE.json").exists():
        return None, f"{EVALS_GPU}/{NOSPAN_JOB}/DONE.json not present (A3 GPU job J3 not finished)", None
    p = d / "predictions.jsonl"
    if not p.exists():
        return None, f"{EVALS_GPU}/{NOSPAN_JOB}/predictions.jsonl missing although DONE.json exists", None
    preds = EE.load_predictions(p)
    missing = [i for i in ids_free if i not in preds]
    if missing:
        return None, f"anchor_test_nospan covers {len(ids_free) - len(missing)}/{len(ids_free)} skill-free sentences", p
    return preds, "", p


# --------------------------------------------------------------------------------------------- figures
DISPLAY = {"bert_baseline": "bert_baseline (token lookup, not BERT)", "gliner_baseline": "gliner_baseline (zero-shot)"}


def display(sid: str) -> str:
    return DISPLAY.get(sid, sid)


def _plot_boundary(tax_by_sys: Dict[str, Dict[str, Any]], out: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = [k for k in TAX_KINDS if any(t["events"][k] for t in tax_by_sys.values())]
    systems = list(tax_by_sys)
    csv_path = out / "fig_boundary_errors.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "kind", "events", "fp", "fn", "n_sentences"])
        for s in systems:
            t = tax_by_sys[s]
            for k in TAX_KINDS:
                w.writerow([s, k, t["events"][k], t["fp"][k], t["fn"][k], t["n_sentences"][k]])
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(systems) + 2.2))
    left = np.zeros(len(systems))
    cmap = plt.get_cmap("tab20")
    for j, k in enumerate(kinds):
        vals = np.array([tax_by_sys[s]["events"][k] for s in systems], dtype=float)
        tot = np.array([max(1, sum(tax_by_sys[s]["events"].values())) for s in systems], dtype=float)
        share = vals / tot
        ax.barh([display(x) for x in systems], share, left=left, color=cmap(j % 20), label=k, edgecolor="white", linewidth=0.4)
        left += share
    ax.set_xlabel("share of error events (974 test sentences)")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.01 * len(systems) / 4), frameon=False)
    ax.set_title("Boundary-error taxonomy per system (component-based, D-A1-04)")
    fig.tight_layout()
    png = out / "fig_boundary_errors.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [C.rel(png), C.rel(csv_path)]


def _plot_f1(rows: List[Dict[str, Any]], out: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = out / "fig_exact_f1_ci.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "exact_span_f1", "ci_low", "ci_high", "token_f1", "char_iou50_f1"])
        for r in rows:
            w.writerow([r["system"], r["exact_span_f1"], r["ci_low"], r["ci_high"], r["token_f1"], r["char_iou50_f1"]])
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1.5))
    y = np.arange(len(rows))
    v = np.array([r["exact_span_f1"] for r in rows])
    lo = np.array([r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] for r in rows])
    ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", color="#1f4e79", ecolor="#7f9fbf", capsize=3)
    ax.scatter([r["char_iou50_f1"] for r in rows], y, marker="x", color="#b35c00", label="char-IoU>=0.5 F1")
    ax.set_yticks(y, [display(r["system"]) for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("exact-span F1 (95% sentence-bootstrap CI), 974 test sentences")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    png = out / "fig_exact_f1_ci.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return [C.rel(png), C.rel(csv_path)]


# --------------------------------------------------------------------------------------------- main
def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def run(seed: int = 42, n_boot: int = 2000, smoke: bool = False, root: Path = C.REPO_ROOT,
        out_root: Path = C.EVALS_ROOT) -> Path:
    gold = load_gold_all(root)
    all_ids = sorted(gold, key=_sent_index)
    sub_ids = [i for i in all_ids if gold[i]["gold_spans"]]
    free_ids = [i for i in all_ids if not gold[i]["gold_spans"]]
    if smoke:
        sub_ids, free_ids = sub_ids[:60], free_ids[:100]
        n_boot = min(n_boot, 200)
    all_eval_ids = sorted(sub_ids + free_ids, key=_sent_index)
    out_dir = out_root / LAYER / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = root / GOLD_REL
    postings_sub = [posting_id(gold[i]) for i in sub_ids]

    gpu_specs, gpu_pending = discover_gpu_systems(root)
    specs = [s for s in SYSTEMS if (root / s["pred"]).exists()] + gpu_specs
    missing_specs = [s for s in SYSTEMS if not (root / s["pred"]).exists()]

    records: List[Dict[str, Any]] = []
    rec = lambda *a, **k: records.append(C.record(LAYER, *a, **k))  # noqa: E731
    evals: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        evals[spec["id"]] = evaluate_system(spec, gold, sub_ids, root)
    for spec in missing_specs:
        rec(spec["id"], "exact_span_f1", None, n_units=0, unit="sentence", split=SPLIT_SUB, dataset=DATASET,
            direction="higher", status="not_available", reason=f"predictions missing: {spec['pred']}")

    # identical-output detection (so duplicates are not counted as independent evidence)
    sig = {sid: tuple(frozenset(_key(x) for x in p) for p in ev["pred_docs"]) for sid, ev in evals.items()}
    identical: Dict[str, List[str]] = {sid: [o for o in evals if o != sid and sig[o] == sig[sid]] for sid in evals}

    # ---------------------------------------------------------------- 1. reproduction
    repro_rows = []
    for sid, ev in evals.items():
        spec = ev["spec"]
        if not spec.get("stored"):
            continue
        if smoke:
            continue
        r = reproduce(spec, gold, ev["preds_all"], sub_ids, all_ids, root)
        repro_rows.append({"system": sid, **{k: v for k, v in r.items() if k != "recomputed"}})
        prov = [spec["pred"], GOLD_REL] + [f for f, _, _ in spec["stored"] if (root / f).exists()]
        rec(sid, "reproduction_max_abs_diff_vs_stored", r["max_abs_diff"], n_units=len(sub_ids), unit="sentence",
            split=SPLIT_SUB, dataset=DATASET, direction="lower", provenance=prov,
            details={"tolerance": REPRO_TOL, "reproduced": r["reproduced"], "files": r["files"],
                     "note": "stored baseline files are computed over all 3,569 sentences and are compared on that population"})
    # ---------------------------------------------------------------- 2+3. metrics with CIs
    f1_rows = []
    tax_by_sys: Dict[str, Dict[str, Any]] = {}
    for sid, ev in evals.items():
        rows = ev["rows"]
        nb = 10000 if (sid in HEADLINE_SYSTEMS and not smoke) else n_boot
        stats = {k: fn for k, (fn, _) in METRIC_TABLE.items()}
        cis = boot_multi(rows, stats, nb, seed)
        post = boot_multi(cluster_rows(rows, postings_sub), {k: stats[k] for k in
                          ("exact_span_f1", "token_f1", "exact_span_precision", "exact_span_recall", "char_iou50_f1")},
                          nb, seed)
        s = rows.sum(axis=0)
        n_gold = int(s[EE.C_TP] + s[EE.C_FN])
        n_pred = int(s[EE.C_TP] + s[EE.C_FP])
        prov = [ev["spec"]["pred"], GOLD_REL]
        cav = list(BASE_CAVEATS)
        if ev["spec"]["id"] == "bert_baseline":
            cav.append("'BERT baseline' is a train-set token->tag lookup table (baselines/bert_baseline.py), not BERT.")
        if identical[sid]:
            cav.append(f"Predictions identical (as span sets) to: {', '.join(identical[sid])}.")
        for metric, (fn, definition) in METRIC_TABLE.items():
            det = {"definition": definition, "role": ev["spec"]["role"], "system_name": ev["spec"]["name"],
                   "n_boot": nb, "n_gold_spans": n_gold, "n_pred_spans_dedup": n_pred,
                   "identical_to": identical[sid], "missing_prediction_rows": len(ev["missing"])}
            if metric in post:
                det["ci_posting"] = list(post[metric])
                det["n_postings"] = len(set(postings_sub))
            if metric in HEADLINE_METRICS and sid == ANCHOR:
                det["headline"] = True
            if metric.startswith("semeval_") and not metric.endswith("one_to_one_f1"):
                mode = metric.split("_")[1] if not metric.startswith("semeval_ent_type") else "ent_type"
                so = _sem_off(mode)
                det["counts"] = dict(zip(SEM_COLS, s[so:so + 5].astype(int).tolist()))
            rec(sid, metric, ev["metrics"][metric], n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB,
                dataset=DATASET, direction="higher", ci=cis[metric],
                ci_method=f"sentence-level cluster percentile bootstrap, {nb} resamples, seed {seed}",
                provenance=prov, caveats=cav, details=det)
        tax = taxonomy(ev["ids"], ev["gold_docs"], ev["pred_docs"], ev["texts"])
        tax_by_sys[sid] = tax
        fp_tot, fn_tot = int(s[EE.C_FP]), int(s[EE.C_FN])
        reconciles = tax["fp_total"] == fp_tot and tax["fn_total"] == fn_tot
        rec(sid, "boundary_error_events_total", float(sum(tax["events"].values())) if reconciles else None,
            n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB, dataset=DATASET, direction="lower",
            provenance=prov, status="measured" if reconciles else "failed",
            reason="" if reconciles else "taxonomy FP/FN do not reconcile with the evaluator",
            caveats=["Events, not spans: a 1:1 boundary error is one event costing one FP and one FN."],
            details={"taxonomy": tax, "evaluator_fp": fp_tot, "evaluator_fn": fn_tot, "reconciles": reconciles,
                     "definition": "component-based taxonomy, decision D-A1-04"})
        for k in TAX_KINDS:
            rec(sid, f"boundary_error_{k}", float(tax["events"][k]), n_units=len(sub_ids), unit="sentence",
                split=SPLIT_SUB, dataset=DATASET, direction="lower", provenance=prov,
                details={"fp": tax["fp"][k], "fn": tax["fn"][k], "n_sentences": tax["n_sentences"][k],
                         "subtypes": tax["subtypes"].get(k, {}), "example_sentence_ids": tax["example_ids"].get(k, []),
                         "share_of_events": tax["events"][k] / max(1, sum(tax["events"].values()))})
        f1_rows.append({"system": sid, "exact_span_f1": ev["metrics"]["exact_span_f1"], "ci_low": cis["exact_span_f1"][0],
                        "ci_high": cis["exact_span_f1"][1], "token_f1": ev["metrics"]["token_f1"],
                        "char_iou50_f1": ev["metrics"]["char_iou50_f1"], "role": ev["spec"]["role"],
                        "precision": ev["metrics"]["exact_span_precision"], "recall": ev["metrics"]["exact_span_recall"],
                        "word_token_f1": ev["metrics"]["word_token_f1"], "partial_span_f1": ev["metrics"]["partial_span_f1"],
                        "char_iou_mean": ev["metrics"]["char_iou_mean_matched"], "macro_f1": ev["metrics"]["macro_f1"],
                        "hard": ev["metrics"]["hard_skill_f1"], "soft": ev["metrics"]["soft_skill_f1"],
                        "strict": ev["metrics"]["semeval_strict_f1"], "sem_partial": ev["metrics"]["semeval_partial_f1"],
                        "sem_type": ev["metrics"]["semeval_ent_type_f1"], "sem_exact": ev["metrics"]["semeval_exact_f1"],
                        "ci_posting": post["exact_span_f1"], "reconciles": reconciles})

    # descriptive paired deltas vs anchor (not a hypothesis; L05/L10 own wrapper inference)
    delta_rows = []
    if ANCHOR in evals:
        a = evals[ANCHOR]["rows"][:, :3]
        for sid, ev in evals.items():
            if sid == ANCHOR:
                continue
            cmp = ST.paired_bootstrap(ev["rows"][:, :3], a, ST.micro_f1, n_boot, seed)
            delta_rows.append({"system": sid, "delta": cmp["delta"], "ci_low": cmp["ci_low"], "ci_high": cmp["ci_high"],
                               "p": cmp["p"], "n_units_differ": cmp["n_units_differ"]})
            rec(sid, "delta_exact_span_f1_vs_anchor", cmp["delta"], n_units=len(sub_ids), unit="sentence",
                split=SPLIT_SUB, dataset=DATASET, direction="higher", ci=(cmp["ci_low"], cmp["ci_high"]),
                ci_method=f"paired sentence bootstrap, {n_boot} resamples, seed {seed}", provenance=[ev["spec"]["pred"], evals[ANCHOR]["spec"]["pred"], GOLD_REL],
                details={"p_boot": cmp["p"], "n_units_differ": cmp["n_units_differ"], "family": "L01_descriptive_vs_anchor",
                         "note": "descriptive only; hypothesis tests live in H1 (here), H2 (L05) and L10"})

    # ---------------------------------------------------------------- SemEval oracle
    oracle = nervaluate_oracle(evals, out_dir) if evals else {"available": False, "reason": "no systems"}
    if oracle.get("available"):
        rec("all", "nervaluate_oracle_agreement", 1.0 if oracle["all_match"] else 0.0, n_units=len(evals),
            unit="system", split=SPLIT_SUB, dataset=DATASET, direction="higher",
            details={k: v for k, v in oracle.items() if k != "available"})
    else:
        rec("all", "nervaluate_oracle_agreement", None, n_units=0, unit="system", split=SPLIT_SUB, dataset=DATASET,
            direction="higher", status="not_available", reason=oracle.get("reason", "unknown"))

    # ---------------------------------------------------------------- 4. skill-free sentences + full test
    free_summary = []
    full_rows_by_sys: Dict[str, np.ndarray] = {}
    free_rows_by_sys: Dict[str, np.ndarray] = {}
    postings_free = [posting_id(gold[i]) for i in free_ids]
    postings_all = [posting_id(gold[i]) for i in all_eval_ids]
    for sid, ev in evals.items():
        preds = ev["preds_all"]
        src_note = ev["spec"]["pred"]
        prov = [ev["spec"]["pred"], GOLD_REL]
        fcav: List[str] = []
        if sid == ANCHOR:
            ap, why, ppath = anchor_full_predictions(root, free_ids)
            if ap is None:
                for metric in ("fp_spans_per_skillfree_sentence", "share_skillfree_sentences_with_prediction",
                               "full_test_exact_span_f1", "full_test_exact_span_precision", "full_test_exact_span_recall"):
                    rec(sid, metric, None, n_units=len(free_ids) if "skillfree" in metric else len(all_eval_ids),
                        unit="sentence", split=SPLIT_FREE if "skillfree" in metric else SPLIT_ALL,
                        dataset=DATASET, direction="lower" if "skillfree" in metric else "higher", status="pending_gpu",
                        reason=why + "; re-run `PYTHONPATH=src python -m skillllm_v2.evals.research.extraction_full` once it exists")
                continue
            preds = {**ap, **{i: ev["preds_all"][i] for i in sub_ids}}
            src_note = f"{C.rel(ppath)} (2,595 skill-free) + {ev['spec']['pred']} (974)"
            prov.append(str(ppath))
            fcav.append("Anchor full-test numbers merge two inference runs: the stored 974-sentence predictions and A3's "
                        "re-inference on the 2,595 skill-free sentences (decision D-A1-08; A3 reports offset-code drift).")
        else:
            if any(i not in preds for i in free_ids):
                if ev["spec"]["role"] == "baseline":
                    rec(sid, "fp_spans_per_skillfree_sentence", None, n_units=0, unit="sentence", split=SPLIT_FREE,
                        dataset=DATASET, direction="lower", status="not_available",
                        reason="no predictions for the skill-free test sentences")
                continue
        fr = skillfree_rows([preds[i] for i in free_ids])
        free_rows_by_sys[sid] = fr
        n_free = len(free_ids)
        ci = boot_multi(fr, {"fp": lambda S: S[..., 0] / n_free, "share": lambda S: S[..., 1] / n_free}, n_boot, seed)
        cip = boot_multi(cluster_rows(fr, postings_free), {"fp": lambda S: S[..., 0] / n_free,
                                                           "share": lambda S: S[..., 1] / n_free}, n_boot, seed)
        fp_rate, share = fr[:, 0].mean(), fr[:, 1].mean()
        rec(sid, "fp_spans_per_skillfree_sentence", float(fp_rate), n_units=n_free, unit="sentence", split=SPLIT_FREE,
            dataset=DATASET, direction="lower", ci=ci["fp"], ci_method=f"sentence bootstrap, {n_boot} resamples",
            provenance=prov, caveats=fcav, details={"total_fp_spans": int(fr[:, 0].sum()), "ci_posting": list(cip["fp"]),
                                      "source": src_note, "definition": "de-duplicated predicted spans on sentences "
                                      "without any gold span, divided by the number of such sentences"})
        rec(sid, "share_skillfree_sentences_with_prediction", float(share), n_units=n_free, unit="sentence",
            split=SPLIT_FREE, dataset=DATASET, direction="lower", ci=ci["share"],
            ci_method=f"sentence bootstrap, {n_boot} resamples", provenance=prov,
            details={"n_sentences_with_prediction": int(fr[:, 1].sum()), "ci_posting": list(cip["share"]), "source": src_note})
        rows_all = np.array([EE.doc_counts(gold[i]["gold_spans"], preds.get(i, []), gold[i]["text"])[:3]
                             for i in all_eval_ids], dtype=float)
        full_rows_by_sys[sid] = rows_all
        ci_all = boot_multi(rows_all, {"f1": ST.micro_f1, "p": ST.micro_precision, "r": ST.micro_recall}, n_boot, seed)
        ci_allp = boot_multi(cluster_rows(rows_all, postings_all), {"f1": ST.micro_f1}, n_boot, seed)
        S = rows_all.sum(axis=0)
        vals = {"f1": float(ST.micro_f1(S)), "p": float(ST.micro_precision(S)), "r": float(ST.micro_recall(S))}
        for key, metric in (("f1", "full_test_exact_span_f1"), ("p", "full_test_exact_span_precision"),
                            ("r", "full_test_exact_span_recall")):
            rec(sid, metric, vals[key], n_units=len(all_eval_ids), unit="sentence", split=SPLIT_ALL, dataset=DATASET,
                direction="higher", ci=ci_all[key], ci_method=f"sentence bootstrap, {n_boot} resamples",
                provenance=prov, caveats=fcav, details={"tp": int(S[0]), "fp": int(S[1]), "fn": int(S[2]), "source": src_note,
                                          "ci_posting": list(ci_allp["f1"]) if key == "f1" else None,
                                          "subset_exact_span_f1": ev["metrics"]["exact_span_f1"]})
        free_summary.append({"system": sid, "fp_per_sentence": fp_rate, "share": share, "full_f1": vals["f1"],
                             "full_p": vals["p"], "full_r": vals["r"], "subset_f1": ev["metrics"]["exact_span_f1"],
                             "ci_full": ci_all["f1"]})

    # ---------------------------------------------------------------- 5. H1
    h1_rows = []
    baselines = [sid for sid, ev in evals.items() if ev["spec"]["role"] == "baseline"]
    h1_boot = n_boot if smoke else 10000
    h1_shuf = 1000 if smoke else 10000
    if ANCHOR in evals and baselines:
        a = evals[ANCHOR]["rows"][:, :3]
        for b in baselines:
            cmp = ST.compare_counts(a, evals[b]["rows"][:, :3], ST.micro_f1, n_boot=h1_boot, n_shuffles=h1_shuf, seed=seed)
            h1_rows.append({"baseline": b, "family": "H1_anchor_vs_baselines", "p": cmp["p_ar"], "p_boot": cmp["p"],
                            "delta": cmp["delta"], "ci_low": cmp["ci_low"], "ci_high": cmp["ci_high"],
                            "value_anchor": cmp["value_a"], "value_baseline": cmp["value_b"],
                            "n_units_differ": cmp["n_units_differ"], "ar_method": cmp["ar_method"], "se": cmp["se"]})
        ST.adjust_families(h1_rows, p_key="p")
        for r in h1_rows:
            r["passes"] = bool(r["delta"] >= H1_MIN_DELTA and r["p_holm"] is not None and r["p_holm"] < H1_ALPHA)
            rec(ANCHOR, f"H1_delta_exact_span_f1_vs_{r['baseline']}", r["delta"], n_units=len(sub_ids), unit="sentence",
                split=SPLIT_SUB, dataset=DATASET, direction="higher", ci=(r["ci_low"], r["ci_high"]),
                ci_method=f"paired sentence bootstrap, {h1_boot} resamples, seed {seed}",
                provenance=[evals[ANCHOR]["spec"]["pred"], evals[r["baseline"]]["spec"]["pred"], GOLD_REL],
                details={k: v for k, v in r.items() if k not in ("baseline",)} | {"baseline": r["baseline"]})
        binding = min(h1_rows, key=lambda r: r["delta"])
        supported = all(r["passes"] for r in h1_rows)
        verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
        fair_present = [r["baseline"] for r in h1_rows if r["baseline"].startswith("gpu_")]
        provisional = bool(gpu_pending)
        rec(ANCHOR, "H1_verdict", binding["delta"], n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB,
            dataset=DATASET, direction="higher", ci=(binding["ci_low"], binding["ci_high"]),
            ci_method=f"paired sentence bootstrap, {h1_boot} resamples (binding = weakest baseline)",
            provenance=[evals[ANCHOR]["spec"]["pred"], GOLD_REL] + [evals[b]["spec"]["pred"] for b in baselines],
            caveats=["Baseline fairness: neither repo baseline is a trained BERT-family tagger; the published SkillSpan "
                     "JobBERT reaches 59.73 span-F1 (COMBINED, STL, full test incl. BIG). A Δ>=0.10 against these "
                     "baselines says little about the anchor vs a fair supervised encoder."] if not fair_present else [],
            details={"verdict": verdict, "rule": H1_RULE, "p_raw": binding["p"], "p_holm": binding["p_holm"],
                     "family": "H1_anchor_vs_baselines", "binding_baseline": binding["baseline"],
                     "comparisons": h1_rows, "fair_bert_family_baselines_present": fair_present,
                     "fair_baselines_pending": gpu_pending, "provisional": provisional,
                     "published_reference": PUBLISHED_SKILLSPAN, "headline": False})
    else:
        rec(ANCHOR, "H1_verdict", None, n_units=0, unit="sentence", split=SPLIT_SUB, dataset=DATASET,
            direction="higher", status="not_available", reason="anchor or baselines missing",
            details={"rule": H1_RULE, "family": "H1_anchor_vs_baselines"})

    # secondary: full-test protocol comparisons (only when the anchor's skill-free predictions exist)
    h1_full = []
    if ANCHOR in full_rows_by_sys:
        for b in baselines:
            if b in full_rows_by_sys:
                cmp = ST.compare_counts(full_rows_by_sys[ANCHOR], full_rows_by_sys[b], ST.micro_f1, n_boot=h1_boot,
                                        n_shuffles=h1_shuf, seed=seed)
                h1_full.append({"baseline": b, "family": "H1_secondary_full_test", "p": cmp["p_ar"], **{
                    k: cmp[k] for k in ("delta", "ci_low", "ci_high", "value_a", "value_b")}})
        ST.adjust_families(h1_full, p_key="p")
        for r in h1_full:
            rec(ANCHOR, f"H1_secondary_full_test_delta_vs_{r['baseline']}", r["delta"], n_units=len(all_eval_ids),
                unit="sentence", split=SPLIT_ALL, dataset=DATASET, direction="higher", ci=(r["ci_low"], r["ci_high"]),
                ci_method=f"paired sentence bootstrap, {h1_boot} resamples", details=r)

    # ---------------------------------------------------------------- claims reconciliation
    claims = {"rows": [], "known_values": {}}
    if not smoke:
        extra = {}
        for sid in ("bert_baseline", "gliner_baseline"):
            if sid in evals:
                p = evals[sid]["preds_all"]
                extra[f"{sid}@all3569"] = evaluate_extraction([gold[i]["gold_spans"] for i in all_ids],
                                                              [p.get(i, []) for i in all_ids])
        claims = reconcile_claims(comparator_values(evals, extra), root)
        with open(out_dir / "claims_reconciliation.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "line", "claimed", "status", "attributed_systems", "metric_family", "anchor_or_wrapper",
                        "nearest_attributed_system", "nearest_attributed_metric", "nearest_value", "abs_diff",
                        "matches_any"])
            for r in claims["rows"]:
                n = r["nearest_attributed"] or {}
                w.writerow([r["file"], r["line"], r["claimed"], r["status"], ";".join(r["attributed_systems"]),
                            r["metric_family"] or "", r["anchor_or_wrapper"], n.get("system", ""), n.get("metric", ""),
                            "" if not n else f"{n['value']:.6f}", "" if not n else f"{n['abs_diff']:.6f}",
                            ";".join(r["matches"])])
        st = Counter(r["status"] for r in claims["rows"])
        aw = [r for r in claims["rows"] if r["anchor_or_wrapper"]]
        aw_mis = [r for r in aw if r["status"] == "mismatch"]
        rec("all", "claims_reconciliation_mismatch_count", float(st.get("mismatch", 0)), n_units=len(claims["rows"]),
            unit="claim", split="docs", dataset="repo documentation (README, MODEL_CARD, reports/skillllm_v2/*.md, docs/**/*.md, docs/thesis/*.tex)",
            direction="lower", provenance=[out_dir / "claims_reconciliation.csv"],
            caveats=["Heuristic claim extraction (numbers with >=3 decimals in extraction-metric contexts); "
                     "system attribution by keywords. 'mismatch' = equals no recomputed metric of any system at the "
                     "claimed precision; audit documents that quote a wrong number in order to correct it are counted too."],
            details={"status_counts": dict(st), "anchor_or_wrapper_claims": len(aw),
                     "anchor_or_wrapper_mismatches": [{k: r[k] for k in ("file", "line", "claimed", "attributed_systems",
                                                                          "metric_family", "nearest_attributed")} for r in aw_mis],
                     "known_values": claims["known_values"]})

    # ---------------------------------------------------------------- per-unit export (compact: one row per
    # sentence x system x label in {ALL, HARD_SKILL, SOFT_SKILL}; exact-span TP/FP/FN, plus char-token counts on ALL rows)
    with open(out_dir / "per_unit_scores.jsonl", "w") as fh:
        for sid, ev in evals.items():
            rows = ev["rows"]
            for i, did in enumerate(ev["ids"]):
                r = rows[i]
                base = {"layer": LAYER, "family": "exact_span", "metric": "exact_span_counts", "unit_id": did,
                        "posting_id": postings_sub[i], "system": sid, "population": "subset974"}
                fh.write(json.dumps({**base, "label": "ALL", "tp": int(r[EE.C_TP]), "fp": int(r[EE.C_FP]),
                                     "fn": int(r[EE.C_FN]), "token_tp": int(r[EE.C_TTP]), "token_fp": int(r[EE.C_TFP]),
                                     "token_fn": int(r[EE.C_TFN])}, separators=(",", ":")) + "\n")
                for lab, (a, b, c) in (("HARD_SKILL", (EE.C_HTP, EE.C_HFP, EE.C_HFN)),
                                       ("SOFT_SKILL", (EE.C_STP, EE.C_SFP, EE.C_SFN))):
                    fh.write(json.dumps({**base, "label": lab, "tp": int(r[a]), "fp": int(r[b]), "fn": int(r[c])},
                                        separators=(",", ":")) + "\n")
            if sid in free_rows_by_sys:
                for i, did in enumerate(free_ids):
                    fh.write(json.dumps({"layer": LAYER, "family": "exact_span", "metric": "exact_span_counts",
                                         "unit_id": did, "posting_id": postings_free[i], "system": sid,
                                         "population": "skillfree2595", "label": "ALL", "tp": 0,
                                         "fp": int(free_rows_by_sys[sid][i, 0]), "fn": 0}, separators=(",", ":")) + "\n")

    # ---------------------------------------------------------------- figures
    role_order = ["anchor", "anchor_rerun", "safe_wrapper", "legacy_wrapper", "qlora_loop", "baseline", "baseline_seed_run"]
    order = sorted(f1_rows, key=lambda r: (role_order.index(r["role"]) if r["role"] in role_order else 9, -r["exact_span_f1"]))
    figs = []
    if order:
        figs += _plot_f1(order, out_dir)
        figs += _plot_boundary({r["system"]: tax_by_sys[r["system"]] for r in order}, out_dir)

    report = build_report(order, repro_rows, oracle, free_summary, h1_rows, h1_full, claims, tax_by_sys, gpu_pending,
                          delta_rows, identical, len(sub_ids), len(free_ids), figs, n_boot, seed, smoke)
    C.write_layer(LAYER, records, report, seed=seed, smoke=smoke, out_root=out_root,
                  extra_meta={"n_boot": n_boot, "n_boot_headline": 10000, "systems": list(evals),
                              "gpu_baselines_pending": gpu_pending})
    return out_dir


def build_report(order, repro_rows, oracle, free_summary, h1_rows, h1_full, claims, tax_by_sys, gpu_pending,
                 delta_rows, identical, n_sub, n_free, figs, n_boot, seed, smoke) -> str:
    L: List[str] = []
    L.append("# L01 - Extraction benchmark (SkillSpan test)\n")
    L.append("Owner A1. Module `src/skillllm_v2/evals/research/extraction_full.py`; decisions D-A1-01..05 and D-A1-08 in `Decisions.md`. "
             f"Population: {n_sub} test sentences with gold spans (from 65 job postings); skill-free sentences: {n_free}. "
             f"CIs: 95% percentile sentence-level bootstrap (seed {seed}; 10,000 resamples for the anchor and safe wrapper, "
             f"{n_boot} otherwise); `ci_posting` in results.json = posting-level cluster bootstrap (sensitivity).\n")
    L.append("## 1. Reproduction of stored metric files\n")
    if repro_rows:
        L.append("| system | reproduced (<=1e-6) | max abs diff | files compared |")
        L.append("|---|---|---|---|")
        for r in repro_rows:
            L.append(f"| {r['system']} | {r['reproduced']} | {r['max_abs_diff']:.2e} | "
                     f"{', '.join(f['file'] + (':' + f['key'] if f.get('key') else '') for f in r['files'])} |")
        L.append("\nStored BERT/GLiNER files are over all 3,569 sentences and reproduce on that population; "
                 "re-scored on the 974-sentence protocol they are different numbers (table 2).\n")
    else:
        L.append("(smoke run: skipped)\n")
    L.append("## 2. All systems on the same 974 sentences\n")
    L.append("| system | role | exact P | exact R | exact F1 [95% CI] | posting CI | token F1 (char) | word-token F1 | "
             "partial F1 (repo) | char-IoU>=.5 F1 | mean char-IoU | HARD F1 | SOFT F1 | macro F1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in order:
        L.append(f"| {display(r['system'])} | {r['role']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | "
                 f"{_fmt(r['exact_span_f1'])} [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | "
                 f"[{_fmt(r['ci_posting'][0])}, {_fmt(r['ci_posting'][1])}] | {_fmt(r['token_f1'])} | "
                 f"{_fmt(r['word_token_f1'])} | {_fmt(r['partial_span_f1'])} | {_fmt(r['char_iou50_f1'])} | "
                 f"{_fmt(r['char_iou_mean'])} | {_fmt(r['hard'])} | {_fmt(r['soft'])} | {_fmt(r['macro_f1'])} |")
    same = sorted({tuple(sorted([s] + v)) for s, v in identical.items() if v})
    if same:
        L.append("\nIdentical prediction sets (not independent evidence): " + "; ".join(" = ".join(g) for g in same) + ".\n")
    L.append("\n### SemEval-2013 / nervaluate schemes (F1)\n")
    L.append("| system | strict | exact (boundary only) | partial | type |")
    L.append("|---|---|---|---|---|")
    for r in order:
        L.append(f"| {r['system']} | {_fmt(r['strict'])} | {_fmt(r['sem_exact'])} | {_fmt(r['sem_partial'])} | {_fmt(r['sem_type'])} |")
    if oracle.get("available"):
        L.append(f"\nOracle: the `nervaluate` {oracle.get('nervaluate_version')} package, run in a throwaway venv on the "
                 f"same spans, gives identical COR/INC/PAR/MIS/SPU counts for every system and mode: "
                 f"**{oracle['all_match']}**. Greedy nervaluate strict F1 can be below the repo exact F1 when an "
                 "earlier boundary-overlapping prediction consumes a gold span; the one-to-one variant is in results.json.\n")
    else:
        L.append(f"\nnervaluate oracle not available: {oracle.get('reason')}\n")
    L.append("## 3. Boundary-error taxonomy (events; component-based, D-A1-04)\n")
    kinds = [k for k in TAX_KINDS if any(tax_by_sys[r['system']]["events"][k] for r in order)]
    L.append("| system | " + " | ".join(kinds) + " | reconciles FP/FN | mislocated-occurrence FP |")
    L.append("|---|" + "---|" * (len(kinds) + 2))
    for r in order:
        t = tax_by_sys[r["system"]]
        L.append(f"| {r['system']} | " + " | ".join(str(t["events"][k]) for k in kinds)
                 + f" | {r['reconciles']} | {t['mislocated_occurrence_fp']} |")
    L.append("\nExample sentence ids per category are in results.json (`boundary_error_*` details); no text is reproduced here.\n")
    L.append("## 4. Skill-free sentences and the full 3,569-sentence test\n")
    if free_summary:
        L.append("| system | FP spans / skill-free sentence | share of skill-free sentences with >=1 prediction | "
                 "full-test exact P | full-test exact R | full-test exact F1 [95% CI] | 974-subset exact F1 |")
        L.append("|---|---|---|---|---|---|---|")
        for r in free_summary:
            L.append(f"| {r['system']} | {_fmt(r['fp_per_sentence'], 3)} | {_fmt(r['share'], 3)} | {_fmt(r['full_p'])} | "
                     f"{_fmt(r['full_r'])} | {_fmt(r['full_f1'])} [{_fmt(r['ci_full'][0])}, {_fmt(r['ci_full'][1])}] | "
                     f"{_fmt(r['subset_f1'])} |")
    if not any(r["system"] == ANCHOR for r in free_summary):
        L.append(f"\n**Anchor: pending_gpu.** Needs `{EVALS_GPU}/{NOSPAN_JOB}/DONE.json` (A3). Re-run: "
                 "`PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.extraction_full`.\n")
    L.append("## 5. H1 - anchor vs baselines (pre-registered)\n")
    L.append(f"Rule: {H1_RULE}\n")
    if h1_rows:
        L.append("| baseline | anchor F1 | baseline F1 | Δ [95% CI] | p (AR) | p (boot) | p Holm | passes |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in h1_rows:
            L.append(f"| {r['baseline']} | {_fmt(r['value_anchor'])} | {_fmt(r['value_baseline'])} | {_fmt(r['delta'])} "
                     f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | {r['p']:.2e} | {r['p_boot']:.2e} | {r['p_holm']:.2e} | {r['passes']} |")
        ver = "SUPPORTED" if all(r["passes"] for r in h1_rows) else "NOT SUPPORTED"
        L.append(f"\n**H1 verdict: {ver}**" + (" (provisional: fair BERT-family baseline(s) pending: "
                                               + ", ".join(gpu_pending) + ")" if gpu_pending else "") + ".\n")
    seed_rows = [r for r in order if r["role"] == "baseline_seed_run"]
    if seed_rows:
        v = np.array([r["exact_span_f1"] for r in seed_rows])
        L.append(f"Fair-baseline seed runs (descriptive; selection used dev only): n={len(v)}, exact F1 on the 974 sentences "
                 f"mean {v.mean():.4f}, sd {v.std(ddof=1) if len(v) > 1 else 0.0:.4f}, range [{v.min():.4f}, {v.max():.4f}].\n")
    if h1_full:
        L.append("Secondary (full 3,569-sentence protocol): " + "; ".join(
            f"vs {r['baseline']}: Δ {r['delta']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}], Holm p {r['p_holm']:.2e}" for r in h1_full) + "\n")
    L.append("### Baseline fairness (straw-man risk)\n")
    L.append("- The repo's 'BERT baseline' is a train-set token->tag lookup table (`baselines/bert_baseline.py`, "
             "`training_mode: token_classification_memorized_baseline`), not a BERT model. The 'GLiNER baseline' is "
             "zero-shot GLiNER medium-v2.1 with no SkillSpan training.")
    L.append("- The stored baseline metrics (0.059150 / 0.075833) were computed over all 3,569 sentences, the anchor's "
             "over 974; on the same 974 sentences they are higher (table 2).")
    pub = PUBLISHED_SKILLSPAN
    L.append(f"- Published reference ({pub['citation']}; {pub['table']}; PDF sha256 `{pub['pdf_sha256'][:16]}...`), TEST span-F1:")
    L.append("\n| model | SKILL STL | KNOWLEDGE STL | COMBINED STL | COMBINED MTL |")
    L.append("|---|---|---|---|---|")
    for m, v in pub["test_rows"].items():
        L.append(f"| {m} | {v['SKILL_STL']} | {v['KNOWLEDGE_STL']} | {v['COMBINED_STL']} | {v['COMBINED_MTL']} |")
    L.append("")
    for c in pub["comparability"]:
        L.append(f"- {c}")
    L.append("- Consequence: a supervised BERT-family tagger is expected around 0.58-0.60 span-F1 on the full test, "
             "i.e. far above 0.06-0.14. Whether the anchor beats such a model by >=0.10 is untested until A3's fair "
             "baseline exists and the anchor's skill-free predictions allow a same-protocol comparison.\n")
    L.append("## 6. Claims vs artifacts\n")
    if claims["rows"]:
        st = Counter(r["status"] for r in claims["rows"])
        L.append("Status counts over " + str(len(claims["rows"])) + " extracted extraction-metric claims: "
                 + ", ".join(f"{k} {v}" for k, v in st.most_common()) + ". Full table: `claims_reconciliation.csv`.\n")
        L.append("Known values:\n")
        L.append("| value | occurrences | files | equals recomputed |")
        L.append("|---|---|---|---|")
        for v, info in claims["known_values"].items():
            L.append(f"| {v} | {info['occurrences']} | {info['n_files']} | {', '.join(info['recomputed_matches'][:4]) or 'nothing (not reproducible)'} |")
        mis = [r for r in claims["rows"] if r["anchor_or_wrapper"] and r["status"] == "mismatch"]
        L.append(f"\nAnchor/wrapper-attributed claims that equal no recomputed value ({len(mis)}), by file:\n")
        for f, n in Counter(r["file"] for r in mis).most_common():
            L.append(f"- `{f}`: {n} values" + (f" - {KNOWN_CAUSES[f]}" if f in KNOWN_CAUSES else ""))
        L.append("- 0.675034 (20 occurrences) is the anchor's exact F1 from an earlier run ('run 1'); it is used as the "
                 "wrapper gate's improvement target and appears in `wrapper_error_analysis.md`; no stored prediction "
                 "file reproduces it. 0.670222 / 0.672602 (significance_f1.md) likewise match no artifact.\n")
        if mis:
            L.append("| file:line | claimed | attributed | nearest recomputed (system:metric) | |diff| |")
            L.append("|---|---|---|---|---|")
            for r in mis[:60]:
                n = r["nearest_attributed"] or {}
                L.append(f"| {r['file']}:{r['line']} | {r['claimed']} | {','.join(r['attributed_systems'])} | "
                         f"{n.get('system', '-')}:{n.get('metric', '-')} = {_fmt(n.get('value'), 6)} | {_fmt(n.get('abs_diff'), 6)} |")
            if len(mis) > 60:
                L.append(f"\n... {len(mis) - 60} more in the CSV.")
    else:
        L.append("(smoke run: skipped)\n")
    if delta_rows:
        L.append("\n## 7. Descriptive paired Δ exact F1 vs anchor (not hypothesis tests)\n")
        L.append("| system | Δ | 95% CI | p (boot) | sentences that differ |")
        L.append("|---|---|---|---|---|")
        for r in sorted(delta_rows, key=lambda r: -r["delta"]):
            L.append(f"| {r['system']} | {r['delta']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p']:.3g} | {r['n_units_differ']} |")
    L.append("\n## Files\n")
    L.append("- `results.json` (all records), `per_unit_scores.jsonl` (rows `layer, family=exact_span, "
             "metric=exact_span_counts, unit_id, posting_id, system, population, label in ALL/HARD_SKILL/SOFT_SKILL, "
             "tp, fp, fn` [+ char-token counts on ALL rows]), "
             "`claims_reconciliation.csv`, " + ", ".join(f"`{Path(f).name}`" for f in figs))
    L.append("\n## Caveats\n")
    for c in BASE_CAVEATS:
        L.append(f"- {c}")
    L.append("- Partial-span F1 and char-level token F1 are repository definitions, not SkillSpan-official metrics.")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    out = run(seed=args.seed, n_boot=args.n_boot, smoke=args.smoke)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
