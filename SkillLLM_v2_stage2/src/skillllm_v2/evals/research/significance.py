"""L10_significance — central, re-runnable significance testing for the whole evaluation.

1. Central extraction re-test from STORED predictions on the 974-doc SkillSpan test subset (paired document bootstrap,
   10,000 resamples, + approximate randomization, 10,000 shuffles or exact enumeration):
     H1_anchor_vs_baselines  anchor − each baseline (zero-shot GLiNER, token-lookup "BERT", A3's fair JobBERT/BERT runs)
     H2_wrapper_vs_anchor    each stored wrapper variant / legacy stage − anchor
     QLoRA_vs_anchor         v1..v4 − anchor
     ensemble_vs_anchor      majority vote (anchor, v1, v2) − anchor (reconstructed from stored predictions)
   Primary metric exact-span micro-F1; token (character) F1 as a separate secondary family.
2. Every ``reports/skillllm_v2/evals/*/per_unit_scores.jsonl`` present is ingested and its declared families are
   re-tested (system − reference), tolerant to missing layers and to schema variants (tp/fp/fn counts,
   token_tp/..., gold_tp/gold_n/pred_tp/pred_n bucket counts, or scalar ``value`` rows).
3. Holm within each declared family, Benjamini-Hochberg across all tests (secondary).
4. Cross-check against the legacy ``reports/skillllm_v2/significance_*.json`` / ``*_significance*.json`` /
   ``ensemble_eval.json``; disagreements are flagged.
5. Forest plot (deltas + 95% CIs, filled = Holm p < 0.05).

    python -m skillllm_v2.evals.research.significance [--smoke] [--seed 42] [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.research import stats as S
from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, load_layer, md_table, record, rel, write_layer

LAYER = "L10_significance"
DATASET = "SkillSpan test, 974 docs with >=1 gold span"
A = "artifacts/skillllm_v2"
PRED = {
    "anchor": f"{A}/mistral_lora/predictions.jsonl",
    "gliner_zero_shot": f"{A}/gliner_baseline/predictions.jsonl",
    "token_lookup_bert": f"{A}/bert_baseline/predictions.jsonl",
    "safe_wrapper_final": f"{A}/safe_wrapper/predictions.jsonl",
    "safe_wrapper_boundary_refiner": f"{A}/safe_wrapper/boundary_refiner/candidate_predictions.jsonl",
    "safe_wrapper_mistral_gliner_taxonomy": f"{A}/safe_wrapper/mistral_gliner_taxonomy/candidate_predictions.jsonl",
    "safe_wrapper_full_cag_graph": f"{A}/safe_wrapper/full_cag_graph/candidate_predictions.jsonl",
    "legacy_full_cag_graph": f"{A}/full_cag_graph/predictions.jsonl",
    "legacy_mistral_gliner_taxonomy": f"{A}/mistral_gliner_taxonomy/predictions.jsonl",
    "qlora_v1": f"{A}/qlora_loop_eval/predictions.jsonl",
    "qlora_v2": f"{A}/qlora_loop_v2_eval/predictions.jsonl",
    "qlora_v3": f"{A}/qlora_loop_v3_eval/predictions.jsonl",
    "qlora_v4": f"{A}/qlora_loop_v4_eval/predictions.jsonl",
}
FAIR_GLOB = f"{A}/evals_gpu/bert_fair"
LEGACY_SIG = {
    "significance_f1": "reports/skillllm_v2/significance_f1.json",
    "qlora_v1_vs_anchor": "reports/skillllm_v2/significance_qlora_vs_anchor.json",
    "qlora_v2_vs_anchor": "reports/skillllm_v2/qlora_loop_v2_significance_vs_anchor.json",
    "qlora_v2_vs_v1": "reports/skillllm_v2/qlora_loop_v2_significance_vs_v1.json",
    "qlora_v3": "reports/skillllm_v2/qlora_loop_v3_significance.json",
    "qlora_v4": "reports/skillllm_v2/qlora_loop_v4_significance.json",
    "ensemble": "reports/skillllm_v2/ensemble_eval.json",
}
REFERENCE_PREFERENCE = ("R0_raw_mistral", "mistral_lora", "anchor", "full", "hybrid", "baseline")
METRIC_COLS = {"exact_span_f1": (0, 1, 2), "token_f1": (3, 4, 5), "hard_skill_f1": (6, 7, 8), "soft_skill_f1": (9, 10, 11)}


def f1_on(cols: Tuple[int, int, int]) -> S.Stat:
    return lambda s: S.micro_f1(s, *cols)


# ============================================================================ central extraction re-test
def read_preds(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r.get("predictions", [])
    return out


def count_matrix(gold: Dict[str, Dict[str, Any]], ids: Sequence[str], preds: Dict[str, List[Dict[str, Any]]]) -> np.ndarray:
    from skillllm_v2.evals import extraction_evals as ev
    rows = []
    for i in ids:
        c = ev.doc_counts(gold[i]["gold_spans"], preds.get(i, []), gold[i]["text"])
        rows.append([c[ev.C_TP], c[ev.C_FP], c[ev.C_FN], c[ev.C_TTP], c[ev.C_TFP], c[ev.C_TFN],
                     c[ev.C_HTP], c[ev.C_HFP], c[ev.C_HFN], c[ev.C_STP], c[ev.C_SFP], c[ev.C_SFN]])
    return np.asarray(rows, dtype=float)


def _key(sp: Dict[str, Any]) -> Tuple[Any, Any, str]:
    return (sp.get("start"), sp.get("end"), sp.get("label"))


def vote(systems: Sequence[Dict[str, List[Dict[str, Any]]]], ids: Sequence[str], k: int) -> Dict[str, List[Dict[str, Any]]]:
    """Span-key vote (same rule as scripts/ensemble_eval.py: keep (start, end, label) predicted by >= k systems)."""
    out = {}
    for i in ids:
        votes: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
        for sysp in systems:
            seen = set()
            for sp in sysp.get(i, []):
                kk = _key(sp)
                if kk in seen:
                    continue
                seen.add(kk)
                votes[kk].append(sp)
        out[i] = [sps[0] for kk, sps in sorted(votes.items(), key=lambda kv: str(kv[0])) if len(sps) >= k]
    return out


def fair_baselines(root: Path) -> Dict[str, Path]:
    """A3's fairly trained BERT-family baselines (any run dir with predictions.jsonl + DONE.json, smoke excluded)."""
    base = root / FAIR_GLOB
    out = {}
    if not base.exists():
        return out
    for p in sorted(base.rglob("predictions.jsonl")):
        if "smoke" in p.parts:
            continue
        if (p.parent / "DONE.json").exists():
            out["fair_" + "_".join(p.parent.relative_to(base).parts)] = p
    # byte-identical files (the top-level predictions.jsonl is a copy of the dev-selected run) count once in the family
    import hashlib
    seen: Dict[str, str] = {}
    for name in sorted(out, key=lambda n: (n == "fair_", n)):  # the generic top-level alias loses
        h = hashlib.sha256(out[name].read_bytes()).hexdigest()
        if h in seen:
            del out[name]
        else:
            seen[h] = name
    return out


def central(root: Path, smoke: bool, n_boot: int, n_shuffles: int, seed: int, log=print) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from skillllm_v2.evals import extraction_evals as ev
    from skillllm_v2.evals.contract import EvalContext
    gold = ev.load_gold(EvalContext(root=root))
    preds = {k: read_preds(root / v) for k, v in PRED.items() if (root / v).exists()}
    ids = [i for i in preds["anchor"] if i in gold]
    if smoke:
        ids = ids[:120]
    fair = fair_baselines(root)
    for k, p in fair.items():
        preds[k] = read_preds(p)
    if all(k in preds for k in ("anchor", "qlora_v1", "qlora_v2")):
        trio = [preds["anchor"], preds["qlora_v1"], preds["qlora_v2"]]
        preds["ensemble_majority_2of3"] = vote(trio, ids, 2)
        preds["ensemble_unanimous_3of3"] = vote(trio, ids, 3)
        preds["ensemble_union_1of3"] = vote(trio, ids, 1)
    M = {k: count_matrix(gold, ids, v) for k, v in preds.items()}
    log(f"central count matrices for {len(M)} systems, {len(ids)} docs")
    plan: List[Tuple[str, str, str, str]] = []  # (family, system, reference, role)
    for b in ["gliner_zero_shot", "token_lookup_bert"] + sorted(fair):
        if b in M:
            plan.append(("H1_anchor_vs_baselines", "anchor", b, "primary"))
    for w in ("safe_wrapper_final", "safe_wrapper_boundary_refiner", "safe_wrapper_mistral_gliner_taxonomy",
              "legacy_full_cag_graph", "legacy_mistral_gliner_taxonomy"):
        if w in M:
            plan.append(("H2_wrapper_vs_anchor", w, "anchor", "primary"))
    for q in ("qlora_v1", "qlora_v2", "qlora_v3", "qlora_v4"):
        if q in M:
            plan.append(("QLoRA_vs_anchor", q, "anchor", "primary"))
    if "ensemble_majority_2of3" in M:
        plan.append(("ensemble_vs_anchor", "ensemble_majority_2of3", "anchor", "primary"))
    # comparisons needed only to cross-check legacy files (not counted in multiplicity)
    xcheck = [("legacy_crosscheck", a, b, "crosscheck") for a, b in (
        ("safe_wrapper_full_cag_graph", "anchor"), ("qlora_v2", "qlora_v1"), ("qlora_v3", "qlora_v2"), ("qlora_v4", "qlora_v2"),
        ("qlora_v3", "qlora_v1"), ("qlora_v4", "qlora_v1"), ("qlora_v4", "qlora_v3"),
        ("ensemble_unanimous_3of3", "anchor"), ("ensemble_union_1of3", "anchor")) if a in M and b in M]
    tests: List[Dict[str, Any]] = []
    fam_m = defaultdict(int)
    for fam, *_ in plan:
        fam_m[fam] += 1
    for j, (fam, a, b, role) in enumerate(plan + xcheck):
        for mi, (metric, cols) in enumerate(METRIC_COLS.items()):
            sub_a, sub_b = M[a][:, list(cols)], M[b][:, list(cols)]
            st = lambda s: S.micro_f1(s, 0, 1, 2)
            nb = family_resamples(n_boot, fam_m.get(fam, 1))
            bt = S.paired_bootstrap(sub_a, sub_b, st, nb, seed + 97 * j + mi)
            ar = S.approximate_randomization(sub_a, sub_b, st, family_resamples(n_shuffles, fam_m.get(fam, 1)), seed + 97 * j + mi + 50)
            family = fam if (metric == "exact_span_f1" or role == "crosscheck") else f"{fam}:{metric}"
            tests.append({"source": "central", "layer": LAYER, "family": family, "role": role, "system": a, "reference": b,
                          "metric": metric, "delta": bt["delta"], "ci_low": bt["ci_low"], "ci_high": bt["ci_high"], "p": bt["p"],
                          "p_shift": bt["p_shift"], "p_ar": ar["p"], "ar_method": ar["method"], "se": bt["se"], "mde_80": S.mde(bt["se"]),
                          "value_system": bt["value_a"], "value_reference": bt["value_b"], "n_units": bt["n_units"],
                          "n_units_differ": bt["n_units_differ"], "unit": "document",
                          "provenance": [PRED.get(a, str(fair.get(a, ""))), PRED.get(b, str(fair.get(b, "")))] if a in PRED or b in PRED else []})
    info = {"ids": ids, "systems": sorted(M), "fair_baselines": {k: rel(v) for k, v in fair.items()},
            "values": {k: {m: float(f1_on(c)(M[k].sum(0))) for m, c in METRIC_COLS.items()} for k in M}}
    return tests, info


# ============================================================================ per_unit_scores ingestion
COUNT_SCHEMAS = (  # (name, tp, fp, fn) or bucket schema
    ("counts", ("tp", "fp", "fn")),
    ("token_counts", ("token_tp", "token_fp", "token_fn")),
)


def _row_kind(r: Dict[str, Any]) -> str:
    if "tp" in r and "fp" in r and "fn" in r:
        return "counts"
    if "gold_tp" in r and "gold_n" in r and "pred_tp" in r and "pred_n" in r:
        return "bucket"
    if "value" in r:
        return "value"
    return "unknown"


def _group_key(r: Dict[str, Any]) -> Tuple:
    return (r.get("layer"), r.get("family"), r.get("metric"), r.get("label"), r.get("bucket"), r.get("population"))


def _pick_reference(systems: Sequence[str]) -> Optional[str]:
    for ref in REFERENCE_PREFERENCE:
        if ref in systems:
            return ref
    return None


def family_resamples(n_base: int, m: int, cap: int = 50000) -> int:
    """Resamples needed so that the smallest attainable +1-smoothed p (2/(B+1) two-sided) times the family size m stays
    below 0.05 -- otherwise Holm can never reject in a large family no matter how large the effect."""
    return int(min(cap, max(n_base, 40 * m + 1)))


def _sub_metrics(kind: str, metric: str, ua: Dict[str, Any], ub: Dict[str, Any], units: Sequence[str]):
    out = []
    if kind == "counts":
        ra = np.array([[ua[u]["tp"], ua[u]["fp"], ua[u]["fn"]] for u in units], dtype=float)
        rb = np.array([[ub[u]["tp"], ub[u]["fp"], ub[u]["fn"]] for u in units], dtype=float)
        out.append((f"{metric}:micro_f1", ra, rb, lambda s: S.micro_f1(s), "micro-F1 from pooled tp/fp/fn"))
        if all("token_tp" in ua[u] and "token_tp" in ub[u] for u in units):
            ta = np.array([[ua[u]["token_tp"], ua[u]["token_fp"], ua[u]["token_fn"]] for u in units], dtype=float)
            tb = np.array([[ub[u]["token_tp"], ub[u]["token_fp"], ub[u]["token_fn"]] for u in units], dtype=float)
            out.append((f"{metric}:token_micro_f1", ta, tb, lambda s: S.micro_f1(s), "micro-F1 from pooled token counts"))
    elif kind == "bucket":
        ra = np.array([[ua[u]["gold_tp"], ua[u]["gold_n"], ua[u]["pred_tp"], ua[u]["pred_n"]] for u in units], dtype=float)
        rb = np.array([[ub[u]["gold_tp"], ub[u]["gold_n"], ub[u]["pred_tp"], ub[u]["pred_n"]] for u in units], dtype=float)
        out.append((f"{metric}:recall", ra, rb, S.ratio_stat(0, 1), "pooled gold_tp/gold_n"))
    else:
        va = np.array([float(ua[u]["value"]) for u in units])
        vb = np.array([float(ub[u]["value"]) for u in units])
        out.append((metric, va[:, None], vb[:, None], S.mean_stat(len(units)), "mean of per-unit values"))
    return out


def ingest(path: Path, n_boot: int, n_shuffles: int, seed: int, skip_layers: Sequence[str] = (), max_tests: int = 4000,
           log=print) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Re-test every declared family of one per_unit_scores.jsonl.

    A reference system named by ``reference`` (or found through the preference list) is looked up first in the same
    group, then in any group of the same layer with the same (metric, label, bucket, population) -- e.g. L05 keeps the
    anchor rows in one family and declares it as the reference of rows in other families. Resample counts grow with
    the family size (``family_resamples``) so Holm is not blocked by the bootstrap p-value floor."""
    groups: Dict[Tuple, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    slots: Dict[Tuple, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    refs: Dict[Tuple, Dict[str, str]] = defaultdict(dict)
    kinds: Dict[Tuple, str] = {}
    n_rows, bad = 0, 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n_rows += 1
            if r.get("layer") in skip_layers:
                continue
            kind = _row_kind(r)
            if kind == "unknown" or "system" not in r or "unit_id" not in r:
                bad += 1
                continue
            if kind == "value" and (r["value"] is None or (isinstance(r["value"], float) and not math.isfinite(r["value"]))):
                continue
            g = _group_key(r)
            groups[g][r["system"]][str(r["unit_id"])] = r
            slots[(g[0],) + g[2:]][r["system"]][str(r["unit_id"])] = r
            kinds[g] = kind
            if r.get("reference"):
                refs[g][r["system"]] = r["reference"]
    meta = {"file": rel(path), "n_rows": n_rows, "n_unparsed_rows": bad, "n_groups": len(groups), "skipped_groups": []}
    jobs: List[Tuple] = []
    for g, by_sys in sorted(groups.items(), key=lambda kv: str(kv[0])):
        slot = slots[(g[0],) + g[2:]]
        systems = sorted(by_sys)
        if refs.get(g):
            pairs, rule = [(s_, r_) for s_, r_ in refs[g].items() if s_ != r_ and r_ in slot], "declared"
        else:
            ref = _pick_reference(systems) or _pick_reference(sorted(slot))
            if ref is not None:
                pairs, rule = [(s_, ref) for s_ in systems if s_ != ref], "preference list"
            elif len(systems) <= 6:
                pairs, rule = [(x, y) for k, x in enumerate(systems) for y in systems[k + 1:]], "all pairs"
            else:
                pairs, rule = [(s_, systems[0]) for s_ in systems[1:]], "vs first system"
        if not pairs:
            meta["skipped_groups"].append({"group": [str(x) for x in g], "reason": f"no comparable pair (systems: {systems})"})
            continue
        for a, b in pairs:
            ua, ub = by_sys[a], (by_sys[b] if b in by_sys else slot[b])
            units = sorted(set(ua) & set(ub))
            if len(units) < 2:
                meta["skipped_groups"].append({"group": [str(x) for x in g], "pair": [a, b], "reason": f"{len(units)} shared units"})
                continue
            for mname, ra, rb, st, how in _sub_metrics(kinds[g], g[2], ua, ub, units):
                jobs.append((g, a, b, mname, ra, rb, st, how, rule, len(set(ua) ^ set(ub)), len(units)))
    jobs = jobs[:max_tests]
    fam_size = defaultdict(int)
    for j in jobs:
        fam_size[f"{j[0][0]}:{j[0][1]}"] += 1
    tests: List[Dict[str, Any]] = []
    for k, (g, a, b, mname, ra, rb, st, how, rule, dropped, n_u) in enumerate(jobs):
        layer, family, metric, label, bucket, population = g
        fam = f"{layer}:{family}"
        nb = family_resamples(n_boot, fam_size[fam])
        nsh = family_resamples(n_shuffles, fam_size[fam])
        sd = seed + 7919 * (k % 997)
        bt = S.paired_bootstrap(ra, rb, st, nb, sd)
        ar = S.approximate_randomization(ra, rb, st, nsh, sd + 1)
        tests.append({"source": rel(path), "layer": layer, "family": fam, "role": "layer", "system": a, "reference": b,
                      "metric": mname + "".join(f"[{kk}={v}]" for kk, v in (("label", label), ("bucket", bucket),
                                                                            ("population", population)) if v not in (None, "ALL")),
                      "delta": bt["delta"], "ci_low": bt["ci_low"], "ci_high": bt["ci_high"], "p": bt["p"], "p_shift": bt["p_shift"],
                      "p_ar": ar["p"], "ar_method": ar["method"], "se": bt["se"], "mde_80": S.mde(bt["se"]),
                      "value_system": bt["value_a"], "value_reference": bt["value_b"], "n_units": n_u, "n_boot": nb,
                      "n_units_differ": bt["n_units_differ"], "unit": "unit_id", "how": how, "reference_rule": rule,
                      "units_dropped_unpaired": dropped, "provenance": [rel(path)]})
    meta["n_tests"] = len(tests)
    meta["family_sizes"] = dict(fam_size)
    return tests, meta


# ============================================================================ legacy cross-check
def legacy_entries(root: Path) -> List[Dict[str, Any]]:
    """Normalise legacy significance files to (file, system, reference, metric, delta, ci, p_boot, p_perm)."""
    out: List[Dict[str, Any]] = []
    mmap = {"exact_span_f1": "exact_span_f1", "token_f1": "token_f1", "hard_skill_f1": "hard_skill_f1", "soft_skill_f1": "soft_skill_f1"}
    sysmap = {"anchor": "anchor", "v1": "qlora_v1", "v2": "qlora_v2", "v3": "qlora_v3", "v4": "qlora_v4"}

    def add(file, system, reference, metric, d):
        b, pm = d.get("bootstrap", {}), d.get("permutation", {})
        out.append({"file": file, "system": system, "reference": reference, "metric": metric,
                    "delta": b.get("observed_delta"), "ci_low": b.get("ci95_low"), "ci_high": b.get("ci95_high"),
                    "p_boot": b.get("p_two_sided"), "p_perm": pm.get("p_two_sided"),
                    "p_holm_legacy": b.get("p_holm_all", b.get("p_holm_5_metrics")), "n_resamples": b.get("n_resamples")})
    p = root / LEGACY_SIG["significance_f1"]
    if p.exists():
        d = json.loads(p.read_text())
        for v, vv in d.get("variants", {}).items():
            for m, mm in vv.get("metrics", {}).items():
                if m in mmap:
                    add(LEGACY_SIG["significance_f1"], f"safe_wrapper_{v}", "anchor", m, mm)
    for tag, sysname, refname in (("qlora_v1_vs_anchor", "qlora_v1", "anchor"), ("qlora_v2_vs_anchor", "qlora_v2", "anchor"),
                                  ("qlora_v2_vs_v1", "qlora_v2", "qlora_v1")):
        p = root / LEGACY_SIG[tag]
        if p.exists():
            d = json.loads(p.read_text())
            for m, mm in d.get("metrics", {}).items():
                if m in mmap:
                    add(LEGACY_SIG[tag], sysname, refname, m, mm)
    for tag in ("qlora_v3", "qlora_v4"):
        p = root / LEGACY_SIG[tag]
        if p.exists():
            d = json.loads(p.read_text())
            for ck, cv in d.get("comparisons", {}).items():
                if "_vs_" not in ck:
                    continue
                a, b = ck.split("_vs_", 1)
                for m, mm in cv.get("metrics", {}).items():
                    if m in mmap and a in sysmap and b in sysmap:
                        add(LEGACY_SIG[tag], sysmap[a], sysmap[b], m, mm)
    p = root / LEGACY_SIG["ensemble"]
    if p.exists():
        d = json.loads(p.read_text())
        name = {"majority(>=2/3)": "ensemble_majority_2of3", "unanimous(3/3)": "ensemble_unanimous_3of3", "union(>=1/3)": "ensemble_union_1of3"}
        for ck, cv in d.get("comparisons", {}).items():
            for m, mm in cv.get("metrics", {}).items():
                if m in mmap and ck in name:
                    add(LEGACY_SIG["ensemble"], name[ck], "anchor", m, mm)
    return out


def crosscheck(legacy: List[Dict[str, Any]], tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    idx = {(t["system"], t["reference"], t["metric"]): t for t in tests if t["source"] == "central"}
    rows = []
    for L in legacy:
        t = idx.get((L["system"], L["reference"], L["metric"]))
        if t is None:
            rows.append({**L, "status": "not_recomputed"})
            continue
        flags = []
        if L["delta"] is not None and abs(L["delta"] - t["delta"]) > 1e-6:
            flags.append(f"DELTA_MISMATCH ({L['delta']:+.6f} vs {t['delta']:+.6f})")
        if L["p_boot"] is not None and (L["p_boot"] < 0.05) != (t["p"] < 0.05):
            flags.append(f"BOOT_DECISION_DIFFERS (legacy p {L['p_boot']:.4f} vs {t['p']:.4f})")
        if L["p_perm"] is not None and (L["p_perm"] < 0.05) != (t["p_ar"] < 0.05):
            flags.append(f"PERM_DECISION_DIFFERS (legacy p {L['p_perm']:.4f} vs AR {t['p_ar']:.4f})")
        if L["ci_low"] is not None and t["ci_low"] is not None and (abs(L["ci_low"] - t["ci_low"]) > 0.003 or abs(L["ci_high"] - t["ci_high"]) > 0.003):
            flags.append("CI_DIFFERS_>0.003")
        if L.get("p_holm_legacy") is not None and t.get("p_holm") is not None and (L["p_holm_legacy"] < 0.05) != (t["p_holm"] < 0.05):
            flags.append(f"HOLM_DECISION_DIFFERS (legacy family {L['p_holm_legacy']:.4f} vs {t['p_holm']:.4f}; families differ)")
        rows.append({**L, "new_delta": t["delta"], "new_ci_low": t["ci_low"], "new_ci_high": t["ci_high"], "new_p": t["p"],
                     "new_p_ar": t["p_ar"], "new_p_holm": t.get("p_holm"), "flags": flags, "status": "agree" if not flags else "flagged"})
    return rows


# ============================================================================ outputs
def forest(path: Path, tests: List[Dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tests:
        if t["role"] != "crosscheck" and t.get("ci_low") is not None:
            fams[t["family"]].append(t)
    fams = {k: v[:40] for k, v in fams.items()}
    heights = [max(1.2, 0.28 * len(v) + 0.8) for v in fams.values()]
    if not fams:
        return
    fig, axes = plt.subplots(len(fams), 1, figsize=(9, sum(heights)), gridspec_kw={"height_ratios": heights}, squeeze=False)
    for ax, (fam, tt) in zip(axes[:, 0], fams.items()):
        y = np.arange(len(tt))[::-1]
        d = np.array([t["delta"] for t in tt])
        lo = np.array([t["ci_low"] for t in tt])
        hi = np.array([t["ci_high"] for t in tt])
        sig = np.array([bool(t.get("p_holm") is not None and t["p_holm"] < 0.05) for t in tt])
        ax.errorbar(d, y, xerr=[np.maximum(0, d - lo), np.maximum(0, hi - d)], fmt="none", ecolor="#7f8c8d", capsize=2, lw=1)
        ax.scatter(d[sig], y[sig], color="#1f5f99", zorder=3, s=18, label="Holm p<0.05")
        ax.scatter(d[~sig], y[~sig], facecolors="white", edgecolors="#1f5f99", zorder=3, s=18, label="n.s.")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t['system']} − {t['reference']} | {t['metric']}"[:70] for t in tt], fontsize=6)
        ax.set_title(fam, fontsize=8, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    axes[0, 0].legend(fontsize=6, loc="best")
    axes[-1, 0].set_xlabel("delta (system − reference) with 95% paired bootstrap CI")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_records(tests: List[Dict[str, Any]], xrows: List[Dict[str, Any]], hcheck: Dict[str, Any], ds: str) -> List[Dict[str, Any]]:
    recs = []
    for t in tests:
        recs.append(record(LAYER, f"{t['system']} vs {t['reference']}", f"delta_{t['metric']}", t["delta"], n_units=t["n_units"],
                           unit=t["unit"], split="test", dataset=ds if t["source"] == "central" else f"from {t['source']}",
                           direction="none", ci=(t["ci_low"], t["ci_high"]) if t["ci_low"] is not None else None,
                           ci_method="paired unit-level percentile bootstrap", provenance=t.get("provenance", []),
                           caveats=(["cross-check only; not counted in multiplicity adjustment"] if t["role"] == "crosscheck" else []),
                           details={k: t.get(k) for k in ("family", "role", "layer", "p", "p_shift", "p_ar", "ar_method", "p_holm",
                                                          "family_size", "p_bh", "se", "mde_80", "value_system", "value_reference",
                                                          "n_units_differ", "how", "reference_rule", "units_dropped_unpaired")}))
    n_flag = sum(1 for x in xrows if x.get("status") == "flagged")
    recs.append(record(LAYER, "legacy_crosscheck", "n_legacy_comparisons_flagged", float(n_flag), n_units=len(xrows), unit="comparison",
                       split="test", dataset=ds, direction="lower", details={"rows": xrows}))
    for h, d in hcheck.items():
        recs.append(record(LAYER, h, f"{h}_central_retest", d.get("value"), n_units=d.get("n_units", 0), unit="document", split="test",
                           dataset=ds, direction="higher", status="measured" if d.get("value") is not None else "not_available",
                           reason="" if d.get("value") is not None else d.get("reason", "not computable"),
                           ci=d.get("ci"), ci_method="paired document bootstrap", details=d))
    return recs


def hypothesis_checks(tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    h1 = [t for t in tests if t["family"] == "H1_anchor_vs_baselines"]
    if h1:
        ok = all(t["delta"] >= 0.10 and t["p_holm"] is not None and t["p_holm"] < 0.05 for t in h1)
        worst = min(h1, key=lambda t: t["delta"])
        l01 = load_layer("L01_extraction")
        owner = next((r for r in (l01 or {}).get("records", []) if r.get("metric") == "H1_verdict"), None)
        owner_v = (owner or {}).get("details", {}).get("verdict")
        central_v = "SUPPORTED" if ok else "NOT SUPPORTED"
        out["H1"] = {"value": worst["delta"], "ci": (worst["ci_low"], worst["ci_high"]), "n_units": worst["n_units"],
                     "central_verdict_by_rule": central_v, "baselines": [t["reference"] for t in h1],
                     "owner_verdict_L01": owner_v,
                     "agrees_with_owner": None if owner_v is None else (owner_v == central_v),
                     "note": "L10 re-applies the H1 rule to the baselines it can see from stored predictions; A1 (L01) owns the verdict."}
    h2 = [t for t in tests if t["family"] == "H2_wrapper_vs_anchor" and t["system"] == "safe_wrapper_final"]
    if h2:
        t = h2[0]
        l05 = load_layer("L05_wrapper")
        owner = next((r for r in (l05 or {}).get("records", []) if r.get("metric") == "H2_verdict"), None)
        out["H2"] = {"value": t["delta"], "ci": (t["ci_low"], t["ci_high"]), "n_units": t["n_units"],
                     "f1_noninferior_margin_-0.005": bool(t["ci_low"] is not None and t["ci_low"] > -0.005),
                     "owner_verdict_L05": (owner or {}).get("details", {}).get("verdict"),
                     "note": "only the F1 non-inferiority half of H2 is re-tested here; APR half lives in L05"}
    return out


def report(tests, xrows, metas, info, hcheck, n_boot, n_shuffles, seed, elapsed) -> str:
    L = ["# L10_significance — central significance re-test", "",
         f"Paired document (or unit) bootstrap, {n_boot} resamples, seed {seed}; approximate randomization {n_shuffles} shuffles "
         "(exact enumeration when ≤16 units differ). p (bootstrap) = percentile inversion, +1 smoothed; Holm within declared family; "
         "BH across all non-cross-check tests (secondary). Statistics module: `research/stats.py` (Decisions D-A2-01).", "",
         f"Central extraction systems ({len(info['systems'])}): {', '.join(info['systems'])}. Fair BERT-family baselines found: "
         f"{info['fair_baselines'] or 'none yet (A3 J5 pending)'}.", "", "## Central extraction families (exact-span F1 primary)", ""]
    cols = ["family", "system − reference", "Δ", "95% CI", "p boot", "p AR", "p Holm", "q BH", "docs differ", "MDE80"]

    def row(t):
        return {"family": t["family"], "system − reference": f"{t['system']} − {t['reference']}", "Δ": t["delta"],
                "95% CI": f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]" if t["ci_low"] is not None else "-", "p boot": t["p"],
                "p AR": t["p_ar"], "p Holm": t.get("p_holm"), "q BH": t.get("p_bh"), "docs differ": t["n_units_differ"], "MDE80": t.get("mde_80")}
    L.append(md_table([row(t) for t in tests if t["source"] == "central" and t["role"] == "primary"], cols))
    L += ["", "## Hypothesis re-tests (owners decide the verdict)", ""]
    for h, d in hcheck.items():
        L.append(f"- {h}: " + json.dumps({k: v for k, v in d.items() if k not in ("ci",)}, default=str))
    L += ["", "## Legacy significance files cross-check", ""]
    xr = [x for x in xrows if x.get("status") != "not_recomputed"]
    L.append(md_table([{"file": Path(x["file"]).name, "comparison": f"{x['system']} − {x['reference']}", "metric": x["metric"],
                        "legacy Δ": x["delta"], "new Δ": x.get("new_delta"), "legacy p boot": x["p_boot"], "new p boot": x.get("new_p"),
                        "legacy p perm": x["p_perm"], "new p AR": x.get("new_p_ar"), "flags": "; ".join(x.get("flags", [])) or "agree"} for x in xr],
                      ["file", "comparison", "metric", "legacy Δ", "new Δ", "legacy p boot", "new p boot", "legacy p perm", "new p AR", "flags"]))
    nr = [x for x in xrows if x.get("status") == "not_recomputed"]
    if nr:
        L.append(f"\n{len(nr)} legacy comparisons not recomputed (system not available centrally): "
                 + ", ".join(sorted({f"{x['system']}−{x['reference']}" for x in nr})))
    L += ["", "## Ingested layer files", ""]
    for m in metas:
        L.append(f"- `{m['file']}`: {m['n_rows']} rows, {m['n_groups']} groups, {m['n_tests']} tests, {len(m['skipped_groups'])} skipped groups, "
                 f"{m['n_unparsed_rows']} unparsed rows.")
    fam_sum: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tests:
        if t["source"] != "central":
            fam_sum[t["family"]].append(t)
    if fam_sum:
        L += ["", "Per ingested family: tests / significant after Holm (p<0.05).", ""]
        L.append(md_table([{"family": f, "tests": len(v), "Holm-significant": sum(1 for t in v if t.get("p_holm") is not None and t["p_holm"] < 0.05),
                            "BH-significant": sum(1 for t in v if t.get("p_bh") is not None and t["p_bh"] < 0.05)} for f, v in sorted(fam_sum.items())],
                          ["family", "tests", "Holm-significant", "BH-significant"]))
    L += ["", "## Caveats", "",
          "- Document resampling captures test-sample noise only; no training-seed variance (single run per system).",
          "- Wrapper deltas rest on ≤25 differing documents: AR (exact when ≤16 differ) is the more trustworthy p-value there.",
          "- Ingested families are re-tested as declared by their owners; descriptive per-unit metrics inside a large declared family make Holm "
          "conservative. Owners' own verdicts take precedence; disagreements are listed above.",
          f"- Runtime {elapsed:.0f}s. Files: `results.json`, `all_tests.csv`, `forest_plot.png`, `legacy_crosscheck.csv`."]
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-shuffles", type=int, default=10000)
    ap.add_argument("--layer-n-boot", type=int, default=2000, help="bootstrap size for ingested layer families")
    args = ap.parse_args(argv)
    os.chdir(REPO_ROOT)
    t0 = time.time()
    n_boot = 300 if args.smoke else args.n_boot
    n_sh = 300 if args.smoke else args.n_shuffles
    lnb = 200 if args.smoke else args.layer_n_boot
    tests, info = central(REPO_ROOT, args.smoke, n_boot, n_sh, args.seed)
    metas = []
    for p in sorted(EVALS_ROOT.glob("*/per_unit_scores.jsonl")):
        if p.parent.name == LAYER:
            continue
        try:
            tt, meta = ingest(p, lnb, min(n_sh, 2000), args.seed, max_tests=300 if args.smoke else 4000)
        except Exception as exc:  # noqa: BLE001 - one malformed layer file must not kill the central test
            meta = {"file": rel(p), "n_rows": 0, "n_groups": 0, "n_tests": 0, "skipped_groups": [], "n_unparsed_rows": 0,
                    "error": repr(exc)}
            tt = []
        tests += tt
        metas.append(meta)
        print(f"ingested {rel(p)}: {meta['n_tests']} tests", flush=True)
    # multiplicity: Holm within family (primary + layer tests), BH across all of them
    adj = [t for t in tests if t["role"] != "crosscheck"]
    S.adjust_families(adj, p_key="p", family_key="family", holm_key="p_holm", bh_key="p_bh")
    S.adjust_families(adj, p_key="p_ar", family_key="family", holm_key="p_ar_holm", bh_key="p_ar_bh")
    for t in tests:
        if t["role"] == "crosscheck":
            t["p_holm"] = t["p_bh"] = None
    xrows = crosscheck(legacy_entries(REPO_ROOT), tests)
    hcheck = hypothesis_checks(tests)
    ds = DATASET if not args.smoke else "SkillSpan test, first 120 docs (smoke)"
    recs = build_records(tests, xrows, hcheck, ds)
    md = report(tests, xrows, metas, info, hcheck, n_boot, n_sh, args.seed, time.time() - t0)
    out = write_layer(LAYER, recs, md, seed=args.seed, smoke=args.smoke,
                      extra_meta={"n_boot": n_boot, "n_shuffles": n_sh, "layer_n_boot": lnb, "ingested": metas})
    keys = ["source", "layer", "family", "role", "system", "reference", "metric", "delta", "ci_low", "ci_high", "p", "p_shift", "p_ar",
            "p_holm", "p_ar_holm", "p_bh", "family_size", "n_units", "n_units_differ", "value_system", "value_reference", "mde_80", "ar_method"]
    with open(out / "all_tests.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for t in tests:
            w.writerow([t.get(k) for k in keys])
    with open(out / "legacy_crosscheck.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        ks = ["file", "system", "reference", "metric", "delta", "new_delta", "ci_low", "ci_high", "new_ci_low", "new_ci_high", "p_boot",
              "new_p", "p_perm", "new_p_ar", "status", "flags"]
        w.writerow(ks)
        for x in xrows:
            w.writerow([("; ".join(x[k]) if k == "flags" and x.get(k) else x.get(k)) for k in ks])
    try:
        forest(out / "forest_plot.png", [t for t in tests if t["source"] == "central"] + [t for t in tests if t["source"] != "central"][:200])
        # compact headline forest: pre-registered primary contrasts only (central exact-span F1, H5, H6 families)
        head = [t for t in tests if t["role"] != "crosscheck" and (
            (t["source"] == "central" and t["metric"] == "exact_span_f1")
            or t["family"] in ("L08_graph_retrieval:H5", "L09_course_rec:H6", "L09_course_rec:H6-ndcg10-noninferiority"))]
        forest(out / "forest_headline.png", head)
    except Exception as exc:  # noqa: BLE001
        print(f"forest plot failed: {exc}")
    print(f"wrote {out}: {len(tests)} tests, {sum(1 for x in xrows if x.get('status') == 'flagged')} legacy flags, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
