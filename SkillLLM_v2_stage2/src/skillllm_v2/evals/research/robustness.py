"""L03_robustness: perturbation robustness, natural boundary-case slice and cross-dataset (out-of-distribution) results.

Inputs
  * perturbed docs: inputs/evals/research/L03_robustness/perturbed_docs.jsonl (research/perturbations.py, seed 42,
    120 stratified test docs x 14 families + identity), not_available.jsonl, MANIFEST.json
  * predictions on them: artifacts/skillllm_v2/evals_gpu/robustness_<system>/predictions.jsonl
    (anchor = GPU job J4; the others = scripts/evals/gpu_baselines_infer.py, CPU)
  * cross-dataset docs (J6): inputs/evals/research/L03_robustness/xdata_{docs,gold}.jsonl and xdata_<system> predictions
  * natural slice: the 974-doc gold test subset with the STORED predictions of every system (+ J1 re-run, + fair BERT)

Statistics: per-document count rows (extraction_evals.doc_counts), clean = the ``identity`` copy scored in the SAME run,
Δ = perturbed - clean on the same documents, paired document bootstrap (research/stats.paired_bootstrap) + approximate
randomization; Holm across the families of one system/metric, BH across the layer. Headline = anchor mean degradation of
exact-span F1 over the non-identity families (joint document bootstrap: one draw of base documents per replicate, every
family's Δ recomputed on the drawn documents to which it applies, then averaged).

  python -m skillllm_v2.evals.research.robustness [--smoke] [--seed 42] [--n-boot 2000]
  python -m skillllm_v2.evals.research.robustness --build-xdata          # J6 inputs (Green / Sayfullina samples)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.extraction_evals import METRICS, N_COLS, doc_counts
from skillllm_v2.evals.research import stats
from skillllm_v2.evals.research.common import REPO_ROOT, md_table, record, rel, sha256_file, write_layer

LAYER = "L03_robustness"
IN_DIR = REPO_ROOT / "inputs/evals/research/L03_robustness"
GPU = REPO_ROOT / "artifacts/skillllm_v2/evals_gpu"
OUT_DIR = REPO_ROOT / "reports/skillllm_v2/evals" / LAYER
SYSTEMS = ("anchor", "safe_wrapper", "bert_fair", "gliner_zeroshot", "token_lookup")
SYSTEM_LABEL = {
    "anchor": "Mistral-24B LoRA anchor",
    "safe_wrapper": "safe wrapper (full_cag_graph recomposed on anchor output)",
    "bert_fair": "fair BERT-family tagger (dev-selected, J5)",
    "gliner_zeroshot": "GLiNER medium v2.1 zero-shot",
    "token_lookup": "token-lookup baseline (memorised BIO tags; stored as 'BERT baseline')",
}
PRIMARY = "exact_span_f1"
REPORT_METRICS = ("exact_span_f1", "partial_span_f1", "token_f1")
SURFACE_CHANGING = {"case_lower", "case_upper", "case_title", "typo_inside", "alias_sub"}
DATASET = "SkillSpan(HF) test, 120-doc stratified subset of the 974 gold-bearing docs, perturbed"


# ----------------------------------------------------------------------------- io
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_preds(path: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return None
    return {r["id"]: r for r in read_jsonl(path)}


def pred_path(job: str, system: str) -> Path:
    return GPU / f"{job}_{system}" / "predictions.jsonl"


def complete(job: str, system: str) -> bool:
    if system == "anchor":
        return (GPU / f"{job}_anchor" / "DONE.json").exists()
    return (GPU / f"{job}_{system}" / "DONE.json").exists()


# ----------------------------------------------------------------------------- counts
def count_rows(ids: Sequence[str], gold: Dict[str, List[Dict[str, Any]]], texts: Dict[str, str],
               preds: Dict[str, Dict[str, Any]], relabel: Optional[str] = None) -> np.ndarray:
    out = np.zeros((len(ids), N_COLS))
    for k, i in enumerate(ids):
        g = gold[i]
        p = preds[i]["predictions"] if i in preds else []
        if relabel:
            g = [{**s, "label": relabel} for s in g]
            p = [{**s, "label": relabel} for s in p]
        out[k] = doc_counts(g, p, texts[i])
    return out


def stat_fn(metric: str) -> Callable[[np.ndarray], np.ndarray]:
    return METRICS[metric]


def doc_f1(row: np.ndarray) -> Optional[float]:
    tp, fp, fn = row[0], row[1], row[2]
    return None if tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn))


# ----------------------------------------------------------------------------- perturbation analysis
def family_tables(rows: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """-> (base ids in order, {family: {base_id: perturbed row id}})."""
    base_ids: List[str] = []
    fam: Dict[str, Dict[str, str]] = defaultdict(dict)
    for r in rows:
        b, f = r["meta"]["base_id"], r["meta"]["family"]
        if b not in base_ids:
            base_ids.append(b)
        fam[f][b] = r["id"]
    return base_ids, dict(fam)


def mean_degradation(base_ids: List[str], fam_ids: Dict[str, Dict[str, str]], counts: Dict[str, np.ndarray],
                     families: Sequence[str], metric: str, n_boot: int, seed: int) -> Dict[str, Any]:
    """mean over families of (clean - perturbed) metric, joint document bootstrap over base docs.

    ``counts[row_id]`` is that row's count vector. For each family, arrays aligned to ``base_ids`` carry zero rows where the
    family does not apply, so a resample's sums only include applicable drawn documents."""
    fn = stat_fn(metric)
    n = len(base_ids)
    pert, clean = {}, {}
    for f in families:
        P = np.zeros((n, N_COLS)); C = np.zeros((n, N_COLS))
        for k, b in enumerate(base_ids):
            if b in fam_ids[f] and b in fam_ids["identity"]:
                P[k] = counts[fam_ids[f][b]]
                C[k] = counts[fam_ids["identity"][b]]
        pert[f], clean[f] = P, C
    obs = float(np.mean([fn(clean[f].sum(0)) - fn(pert[f].sum(0)) for f in families]))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    chunk = max(1, 200_000 // (n * N_COLS))
    for lo in range(0, n_boot, chunk):
        b = min(chunk, n_boot - lo)
        idx = rng.integers(0, n, size=(b, n))
        acc = np.zeros(b)
        for f in families:
            acc += fn(clean[f][idx].sum(1)) - fn(pert[f][idx].sum(1))
        samples[lo:lo + b] = acc / len(families)
    out = stats.summarize_samples(samples, obs, paired=True)
    out["n_units"] = n
    out["samples"] = samples
    return out


def analyse_perturbations(n_boot: int, seed: int, smoke: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {"family_rows": [], "per_unit": [], "systems_available": [], "na_counts": {}}
    pfile = IN_DIR / "perturbed_docs.jsonl"
    if not pfile.exists():
        recs.append(record(LAYER, "anchor", "mean_degradation_exact_f1", None, n_units=0, unit="document", split="test",
                           dataset=DATASET, direction="lower", status="pending_gpu",
                           reason="perturbation set not frozen yet", details={"headline": True}))
        return recs, info
    rows = read_jsonl(pfile)
    manifest = json.loads((IN_DIR / "MANIFEST.json").read_text())
    na = read_jsonl(IN_DIR / "not_available.jsonl") if (IN_DIR / "not_available.jsonl").exists() else []
    info["na_counts"] = dict(Counter(r["family"] for r in na))
    info["na_reasons"] = {f: dict(Counter(r["reason"] for r in na if r["family"] == f)) for f in info["na_counts"]}
    info["manifest"] = {k: v for k, v in manifest.items() if k != "base_ids"}
    base_ids, fam_ids = family_tables(rows)
    if smoke:
        base_ids = base_ids[:20]
        fam_ids = {f: {b: r for b, r in m.items() if b in base_ids} for f, m in fam_ids.items()}
    keep = {rid for m in fam_ids.values() for rid in m.values()}
    rows = [r for r in rows if r["id"] in keep]
    gold = {r["id"]: r["gold_spans"] for r in rows}
    texts = {r["id"]: r["text"] for r in rows}
    families = [f for f in fam_ids if f != "identity"]
    prov_inputs = [pfile, IN_DIR / "MANIFEST.json"]
    info["base_ids"] = base_ids
    for system in SYSTEMS:
        job = "robustness"
        ppath = pred_path(job, system)
        preds = load_preds(ppath) if complete(job, system) else None
        if preds is None or any(r["id"] not in preds for r in rows):
            missing = "not run yet" if preds is None else f"{sum(r['id'] not in preds for r in rows)} rows missing"
            recs.append(record(LAYER, system, "mean_degradation_exact_f1", None, n_units=len(base_ids), unit="document",
                               split="test", dataset=DATASET, direction="lower",
                               status="pending_gpu" if system in ("anchor", "safe_wrapper") else "not_available",
                               reason=f"predictions on perturbed docs {missing}: {rel(ppath)}",
                               details={"headline": system == "anchor"}))
            continue
        info["systems_available"].append(system)
        ids = [r["id"] for r in rows]
        mat = count_rows(ids, gold, texts, preds)
        counts = {i: mat[k] for k, i in enumerate(ids)}
        prov = prov_inputs + [ppath]
        status_by_row = {i: preds[i].get("status", "ok") for i in ids}
        fam_results = []
        for f in families:
            bs = [b for b in base_ids if b in fam_ids[f] and b in fam_ids["identity"]]
            if not bs:
                continue
            P = np.array([counts[fam_ids[f][b]] for b in bs])
            C = np.array([counts[fam_ids["identity"][b]] for b in bs])
            for metric in REPORT_METRICS:
                fn = stat_fn(metric)
                pb = stats.paired_bootstrap(P, C, fn, n_boot=n_boot, seed=seed)
                ar = stats.approximate_randomization(P, C, fn, n_shuffles=2000 if smoke else 10000, seed=seed)
                fam_results.append({"system": system, "family": f, "metric": metric, "n_docs": len(bs),
                                    "clean": pb["value_b"], "perturbed": pb["value_a"], "delta": pb["delta"],
                                    "ci_low": pb["ci_low"], "ci_high": pb["ci_high"], "p_boot": pb["p"], "p": ar["p"],
                                    "n_units_differ": pb["n_units_differ"],
                                    "parse_errors": sum(status_by_row[fam_ids[f][b]] != "ok" for b in bs),
                                    "mean_pred_spans_clean": float(np.mean([C[k][0] + C[k][1] for k in range(len(bs))])),
                                    "mean_pred_spans_perturbed": float(np.mean([P[k][0] + P[k][1] for k in range(len(bs))]))})
            for b in bs:
                for cond, rid in (("perturbed", fam_ids[f][b]), ("clean", fam_ids["identity"][b])):
                    c = counts[rid]
                    info["per_unit"].append({"layer": LAYER, "family": f"perturbation:{f}", "system": system,
                                             "metric": "exact_span", "condition": cond, "unit_id": b, "row_id": rid,
                                             "tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "value": doc_f1(c)})
        for metric in REPORT_METRICS:
            sub = [x for x in fam_results if x["metric"] == metric]
            for x, ph, pbh in zip(sub, stats.holm([x["p"] for x in sub]), stats.benjamini_hochberg([x["p"] for x in sub])):
                x["p_holm"], x["p_bh_within"] = ph, pbh
        info["family_rows"] += fam_results
        for x in fam_results:
            recs.append(record(
                LAYER, system, f"delta_{x['metric']}:{x['family']}", x["delta"], n_units=x["n_docs"], unit="document",
                split="test", dataset=DATASET, direction="higher", ci=(x["ci_low"], x["ci_high"]),
                ci_method="paired document bootstrap (perturbed vs identity copy, same run), 95% percentile",
                provenance=prov, caveats=(["gold surface changes in this family (same skill, different string)"]
                                          if x["family"] in SURFACE_CHANGING else []),
                details={"definition": "metric(perturbed) - metric(identity copy) on the same documents",
                         "clean": x["clean"], "perturbed": x["perturbed"], "p_ar": x["p"], "p_boot": x["p_boot"],
                         "p_holm": x["p_holm"], "family": f"L03:{system}:{x['metric']}", "parse_errors": x["parse_errors"],
                         "n_units_differ": x["n_units_differ"], "mean_pred_spans_clean": x["mean_pred_spans_clean"],
                         "mean_pred_spans_perturbed": x["mean_pred_spans_perturbed"]}))
        # clean reference + absolute scores on the identity copies
        idrows = np.array([counts[fam_ids["identity"][b]] for b in base_ids])
        for metric in REPORT_METRICS:
            ci = stats.bootstrap_ci(idrows, stat_fn(metric), n_boot=n_boot, seed=seed)
            recs.append(record(LAYER, system, f"clean_{metric}", ci["value"], n_units=len(base_ids), unit="document",
                               split="test", dataset=DATASET, direction="higher", ci=(ci["ci_low"], ci["ci_high"]),
                               ci_method="document bootstrap", provenance=prov,
                               details={"definition": "metric on the identity copies of the 120 robustness docs"}))
        sets = {"all": families, "surface_preserving": [f for f in families if f not in SURFACE_CHANGING]}
        info.setdefault("mean_deg", {})[system] = {}
        for name, fams in sets.items():
            md = mean_degradation(base_ids, fam_ids, counts, fams, PRIMARY, n_boot if not smoke else 200, seed)
            info["mean_deg"][system][name] = md
            metric = "mean_degradation_exact_f1" + ("" if name == "all" else "_surface_preserving")
            recs.append(record(
                LAYER, system, metric, md["value"], n_units=len(base_ids), unit="document", split="test", dataset=DATASET,
                direction="lower", ci=(md["ci_low"], md["ci_high"]),
                ci_method="joint document bootstrap over base docs (all families recomputed per replicate), 95% percentile",
                provenance=prov, caveats=["unweighted mean over families; families apply to different numbers of docs",
                                          "perturbations are synthetic; see the human validity sample"],
                details={"headline": system == "anchor" and name == "all", "families": fams, "p_boot": md["p"],
                         "definition": "mean over families of [exact F1(identity copy) - exact F1(perturbed)]; positive = worse"}))
        if system == "anchor":
            info["determinism"] = determinism_check(fam_ids, preds, base_ids)
            d = info["determinism"]
            if d.get("n_docs"):
                recs.append(record(LAYER, "anchor", "identity_rerun_identical_doc_rate", d["identical_rate"],
                                   n_units=d["n_docs"], unit="document", split="test", dataset=DATASET, direction="higher",
                                   provenance=[ppath, GPU / "anchor_test_logprobs/predictions.jsonl"],
                                   details={"definition": "share of docs whose (start,end,label) set in the J4 identity copy "
                                            "equals J1's output for the same doc (both re-runs, current offset code)", **d}))
    # robustness of baselines relative to the anchor (difference in mean degradation, joint bootstrap draws share seed)
    md_all = info.get("mean_deg", {})
    if "anchor" in md_all:
        for system in md_all:
            if system == "anchor":
                continue
            diff = md_all["anchor"]["all"]["samples"] - md_all[system]["all"]["samples"]
            val = md_all["anchor"]["all"]["value"] - md_all[system]["all"]["value"]
            s = stats.summarize_samples(diff, val, paired=True)
            recs.append(record(LAYER, f"anchor_vs_{system}", "diff_mean_degradation_exact_f1", val, n_units=len(base_ids),
                               unit="document", split="test", dataset=DATASET, direction="lower",
                               ci=(s["ci_low"], s["ci_high"]), ci_method="joint document bootstrap, same draws for both systems",
                               details={"definition": "anchor mean degradation - baseline mean degradation (negative = anchor more robust)",
                                        "p_boot": s["p"]}))
    for sysname in md_all:
        for name in md_all[sysname]:
            md_all[sysname][name].pop("samples", None)
    return recs, info


def determinism_check(fam_ids: Dict[str, Dict[str, str]], preds: Dict[str, Dict[str, Any]], base_ids: List[str]) -> Dict[str, Any]:
    j1 = load_preds(GPU / "anchor_test_logprobs/predictions.jsonl") if (GPU / "anchor_test_logprobs/DONE.json").exists() else None
    if j1 is None:
        return {"status": "J1 not finished"}
    same = diff = 0
    differing = []
    for b in base_ids:
        rid = fam_ids["identity"].get(b)
        if rid is None or b not in j1:
            continue
        a = {(p["start"], p["end"], p["label"]) for p in preds[rid]["predictions"]}
        c = {(p["start"], p["end"], p["label"]) for p in j1[b]["predictions"]}
        if a == c:
            same += 1
        else:
            diff += 1
            differing.append(b)
    n = same + diff
    return {"n_docs": n, "identical": same, "identical_rate": same / n if n else None, "differing_ids": differing[:20]}


# ----------------------------------------------------------------------------- natural boundary-case slice
COORD = {"and", "or", "/", "&", ",", "+"}


def slice_flags(doc: Dict[str, Any]) -> Dict[str, bool]:
    spans = doc["gold_spans"]
    text = doc["text"]
    coord = any(re.search(r"/| and | or | & ", s["surface_form"]) for s in spans)
    ss = sorted(spans, key=lambda s: (s["start"], s["end"]))
    for a, b in zip(ss, ss[1:]):
        between = text[a["end"]:b["start"]].strip().lower()
        if a["end"] <= b["start"] and between in COORD:
            coord = True
    paren = any("(" in s["surface_form"] or ")" in s["surface_form"] for s in spans) or any(
        text[max(0, s["start"] - 2):s["start"]].strip() == "(" for s in spans)
    nested = any(a is not b and a["start"] < b["end"] and b["start"] < a["end"] for a in spans for b in spans if a is not b)
    return {"coordination": coord, "parentheses": paren, "nested_overlap": nested, "any_boundary_case": coord or paren or nested}


def boundary_gold_subset(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gold spans that are themselves boundary cases (coordinated / parenthesised / overlapping)."""
    spans, text = doc["gold_spans"], doc["text"]
    out = []
    ss = sorted(spans, key=lambda s: (s["start"], s["end"]))
    adj = set()
    for a, b in zip(ss, ss[1:]):
        if a["end"] <= b["start"] and text[a["end"]:b["start"]].strip().lower() in COORD:
            adj.add(id(a)); adj.add(id(b))
    for s in spans:
        if (re.search(r"/| and | or | & |\(|\)", s["surface_form"]) or id(s) in adj
                or any(o is not s and o["start"] < s["end"] and s["start"] < o["end"] for o in spans)):
            out.append(s)
    return out


def natural_slice_systems() -> Dict[str, Path]:
    s = {
        "anchor": REPO_ROOT / "artifacts/skillllm_v2/mistral_lora/predictions.jsonl",
        "anchor_rerun_J1": GPU / "anchor_test_logprobs/predictions.jsonl",
        "safe_wrapper": REPO_ROOT / "artifacts/skillllm_v2/safe_wrapper/full_cag_graph/predictions.jsonl",
        "gliner_zeroshot": REPO_ROOT / "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl",
        "token_lookup": REPO_ROOT / "artifacts/skillllm_v2/bert_baseline/predictions.jsonl",
        "bert_fair": GPU / "bert_fair/predictions.jsonl",
    }
    return s


def analyse_natural_slice(n_boot: int, seed: int, smoke: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    docs = [d for d in load_skillspan_split(REPO_ROOT / "data/source/skillspan_hf", "test") if d["gold_spans"]]
    if smoke:
        docs = docs[:200]
    flags = {d["id"]: slice_flags(d) for d in docs}
    texts = {d["id"]: d["text"] for d in docs}
    gold = {d["id"]: d["gold_spans"] for d in docs}
    recs, table = [], []
    ds = "SkillSpan(HF) test, 974 gold-bearing docs: natural boundary-case slices (gold-defined)"
    for system, path in natural_slice_systems().items():
        preds = load_preds(path)
        if preds is None or (system == "bert_fair" and not (GPU / "bert_fair/SELECTED.json").exists()) or (
                system == "anchor_rerun_J1" and not (GPU / "anchor_test_logprobs/DONE.json").exists()):
            recs.append(record(LAYER, system, "boundary_slice_exact_f1", None, n_units=0, unit="document", split="test",
                               dataset=ds, direction="higher", status="pending_gpu", reason=f"missing {rel(path)}"))
            continue
        ids = [d["id"] for d in docs]
        mat = count_rows(ids, gold, texts, preds)
        for sl in ("coordination", "parentheses", "nested_overlap", "any_boundary_case"):
            m_in = np.array([flags[i][sl] for i in ids])
            if m_in.sum() == 0:
                continue
            A, B = mat[m_in], mat[~m_in]
            fn = stat_fn(PRIMARY)
            ra = stats.bootstrap_ci(A, fn, n_boot=n_boot, seed=seed)
            rb = stats.bootstrap_ci(B, fn, n_boot=n_boot, seed=seed + 1)
            sa = stats._boot_stats([A], [fn], n_boot, seed)[0]
            sb = stats._boot_stats([B], [fn], n_boot, seed + 1)[0]
            d = stats.summarize_samples(sa - sb, ra["value"] - rb["value"], paired=True)
            # recall on the boundary-case gold spans themselves (span subset; docs are the resampling unit)
            rows_r = []
            for k, i in enumerate(ids):
                if not m_in[k]:
                    continue
                sub = boundary_gold_subset({"gold_spans": gold[i], "text": texts[i]}) if sl == "any_boundary_case" else None
                if sub is None:
                    continue
                pk = {(p.get("start"), p.get("end"), p.get("label")) for p in preds.get(i, {}).get("predictions", [])}
                gk = {(s["start"], s["end"], s["label"]) for s in sub}
                rows_r.append([len(gk & pk), len(gk)])
            rec_sub = None
            if rows_r:
                rr = np.array(rows_r, float)
                rec_sub = stats.bootstrap_ci(rr, stats.ratio_stat(0, 1), n_boot=n_boot, seed=seed)
            row = {"system": system, "slice": sl, "n_docs_in": int(m_in.sum()), "n_docs_out": int((~m_in).sum()),
                   "f1_in": ra["value"], "f1_in_ci": (ra["ci_low"], ra["ci_high"]), "f1_out": rb["value"],
                   "delta_in_minus_out": d["value"], "delta_ci": (d["ci_low"], d["ci_high"]),
                   "mean_gold_span_words_in": float(np.mean([len(sp["surface_form"].split()) for k2, i2 in enumerate(ids) if m_in[k2] for sp in gold[i2]])),
                   "mean_gold_span_words_out": float(np.mean([len(sp["surface_form"].split()) for k2, i2 in enumerate(ids) if not m_in[k2] for sp in gold[i2]])),
                   "boundary_span_recall": rec_sub["value"] if rec_sub else None,
                   "boundary_span_recall_ci": (rec_sub["ci_low"], rec_sub["ci_high"]) if rec_sub else None}
            table.append(row)
            recs.append(record(LAYER, system, f"boundary_slice_exact_f1:{sl}", ra["value"], n_units=int(m_in.sum()),
                               unit="document", split="test", dataset=ds, direction="higher", ci=row["f1_in_ci"],
                               ci_method="document bootstrap", provenance=[path],
                               details={"slice_definition": SLICE_DEF[sl], "f1_outside_slice": rb["value"],
                                        "delta_in_minus_out": d["value"], "delta_ci": row["delta_ci"],
                                        "delta_ci_method": "independent bootstraps of the two disjoint doc sets",
                                        "boundary_span_recall": row["boundary_span_recall"],
                                        "mean_gold_span_words_in_vs_out": (row["mean_gold_span_words_in"], row["mean_gold_span_words_out"]),
                                        "confound_note": "boundary-case docs have longer gold spans; part of the gap is span length",
                                        "boundary_span_recall_ci": row["boundary_span_recall_ci"]},
                               caveats=(["stored predictions were aligned with the pre-1f8bd68 find_offsets (first substring match)"]
                                        if system == "anchor" else [])))
    return recs, table


SLICE_DEF = {
    "coordination": "doc has a gold span containing '/', ' and ', ' or ', ' & ', or two gold spans separated only by and/or/'/'/&/,/+",
    "parentheses": "doc has a gold span containing '(' or ')' or directly preceded by '('",
    "nested_overlap": "doc has two overlapping gold spans (SkillSpan skill and knowledge spans may overlap)",
    "any_boundary_case": "union of the three; boundary_span_recall = exact recall on those gold spans only",
}


# ----------------------------------------------------------------------------- cross-dataset (J6)
XDATA = {
    "green": {"path": "data/external/green_hf/test.json", "primary": {"SKILL"}, "secondary": {"SKILL", "DOMAIN", "QUALIFICATION"},
              "precision_valid": True},
    "sayfullina": {"path": "data/external/sayfullina_hf/test.json", "primary": {"SOFT"}, "secondary": None,
                   "precision_valid": False},
}
XDATA_N = 300


def build_xdata(seed: int = 42) -> Dict[str, Any]:
    from skillllm_v2.train.build_verifier_dataset import _spans_from_tags, _tokens_to_text
    IN_DIR.mkdir(parents=True, exist_ok=True)
    docs, golds, manifest = [], [], {"seed": seed, "n_per_dataset": XDATA_N, "datasets": {}}
    for name, cfg in XDATA.items():
        path = REPO_ROOT / cfg["path"]
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(rows)), min(XDATA_N, len(rows))))
        types = sorted({t.split("-", 1)[1] for r in rows for t in r["tags_skill"] if "-" in t})
        for i in idx:
            r = rows[i]
            text, offs = _tokens_to_text(r["tokens"])
            spans = []
            for ty in types:
                tags = [t if t.endswith("-" + ty) else "O" for t in r["tags_skill"]]
                spans += _spans_from_tags(text, offs, tags, ty)
            did = f"{name}-test-{i}"
            docs.append({"id": did, "text": text, "tokens": r["tokens"], "token_offsets": [list(o) for o in offs],
                         "meta": {"dataset": name, "split": "test", "row": i}})
            golds.append({"id": did, "dataset": name, "gold_spans": sorted(spans, key=lambda s: (s["start"], s["end"], s["label"]))})
        manifest["datasets"][name] = {"source": cfg["path"], "source_sha256": sha256_file(path), "n_rows": len(rows),
                                      "n_sampled": len(idx), "types": types}
    with open(IN_DIR / "xdata_docs.jsonl", "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(IN_DIR / "xdata_gold.jsonl", "w", encoding="utf-8") as fh:
        for g in golds:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    manifest["xdata_docs_sha256"] = sha256_file(IN_DIR / "xdata_docs.jsonl")
    manifest["xdata_gold_sha256"] = sha256_file(IN_DIR / "xdata_gold.jsonl")
    (IN_DIR / "XDATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def analyse_xdata(n_boot: int, seed: int, smoke: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    recs, table, per_unit = [], [], []
    gpath = IN_DIR / "xdata_gold.jsonl"
    if not gpath.exists():
        recs.append(record(LAYER, "anchor", "xdata_exact_f1", None, n_units=0, unit="document", split="test",
                           dataset="cross-dataset", direction="higher", status="pending_gpu", reason="xdata inputs not built"))
        return recs, table, per_unit
    golds = read_jsonl(gpath)
    docs = {d["id"]: d for d in read_jsonl(IN_DIR / "xdata_docs.jsonl")}
    for name, cfg in XDATA.items():
        g_ds = [g for g in golds if g["dataset"] == name][: (50 if smoke else None)]
        ids = [g["id"] for g in g_ds]
        texts = {i: docs[i]["text"] for i in ids}
        ds = f"{name} test (seeded sample n={len(ids)}), label-agnostic span match"
        for system in SYSTEMS:
            ppath = pred_path("xdata", system)
            preds = load_preds(ppath) if complete("xdata", system) else None
            if preds is None:
                recs.append(record(LAYER, system, f"xdata_exact_recall:{name}", None, n_units=len(ids), unit="document",
                                   split="test", dataset=ds, direction="higher",
                                   status="pending_gpu" if system in ("anchor", "safe_wrapper") else "not_available",
                                   reason=f"missing {rel(ppath)}"))
                continue
            for target, types in (("primary", cfg["primary"]), ("secondary", cfg["secondary"])):
                if not types:
                    continue
                gold = {g["id"]: [s for s in g["gold_spans"] if s["label"] in types] for g in g_ds}
                mat = count_rows(ids, gold, texts, preds, relabel="SKILL")
                metrics = ["exact_span_recall", "partial_span_recall"]
                if cfg["precision_valid"]:
                    metrics += ["exact_span_precision", "exact_span_f1", "partial_span_f1"]
                for m in metrics:
                    ci = stats.bootstrap_ci(mat, stat_fn(m), n_boot=n_boot, seed=seed)
                    table.append({"dataset": name, "target": target, "types": sorted(types), "system": system, "metric": m,
                                  "value": ci["value"], "ci": (ci["ci_low"], ci["ci_high"]), "n_docs": len(ids)})
                    recs.append(record(
                        LAYER, system, f"xdata_{m.replace('_span', '')}:{name}:{target}", ci["value"], n_units=len(ids),
                        unit="document", split="test", dataset=ds, direction="higher", ci=(ci["ci_low"], ci["ci_high"]),
                        ci_method="document bootstrap", provenance=[gpath, ppath],
                        caveats=XDATA_CAVEATS[name], details={"gold_types": sorted(types), "label_agnostic": True,
                                                              "out_of_distribution": True}))
                if target == "primary":
                    for k, i in enumerate(ids):
                        c = mat[k]
                        per_unit.append({"layer": LAYER, "family": f"xdata:{name}", "system": system, "metric": "exact_span_label_agnostic",
                                         "unit_id": i, "tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "value": doc_f1(c)})
            if not cfg["precision_valid"]:
                recs.append(record(LAYER, system, f"xdata_exact_precision:{name}", None, n_units=len(ids), unit="document",
                                   split="test", dataset=ds, direction="higher", status="not_available",
                                   reason="only one soft-skill span per sentence is annotated; other mentions are unlabelled, so precision/F1 are not measurable"))
    return recs, table, per_unit


XDATA_CAVEATS = {
    "green": ["annotation scheme differs from SkillSpan (SKILL also covers tasks; no knowledge/skill split); precision is "
              "attenuated by scheme mismatch, compare systems relative to each other",
              "labels ignored (any predicted skill span counts as SKILL)"],
    "sayfullina": ["sentences are lemmatised ('be look for ...'), one soft skill per sentence annotated: recall only",
                   "labels ignored"],
}


# ----------------------------------------------------------------------------- validity sample
def write_validity_sample(seed: int, out: Path, n: int = 50) -> Optional[Path]:
    pfile = IN_DIR / "perturbed_docs.jsonl"
    if not pfile.exists():
        return None
    rows = [r for r in read_jsonl(pfile) if r["meta"]["family"] != "identity"]
    by_f: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_f[r["meta"]["family"]].append(r)
    fams = sorted(by_f)
    rng = random.Random(seed)
    per = {f: n // len(fams) for f in fams}
    for f in rng.sample(fams, n - sum(per.values())):
        per[f] += 1
    picked = []
    for f in fams:
        pool = sorted(by_f[f], key=lambda r: r["id"])
        picked += rng.sample(pool, min(per[f], len(pool)))
    base = {}
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    for d in load_skillspan_split(REPO_ROOT / "data/source/skillspan_hf", "test"):
        base[d["id"]] = d
    out.mkdir(parents=True, exist_ok=True)
    path = out / "perturbation_validity_sample.csv"
    cols = ["item_id", "family", "base_id", "perturbed_id", "original_text", "perturbed_text", "gold_spans_original",
            "gold_spans_perturbed", "perturbation_note", "label_preserved_yes_no_unsure", "skills_still_skills_yes_no",
            "text_natural_1to5", "comment"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for k, r in enumerate(picked):
            b = base[r["meta"]["base_id"]]
            fmt = lambda spans: " | ".join(f"{s['surface_form']} [{s['label']}]" for s in spans)  # noqa: E731
            w.writerow({"item_id": f"L03-V{k + 1:03d}", "family": r["meta"]["family"], "base_id": b["id"], "perturbed_id": r["id"],
                        "original_text": b["text"], "perturbed_text": r["text"], "gold_spans_original": fmt(b["gold_spans"]),
                        "gold_spans_perturbed": fmt(r["gold_spans"]), "perturbation_note": r["meta"]["note"],
                        "label_preserved_yes_no_unsure": "", "skills_still_skills_yes_no": "", "text_natural_1to5": "", "comment": ""})
    return path


# ----------------------------------------------------------------------------- figures
def plot_family_deltas(rows: List[Dict[str, Any]], out: Path) -> Optional[Path]:
    rows = [r for r in rows if r["metric"] == PRIMARY]
    if not rows:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = sorted({r["family"] for r in rows})
    systems = [s for s in SYSTEMS if any(r["system"] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = 0.8 / max(1, len(systems))
    for j, s in enumerate(systems):
        xs, ys, lo, hi = [], [], [], []
        for k, f in enumerate(fams):
            r = next((x for x in rows if x["system"] == s and x["family"] == f), None)
            if r is None:
                continue
            xs.append(k + (j - (len(systems) - 1) / 2) * w)
            ys.append(r["delta"]); lo.append(r["delta"] - r["ci_low"]); hi.append(r["ci_high"] - r["delta"])
        ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, label=SYSTEM_LABEL[s])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=40, ha="right")
    ax.set_ylabel("Δ exact-span F1 (perturbed − identity)")
    ax.set_title("L03 robustness: exact-span F1 change per perturbation family (95% paired doc-bootstrap CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "family_deltas_exact_f1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- report
def fmt(v: Any, nd: int = 3) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def build_report(recs, info, slice_table, xtable, validity_path, fig) -> str:  # noqa: ANN001
    lines = ["# L03_robustness — perturbation robustness, boundary cases, cross-dataset", ""]
    head = next((r for r in recs if r["system"] == "anchor" and r["metric"] == "mean_degradation_exact_f1"), None)
    if head and head["status"] == "measured":
        lines += [f"**Headline:** anchor mean degradation of exact-span F1 over {len(head['details']['families'])} perturbation "
                  f"families = **{head['value']:.4f}** [95% CI {head['ci_low']:.4f}, {head['ci_high']:.4f}] "
                  f"(n = {head['n_units']} test docs; positive = worse than the identity copy).", ""]
    else:
        lines += [f"**Headline:** pending — {head['reason'] if head else 'no record'}", ""]
    m = info.get("manifest")
    if m:
        lines += ["## Design", "",
                  f"- {m['n_base_docs']} stratified test docs (seed {m['seed']}), {m['n_rows']} perturbed rows incl. identity copies; "
                  f"{m['n_not_available']} (doc, family) pairs not applicable (reasons in `inputs/evals/research/L03_robustness/not_available.jsonl`).",
                  f"- Rows per family: {m['rows_per_family']}",
                  f"- Alias sources: {m['alias_sources']}",
                  "- Clean reference = identity copy of the same doc scored in the same run; Δ = perturbed − clean; paired doc bootstrap "
                  "+ approximate randomization (10k), Holm across families per system/metric.",
                  "- Families marked † change the gold surface (case, typo inside span, alias): exact-match loss there partly "
                  "reflects a different target string, not only brittleness.", ""]
    fam_rows = [r for r in info.get("family_rows", []) if r["metric"] == PRIMARY]
    if fam_rows:
        lines += ["## Δ exact-span F1 per family", ""]
        tab = []
        for r in sorted(fam_rows, key=lambda x: (x["family"], SYSTEMS.index(x["system"]))):
            tab.append({"family": r["family"] + (" †" if r["family"] in SURFACE_CHANGING else ""), "system": r["system"],
                        "n": r["n_docs"], "clean": r["clean"], "perturbed": r["perturbed"], "delta": r["delta"],
                        "CI": f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]", "p_holm": r.get("p_holm"),
                        "parse_err": r["parse_errors"]})
        lines += [md_table(tab, ["family", "system", "n", "clean", "perturbed", "delta", "CI", "p_holm", "parse_err"]), ""]
        if fig:
            lines += [f"![family deltas]({fig.name})", ""]
    md = info.get("mean_deg", {})
    if md:
        lines += ["## Mean degradation per system", ""]
        tab = [{"system": s, "set": k, "mean_degradation": v["value"], "CI": f"[{fmt(v['ci_low'])}, {fmt(v['ci_high'])}]"}
               for s, d in md.items() for k, v in d.items()]
        lines += [md_table(tab, ["system", "set", "mean_degradation", "CI"]), ""]
    det = info.get("determinism")
    if det:
        lines += [f"Determinism control (J4 identity copies vs J1 re-run of the same docs): {det}", ""]
    if slice_table:
        lines += ["## Natural boundary cases (974-doc test subset, no synthetic edits)", "",
                  "Slices are defined on GOLD spans: " + "; ".join(f"**{k}** = {v}" for k, v in SLICE_DEF.items()), ""]
        tab = [{"system": r["system"], "slice": r["slice"], "n_in": r["n_docs_in"], "F1_in": r["f1_in"], "F1_out": r["f1_out"],
                "Δ in−out": r["delta_in_minus_out"], "Δ CI": f"[{fmt(r['delta_ci'][0])}, {fmt(r['delta_ci'][1])}]",
                "bnd_span_recall": r["boundary_span_recall"],
                "span_words in/out": f"{r['mean_gold_span_words_in']:.1f}/{r['mean_gold_span_words_out']:.1f}"} for r in slice_table]
        lines += [md_table(tab, ["system", "slice", "n_in", "F1_in", "F1_out", "Δ in−out", "Δ CI", "bnd_span_recall", "span_words in/out"]), "",
                  "Δ in−out compares disjoint document sets (independent bootstraps); boundary-case docs also have longer gold "
                  "spans, so the gap is not attributable to coordination/nesting alone.", ""]
    if xtable:
        lines += ["## Cross-dataset (out-of-distribution), label-agnostic", "",
                  "Green et al. 2022 (CC BY 4.0) test sample and Sayfullina et al. 2018 soft-skill test sample (recall only). "
                  "Different annotation guidelines: absolute numbers are NOT comparable with SkillSpan; compare systems.", "",
                  "**Label mapping (pre-registered, D-A3-05).** Scoring ignores predicted/gold labels: every predicted span "
                  "(HARD_SKILL or SOFT_SKILL) is relabelled SKILL and matched to the target gold spans. Green gold: *primary* "
                  "target = SKILL tags; *secondary* = SKILL + DOMAIN + QUALIFICATION (SkillSpan 'knowledge' partly covers domains "
                  "and degrees); OCCUPATION and EXPERIENCE are never targets. Sayfullina gold: the single annotated SOFT span per "
                  "sentence; only recall is valid (precision/F1 not_available). Samples: 300 sentences per dataset, seed 42; the "
                  "same docs for every system (fair BERT = dev-selected JobBERT trained on SkillSpan only).", ""]
        tab = [{"dataset": r["dataset"], "target": r["target"], "system": r["system"], "metric": r["metric"], "value": r["value"],
                "CI": f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}]", "n": r["n_docs"]} for r in xtable]
        lines += [md_table(tab, ["dataset", "target", "system", "metric", "value", "CI", "n"]), ""]
    pend = [r for r in recs if r["status"] != "measured"]
    if pend:
        lines += ["## Pending / not available", ""] + [f"- {r['system']} / {r['metric']}: {r['status']} — {r['reason']}" for r in pend] + [""]
    if validity_path:
        lines += [f"Human validity sample (50 items, for L11): `{validity_path.name}` — columns label_preserved / skills_still_skills / "
                  "text_natural are blank for annotators.", ""]
    lines += ["## Caveats", "",
              "- Perturbations are synthetic and rule-based; validity must be confirmed on the human sample before strong claims.",
              "- 120 docs: CIs for families that apply to few docs (alias_sub, reorder_clauses) are wide; read them as indicative.",
              "- The token-lookup baseline is the repository's stored 'BERT baseline' (memorised BIO tags), not a neural BERT.",
              "- The safe wrapper is recomposed (full_cag_graph composition, gate not applied) from the run_safe_wrapper code on top of "
              "the perturbed-input anchor predictions; its verifier/calibrator were not re-fit.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--build-xdata", action="store_true")
    args = ap.parse_args()
    if args.build_xdata:
        print(json.dumps(build_xdata(args.seed), indent=2))
        return
    n_boot = 200 if args.smoke else args.n_boot
    out = OUT_DIR / ("smoke" if args.smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    recs, info = analyse_perturbations(n_boot, args.seed, args.smoke)
    srecs, stable = analyse_natural_slice(n_boot, args.seed, args.smoke)
    xrecs, xtable, xunits = analyse_xdata(n_boot, args.seed, args.smoke)
    recs += srecs + xrecs
    validity = write_validity_sample(args.seed, out)
    fig = plot_family_deltas(info.get("family_rows", []), out)
    if info.get("family_rows"):
        with open(out / "family_deltas.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(info["family_rows"][0].keys()))
            w.writeheader()
            w.writerows(info["family_rows"])
    if stable:
        (out / "boundary_slices.json").write_text(json.dumps(stable, indent=2) + "\n")
    if xtable:
        (out / "xdata_results.json").write_text(json.dumps(xtable, indent=2) + "\n")
    with open(out / "per_unit_scores.jsonl", "w", encoding="utf-8") as fh:
        for r in info.get("per_unit", []) + xunits:
            fh.write(json.dumps(r) + "\n")
    # BH across every p in the layer (secondary)
    ps = [r["details"].get("p_ar") for r in recs]
    for r, q in zip(recs, stats.benjamini_hochberg(ps)):
        if q is not None:
            r["details"]["p_bh_layer"] = q
    report = build_report(recs, info, stable, xtable, validity, fig)
    write_layer(LAYER, recs, report, seed=args.seed, smoke=args.smoke,
                extra_meta={"n_boot": n_boot, "systems_available": info.get("systems_available", [])})
    print(f"wrote {out} ({len(recs)} records)")


if __name__ == "__main__":
    main()
