"""L02_generalization - seen vs unseen skill mentions on the SkillSpan test set (owner: A1).

Run (CPU, repo root)::

    PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.generalization [--smoke] [--seed 42] [--n-boot 2000]

Every TEST gold span is placed in exactly one bucket relative to the TRAIN gold spans (decision D-A1-06):
S1 exact_seen, S2 normalized_seen, S3 canonical_seen_surface_unseen (taxonomy maps it to a canonical id that a train
surface also maps to), S4 fully_unseen. Per bucket and system: gold-side exact recall (primary), prediction-side
precision with predictions bucketed by their own surface, bucket F1, sentence-level cluster-bootstrap CIs
(decision D-A1-07). Headline: unseen (S4) exact F1. H3: seen (S1+S2) minus unseen (S4) exact F1 for the anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from skillllm_v2.eval.extraction_eval import _key
from skillllm_v2.evals import extraction_evals as EE
from skillllm_v2.evals.research import common as C
from skillllm_v2.evals.research import extraction_full as XF
from skillllm_v2.schema import normalize_text
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

LAYER = "L02_generalization"
DATASET = XF.DATASET
SPLIT_SUB = XF.SPLIT_SUB
DATA_ROOT = "data/source/skillspan_hf"
ALIAS_TABLE = "data/skillllm_v2/taxonomy/skill_aliases.jsonl"
ALIAS_SOURCES_OK = ("manual_curated", "skillspan_gold_train_dev")
ALIAS_SOURCES_TEST_DERIVED = ("mistral_high_confidence",)
MANIFEST = "data/external/MANIFEST.jsonl"
BUCKETS = ("S1", "S2", "S3", "S4")
BUCKET_NAMES = {"S1": "exact_seen", "S2": "normalized_seen", "S3": "canonical_seen_surface_unseen", "S4": "fully_unseen"}
STRATA = ("0", "1", "2-5", ">5")
GROUPS = {"seen": ("S1", "S2"), "unseen": ("S4",), "S3": ("S3",)}
LABELS = XF.LABELS
H3_MIN_DELTA = 0.05
H3_RULE = ("H3 SUPPORTED iff, for the anchor, exact F1 on seen mentions (S1 exact_seen + S2 normalized_seen) minus exact "
           "F1 on fully unseen mentions (S4) is >= 0.05 AND its 95% paired sentence-bootstrap CI (10,000) excludes 0. "
           "S3 is reported separately.")
EDGE_CHARS = set(".,;:!?\"'`()[]{}<>*•·…–—-/\\|“”‘’«»")
NOTE_DEF = ("Bucket rules: decision D-A1-06 (Decisions.md). Gold-side recall = exact (start,end,label) matches / gold spans "
            "in the bucket; prediction-side precision = exact matches among de-duplicated predictions whose own surface "
            "falls in the bucket; F1 = harmonic mean (D-A1-07).")


# --------------------------------------------------------------------------------------------- normalisation
def norm_surface(s: str) -> str:
    """S2 normaliser (D-A1-06): NFKC, casefold, whitespace collapse, edge-punctuation strip, last-token plural rule."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    changed = True
    while s and changed:
        changed = False
        if s and s[-1] in EDGE_CHARS:
            s, changed = s[:-1].rstrip(), True
        if s and s[0] in EDGE_CHARS and s[0] != ".":
            s, changed = s[1:].lstrip(), True
    if not s:
        return s
    toks = s.split(" ")
    last = toks[-1]
    if len(last) >= 4:
        if last.endswith("ies"):
            last = last[:-3] + "y"
        elif last.endswith(("sses", "shes", "ches", "xes", "zes")):
            last = last[:-2]
        elif last.endswith("s") and not last.endswith(("ss", "us", "is")):
            last = last[:-1]
    toks[-1] = last
    return " ".join(toks)


def stratum(count: int) -> str:
    return "0" if count == 0 else "1" if count == 1 else "2-5" if count <= 5 else ">5"


# --------------------------------------------------------------------------------------------- taxonomies
def load_alias_index(root: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """normalize_text(alias) -> {'alias:<canonical id>'} from non-test-derived rows; plus an audit summary."""
    path = root / ALIAS_TABLE
    idx: Dict[str, Set[str]] = defaultdict(set)
    by_source: Counter = Counter()
    if not path.exists():
        return {}, {"available": False}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_source[r.get("source", "")] += 1
        if r.get("source") in ALIAS_SOURCES_OK:
            idx[r["normalized_alias"]].add("alias:" + r["canonical_skill_id"])
    return dict(idx), {"available": True, "rows_by_source": dict(by_source),
                       "test_derived_rows": sum(by_source.get(s, 0) for s in ALIAS_SOURCES_TEST_DERIVED),
                       "used_sources": list(ALIAS_SOURCES_OK)}


def esco_manifest_entry(root: Path) -> Optional[Dict[str, Any]]:
    p = root / MANIFEST
    if not p.exists():
        return None
    best = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "esco" in str(e.get("name", "")).lower() and any("skills_en" in f.get("path", "") for f in e.get("files", [])):
            best = e  # last complete entry wins
    return best


def load_esco_index(root: Path) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    entry = esco_manifest_entry(root)
    if entry is None:
        return {}, {"available": False, "reason": "no ESCO entry with skills_en in data/external/MANIFEST.jsonl yet (A4)"}
    files = [f for f in entry["files"] if f.get("path", "").endswith("skills_en.jsonl")]
    if not files:
        return {}, {"available": False, "reason": "ESCO manifest entry has no skills_en.jsonl"}
    f = sorted(files, key=lambda x: x["path"])[-1]
    path = root / "data/external" / f["path"]
    if not path.exists():
        return {}, {"available": False, "reason": f"ESCO file missing: {path}"}
    idx: Dict[str, Set[str]] = defaultdict(set)
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        uri = r.get("uri") or r.get("conceptUri")
        labels = [r.get("preferred_label") or r.get("preferredLabel")] + list(r.get("alt_labels") or r.get("altLabels") or [])
        n += 1
        for lab in labels:
            if lab:
                k = normalize_text(lab)
                if k:
                    idx[k].add("esco:" + str(uri))
    return dict(idx), {"available": True, "file": C.rel(path), "n_concepts": n, "manifest_name": entry.get("name"),
                       "version": entry.get("version_or_commit"), "sha256": f.get("sha256")}


# --------------------------------------------------------------------------------------------- bucketing
class Bucketer:
    def __init__(self, train_surfaces: Sequence[Tuple[str, str]], alias_idx: Dict[str, Set[str]],
                 esco_idx: Dict[str, Set[str]]):
        self.exact = {s for s, _ in train_surfaces}
        self.exact_label = {(s, lab) for s, lab in train_surfaces}
        self.norm_count = Counter(norm_surface(s) for s, _ in train_surfaces)
        self.alias_idx, self.esco_idx = alias_idx, esco_idx
        self.train_canon: Dict[str, Set[str]] = {"alias": set(), "esco": set()}
        for s in {s for s, _ in train_surfaces}:
            for src, ids in self._canon(s).items():
                self.train_canon[src] |= ids

    def _canon(self, surface: str) -> Dict[str, Set[str]]:
        k = normalize_text(surface)
        return {"alias": set(self.alias_idx.get(k, ())), "esco": set(self.esco_idx.get(k, ()))}

    def bucket(self, surface: str) -> Tuple[str, str]:
        """(bucket, S3 mapping source or '')."""
        if surface in self.exact:
            return "S1", ""
        if self.norm_count.get(norm_surface(surface), 0) > 0:
            return "S2", ""
        canon = self._canon(surface)
        hit = [src for src in ("alias", "esco") if canon[src] & self.train_canon[src]]
        if hit:
            return "S3", "+".join(hit)
        return "S4", ""

    def freq(self, surface: str) -> str:
        return stratum(self.norm_count.get(norm_surface(surface), 0))


def span_surface(sp: Dict[str, Any], text: str) -> str:
    if sp.get("start") is not None and sp.get("end") is not None:
        return text[sp["start"]:sp["end"]]
    return str(sp.get("surface_form") or "")


# column layout: per bucket [gold_n, gold_tp, pred_n, pred_tp]; then per stratum the same; then per label x {seen, unseen}
COLS_PER = 4
OFF_BUCKET = 0
OFF_STRATA = OFF_BUCKET + len(BUCKETS) * COLS_PER
OFF_LABEL = OFF_STRATA + len(STRATA) * COLS_PER
LABEL_GROUPS = [(lab, g) for lab in LABELS for g in ("seen", "unseen")]
N_COLS = OFF_LABEL + len(LABEL_GROUPS) * COLS_PER
LEN_BANDS = ("1", "2", "3+")


def len_band(surface: str) -> str:
    n = len(surface.split())
    return "1" if n <= 1 else "2" if n == 2 else "3+"


def length_rows(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], text: str, B: "Bucketer",
                cache: Dict[str, Tuple[str, str, str]]) -> np.ndarray:
    """Gold-side [n, tp] per (seen|unseen) x word-length band: the length-confounder check for H3 (descriptive)."""
    row = np.zeros(2 * len(LEN_BANDS) * 2)
    pkeys = {_key(p) for p in XF.dedup(pred)}
    for g in gold:
        surf = span_surface(g, text)
        if surf not in cache:
            b, src = B.bucket(surf)
            cache[surf] = (b, src, B.freq(surf))
        b = cache[surf][0]
        grp = 0 if b in GROUPS["seen"] else 1 if b in GROUPS["unseen"] else None
        if grp is None:
            continue
        off = (grp * len(LEN_BANDS) + LEN_BANDS.index(len_band(surf))) * 2
        row[off] += 1
        row[off + 1] += 1.0 if _key(g) in pkeys else 0.0
    return row


def doc_rows(gold: Sequence[Dict[str, Any]], pred: Sequence[Dict[str, Any]], text: str, B: Bucketer,
             cache: Dict[str, Tuple[str, str, str]]) -> np.ndarray:
    def info(surface: str) -> Tuple[str, str, str]:
        if surface not in cache:
            b, src = B.bucket(surface)
            cache[surface] = (b, src, B.freq(surface))
        return cache[surface]

    row = np.zeros(N_COLS)
    preds = XF.dedup(pred)
    pkeys, gkeys = {_key(p) for p in preds}, {_key(g) for g in gold}
    for side, items, other in (("gold", gold, pkeys), ("pred", preds, gkeys)):
        base = 0 if side == "gold" else 2
        for sp in items:
            b, _, fr = info(span_surface(sp, text))
            hit = 1.0 if _key(sp) in other else 0.0
            for off in (OFF_BUCKET + BUCKETS.index(b) * COLS_PER, OFF_STRATA + STRATA.index(fr) * COLS_PER):
                row[off + base] += 1
                row[off + base + 1] += hit
            grp = "seen" if b in GROUPS["seen"] else "unseen" if b in GROUPS["unseen"] else None
            if grp and sp.get("label") in LABELS:
                off = OFF_LABEL + LABEL_GROUPS.index((sp["label"], grp)) * COLS_PER
                row[off + base] += 1
                row[off + base + 1] += hit
    return row


def _div(a: Any, b: Any) -> np.ndarray:
    return XF._div(a, b)


def group_cols(S: np.ndarray, offsets: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gn = sum(S[..., o] for o in offsets)
    gt = sum(S[..., o + 1] for o in offsets)
    pn = sum(S[..., o + 2] for o in offsets)
    pt = sum(S[..., o + 3] for o in offsets)
    return gn, gt, pn, pt


def prf_from(S: np.ndarray, offsets: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gn, gt, pn, pt = group_cols(S, offsets)
    r, p = _div(gt, gn), _div(pt, pn)
    return p, r, XF._f1(p, r)


def bucket_offsets(buckets: Sequence[str]) -> List[int]:
    return [OFF_BUCKET + BUCKETS.index(b) * COLS_PER for b in buckets]


def stat_table() -> Dict[str, Any]:
    t: Dict[str, Any] = {}
    targets = {**{b: bucket_offsets([b]) for b in BUCKETS}, **{g: bucket_offsets(bs) for g, bs in GROUPS.items()},
               **{f"freq_{s}": [OFF_STRATA + STRATA.index(s) * COLS_PER] for s in STRATA},
               **{f"{lab}_{g}": [OFF_LABEL + LABEL_GROUPS.index((lab, g)) * COLS_PER] for lab, g in LABEL_GROUPS}}
    for name, offs in targets.items():
        for k, nm in enumerate(("precision", "recall", "f1")):
            t[f"{name}_{nm}"] = (lambda S, offs=offs, k=k: prf_from(S, offs)[k])
    return t


STATS = stat_table()


def _p_two_sided(samples: np.ndarray) -> float:
    d = samples[np.isfinite(samples)]
    b = len(d)
    if b == 0:
        return float("nan")
    return float(min(1.0, 2 * min((1 + (d <= 0).sum()) / (b + 1), (1 + (d >= 0).sum()) / (b + 1))))


# --------------------------------------------------------------------------------------------- overlap stats
def overlap_stats(train: Sequence[Tuple[str, str]], other: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    tr_exact = {s for s, _ in train}
    tr_norm = {norm_surface(s) for s, _ in train}
    tr_norm_lab = {(norm_surface(s), lab) for s, lab in train}
    ot_norm = [norm_surface(s) for s, _ in other]
    mention_exact = sum(1 for s, _ in other if s in tr_exact) / max(1, len(other))
    mention_norm = sum(1 for n in ot_norm if n in tr_norm) / max(1, len(other))
    mention_norm_lab = sum(1 for (s, lab) in other if (norm_surface(s), lab) in tr_norm_lab) / max(1, len(other))
    T, R = set(ot_norm), tr_norm
    Te, Re = {s for s, _ in other}, tr_exact
    return {"n_mentions": len(other), "mention_share_exact_seen": mention_exact,
            "mention_share_normalized_seen": mention_norm, "mention_share_normalized_seen_same_label": mention_norm_lab,
            "n_types_normalized": len(T), "n_train_types_normalized": len(R),
            "type_share_seen_normalized": len(T & R) / max(1, len(T)), "type_jaccard_normalized": len(T & R) / max(1, len(T | R)),
            "type_share_seen_exact": len(Te & Re) / max(1, len(Te)), "type_jaccard_exact": len(Te & Re) / max(1, len(Te | Re))}


# --------------------------------------------------------------------------------------------- figures
MAIN_FIG_SYSTEMS = ("mistral_lora", "safe_wrapper", "qlora_loop_v4", "qlora_loop_v1", "bert_baseline", "gliner_baseline")


def plot_buckets(rows: List[Dict[str, Any]], out: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = out / "fig_f1_by_bucket.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "bucket", "f1", "f1_ci_low", "f1_ci_high", "recall", "recall_ci_low", "recall_ci_high",
                    "precision", "n_gold", "n_pred"])
        for r in rows:
            w.writerow([r["system"], r["bucket"], r["f1"], r["f1_ci"][0], r["f1_ci"][1], r["recall"], r["recall_ci"][0],
                        r["recall_ci"][1], r["precision"], r["n_gold"], r["n_pred"]])
    systems = [s for s in MAIN_FIG_SYSTEMS if any(r["system"] == s for r in rows)]
    buckets = list(BUCKETS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    width = 0.8 / max(1, len(systems))
    colors = plt.get_cmap("tab10")
    for ax, key in zip(axes, ("f1", "recall")):
        for j, s in enumerate(systems):
            vals, lo, hi = [], [], []
            for b in buckets:
                r = next((x for x in rows if x["system"] == s and x["bucket"] == b), None)
                v = r[key] if r else np.nan
                ci = r[f"{key}_ci"] if r else (np.nan, np.nan)
                vals.append(v)
                lo.append(v - ci[0] if ci[0] is not None else 0)
                hi.append(ci[1] - v if ci[1] is not None else 0)
            x = np.arange(len(buckets)) + (j - (len(systems) - 1) / 2) * width
            ax.bar(x, vals, width, yerr=[lo, hi], capsize=2, label=XF.display(s), color=colors(j % 10), edgecolor="white", linewidth=0.4)
        ax.set_xticks(np.arange(len(buckets)), [f"{b}\n{BUCKET_NAMES[b]}" for b in buckets], fontsize=8)
        ax.set_title("exact F1 (gold recall x bucketed precision)" if key == "f1" else "gold-side exact recall")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score (95% sentence-bootstrap CI)")
    axes[0].set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Seen vs unseen skill mentions (SkillSpan test, 974 sentences; buckets vs train gold)", fontsize=10)
    fig.tight_layout()
    png = out / "fig_f1_by_bucket.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return [C.rel(png), C.rel(csv_path)]


def plot_strata(rows: List[Dict[str, Any]], out: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = out / "fig_recall_by_train_frequency.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "train_frequency", "recall", "ci_low", "ci_high", "n_gold"])
        for r in rows:
            w.writerow([r["system"], r["stratum"], r["recall"], r["ci"][0], r["ci"][1], r["n_gold"]])
    systems = [s for s in MAIN_FIG_SYSTEMS if any(r["system"] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, s in enumerate(systems):
        rr = [next(x for x in rows if x["system"] == s and x["stratum"] == st) for st in STRATA]
        v = np.array([x["recall"] for x in rr])
        lo = np.array([x["ci"][0] for x in rr])
        hi = np.array([x["ci"][1] for x in rr])
        ax.errorbar(np.arange(len(STRATA)) + 0.05 * j, v, yerr=[v - lo, hi - v], marker="o", capsize=3, label=XF.display(s))
    ax.set_xticks(np.arange(len(STRATA)), [f"train count {s}" for s in STRATA], fontsize=8)
    ax.set_ylabel("gold-side exact recall")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Recall by train frequency of the normalised surface", fontsize=10)
    fig.tight_layout()
    png = out / "fig_recall_by_train_frequency.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return [C.rel(png), C.rel(csv_path)]


# --------------------------------------------------------------------------------------------- main
def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def gold_surfaces(docs: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    return [(d["text"][g["start"]:g["end"]], g["label"]) for d in docs for g in d["gold_spans"]]


def run(seed: int = 42, n_boot: int = 2000, smoke: bool = False, root: Path = C.REPO_ROOT,
        out_root: Path = C.EVALS_ROOT) -> Path:
    data = root / DATA_ROOT
    train_docs = load_skillspan_split(data, "train")
    dev_docs = load_skillspan_split(data, "dev")
    gold = XF.load_gold_all(root)
    all_ids = sorted(gold, key=XF._sent_index)
    sub_ids = [i for i in all_ids if gold[i]["gold_spans"]]
    if smoke:
        sub_ids = sub_ids[:60]
        n_boot = min(n_boot, 200)
    out_dir = out_root / LAYER / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    postings = [XF.posting_id(gold[i]) for i in sub_ids]

    train_s = gold_surfaces(train_docs)
    dev_s = gold_surfaces(dev_docs)
    test_s = gold_surfaces(gold[i] for i in sub_ids)
    alias_idx, alias_audit = load_alias_index(root)
    esco_idx, esco_info = load_esco_index(root)
    B = Bucketer(train_s, alias_idx, esco_idx)
    cache: Dict[str, Tuple[str, str, str]] = {}
    prov_data = [f"{DATA_ROOT}/train.json", f"{DATA_ROOT}/dev.json", XF.GOLD_REL, ALIAS_TABLE]
    if esco_info.get("available"):
        prov_data.append(esco_info["file"])

    records: List[Dict[str, Any]] = []
    rec = lambda *a, **k: records.append(C.record(LAYER, *a, **k))  # noqa: E731

    # ---------------------------------------------------------------- bucket sizes + overlap
    sizes: Counter = Counter()
    s3_src: Counter = Counter()
    lab_by_bucket: Dict[str, Counter] = defaultdict(Counter)
    len_by_bucket: Dict[str, List[int]] = defaultdict(list)
    strata_sizes: Counter = Counter()
    exact_label_seen = 0
    for s, lab in test_s:
        b, src = B.bucket(s)
        sizes[b] += 1
        if src:
            s3_src[src] += 1
        lab_by_bucket[b][lab] += 1
        len_by_bucket[b].append(len(s.split()))
        strata_sizes[B.freq(s)] += 1
        exact_label_seen += (s, lab) in B.exact_label
    n_test = len(test_s)
    for b in BUCKETS:
        rec("dataset", f"bucket_size_{b}_{BUCKET_NAMES[b]}", float(sizes[b]), n_units=len(sub_ids), unit="sentence",
            split=SPLIT_SUB, dataset=DATASET, direction="none", provenance=prov_data,
            details={"share_of_test_gold_mentions": sizes[b] / max(1, n_test), "n_test_gold_mentions": n_test,
                     "label_counts": dict(lab_by_bucket[b]),
                     "mean_span_words": float(np.mean(len_by_bucket[b])) if len_by_bucket[b] else None,
                     "s3_mapping_sources": dict(s3_src) if b == "S3" else None, "definition": NOTE_DEF})
    for st in STRATA:
        rec("dataset", f"frequency_stratum_size_{st}", float(strata_sizes[st]), n_units=len(sub_ids), unit="sentence",
            split=SPLIT_SUB, dataset=DATASET, direction="none", provenance=prov_data,
            details={"share": strata_sizes[st] / max(1, n_test), "definition": "train mention count of norm(surface)"})
    ov_test = overlap_stats(train_s, test_s)
    ov_dev = overlap_stats(train_s, dev_s)
    ov_test["mention_share_exact_seen_same_label"] = exact_label_seen / max(1, n_test)
    for name, ov, split in (("train_test", ov_test, SPLIT_SUB), ("train_dev", ov_dev, "dev[all gold spans]")):
        for k in ("mention_share_exact_seen", "mention_share_normalized_seen", "type_share_seen_normalized",
                  "type_jaccard_normalized", "type_jaccard_exact"):
            rec("dataset", f"{name}_{k}", float(ov[k]), n_units=ov["n_mentions"], unit="mention", split=split,
                dataset=DATASET, direction="none", provenance=prov_data, details=ov)
    rec("dataset", "alias_table_test_derived_rows", float(alias_audit.get("test_derived_rows", 0)),
        n_units=sum(alias_audit.get("rows_by_source", {}).values()), unit="alias_row", split="taxonomy",
        dataset=ALIAS_TABLE, direction="lower", provenance=[ALIAS_TABLE, "src/skillllm_v2/normalize/skill_alias_builder.py",
                                                             "scripts/build_offline_aliases.py"],
        caveats=["Rows with source mistral_high_confidence are mined from the anchor's TEST predictions "
                 "(skill_alias_builder.py default input artifacts/skillllm_v2/mistral_lora/predictions.jsonl); excluded "
                 "from S3. scripts/build_offline_aliases.py writes a different file (src/skillllm_v2/resources/"
                 "offline_aliases.jsonl) from hand-curated lists only and uses no test data."],
        details=alias_audit)
    if esco_info.get("available"):
        rec("dataset", "esco_concepts_loaded", float(esco_info["n_concepts"]), n_units=esco_info["n_concepts"],
            unit="concept", split="taxonomy", dataset="ESCO", direction="none", provenance=[esco_info["file"]], details=esco_info)
    else:
        rec("dataset", "esco_concepts_loaded", None, n_units=0, unit="concept", split="taxonomy", dataset="ESCO",
            direction="none", status="not_available", reason=esco_info.get("reason", "ESCO not available"))

    # ---------------------------------------------------------------- per system
    gpu_specs, _ = XF.discover_gpu_systems(root)
    specs = [s for s in XF.SYSTEMS if (root / s["pred"]).exists()] + gpu_specs
    bucket_rows: List[Dict[str, Any]] = []
    strata_rows: List[Dict[str, Any]] = []
    per_unit: List[Dict[str, Any]] = []
    h3 = None
    summary: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        sid = spec["id"]
        preds = EE.load_predictions(root / spec["pred"])
        rows = np.array([doc_rows(gold[i]["gold_spans"], preds.get(i, []), gold[i]["text"], B, cache) for i in sub_ids])
        nb = 10000 if (sid == XF.ANCHOR and not smoke) else n_boot
        lrows = np.array([length_rows(gold[i]["gold_spans"], preds.get(i, []), gold[i]["text"], B, cache) for i in sub_ids])
        lstats = {}
        for bi, band in enumerate(LEN_BANDS):
            so, uo = bi * 2, (len(LEN_BANDS) + bi) * 2
            lstats[f"seen_len{band}"] = lambda S, o=so: _div(S[..., o + 1], S[..., o])
            lstats[f"unseen_len{band}"] = lambda S, o=uo: _div(S[..., o + 1], S[..., o])
            lstats[f"delta_len{band}"] = lambda S, a=so, b=uo: _div(S[..., a + 1], S[..., a]) - _div(S[..., b + 1], S[..., b])
        lcis = XF.boot_multi(lrows, lstats, nb, seed)
        LS = lrows.sum(axis=0)
        length_check = {}
        for bi, band in enumerate(LEN_BANDS):
            so, uo = bi * 2, (len(LEN_BANDS) + bi) * 2
            length_check[band] = {"seen_recall": float(lstats[f"seen_len{band}"](LS)), "seen_n": int(LS[so]),
                                  "unseen_recall": float(lstats[f"unseen_len{band}"](LS)), "unseen_n": int(LS[uo]),
                                  "delta": float(lstats[f"delta_len{band}"](LS)), "delta_ci": list(lcis[f"delta_len{band}"])}
            if LS[so] > 0 and LS[uo] > 0:
                rec(sid, f"delta_seen_minus_unseen_recall_len{band}words", length_check[band]["delta"],
                    n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB, dataset=DATASET, direction="none",
                    ci=lcis[f"delta_len{band}"], ci_method=f"paired sentence bootstrap, {nb} resamples",
                    provenance=[spec["pred"]] + prov_data,
                    details={**length_check[band], "purpose": "length-confounder check for H3 (gold-side recall, "
                                                              "spans of the same word length)"})
        stats = dict(STATS)
        stats["delta_seen_minus_unseen_f1"] = lambda S: STATS["seen_f1"](S) - STATS["unseen_f1"](S)
        stats["delta_seen_minus_unseen_recall"] = lambda S: STATS["seen_recall"](S) - STATS["unseen_recall"](S)
        cis, samples = XF.boot_multi(rows, stats, nb, seed, return_samples=True)
        post = XF.boot_multi(XF.cluster_rows(rows, postings), {k: stats[k] for k in
                             ("unseen_f1", "seen_f1", "delta_seen_minus_unseen_f1")}, nb, seed)
        S = rows.sum(axis=0)
        vals = {k: float(fn(S)) for k, fn in stats.items()}
        prov = [spec["pred"]] + prov_data
        summary[sid] = {"vals": vals, "cis": cis, "post": post, "role": spec["role"], "length_check": length_check}
        cav = list(XF.BASE_CAVEATS) + ["Seen and unseen mentions differ in label mix and length (see bucket_size "
                                       "details); bucket differences are descriptive, not causal."]
        for name in list(BUCKETS) + list(GROUPS):
            offs = bucket_offsets(GROUPS[name]) if name in GROUPS else bucket_offsets([name])
            gn, gt, pn, pt = (float(x) for x in group_cols(S, offs))
            label = BUCKET_NAMES.get(name, name)
            for nm in ("recall", "precision", "f1"):
                metric = f"{name}_{label}_exact_{nm}" if name in BUCKETS else f"{name}_exact_{nm}"
                det = {"n_gold_spans": int(gn), "n_gold_tp": int(gt), "n_pred_spans": int(pn), "n_pred_tp": int(pt),
                       "definition": NOTE_DEF, "primary": nm == "recall", "n_boot": nb}
                if name == "unseen" and nm == "f1":
                    metric = "unseen_exact_f1"
                    det.update({"headline": sid == XF.ANCHOR, "bucket": "S4 fully_unseen",
                                "ci_posting": list(post["unseen_f1"])})
                if name == "seen" and nm == "f1":
                    det["ci_posting"] = list(post["seen_f1"])
                undefined = (nm == "recall" and gn == 0) or (nm == "precision" and pn == 0) or (nm == "f1" and (gn == 0 or pn == 0))
                if undefined:
                    vals[f"{name}_{nm}"] = float("nan")
                    rec(sid, metric, None, n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB, dataset=DATASET,
                        direction="higher", status="not_available", provenance=prov, details=det,
                        reason=f"undefined: {int(gn)} gold and {int(pn)} predicted mentions in this bucket")
                    continue
                rec(sid, metric, vals[f"{name}_{nm}"], n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB,
                    dataset=DATASET, direction="higher", ci=cis[f"{name}_{nm}"],
                    ci_method=f"sentence-level cluster percentile bootstrap, {nb} resamples, seed {seed}",
                    provenance=prov, caveats=cav, details=det)
            if name in BUCKETS:
                nanci = lambda k: cis[k] if np.isfinite(vals[k]) else (float("nan"), float("nan"))  # noqa: E731
                bucket_rows.append({"system": sid, "bucket": name, "f1": vals[f"{name}_f1"], "f1_ci": nanci(f"{name}_f1"),
                                    "recall": vals[f"{name}_recall"], "recall_ci": nanci(f"{name}_recall"),
                                    "precision": vals[f"{name}_precision"], "n_gold": int(gn), "n_pred": int(pn)})
        for st in STRATA:
            off = [OFF_STRATA + STRATA.index(st) * COLS_PER]
            gn, gt, pn, pt = (float(x) for x in group_cols(S, off))
            rec(sid, f"train_freq_{st}_exact_recall", vals[f"freq_{st}_recall"], n_units=len(sub_ids), unit="sentence",
                split=SPLIT_SUB, dataset=DATASET, direction="higher", ci=cis[f"freq_{st}_recall"],
                ci_method=f"sentence bootstrap, {nb} resamples", provenance=prov,
                details={"n_gold_spans": int(gn), "precision": vals[f"freq_{st}_precision"], "f1": vals[f"freq_{st}_f1"],
                         "n_pred_spans": int(pn)})
            strata_rows.append({"system": sid, "stratum": st, "recall": vals[f"freq_{st}_recall"],
                                "ci": cis[f"freq_{st}_recall"], "n_gold": int(gn)})
        for lab, g in LABEL_GROUPS:
            rec(sid, f"{lab}_{g}_exact_f1", vals[f"{lab}_{g}_f1"], n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB,
                dataset=DATASET, direction="higher", ci=cis[f"{lab}_{g}_f1"], ci_method=f"sentence bootstrap, {nb} resamples",
                provenance=prov, details={"recall": vals[f"{lab}_{g}_recall"], "precision": vals[f"{lab}_{g}_precision"],
                                          "purpose": "confounder check: seen/unseen gap within one label"})
        d = samples["delta_seen_minus_unseen_f1"]
        rec(sid, "delta_seen_minus_unseen_exact_f1", vals["delta_seen_minus_unseen_f1"], n_units=len(sub_ids),
            unit="sentence", split=SPLIT_SUB, dataset=DATASET, direction="none", ci=cis["delta_seen_minus_unseen_f1"],
            ci_method=f"paired sentence bootstrap (same draws for both groups), {nb} resamples", provenance=prov,
            details={"p_boot": _p_two_sided(d), "ci_posting": list(post["delta_seen_minus_unseen_f1"]),
                     "delta_recall": vals["delta_seen_minus_unseen_recall"],
                     "delta_recall_ci": list(cis["delta_seen_minus_unseen_recall"])})
        if sid == XF.ANCHOR:
            delta = vals["delta_seen_minus_unseen_f1"]
            ci = cis["delta_seen_minus_unseen_f1"]
            supported = delta >= H3_MIN_DELTA and ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
            h3 = {"delta": delta, "ci": ci, "p": _p_two_sided(d), "supported": supported,
                  "seen_f1": vals["seen_f1"], "unseen_f1": vals["unseen_f1"], "s3_f1": vals["S3_f1"],
                  "ci_posting": post["delta_seen_minus_unseen_f1"], "length_check": length_check, "per_label": {
                      lab: {"seen_f1": vals[f"{lab}_seen_f1"], "unseen_f1": vals[f"{lab}_unseen_f1"]} for lab in LABELS}}
            rec(sid, "H3_verdict", delta, n_units=len(sub_ids), unit="sentence", split=SPLIT_SUB, dataset=DATASET,
                direction="none", ci=ci, ci_method=f"paired sentence bootstrap, {nb} resamples, seed {seed}",
                provenance=prov, caveats=cav,
                details={"verdict": "SUPPORTED" if supported else "NOT SUPPORTED", "rule": H3_RULE, "p_raw": h3["p"],
                         "p_holm": h3["p"], "family": "H3_seen_vs_unseen (single test; Holm = raw)",
                         "seen_exact_f1": vals["seen_f1"], "unseen_exact_f1": vals["unseen_f1"],
                         "s3_exact_f1": vals["S3_f1"], "ci_posting": list(post["delta_seen_minus_unseen_f1"]),
                         "per_label": h3["per_label"], "length_check_recall": length_check,
                         "esco_used": bool(esco_info.get("available"))})
        for i, did in enumerate(sub_ids):
            for b in BUCKETS:
                o = OFF_BUCKET + BUCKETS.index(b) * COLS_PER
                r = rows[i, o:o + 4]
                if r.any():
                    per_unit.append({"layer": LAYER, "family": "bucket_counts", "system": sid, "metric": "bucket_counts",
                                     "unit_id": did, "posting_id": postings[i], "population": "subset974",
                                     "label": "ALL", "bucket": b, "gold_n": int(r[0]),
                                     "gold_tp": int(r[1]), "pred_n": int(r[2]), "pred_tp": int(r[3])})
    if h3 is None:
        rec(XF.ANCHOR, "H3_verdict", None, n_units=0, unit="sentence", split=SPLIT_SUB, dataset=DATASET,
            direction="none", status="not_available", reason="anchor predictions missing", details={"rule": H3_RULE})

    with open(out_dir / "per_unit_scores.jsonl", "w") as fh:
        for r in per_unit:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    figs = plot_buckets(bucket_rows, out_dir) + plot_strata(strata_rows, out_dir) if bucket_rows else []
    report = build_report(sizes, n_test, lab_by_bucket, len_by_bucket, s3_src, strata_sizes, ov_test, ov_dev,
                          alias_audit, esco_info, summary, h3, figs, len(sub_ids), n_boot, seed)
    C.write_layer(LAYER, records, report, seed=seed, smoke=smoke, out_root=out_root,
                  extra_meta={"n_boot": n_boot, "n_boot_anchor": 10000, "esco": esco_info,
                              "systems": [s["id"] for s in specs]})
    return out_dir


def build_report(sizes, n_test, lab_by_bucket, len_by_bucket, s3_src, strata_sizes, ov_test, ov_dev, alias_audit,
                 esco_info, summary, h3, figs, n_sub, n_boot, seed) -> str:
    L: List[str] = []
    L.append("# L02 - Generalisation to unseen skill mentions (SkillSpan test vs train)\n")
    L.append(f"Owner A1. Module `src/skillllm_v2/evals/research/generalization.py`; rules D-A1-06/07 in `Decisions.md`. "
             f"{n_sub} test sentences, {n_test} gold mentions; CIs = 95% sentence-level cluster bootstrap (seed {seed}; "
             f"10,000 for the anchor, {n_boot} otherwise); `ci_posting` in results.json = 65-posting cluster bootstrap.\n")
    L.append("## Buckets (test gold mentions vs train gold spans)\n")
    L.append("| bucket | mentions | share | HARD | SOFT | mean words |")
    L.append("|---|---|---|---|---|---|")
    for b in BUCKETS:
        lens = len_by_bucket.get(b) or []
        L.append(f"| {b} {BUCKET_NAMES[b]} | {sizes[b]} | {sizes[b] / max(1, n_test):.3f} | {lab_by_bucket[b].get('HARD_SKILL', 0)} | "
                 f"{lab_by_bucket[b].get('SOFT_SKILL', 0)} | {np.mean(lens) if lens else float('nan'):.2f} |")
    L.append(f"\nS3 mapping sources: {dict(s3_src) or 'none'}. ESCO: "
             + (f"used ({esco_info.get('n_concepts')} concepts, {esco_info.get('file')})" if esco_info.get("available")
                else f"not used ({esco_info.get('reason')})") + ".")
    L.append(f"Train-frequency strata of test mentions (normalised surface): "
             + ", ".join(f"{s}: {strata_sizes[s]}" for s in STRATA) + ".\n")
    L.append("### Train/test and train/dev overlap\n")
    L.append("| pair | mentions | mention share seen (exact) | mention share seen (normalised) | type share seen (norm.) | type Jaccard (norm.) | type Jaccard (exact) |")
    L.append("|---|---|---|---|---|---|---|")
    for name, ov in (("train-test (974 subset)", ov_test), ("train-dev", ov_dev)):
        L.append(f"| {name} | {ov['n_mentions']} | {ov['mention_share_exact_seen']:.3f} | {ov['mention_share_normalized_seen']:.3f} | "
                 f"{ov['type_share_seen_normalized']:.3f} | {ov['type_jaccard_normalized']:.3f} | {ov['type_jaccard_exact']:.3f} |")
    L.append(f"\nLabel-aware exact-seen share (test): {ov_test.get('mention_share_exact_seen_same_label', float('nan')):.3f}; "
             f"normalised + same label: {ov_test['mention_share_normalized_seen_same_label']:.3f}.\n")
    L.append("### Leakage audit of the taxonomy\n")
    L.append(f"- `{ALIAS_TABLE}` rows by source: {alias_audit.get('rows_by_source')}. The "
             f"{alias_audit.get('test_derived_rows')} `mistral_high_confidence` rows are mined from the anchor's TEST "
             "predictions (`src/skillllm_v2/normalize/skill_alias_builder.py`), i.e. transductive use of test inputs; they "
             "are excluded from S3. `scripts/build_offline_aliases.py` builds a different file "
             "(`src/skillllm_v2/resources/offline_aliases.jsonl`) from hand-curated lists only (no SkillSpan or prediction data).")
    L.append("- 601 rows come from SkillSpan train+dev gold spans (count >= 2): using them for S3 is not test leakage.\n")
    L.append("## Exact F1 by bucket (gold recall x bucketed precision)\n")
    L.append("| system | S1 F1 [CI] | S2 F1 [CI] | S3 F1 [CI] | S4 unseen F1 [CI] | seen (S1+S2) F1 | Δ seen-unseen [CI] |")
    L.append("|---|---|---|---|---|---|---|")
    for sid, s in summary.items():
        v, c = s["vals"], s["cis"]
        cell = lambda k: (f"{_fmt(v[k])} [{_fmt(c[k][0])}, {_fmt(c[k][1])}]" if np.isfinite(v[k]) else "n/a")  # noqa: E731
        L.append(f"| {sid} | {cell('S1_f1')} | {cell('S2_f1')} | {cell('S3_f1')} | {cell('S4_f1')} | {_fmt(v['seen_f1'])} | "
                 f"{cell('delta_seen_minus_unseen_f1')} |")
    L.append("\n## Gold-side exact recall by bucket (primary)\n")
    L.append("| system | S1 | S2 | S3 | S4 | train freq 1 | 2-5 | >5 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for sid, s in summary.items():
        v = s["vals"]
        L.append(f"| {sid} | {_fmt(v['S1_recall'])} | {_fmt(v['S2_recall'])} | {_fmt(v['S3_recall'])} | {_fmt(v['S4_recall'])} | "
                 f"{_fmt(v['freq_1_recall'])} | {_fmt(v['freq_2-5_recall'])} | {_fmt(v['freq_>5_recall'])} |")
    L.append("\n## H3 - seen vs unseen (anchor, pre-registered)\n")
    L.append(f"Rule: {H3_RULE}\n")
    if h3:
        L.append(f"- seen (S1+S2) exact F1 {h3['seen_f1']:.4f}; unseen (S4) exact F1 {h3['unseen_f1']:.4f}; S3 {h3['s3_f1']:.4f}.")
        L.append(f"- Δ = {h3['delta']:.4f}, 95% CI [{h3['ci'][0]:.4f}, {h3['ci'][1]:.4f}], p = {h3['p']:.4g}; posting-level CI "
                 f"[{h3['ci_posting'][0]:.4f}, {h3['ci_posting'][1]:.4f}].")
        L.append("- Within-label check: " + "; ".join(f"{lab}: seen {d['seen_f1']:.4f} vs unseen {d['unseen_f1']:.4f}"
                                                   for lab, d in h3["per_label"].items()) + ".")
        L.append("- Length check (gold-side recall, same word length): " + "; ".join(
            f"{band} word(s): seen {d['seen_recall']:.3f} (n={d['seen_n']}) vs unseen {d['unseen_recall']:.3f} (n={d['unseen_n']}), "
            f"Δ {d['delta']:.3f} [{d['delta_ci'][0]:.3f}, {d['delta_ci'][1]:.3f}]" for band, d in h3["length_check"].items()) + ".")
        L.append(f"\n**H3 verdict: {'SUPPORTED' if h3['supported'] else 'NOT SUPPORTED'}.**\n")
    L.append("## Files\n")
    L.append("- `results.json`, `per_unit_scores.jsonl` (sentence x system x bucket counts), " + ", ".join(f"`{Path(f).name}`" for f in figs))
    L.append("\n## Caveats\n")
    L.append("- Buckets are lexical: an unseen surface may still be a paraphrase of a seen skill; a seen surface may be "
             "annotated differently in context. S3 coverage depends on the taxonomy and is small without ESCO.")
    L.append("- Unseen mentions are longer and more often SOFT_SKILL, so part of the gap may reflect span length / label "
             "difficulty rather than memorisation (see the within-label check).")
    L.append("- Predictions are bucketed by their own surface; a false positive with an unseen surface lowers unseen precision.")
    for c in XF.BASE_CAVEATS:
        L.append(f"- {c}")
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
