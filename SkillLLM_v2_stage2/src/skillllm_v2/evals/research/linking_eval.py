"""L06 taxonomy linking: audit of the legacy "Linking Accuracy" and a gold-based evaluation against ESCO.

Gold: Decorte et al. (2022) TECH / HOUSE sets (HF ``TechWolf/skill-extraction-{tech,house}``, CC-BY-4.0): every
SkillSpan *test* skill span of the TECH and HOUSE postings was manually labelled with the most specific ESCO v1.1.0
skill, or ``LABEL NOT PRESENT`` (no fitting ESCO skill) / ``UNDERSPECIFIED`` (span is not a skill). Their
``validation`` file (20 %) is used for every threshold; their ``test`` file (80 %) is reporting-only.

Taxonomy: ESCO v1.1.0 skills pillar (the version the gold uses), English preferred + alternative labels, fetched
from the official ESCO REST API (``data/external/esco/v1.1.0``; v1.2.1 = latest is also downloaded for diagnostics).

Linkers (all rank ESCO concepts for a mention surface, context-free, top-10):
* ``random``            seeded random concepts (negative control).
* ``exact``             normalised surface == normalised ESCO preferred/alternative label; otherwise NIL.
* ``bm25``              BM25 (k1 1.5, b 0.75, not tuned) over preferred + alternative labels of each concept.
* ``dense_minilm``      all-MiniLM-L6-v2 cosine to the nearest label of each concept (max over labels).
* ``cascade_esco``      the repository's own ``TaxonomyLinker`` cascade (alias -> normalised -> fuzzy>=0.88 -> BM25)
                         loaded with ESCO as its taxonomy; ranked list replayed by ``evals.linking_evals.rank_candidates``.
* ``shipped_bridge``    the linker as shipped (696-entry repo alias table) + a deterministic label bridge from repo
                         canonical entries to ESCO URIs (exact normalised label match, built WITHOUT any gold).
NIL (abstention) for the score-based linkers uses a threshold tuned on Decorte *validation* (maximises NIL-aware
accuracy); "@never" variants never abstain. Nothing is tuned on test.

Run: ``python -m skillllm_v2.evals.research.linking_eval [--smoke] [--seed 42] [--n-boot 2000]``.
Dense embeddings are computed once with ``--embed`` in the GPU/torch venv (CPU is fine) and cached under
``data/external/esco/<version>/derived/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from skillllm_v2.evals.research.common import (EVALS_ROOT, REPO_ROOT, md_table, provenance_entry, record, rel,
                                               write_layer)

LAYER = "L06_linking"
ESCO_VERSION = "v1.1.0"  # version the Decorte gold was annotated with
ESCO_LATEST = "v1.2.1"
ESCO_ROOT = REPO_ROOT / "data" / "external" / "esco"
DECORTE_ROOT = REPO_ROOT / "data" / "external" / "decorte_skill_extraction"
SKILLSPAN_ROOT = REPO_ROOT / "data" / "source" / "skillspan_hf"
REPO_ALIAS_TABLE = REPO_ROOT / "data" / "skillllm_v2" / "taxonomy" / "skill_aliases.jsonl"
PRED_LINKED = REPO_ROOT / "artifacts" / "skillllm_v2" / "safe_wrapper" / "linked_skills.jsonl"
DERIVED_INPUTS = REPO_ROOT / "inputs" / "evals" / "research" / LAYER
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
NIL_MARKERS = {"LABEL NOT PRESENT": "not_present", "UNDERSPECIFIED": "underspecified"}
K = 10
KS = (1, 3, 5, 10)
SUBSETS = ("tech", "house")
DATASET = "Decorte et al. 2022 TECH+HOUSE (SkillSpan test spans -> ESCO v1.1.0)"

AUDIT_FILES = [
    "reports/skillllm_v2/money_table.md", "reports/skillllm_v2/full_cag_graph_metrics.json",
    "reports/skillllm_v2/mistral_gliner_taxonomy_metrics.json", "reports/skillllm_v2/linking_eval.csv",
    "reports/skillllm_v2/linking_eval.md", "reports/skillllm_v2/linking_error_analysis.md",
    "data/skillllm_v2/linker_train.jsonl", "data/skillllm_v2/linker_dev.jsonl",
    "data/skillllm_v2/taxonomy/skill_aliases.jsonl", "src/skillllm_v2/pipeline/smoke_runner.py",
    "src/skillllm_v2/train/build_linker_dataset.py", "src/skillllm_v2/train/train_taxonomy_reranker.py",
    "src/skillllm_v2/evals/linking_evals.py", "scripts/build_offline_aliases.py",
]


def _norm(s: str) -> str:
    from skillllm_v2.schema import normalize_text
    return normalize_text(s)


def _terms(s: str) -> List[str]:
    return re.findall(r"[a-z0-9+#.-]+", _norm(s))


# ============================================================================================ data
def esco_dir(version: str = ESCO_VERSION) -> Path:
    return ESCO_ROOT / version


def load_esco(version: str = ESCO_VERSION) -> List[Dict[str, Any]]:
    path = esco_dir(version) / "skills_en.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["labels"] = list(dict.fromkeys([r["preferred_label"]] + [a for a in r.get("alt_labels", []) if a]))
    return sorted(rows, key=lambda r: r["uri"])


def label_index(concepts: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """normalised preferred label -> uris, normalised alternative label -> uris."""
    pref: Dict[str, List[str]] = defaultdict(list)
    alt: Dict[str, List[str]] = defaultdict(list)
    for c in concepts:
        pref[_norm(c["preferred_label"])].append(c["uri"])
        for a in c.get("alt_labels", []):
            alt[_norm(a)].append(c["uri"])
    return dict(pref), dict(alt)


def load_skillspan(split: str) -> List[Dict[str, Any]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    return load_skillspan_split(SKILLSPAN_ROOT, split, limit=None)


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_decorte(split: str, subsets: Sequence[str] = SUBSETS, test_docs: Optional[List[Dict[str, Any]]] = None,
                 concepts: Optional[List[Dict[str, Any]]] = None, train_surfaces: Optional[set] = None,
                 repo_aliases: Optional[set] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Mentions (one per (subset, sentence, span)) with gold ESCO uris, aligned to SkillSpan test doc ids."""
    test_docs = test_docs if test_docs is not None else load_skillspan("test")
    concepts = concepts if concepts is not None else load_esco()
    by_text: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in test_docs:
        by_text[_ws(d["text"])].append(d)
    pref_exact: Dict[str, List[str]] = defaultdict(list)
    for c in concepts:
        pref_exact[c["preferred_label"]].append(c["uri"])
    pref_n, alt_n = label_index(concepts)
    stats: Counter = Counter()
    unresolved: Counter = Counter()
    mentions: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for sub in subsets:
        path = DECORTE_ROOT / sub / f"{split}.csv"
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        stats[f"rows_{sub}"] += len(rows)
        for r in rows:
            key = (sub, r["sentence"], r["span"])
            m = mentions.get(key)
            if m is None:
                m = {"subset": sub, "split": split, "sentence": r["sentence"], "span": r["span"],
                     "gold_labels": set(), "gold_uris": set(), "nil_kinds": set(), "sub_spans": set(), "n_rows": 0}
                mentions[key] = m
            m["n_rows"] += 1
            if r.get("sub_span"):
                m["sub_spans"].add(r["sub_span"])
            lab = r["label"].strip()
            if lab in NIL_MARKERS:
                m["nil_kinds"].add(NIL_MARKERS[lab])
                continue
            m["gold_labels"].add(lab)
            uris = pref_exact.get(lab) or pref_n.get(_norm(lab)) or alt_n.get(_norm(lab)) or []
            how = "pref_exact" if pref_exact.get(lab) else "pref_norm" if pref_n.get(_norm(lab)) else "alt_norm" if uris else "unresolved"
            stats[f"label_resolution_{how}"] += 1
            if not uris:
                unresolved[lab] += 1
            m["gold_uris"].update(uris)
    out: List[Dict[str, Any]] = []
    for i, (key, m) in enumerate(sorted(mentions.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2]))):
        docs = by_text.get(_ws(m["sentence"]), [])
        same_src = [d for d in docs if d["source"] == m["subset"]] or docs
        doc = same_src[0] if same_src else None
        m["skillspan_ids"] = [d["id"] for d in docs]
        m["doc_id"] = doc["id"] if doc else None
        m["cluster"] = f"{doc['source']}:{doc['document_id']}" if doc else f"unaligned:{_ws(m['sentence'])[:40]}"
        stats["aligned" if doc else "unaligned"] += 1
        hard_soft = None
        if doc:
            text = doc["text"]
            starts = [mm.start() for mm in re.finditer(re.escape(m["span"]), text)] or \
                     [mm.start() for mm in re.finditer(re.escape(m["span"]), text, flags=re.I)]
            labs = set()
            gold_pos = {(g["start"], g["end"]): g for g in doc["gold_spans"]}
            for s in starts:
                e = s + len(m["span"])
                for g in doc["gold_spans"]:
                    if g["start"] == s and g["end"] == e:
                        labs.add(g["label"])
            if not labs:  # fall back to overlap with any gold span
                for s in starts:
                    e = s + len(m["span"])
                    for (gs, ge), g in gold_pos.items():
                        if max(gs, s) < min(ge, e):
                            labs.add(g["label"])
                stats["label_by_overlap" if labs else "label_unknown"] += 1
            hard_soft = "BOTH" if len(labs) > 1 else (next(iter(labs)) if labs else "UNKNOWN")
            m["offsets"] = [[s, s + len(m["span"])] for s in starts]
        m["label"] = hard_soft or "UNKNOWN"
        m["linkable"] = bool(m["gold_uris"])
        m["nil"] = not m["linkable"] and not m["gold_labels"]
        m["nil_kind"] = None if m["linkable"] else ("underspecified" if "underspecified" in m["nil_kinds"] else "not_present")
        m["partial_nil"] = m["linkable"] and bool(m["nil_kinds"])
        ns = _norm(m["span"])
        m["seen"] = ns in train_surfaces if train_surfaces is not None else None
        m["in_repo_alias_table"] = ns in repo_aliases if repo_aliases is not None else None
        m["mention_id"] = f"{split}:{m['subset']}:{hashlib.sha1((m['sentence'] + chr(0) + m['span']).encode()).hexdigest()[:12]}"
        m["gold_labels"], m["gold_uris"] = sorted(m["gold_labels"]), sorted(m["gold_uris"])
        m["nil_kinds"], m["sub_spans"] = sorted(m["nil_kinds"]), sorted(m["sub_spans"])
        out.append(m)
    stats["mentions"] = len(out)
    stats["linkable"] = sum(m["linkable"] for m in out)
    stats["nil"] = sum(m["nil"] for m in out)
    stats["gold_label_unresolved"] = sum(1 for m in out if m["gold_labels"] and not m["gold_uris"])
    return out, {"counts": dict(stats), "unresolved_labels": dict(unresolved.most_common(20))}


def train_gold_surfaces() -> set:
    return {_norm(g["surface_form"]) for d in load_skillspan("train") for g in d["gold_spans"]}


def repo_alias_surfaces() -> set:
    rows = [json.loads(l) for l in REPO_ALIAS_TABLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["normalized_alias"] for r in rows} | {_norm(r["canonical_skill_name"]) for r in rows}


def assign_l02_buckets(mentions: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    """Tag each mention with the L02 bucket (S1 exact seen / S2 normalised seen / S3 canonical seen / S4 fully unseen).

    Uses A1's ``generalization.Bucketer`` unchanged (same train surfaces, same alias + ESCO indices), applied to the span
    surface as it appears in the sentence, so the seen/unseen strata of L06 are the L02 strata (D-B3-01)."""
    from skillllm_v2.evals.research import generalization as G
    train_docs = load_skillspan("train")
    alias_idx, _ = G.load_alias_index(REPO_ROOT)
    esco_idx, esco_info = G.load_esco_index(REPO_ROOT)
    B = G.Bucketer(G.gold_surfaces(train_docs), alias_idx, esco_idx)
    cnt: Counter = Counter()
    for m in mentions:
        b, _src = B.bucket(m["span"])
        m["l02_bucket"] = b
        cnt[b] += 1
    return {"bucket_counts": dict(cnt), "esco_index": esco_info}


# ============================================================================================ linkers
class RandomLinker:
    name = "random"

    def __init__(self, concepts: Sequence[Dict[str, Any]], seed: int) -> None:
        self.uris = [c["uri"] for c in concepts]
        self.seed = seed

    def rank(self, mention: Dict[str, Any]) -> Tuple[List[Tuple[str, float]], Optional[float], str]:
        rng = random.Random(f"{self.seed}:{mention['mention_id']}")
        return [(u, 0.0) for u in rng.sample(self.uris, K)], 0.0, "random"


class AlwaysNilLinker:
    """Trivial NIL baseline: abstains on everything (NIL F1 floor given the NIL prevalence)."""
    name = "always_nil"

    def rank(self, mention: Dict[str, Any]):
        return [], None, "none"


class ExactLinker:
    name = "exact"

    def __init__(self, concepts: Sequence[Dict[str, Any]]) -> None:
        self.pref, self.alt = label_index(concepts)

    def rank(self, mention: Dict[str, Any]):
        n = _norm(mention["span"])
        out: List[Tuple[str, float]] = []
        for u in sorted(self.pref.get(n, [])):
            out.append((u, 1.0))
        for u in sorted(self.alt.get(n, [])):
            if u not in {x for x, _ in out}:
                out.append((u, 0.9))
        return out[:K], (out[0][1] if out else None), ("pref" if out and out[0][1] == 1.0 else "alt" if out else "none")


class BM25Linker:
    name = "bm25"

    def __init__(self, concepts: Sequence[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.uris = [c["uri"] for c in concepts]
        docs = [_terms(" ".join(c["labels"])) for c in concepts]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        self.k1, self.b = k1, b
        self.post: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, tf in Counter(d).items():
                self.post[t].append((i, tf))
        self.n = len(docs)

    def scores(self, text: str) -> np.ndarray:
        s = np.zeros(self.n)
        for t in _terms(text):
            plist = self.post.get(t)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            idx = np.fromiter((i for i, _ in plist), dtype=int, count=df)
            tf = np.fromiter((f for _, f in plist), dtype=float, count=df)
            s[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len[idx] / self.avg))
        return s

    def rank(self, mention: Dict[str, Any]):
        s = self.scores(mention["span"])
        if not s.any():
            return [], None, "none"
        top = np.lexsort((np.arange(self.n), -s))[:K]
        out = [(self.uris[i], float(s[i])) for i in top if s[i] > 0]
        return out, out[0][1], "bm25"


class DenseLinker:
    name = "dense_minilm"

    def __init__(self, concepts: Sequence[Dict[str, Any]], cache: Dict[str, Any]) -> None:
        self.uris = [c["uri"] for c in concepts]
        uri_pos = {u: i for i, u in enumerate(self.uris)}
        lab_uri = cache["label_uris"]
        order = np.argsort(np.array([uri_pos[u] for u in lab_uri]), kind="stable")
        self.L = cache["label_vecs"][order]
        owner = np.array([uri_pos[lab_uri[i]] for i in order])
        self.starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        self.owner = owner[self.starts]
        self.q = cache["query_vecs"]

    def rank(self, mention: Dict[str, Any]):
        v = self.q.get(mention["span"])
        if v is None:
            return [], None, "missing_embedding"
        sims = self.L @ v
        best = np.maximum.reduceat(sims, self.starts)
        top = np.lexsort((self.owner, -best))[:K]
        out = [(self.uris[self.owner[i]], float(best[i])) for i in top]
        return out, out[0][1], "dense"


def esco_taxonomy_file(concepts: Sequence[Dict[str, Any]], version: str = ESCO_VERSION) -> Path:
    """ESCO in the legacy taxonomy-file format TaxonomyLinker reads (id, name, aliases)."""
    path = esco_dir(version) / "derived" / "taxonomy_linker_esco.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"canonical_skill_id": c["uri"], "canonical_skill_name": c["preferred_label"],
                         "aliases": c["labels"], "taxonomy": "ESCO"}, ensure_ascii=False) for c in concepts]
    body = "\n".join(lines) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != body:
        path.write_text(body, encoding="utf-8")
    return path


_POOL_LINKER = None


def _pool_rank(surface: str):
    from skillllm_v2.evals.linking_evals import rank_candidates
    r = rank_candidates(_POOL_LINKER, surface)
    return surface, [(c["canonical_skill_id"], c["confidence"], c["method"]) for c in r[:K]]


def _pool_link(surface: str):
    out = _POOL_LINKER.link(surface)
    return surface, (out["canonical_skill_id"], out["link_confidence"], out["link_method"]) if out else None


def _cached_rank(cache_path: Path, meta: Dict[str, Any], surfaces: Sequence[str], compute) -> Dict[str, Any]:
    """Surface -> ranked list cache (JSON); only missing surfaces are computed. Invalidated when ``meta`` changes."""
    store: Dict[str, Any] = {}
    if cache_path.exists():
        try:
            d = json.loads(cache_path.read_text(encoding="utf-8"))
            if d.get("meta") == meta:
                store = d["ranked"]
        except (ValueError, KeyError):
            store = {}
    missing = sorted(set(surfaces) - set(store))
    if missing:
        store.update({k: [list(x) for x in v] for k, v in compute(missing).items()})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"meta": meta, "ranked": store}, ensure_ascii=False), encoding="utf-8")
    return {k: [tuple(x) for x in store[k]] for k in set(surfaces)}


def cascade_rank_all(linker: Any, surfaces: Sequence[str], workers: int = 8) -> Dict[str, List[Tuple[str, float, str]]]:
    """rank_candidates for many surfaces in a fork pool (the fuzzy stage is O(#aliases) per surface)."""
    global _POOL_LINKER
    _POOL_LINKER = linker
    uniq = sorted(set(surfaces))
    if workers <= 1 or len(uniq) < 20:
        return dict(_pool_rank(s) for s in uniq)
    with mp.get_context("fork").Pool(workers) as pool:
        return dict(pool.map(_pool_rank, uniq, chunksize=4))


def cascade_link_all(linker: Any, surfaces: Sequence[str], workers: int = 8) -> Dict[str, Any]:
    global _POOL_LINKER
    _POOL_LINKER = linker
    uniq = sorted(set(surfaces))
    if workers <= 1 or len(uniq) < 20:
        return dict(_pool_link(s) for s in uniq)
    with mp.get_context("fork").Pool(workers) as pool:
        return dict(pool.map(_pool_link, uniq, chunksize=2))


class PrecomputedLinker:
    """Wraps precomputed ranked lists {surface: [(uri, conf, method)]}."""

    def __init__(self, name: str, ranked: Dict[str, List[Tuple[str, float, str]]]) -> None:
        self.name, self.ranked = name, ranked

    def rank(self, mention: Dict[str, Any]):
        r = self.ranked.get(mention["span"], [])
        return [(u, c) for u, c, _ in r], (r[0][1] if r else None), (r[0][2] if r else "none")


def build_bridge(concepts: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Repo canonical id -> ESCO uris by exact normalised label match (no gold used).

    For each repo entry: (1) canonical name == ESCO preferred label, (2) canonical name == ESCO alt label,
    (3) any repo alias == ESCO preferred label, (4) any repo alias == ESCO alt label. First non-empty stage wins;
    uris inside a stage are sorted for determinism.
    """
    from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker
    pref, alt = label_index(concepts)
    linker = TaxonomyLinker(alias_table_paths=[str(REPO_ALIAS_TABLE)])
    bridge: Dict[str, List[str]] = {}
    how: Counter = Counter()
    rows = []
    for e in linker.entries:
        name = _norm(e["canonical_skill_name"])
        aliases = sorted(e["aliases_norm"])
        stages = [("name_pref", pref.get(name, [])), ("name_alt", alt.get(name, [])),
                  ("alias_pref", sorted({u for a in aliases for u in pref.get(a, [])})),
                  ("alias_alt", sorted({u for a in aliases for u in alt.get(a, [])}))]
        uris, stage = [], "unmapped"
        for st, us in stages:
            if us:
                uris, stage = sorted(set(us)), st
                break
        bridge[e["canonical_skill_id"]] = uris
        how[stage] += 1
        rows.append({"canonical_skill_id": e["canonical_skill_id"], "canonical_skill_name": e["canonical_skill_name"],
                     "esco_uris": uris, "stage": stage})
    DERIVED_INPUTS.mkdir(parents=True, exist_ok=True)
    (DERIVED_INPUTS / "repo_to_esco_bridge.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return bridge, {"entries": len(bridge), "stage_counts": dict(how),
                    "mapped_share": 1 - how["unmapped"] / max(1, len(bridge)), "multi_uri_entries": sum(len(v) > 1 for v in bridge.values())}


def shipped_ranked(surfaces: Sequence[str], bridge: Dict[str, List[str]]) -> Tuple[Dict[str, List[Tuple[str, float, str]]], Dict[str, Any]]:
    from skillllm_v2.evals.linking_evals import rank_candidates
    from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker
    linker = TaxonomyLinker(alias_table_paths=[str(REPO_ALIAS_TABLE)])
    out: Dict[str, List[Tuple[str, float, str]]] = {}
    diag: Counter = Counter()
    for s in sorted(set(surfaces)):
        top = linker.link(s)
        ranked = rank_candidates(linker, s)
        if top is None:
            diag["repo_unlinked"] += 1
            out[s] = []
            continue
        diag["repo_linked"] += 1
        if ranked and ranked[0]["canonical_skill_id"] != top["canonical_skill_id"]:
            diag["rank_inconsistent"] += 1
        if not bridge.get(top["canonical_skill_id"]):
            diag["top1_unbridgeable"] += 1
            out[s] = []  # the shipped output cannot be expressed in ESCO -> treated as NIL
            continue
        lst, seen = [], set()
        for c in ranked:
            for u in bridge.get(c["canonical_skill_id"], []):
                if u not in seen:
                    seen.add(u)
                    lst.append((u, float(c["confidence"]), c["method"]))
        out[s] = lst[:K]
    return out, dict(diag)


# ============================================================================================ embeddings
def embedding_cache_path(version: str = ESCO_VERSION) -> Path:
    return esco_dir(version) / "derived" / "minilm_cache.npz"


def embed_main(version: str = ESCO_VERSION, device: str = "cpu", batch: int = 512) -> None:
    """Run in the torch venv: mean-pooled, L2-normalised all-MiniLM-L6-v2 vectors (= sentence-transformers recipe)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    concepts = load_esco(version)
    label_texts, label_uris = [], []
    for c in concepts:
        for lab in c["labels"]:
            label_texts.append(lab)
            label_uris.append(c["uri"])
    queries = set()
    for split in ("validation", "test"):
        for sub in SUBSETS:
            with open(DECORTE_ROOT / sub / f"{split}.csv", encoding="utf-8", newline="") as fh:
                queries.update(r["span"] for r in csv.DictReader(fh))
    if PRED_LINKED.exists():
        for line in PRED_LINKED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                queries.update(s["surface_form"] for s in json.loads(line)["linked_skills"])
    queries = sorted(queries)
    tok = AutoTokenizer.from_pretrained(MINILM)
    model = AutoModel.from_pretrained(MINILM).to(device).eval()

    def enc(texts: List[str]) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                h = model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                outs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy().astype(np.float32))
        return np.concatenate(outs) if outs else np.zeros((0, 384), np.float32)

    torch.manual_seed(0)
    lv, qv = enc(label_texts), enc(queries)
    path = embedding_cache_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, label_vecs=lv, label_uris=np.array(label_uris), label_texts=np.array(label_texts),
                        query_vecs=qv, queries=np.array(queries), model=np.array(MINILM))
    print(json.dumps({"labels": len(label_texts), "queries": len(queries), "path": str(path)}))


def load_embedding_cache(version: str = ESCO_VERSION) -> Optional[Dict[str, Any]]:
    path = embedding_cache_path(version)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {"label_vecs": z["label_vecs"], "label_uris": [str(u) for u in z["label_uris"]],
            "query_vecs": {str(q): z["query_vecs"][i] for i, q in enumerate(z["queries"])}, "path": path}


# ============================================================================================ scoring
def score_mentions(mentions: Sequence[Dict[str, Any]], linker: Any, tau: Optional[float]) -> List[Dict[str, Any]]:
    """Per-mention outcome. Abstain if no candidate, or top-1 score < tau (tau None = never abstain on a candidate)."""
    out = []
    for m in mentions:
        ranked, s1, method = linker.rank(m)
        uris = [u for u, _ in ranked]
        abstain = (not uris) or (tau is not None and s1 is not None and s1 < tau)
        gold = set(m["gold_uris"])
        row = {"mention_id": m["mention_id"], "cluster": m["cluster"], "linkable": m["linkable"], "nil": m["nil"],
               "top": uris[:5], "s1": s1, "method": method, "abstain": abstain}
        if m["linkable"]:
            ranks = [i for i, u in enumerate(uris) if u in gold]
            r = ranks[0] + 1 if ranks else None
            for k in KS:
                row[f"hit@{k}"] = float(r is not None and r <= k)
            row["rr"] = 1.0 / r if r else 0.0
            row["rp@5"] = sum(1 for u in uris[:5] if u in gold) / min(5, len(gold))
            row["e2e_correct"] = float(not abstain and r == 1)
        else:
            row["e2e_correct"] = float(abstain)
        out.append(row)
    return out


def tune_tau(mentions: Sequence[Dict[str, Any]], linker: Any) -> Dict[str, Any]:
    """Pick the abstention threshold maximising NIL-aware accuracy on (validation) mentions; ties -> lowest tau."""
    base = score_mentions(mentions, linker, None)
    s1 = sorted({r["s1"] for r in base if r["s1"] is not None})
    grid = [None] + s1
    best = None
    for tau in grid:
        acc = 0.0
        for r in base:
            ab = r["abstain"] or (tau is not None and r["s1"] is not None and r["s1"] < tau)
            if r["linkable"]:
                acc += float(not ab and r.get("hit@1", 0.0) == 1.0)
            else:
                acc += float(ab)
        acc /= max(1, len(base))
        if best is None or acc > best[1] + 1e-12:
            best = (tau, acc)
    never = sum(r["e2e_correct"] for r in base) / max(1, len(base))
    return {"tau": best[0], "val_nil_aware_acc": best[1], "val_nil_aware_acc_never_abstain": never, "grid_size": len(grid)}


def cluster_rows(rows: Sequence[Dict[str, Any]], num: str, sel=lambda r: True) -> Tuple[np.ndarray, List[str]]:
    agg: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        if sel(r):
            agg[r["cluster"]][0] += r[num]
            agg[r["cluster"]][1] += 1
    keys = sorted(agg)
    return np.array([agg[k] for k in keys], dtype=float).reshape(-1, 2), keys


def nil_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    agg: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in rows:
        a = agg[r["cluster"]]
        if r["nil"] and r["abstain"]:
            a[0] += 1
        elif r["abstain"] and r["linkable"]:
            a[1] += 1
        elif r["nil"] and not r["abstain"]:
            a[2] += 1
    keys = sorted({r["cluster"] for r in rows})
    return np.array([agg[k] for k in keys], dtype=float).reshape(-1, 3), keys


# ============================================================================================ audit
def audit_legacy() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Legacy 'Linking Accuracy' (money table ~0.890): what it is computed from."""
    recs = []
    rows = [json.loads(l) for l in REPO_ALIAS_TABLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    src = Counter(r["source"] for r in rows)
    smoke = (REPO_ROOT / "src/skillllm_v2/pipeline/smoke_runner.py").read_text(encoding="utf-8")
    legacy_is_coverage = 'metrics["linking_accuracy"] = linked_count / total' in smoke and 'metrics["linking_accuracy"] = linked / total' in smoke
    linker_train = [json.loads(l) for l in (REPO_ROOT / "data/skillllm_v2/linker_train.jsonl").read_text().splitlines() if l.strip()]
    fc = json.loads((REPO_ROOT / "reports/skillllm_v2/full_cag_graph_metrics.json").read_text())
    facts = {
        "legacy_value_full_cag_graph": fc.get("linking_accuracy"),
        "computed_as": "pipeline/smoke_runner.py: linked_count / total over PREDICTED spans (share with any non-null canonical id)",
        "legacy_is_coverage_not_accuracy": legacy_is_coverage,
        "gold_ids_used": False,
        "alias_table_rows": len(rows), "alias_table_sources": dict(src),
        "alias_rows_mined_from_test_predictions": src.get("mistral_high_confidence", 0),
        "linker_train_rows": len(linker_train), "linker_dev_rows": len(linker_train),
        "taxonomy_reranker": "placeholder (train_taxonomy_reranker writes a README only); build_linker_dataset writes an empty file",
        "linking_eval_csv": "taxonomy_linking_accuracy=0.0 because supervised reranker labels are unavailable (linking_eval.md)",
    }
    verdict = ("NOT AN ACCURACY: the reported 'Linking Accuracy' (0.890374 Full CAG/Graph, 0.889834 safe wrapper) is "
               "coverage = share of predicted spans that received any canonical id. No gold link was compared. "
               "It is additionally partly circular: the 711-row alias table it links into was built from SkillSpan "
               "train/dev gold surfaces (601 rows) and 76 high-confidence Mistral predictions on the SAME test split, and "
               "its BM25 fallback links almost any string that shares a token with an entry.")
    recs.append(record(LAYER, "legacy_money_table", "legacy_linking_accuracy_is_coverage",
                       float(fc.get("linking_accuracy") or 0.0), n_units=0, unit="n/a", split="test", dataset="audit",
                       direction="none", provenance=[REPO_ROOT / f for f in AUDIT_FILES if (REPO_ROOT / f).exists()],
                       caveats=["Audit record: value is the legacy number being audited, NOT a linking accuracy."],
                       details={"verdict": verdict, **facts}))
    return recs, {"verdict": verdict, **facts}


# ============================================================================================ circularity audit
LEGACY_ARTIFACTS = {"full_cag_graph": "artifacts/skillllm_v2/full_cag_graph/linked_skills.jsonl",
                    "mistral_gliner_taxonomy": "artifacts/skillllm_v2/mistral_gliner_taxonomy/linked_skills.jsonl",
                    "safe_wrapper": "artifacts/skillllm_v2/safe_wrapper/linked_skills.jsonl"}


def _read_linked(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def audit_circularity(test_docs: Sequence[Dict[str, Any]], decorte_test: Sequence[Dict[str, Any]], seed: int,
                      workers: int = 8, n_null: int = 1000) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Label-free DIAGNOSTICS on the legacy 'Linking Accuracy' (= coverage). None of these numbers is a linking accuracy.

    (a) reproduce the legacy value from the stored linked_skills files and decompose it by link method / taxonomy tag;
    (b) alias-table provenance: how much of the dictionary is gold-derived / mined from the test split;
    (c) discrimination: coverage on predicted spans that match a gold span vs those that do not (gold SPANS only);
    (d) null control: coverage of the shipped linker on strings that have no specific ESCO skill (Decorte UNDERSPECIFIED mentions and
        seeded random non-gold n-grams of the test sentences).
    """
    from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker
    recs: List[Dict[str, Any]] = []
    alias_rows = [json.loads(l) for l in REPO_ALIAS_TABLE.read_text(encoding="utf-8").splitlines() if l.strip()]
    src_of_norm: Dict[str, set] = defaultdict(set)
    for r in alias_rows:
        src_of_norm[r["normalized_alias"]].add(r["source"])
        src_of_norm[_norm(r["canonical_skill_name"])].add(r["source"])
    gold_by_doc = {d["id"]: {(g["start"], g["end"]) for g in d["gold_spans"]} for d in test_docs}
    gold_surfs = {_norm(g["surface_form"]) for d in test_docs for g in d["gold_spans"]}
    out: Dict[str, Any] = {"artifacts": {}}
    prov = [REPO_ROOT / f for f in LEGACY_ARTIFACTS.values() if (REPO_ROOT / f).exists()] + [REPO_ALIAS_TABLE]
    for name, rel_path in LEGACY_ARTIFACTS.items():
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        spans = []
        for row in _read_linked(path):
            for sp in row["linked_skills"]:
                tl = (sp.get("metadata") or {}).get("taxonomy_link") or {}
                spans.append({"doc": row["id"], "start": sp.get("start"), "end": sp.get("end"), "surface": sp["surface_form"],
                              "linked": bool(sp.get("canonical_skill_name")), "method": tl.get("link_method") or "none",
                              "tax": tl.get("taxonomy") or "none",
                              "gold": (sp.get("start"), sp.get("end")) in gold_by_doc.get(row["id"], set())})
        n = len(spans)
        cov = sum(x["linked"] for x in spans) / n
        linked_sp = [x for x in spans if x["linked"]]
        meth = Counter(x["method"] for x in linked_sp)
        tax = Counter(x["tax"] for x in linked_sp)
        tp = [x for x in spans if x["gold"]]
        fp = [x for x in spans if not x["gold"]]
        in_alias = [x for x in spans if _norm(x["surface"]) in src_of_norm]
        mined_alias = [x for x in in_alias if "mistral_high_confidence" in src_of_norm[_norm(x["surface"])]]
        d = {"n_pred_spans": n, "legacy_coverage_reproduced": cov,
             "link_method_share_of_linked": {k: v / len(linked_sp) for k, v in meth.items()},
             "taxonomy_tag_share_of_linked": {k: v / len(linked_sp) for k, v in tax.items()},
             "share_of_all_spans_linked_only_by_bm25_or_fuzzy": sum(x["linked"] and x["method"] in ("bm25", "fuzzy") for x in spans) / n,
             "coverage_on_gold_matching_spans": sum(x["linked"] for x in tp) / max(1, len(tp)), "n_gold_matching_spans": len(tp),
             "coverage_on_non_gold_spans": sum(x["linked"] for x in fp) / max(1, len(fp)), "n_non_gold_spans": len(fp),
             "bm25_share_of_linked_on_non_gold_spans": sum(x["method"] == "bm25" for x in fp if x["linked"]) / max(1, sum(x["linked"] for x in fp)),
             "share_spans_surface_in_alias_table": len(in_alias) / n,
             "share_spans_surface_in_test_mined_alias_rows": len(mined_alias) / n}
        out["artifacts"][name] = d
        recs.append(record(LAYER, f"legacy:{name}", "diagnostic_legacy_coverage_reproduced", cov, n_units=n, unit="predicted span",
                           split="test", dataset="audit (no gold links)", direction="none", provenance=prov, status="measured",
                           caveats=["DIAGNOSTIC, NOT ACCURACY: share of predicted spans that received any canonical id (= money-table 'Linking Accuracy')."],
                           details=d))
    # (b) alias-table provenance
    src = Counter(r["source"] for r in alias_rows)
    mined = [r for r in alias_rows if r["source"] == "mistral_high_confidence"]
    mined_in_gold = sum(r["normalized_alias"] in gold_surfs for r in mined)
    test_gold_norm = [_norm(g["surface_form"]) for d in test_docs for g in d["gold_spans"]]
    in_tbl = [x for x in test_gold_norm if x in src_of_norm]
    by_src = Counter()
    for x in in_tbl:
        by_src["|".join(sorted(src_of_norm[x]))] += 1
    out["alias_table"] = {"rows": len(alias_rows), "by_source": dict(src),
                          "taxonomy_tag_rows": dict(Counter(r["taxonomy"] for r in alias_rows)),
                          "mined_rows_whose_alias_is_a_test_gold_surface": f"{mined_in_gold}/{len(mined)}",
                          "test_gold_mentions_with_surface_in_table": len(in_tbl) / max(1, len(test_gold_norm)),
                          "test_gold_mentions_in_table_by_source": dict(by_src)}
    recs.append(record(LAYER, "alias_table", "diagnostic_test_gold_mentions_in_alias_table", len(in_tbl) / max(1, len(test_gold_norm)),
                       n_units=len(test_gold_norm), unit="test gold mention", split="test", dataset="audit (no gold links)",
                       direction="none", provenance=[REPO_ALIAS_TABLE], status="measured",
                       caveats=["DIAGNOSTIC: dictionary overlap with test gold SURFACES; the 601 train/dev rows are legitimate, the "
                                "mistral_high_confidence rows are mined from anchor predictions on this test split."],
                       details=out["alias_table"]))
    # (d) null control through the shipped linker
    rng = random.Random(seed)
    under = sorted({m["span"] for m in decorte_test if m["nil_kind"] == "underspecified"})
    pool = []
    for d in test_docs:
        toks = [(mm.start(), mm.end()) for mm in re.finditer(r"\S+", d["text"])]
        gold = [(g["start"], g["end"]) for g in d["gold_spans"]]
        for i in range(len(toks)):
            for L in (1, 2, 3):
                if i + L > len(toks):
                    continue
                a, b = toks[i][0], toks[i + L - 1][1]
                if any(max(a, ga) < min(b, gb) for ga, gb in gold):
                    continue
                w = d["text"][a:b]
                if re.search(r"[A-Za-z]", w):
                    pool.append(w)
    rnd = sorted(set(rng.sample(pool, min(n_null, len(pool)))))
    shipped = TaxonomyLinker(alias_table_paths=[str(REPO_ALIAS_TABLE)])
    res = cascade_link_all(shipped, list(under) + rnd, workers=workers)
    def cov_of(ss):
        hit = [s for s in ss if res.get(s)]
        return len(hit) / max(1, len(ss)), Counter(res[s][2] for s in hit)
    nc = {}
    for nm, ss in (("decorte_underspecified_mentions", under), ("random_non_gold_ngrams", rnd)):
        c, mm = cov_of(ss)
        nc[nm] = {"n": len(ss), "shipped_linker_coverage": c, "method_mix": dict(mm)}
        recs.append(record(LAYER, "shipped_linker", f"diagnostic_null_control_coverage[{nm}]", c, n_units=len(ss), unit="string",
                           split="test", dataset="audit (non-skill strings)", direction="none", status="measured",
                           provenance=[REPO_ALIAS_TABLE],
                           caveats=["DIAGNOSTIC: share of strings with no specific ESCO skill which the shipped linker nevertheless links; "
                                    "a coverage-style 'accuracy' rewards this. (UNDERSPECIFIED = Decorte label for spans too vague to link, not necessarily non-skills.)"], details=nc[nm]))
    out["null_control"] = nc
    fc = out["artifacts"].get("full_cag_graph", {})
    out["circularity_findings"] = {
        "labels_exist": False,
        "score_is_function_of_linker_and_its_dictionary_only": True,
        "dictionary_is_gold_derived": f"{out['alias_table']['by_source'].get('skillspan_gold_train_dev', 0)} of {len(alias_rows)} alias rows come from SkillSpan train/dev gold spans; "
                                      f"{round(100 * fc.get('taxonomy_tag_share_of_linked', {}).get('SKILLSPAN_GOLD', 0))}% of linked predicted spans carry the taxonomy tag SKILLSPAN_GOLD",
        "test_leak": f"{out['alias_table']['mined_rows_whose_alias_is_a_test_gold_surface']} test-mined alias rows are test gold surfaces",
    }
    return recs, out


# ============================================================================================ annotation sample
def mark_span(text: str, start: Optional[int], end: Optional[int]) -> str:
    """Highlight the mention occurrence with [[...]] so annotators judge the right occurrence."""
    if start is None or end is None or not (0 <= start < end <= len(text)):
        return text
    return f"{text[:start]}[[{text[start:end]}]]{text[end:]}"


def conf_bin(method: str, conf: Optional[float]) -> str:
    if method in ("alias", "normalized"):
        return "exact_alias"
    if method == "fuzzy":
        return "fuzzy"
    if method == "bm25":
        return "bm25_high" if (conf or 0) >= 0.7 else "bm25_low"
    return "none"


def build_annotation_sample(cascade: Dict[str, List[Tuple[str, float, str]]], tau: Optional[float],
                            concepts: Sequence[Dict[str, Any]], gold_test: Sequence[Dict[str, Any]],
                            train_surfaces: set, seed: int, n_target: int = 300) -> Dict[str, Any]:
    """Stratified sample of PREDICTED spans (safe-wrapper output on SkillSpan test) with cascade_esco top-5 candidates.

    Blinded CSV: item id, sentence, mention, candidates A-E in a per-item random order (preferred label + short
    description), no scores, no system pick. Answer key (separate file): strata, which letter is the linker's rank r,
    confidence, abstain-at-tau, and Decorte gold labels when the predicted span exactly matches a gold-annotated span
    of the same sentence (control items)."""
    by_uri = {c["uri"]: c for c in concepts}
    gold_by = {(m["doc_id"], _norm(m["span"])): m for m in gold_test if m.get("doc_id")}
    pool = []
    for line in PRED_LINKED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for sp in row["linked_skills"]:
            surf = sp["surface_form"]
            ranked = cascade.get(surf, [])
            method = ranked[0][2] if ranked else "none"
            conf = ranked[0][1] if ranked else None
            nil_pred = (not ranked) or (tau is not None and conf is not None and conf < tau)
            g = gold_by.get((row["id"], _norm(surf)))
            pool.append({"doc_id": row["id"], "text": row["text"], "surface": surf, "start": sp.get("start"),
                         "end": sp.get("end"), "label": sp.get("label"), "ranked": ranked[:5], "method": method, "conf": conf,
                         "conf_bin": conf_bin(method, conf), "nil_pred": nil_pred, "seen": _norm(surf) in train_surfaces,
                         "gold": g})
    strata: Dict[Tuple, List[int]] = defaultdict(list)
    for i, p in enumerate(pool):
        strata[(p["label"], p["conf_bin"], p["nil_pred"], p["seen"])].append(i)
    rng = random.Random(seed)
    keys = sorted(strata, key=str)
    # allocation: at least min(len, 4) per stratum, rest proportional to sqrt(size) (oversamples rare cells)
    alloc = {k: min(len(strata[k]), 4) for k in keys}
    left = n_target - sum(alloc.values())
    w = {k: math.sqrt(len(strata[k])) for k in keys}
    tot = sum(w.values())
    for k in keys:
        alloc[k] = min(len(strata[k]), alloc[k] + int(round(left * w[k] / tot)))
    chosen: List[int] = []
    for k in keys:
        idx = list(strata[k])
        rng.shuffle(idx)
        # de-duplicate identical (doc, surface) inside a stratum
        seen_pairs = set()
        picked = []
        for i in idx:
            pair = (pool[i]["doc_id"], _norm(pool[i]["surface"]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            picked.append(i)
            if len(picked) >= alloc[k]:
                break
        chosen.extend(picked)
    rng.shuffle(chosen)
    blinded, key = [], []
    for n, i in enumerate(chosen):
        p = pool[i]
        item = f"L06-{n + 1:04d}"
        cands = [u for u, _, _ in p["ranked"]]
        order = list(range(len(cands)))
        rng.shuffle(order)
        letters = "ABCDE"
        row = {"item_id": item, "sentence": mark_span(p["text"], p["start"], p["end"]), "mention": p["surface"]}
        letter_of_rank = {}
        for j in range(5):
            if j < len(order):
                c = by_uri.get(cands[order[j]], {})
                desc = (c.get("description") or "").replace("\n", " ")
                row[f"candidate_{letters[j]}"] = f"{c.get('preferred_label', cands[order[j]])} — {desc[:160]}"
                letter_of_rank[order[j] + 1] = letters[j]
            else:
                row[f"candidate_{letters[j]}"] = ""
        row["options"] = "one or more of A-E | NONE_OF_THESE (skill, but no listed ESCO concept fits) | NOT_A_SKILL"
        blinded.append(row)
        stratum = {"label": p["label"], "conf_bin": p["conf_bin"], "nil_pred_at_tau": p["nil_pred"], "seen": p["seen"]}
        key.append({"item_id": item, "doc_id": p["doc_id"], "start": p["start"], "end": p["end"], "stratum": stratum,
                    "linker": "cascade_esco", "tau": tau, "linker_rank_to_letter": letter_of_rank,
                    "candidate_uris_by_rank": cands, "top1_conf": p["conf"], "top1_method": p["method"],
                    "control_item": p["gold"] is not None,
                    "decorte_gold_labels": (p["gold"] or {}).get("gold_labels"),
                    "decorte_gold_uris": (p["gold"] or {}).get("gold_uris"),
                    "decorte_nil_kind": (p["gold"] or {}).get("nil_kind")})
    out_dir = EVALS_ROOT / LAYER
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "annotation_sample.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(blinded[0].keys()) if blinded else ["item_id"])
        w.writeheader()
        w.writerows(blinded)
    (out_dir / "annotation_answer_key.jsonl").write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in key) + "\n",
                                                        encoding="utf-8")
    cells = Counter((str(k["stratum"]["label"]), k["stratum"]["conf_bin"], k["stratum"]["nil_pred_at_tau"], k["stratum"]["seen"]) for k in key)
    return {"n_items": len(key), "pool_spans": len(pool), "n_strata": len(keys), "control_items": sum(k["control_item"] for k in key),
            "cells": {"|".join(map(str, c)): v for c, v in sorted(cells.items(), key=str)},
            "pool_strata_sizes": {"|".join(map(str, k)): len(v) for k, v in sorted(strata.items(), key=str)}}


# ============================================================================================ main
def _ci(rows: np.ndarray, stat, n_boot: int, seed: int) -> Dict[str, Any]:
    from skillllm_v2.evals.research.stats import bootstrap_ci
    return bootstrap_ci(rows, stat, n_boot=n_boot, seed=seed)


def run(args: argparse.Namespace) -> Path:
    from skillllm_v2.evals.research import stats as S
    from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker

    seed, n_boot = args.seed, args.n_boot
    records: List[Dict[str, Any]] = []
    audit_recs, audit = audit_legacy()
    records += audit_recs

    concepts = load_esco(ESCO_VERSION)
    test_docs = load_skillspan("test")
    circ: Dict[str, Any] = {}
    train_surf = train_gold_surfaces()
    repo_alias = repo_alias_surfaces()
    val, val_info = load_decorte("validation", test_docs=test_docs, concepts=concepts, train_surfaces=train_surf, repo_aliases=repo_alias)
    test, test_info = load_decorte("test", test_docs=test_docs, concepts=concepts, train_surfaces=train_surf, repo_aliases=repo_alias)
    if args.smoke:
        rng = random.Random(seed)
        val = rng.sample(val, min(60, len(val)))
        test = rng.sample(test, min(120, len(test)))
        concepts_used = concepts
    else:
        concepts_used = concepts
    l02_info = assign_l02_buckets(val + test, seed)
    prov_gold = [DECORTE_ROOT / s / f"{sp}.csv" for s in SUBSETS for sp in ("validation", "test")]
    prov_esco = [esco_dir(ESCO_VERSION) / "skills_en.jsonl"]

    # ---- linkers
    linkers: Dict[str, Any] = {"random": RandomLinker(concepts_used, seed), "exact": ExactLinker(concepts_used),
                               "bm25": BM25Linker(concepts_used), "always_nil": AlwaysNilLinker()}
    cache = load_embedding_cache(ESCO_VERSION)
    if cache is not None:
        linkers["dense_minilm"] = DenseLinker(concepts_used, cache)
    tax_path = esco_taxonomy_file(concepts_used)
    esco_linker = TaxonomyLinker(taxonomy_paths=[str(tax_path)], alias_table_paths=[])
    surfaces = sorted({m["span"] for m in val + test})
    pred_surfaces: List[str] = []
    if PRED_LINKED.exists() and not args.smoke:
        for line in PRED_LINKED.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pred_surfaces += [s["surface_form"] for s in json.loads(line)["linked_skills"]]
    tax_sha = hashlib.sha256(tax_path.read_bytes()).hexdigest()
    cascade = _cached_rank(esco_dir(ESCO_VERSION) / "derived" / "cascade_rank_cache.json", {"taxonomy_sha256": tax_sha, "K": K},
                           surfaces + pred_surfaces, lambda miss: cascade_rank_all(esco_linker, miss, workers=args.workers))
    linkers["cascade_esco"] = PrecomputedLinker("cascade_esco", cascade)
    # consistency of the replayed ranking with TaxonomyLinker.link on a seeded subsample
    rng = random.Random(seed)
    chk = rng.sample(surfaces, min(len(surfaces), 40 if args.smoke else 150))
    live = cascade_link_all(esco_linker, chk, workers=args.workers)
    consistent = sum(1 for s in chk if ((cascade.get(s) or [None])[0] or (None,))[0] == (live.get(s) or (None,))[0])
    bridge, bridge_info = build_bridge(concepts_used)
    shipped, shipped_diag = shipped_ranked(surfaces, bridge)
    linkers["shipped_bridge"] = PrecomputedLinker("shipped_bridge", shipped)

    if not args.smoke:
        circ_recs, circ = audit_circularity(test_docs, test, seed, workers=args.workers)
        records += circ_recs
        audit = {**audit, "circularity": circ}

    # ---- thresholds on validation only
    tuned: Dict[str, Dict[str, Any]] = {}
    for name in ("bm25", "dense_minilm", "cascade_esco"):
        if name in linkers:
            tuned[name] = tune_tau(val, linkers[name])
    configs: List[Tuple[str, str, Optional[float]]] = [("random", "random", None), ("always_nil", "always_nil", None), ("exact", "exact", None),
                                                       ("shipped_bridge", "shipped_bridge", None)]
    for name in ("bm25", "dense_minilm", "cascade_esco"):
        if name in linkers:
            configs.append((f"{name}@never", name, None))
            configs.append((f"{name}@tau_val", name, tuned[name]["tau"]))
    if "dense_minilm" not in linkers:
        records.append(record(LAYER, "dense_minilm", "top1_acc_linkable", None, n_units=0, unit="mention", split="test",
                              dataset=DATASET, direction="higher", status="not_available",
                              reason=f"embedding cache {rel(embedding_cache_path())} missing; run --embed in the torch venv"))

    strata = {
        "all": lambda m: True,
        "tech": lambda m: m["subset"] == "tech", "house": lambda m: m["subset"] == "house",
        "HARD": lambda m: m["label"] == "HARD_SKILL", "SOFT": lambda m: m["label"] == "SOFT_SKILL",
        "seen": lambda m: m["l02_bucket"] in ("S1", "S2"), "unseen": lambda m: m["l02_bucket"] == "S4",
        "S1": lambda m: m["l02_bucket"] == "S1", "S2": lambda m: m["l02_bucket"] == "S2",
        "S3": lambda m: m["l02_bucket"] == "S3", "S4": lambda m: m["l02_bucket"] == "S4",
        "HARD_seen": lambda m: m["label"] == "HARD_SKILL" and m["l02_bucket"] in ("S1", "S2"),
        "HARD_unseen": lambda m: m["label"] == "HARD_SKILL" and m["l02_bucket"] == "S4",
        "SOFT_seen": lambda m: m["label"] == "SOFT_SKILL" and m["l02_bucket"] in ("S1", "S2"),
        "SOFT_unseen": lambda m: m["label"] == "SOFT_SKILL" and m["l02_bucket"] == "S4",
        "variant_surface": lambda m: m["linkable"] and all(_norm(m["span"]) != _norm(l) for l in m["gold_labels"]),
        "alias_listed": None,  # filled below (needs ESCO alt labels)
    }
    by_uri = {c["uri"]: c for c in concepts}

    def alias_listed(m: Dict[str, Any]) -> bool:
        if not m["linkable"]:
            return False
        n = _norm(m["span"])
        if any(n == _norm(by_uri[u]["preferred_label"]) for u in m["gold_uris"] if u in by_uri):
            return False
        return any(n in {_norm(a) for a in by_uri[u].get("alt_labels", [])} for u in m["gold_uris"] if u in by_uri)

    strata["alias_listed"] = alias_listed
    strata["pref_label_surface"] = lambda m: m["linkable"] and any(_norm(m["span"]) == _norm(l) for l in m["gold_labels"])
    test_by_id = {m["mention_id"]: m for m in test}
    results: Dict[str, List[Dict[str, Any]]] = {}
    per_unit: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    ratio = S.ratio_stat(0, 1)
    for sysname, lname, tau in configs:
        rows = score_mentions(test, linkers[lname], tau)
        results[sysname] = rows
        for r in rows:
            m = test_by_id[r["mention_id"]]
            if m["linkable"]:
                for met in ("hit@1", "hit@5", "rr"):
                    per_unit.append({"layer": LAYER, "family": "L06_linking", "system": sysname, "metric": met,
                                     "unit_id": r["mention_id"], "cluster_id": r["cluster"], "value": r[met]})
            per_unit.append({"layer": LAYER, "family": "L06_linking", "system": sysname, "metric": "e2e_correct",
                             "unit_id": r["mention_id"], "cluster_id": r["cluster"], "value": r["e2e_correct"]})
        srow = {"system": sysname, "tau": tau}
        for sname, fn in strata.items():
            sel_rows = [r for r in rows if fn(test_by_id[r["mention_id"]])]
            lk = [r for r in sel_rows if r["linkable"]]
            if not lk:
                continue
            metrics = [("top1_acc_linkable", "hit@1"), ("recall@3", "hit@3"), ("recall@5", "hit@5"), ("recall@10", "hit@10"),
                       ("mrr@10", "rr"), ("rp@5", "rp@5")]
            if sname != "all":
                metrics = [("top1_acc_linkable", "hit@1"), ("recall@5", "hit@5"), ("mrr@10", "rr")]
            for mname, key in metrics:
                arr, _ = cluster_rows(lk, key)
                ci = _ci(arr, ratio, n_boot, seed)
                headline = sname == "all" and mname in ("top1_acc_linkable", "recall@5", "mrr@10") and sysname in ("cascade_esco@tau_val", "shipped_bridge")
                records.append(record(LAYER, sysname, mname if sname == "all" else f"{mname}[{sname}]", ci["value"],
                                      n_units=len(arr), unit="posting (cluster); mentions in details", split="test",
                                      dataset=DATASET, direction="higher", ci=(ci["ci_low"], ci["ci_high"]),
                                      ci_method=f"cluster (posting) percentile bootstrap, n_boot={n_boot}, seed={seed}",
                                      provenance=prov_gold + prov_esco,
                                      caveats=["Context-free linking of GOLD SkillSpan spans (extraction errors excluded).",
                                               "Gold = one annotation pass (no IAA reported by Decorte et al.)."],
                                      details={"n_mentions": len(lk), "stratum": sname, "tau": tau,
                                               **({"headline": True} if headline else {})}))
                if sname == "all":
                    srow[mname] = ci["value"]
                    srow[mname + "_ci"] = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
                elif mname == "top1_acc_linkable":
                    srow[f"top1[{sname}]"] = ci["value"]
                    srow[f"n[{sname}]"] = len(lk)
            if sname == "all":
                arr, _ = cluster_rows(sel_rows, "e2e_correct")
                ci = _ci(arr, ratio, n_boot, seed)
                records.append(record(LAYER, sysname, "nil_aware_accuracy", ci["value"], n_units=len(arr), unit="posting",
                                      split="test", dataset=DATASET, direction="higher", ci=(ci["ci_low"], ci["ci_high"]),
                                      ci_method=f"cluster (posting) percentile bootstrap, n_boot={n_boot}",
                                      provenance=prov_gold + prov_esco,
                                      details={"n_mentions": len(sel_rows), "definition": "linkable: not abstained and top-1 in gold; NIL (LABEL NOT PRESENT / UNDERSPECIFIED): abstained", "tau": tau}))
                srow["nil_aware_acc"] = ci["value"]
                nr, _ = nil_rows(sel_rows)
                for mname, st in (("nil_precision", S.micro_precision), ("nil_recall", S.micro_recall), ("nil_f1", S.micro_f1)):
                    c2 = _ci(nr, st, n_boot, seed)
                    records.append(record(LAYER, sysname, mname, c2["value"], n_units=len(nr), unit="posting", split="test",
                                          dataset=DATASET, direction="higher", ci=(c2["ci_low"], c2["ci_high"]),
                                          ci_method="cluster (posting) percentile bootstrap", provenance=prov_gold,
                                          details={"positive_class": "NIL (gold has no ESCO label)", "tau": tau,
                                                   "tp_fp_fn": nr.sum(axis=0).tolist()}))
                    srow[mname] = c2["value"]
                    srow[mname + "_ci"] = f"[{c2['ci_low']:.3f}, {c2['ci_high']:.3f}]"
                srow["abstain_rate"] = sum(r["abstain"] for r in sel_rows) / len(sel_rows)
            elif sname in ("tech", "house", "HARD", "SOFT", "seen", "unseen"):
                nr, _ = nil_rows(sel_rows)
                for mname, st in (("nil_precision", S.micro_precision), ("nil_recall", S.micro_recall), ("nil_f1", S.micro_f1)):
                    c2 = _ci(nr, st, n_boot, seed)
                    records.append(record(LAYER, sysname, f"{mname}[{sname}]", c2["value"], n_units=len(nr), unit="posting",
                                          split="test", dataset=DATASET, direction="higher", ci=(c2["ci_low"], c2["ci_high"]),
                                          ci_method="cluster (posting) percentile bootstrap", provenance=prov_gold,
                                          details={"positive_class": "NIL", "tau": tau, "stratum": sname,
                                                   "tp_fp_fn": nr.sum(axis=0).tolist(), "n_mentions": len(sel_rows)}))
                    if mname == "nil_f1":
                        srow[f"nil_f1[{sname}]"] = c2["value"]
        summary_rows.append(srow)

    # ---- hierarchy near-miss (secondary): top-1 wrong but shares a direct ESCO parent group with a gold concept
    def parents(u: str) -> set:
        c = by_uri.get(u, {})
        return {x["uri"] for x in c.get("broader_hierarchy", []) + c.get("broader_skill", []) if x.get("uri")}

    for sysname in ("cascade_esco@tau_val", "shipped_bridge", "dense_minilm@tau_val", "bm25@tau_val"):
        rows = results.get(sysname)
        if not rows:
            continue
        wrong = [r for r in rows if r["linkable"] and not r["abstain"] and r["hit@1"] == 0.0 and r["top"]]
        near = [r for r in wrong if parents(r["top"][0]) & set().union(*[parents(u) for u in test_by_id[r["mention_id"]]["gold_uris"]])]
        records.append(record(LAYER, sysname, "error_share_same_parent_group", len(near) / len(wrong) if wrong else 0.0,
                              n_units=len(wrong), unit="wrong top-1 mention", split="test", dataset=DATASET,
                              direction="none", details={"definition": "share of wrong (non-abstained) top-1 links whose concept shares a direct broader ESCO group/skill with a gold concept (near misses)",
                                                         "n_near": len(near)},
                              caveats=["Secondary diagnostic; no CI (descriptive)."]))

    # ---- paired comparisons (family L06_linkers): cascade_esco@tau_val vs each other system, top-1 on linkable
    ref = "cascade_esco@tau_val"
    comps = []
    if ref in results:
        for other in [s for s, _, _ in configs if s != ref]:
            for met, sel in (("top1_acc_linkable", lambda r: r["linkable"]), ("nil_aware_accuracy", lambda r: True)):
                key = "hit@1" if met == "top1_acc_linkable" else "e2e_correct"
                a, ka = cluster_rows(results[ref], key, sel)
                b, kb = cluster_rows(results[other], key, sel)
                assert ka == kb
                pb = S.paired_bootstrap(a, b, ratio, n_boot=n_boot, seed=seed)
                comps.append({"a": ref, "b": other, "metric": met, "delta": pb["delta"], "ci_low": pb["ci_low"],
                              "ci_high": pb["ci_high"], "p": pb["p"], "family": f"L06_{met}", "n_units": pb["n_units"]})
        S.adjust_families(comps, p_key="p", family_key="family")
        for c in comps:
            records.append(record(LAYER, f"{c['a']} - {c['b']}", f"delta_{c['metric']}", c["delta"], n_units=c["n_units"],
                                  unit="posting", split="test", dataset=DATASET, direction="higher",
                                  ci=(c["ci_low"], c["ci_high"]), ci_method="paired cluster bootstrap (shared draws)",
                                  provenance=prov_gold,
                                  details={"p_raw": c["p"], "p_holm": c.get("p_holm"), "p_bh": c.get("p_bh"), "family": c["family"]}))

    # ---- label-free diagnostics on the system's predicted spans (NOT accuracy)
    diag = {}
    if pred_surfaces:
        tau_c = tuned.get("cascade_esco", {}).get("tau")
        n = len(pred_surfaces)
        linked = sum(1 for s in pred_surfaces if cascade.get(s) and (tau_c is None or cascade[s][0][1] >= tau_c))
        diag = {"pred_spans": n, "cascade_esco_coverage_at_tau": linked / n, "cascade_esco_nil_rate_at_tau": 1 - linked / n,
                "method_mix": dict(Counter((cascade.get(s) or [(None, None, "none")])[0][2] for s in pred_surfaces))}
        sh_pred, _ = shipped_ranked(pred_surfaces, bridge)
        diag["shipped_bridge_coverage"] = sum(1 for s in pred_surfaces if sh_pred.get(s)) / n
        records.append(record(LAYER, "cascade_esco@tau_val", "diagnostic_coverage_predicted_spans", linked / n, n_units=n,
                              unit="predicted span", split="test", dataset="safe_wrapper linked_skills (no gold)",
                              direction="none", provenance=[PRED_LINKED],
                              caveats=["DIAGNOSTIC, NOT ACCURACY: share of predicted spans given an ESCO link at tau_val."],
                              details=diag))

    # ---- data / bridge / consistency records
    records.append(record(LAYER, "data", "decorte_alignment_to_skillspan_test", test_info["counts"].get("aligned", 0) / max(1, len(test)),
                          n_units=len(test), unit="mention", split="test", dataset=DATASET, direction="none",
                          provenance=prov_gold, details={"test": test_info, "validation": val_info}))
    records.append(record(LAYER, "shipped_bridge", "bridge_mapped_share", bridge_info["mapped_share"], n_units=bridge_info["entries"],
                          unit="repo canonical entry", split="n/a", dataset="skill_aliases.jsonl -> ESCO v1.1.0",
                          direction="none", provenance=[REPO_ALIAS_TABLE, DERIVED_INPUTS / "repo_to_esco_bridge.jsonl"],
                          details={**bridge_info, "shipped_diagnostics_on_gold_surfaces": shipped_diag}))
    records.append(record(LAYER, "cascade_esco", "rank_replay_consistency", consistent / max(1, len(chk)), n_units=len(chk),
                          unit="surface", split="test", dataset=DATASET, direction="none",
                          details={"definition": "rank-1 of replayed cascade == TaxonomyLinker.link on a seeded subsample"},
                          status="measured"))
    for name, t in tuned.items():
        records.append(record(LAYER, name, "tau_tuned_on_validation", float(t["tau"]) if t["tau"] is not None else -1.0,
                              n_units=len(val), unit="mention", split="validation", dataset=DATASET, direction="none",
                              provenance=prov_gold, details=t))

    # ---- annotation sample
    ann = {}
    if not args.smoke and PRED_LINKED.exists():
        ann = build_annotation_sample(cascade, tuned.get("cascade_esco", {}).get("tau"), concepts, test + val, train_surf, seed)
        records.append(record(LAYER, "annotation_sample", "n_items", float(ann["n_items"]), n_units=ann["n_items"], unit="item",
                              split="test", dataset="safe_wrapper predicted spans", direction="none", status="measured",
                              details=ann, caveats=["Human labels pending (A6 protocol); no accuracy derived from this file."]))
        records.append(record(LAYER, "cascade_esco@tau_val", "human_linking_accuracy_predicted_spans", None, n_units=0,
                              unit="item", split="test", dataset="annotation_sample.csv", direction="higher",
                              status="pending_human", reason="needs human annotation of annotation_sample.csv (A6 / L11)"))

    # ---- outputs
    out_dir = EVALS_ROOT / LAYER / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        with open(out_dir / "per_unit_scores.jsonl", "w", encoding="utf-8") as fh:
            for r in per_unit:
                fh.write(json.dumps(r) + "\n")
        with open(out_dir / "per_mention_outputs.jsonl", "w", encoding="utf-8") as fh:
            for sysname, rows in results.items():
                for r in rows:
                    fh.write(json.dumps({"system": sysname, **{k: v for k, v in r.items()}}) + "\n")
        DERIVED_INPUTS.mkdir(parents=True, exist_ok=True)
        with open(DERIVED_INPUTS / "decorte_mentions_aligned.jsonl", "w", encoding="utf-8") as fh:
            for m in val + test:
                fh.write(json.dumps({k: v for k, v in m.items() if k != "sentence"} | {"sentence_doc": m.get("doc_id")}, ensure_ascii=False) + "\n")
    _figures(summary_rows, results, test_by_id, out_dir)
    report = _report(summary_rows, comps, audit, tuned, test_info, val_info, bridge_info, shipped_diag, consistent, len(chk),
                     diag, ann, len(test), len(val), sum(m["linkable"] for m in test))
    return write_layer(LAYER, records, report, seed=seed, smoke=args.smoke,
                       extra_meta={"esco_version": ESCO_VERSION, "n_boot": n_boot})


def _figures(summary_rows: List[Dict[str, Any]], results: Dict[str, List[Dict[str, Any]]],
             test_by_id: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary_rows if "top1_acc_linkable" in r]
    with open(out_dir / "linker_summary.csv", "w", encoding="utf-8", newline="") as fh:
        cols = sorted({k for r in rows for k in r}, key=lambda k: (k != "system", k))
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["system"] for r in rows]
    x = np.arange(len(names))
    for j, (k, lab) in enumerate((("top1_acc_linkable", "top-1"), ("recall@5", "recall@5"), ("nil_aware_acc", "NIL-aware acc"))):
        ax.bar(x + (j - 1) * 0.27, [r.get(k) or 0 for r in rows], 0.27, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (Decorte TECH+HOUSE test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "linker_comparison.png", dpi=150)
    plt.close(fig)
    # accuracy-coverage curve for score-based linkers on linkable mentions
    curve = []
    for sysname in ("cascade_esco@never", "dense_minilm@never", "bm25@never"):
        rr = [r for r in results.get(sysname, []) if r["linkable"] and r["s1"] is not None]
        if not rr:
            continue
        rr = sorted(rr, key=lambda r: -r["s1"])
        for q in np.linspace(0.05, 1.0, 20):
            kept = rr[: max(1, int(round(q * len(rr))))]
            curve.append({"system": sysname, "coverage": len(kept) / len(rr), "top1_acc": sum(r["hit@1"] for r in kept) / len(kept),
                          "min_score": kept[-1]["s1"]})
    if curve:
        with open(out_dir / "accuracy_coverage.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
            w.writeheader()
            w.writerows(curve)
        fig, ax = plt.subplots(figsize=(5, 4))
        for s in sorted({c["system"] for c in curve}):
            pts = [c for c in curve if c["system"] == s]
            ax.plot([p["coverage"] for p in pts], [p["top1_acc"] for p in pts], marker="o", ms=3, label=s.replace("@never", ""))
        ax.set_xlabel("coverage (share of linkable gold mentions kept, by top-1 score)")
        ax.set_ylabel("top-1 accuracy on kept")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_coverage.png", dpi=150)
        plt.close(fig)


def _circ_lines(c: Dict[str, Any]) -> List[str]:
    if not c:
        return ["(circularity audit not run in smoke mode)"]
    fc = c["artifacts"].get("full_cag_graph", {})
    at = c["alias_table"]
    nc = c["null_control"]
    f = c["circularity_findings"]
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    return [
        "**Is the ~0.890 circular? Yes, in the sense that matters:** there are no gold links at all; the score is a function of "
        "the linker and the dictionary it links into, so it cannot detect a wrong link.",
        "",
        f"- Reproduced exactly from `artifacts/skillllm_v2/full_cag_graph/linked_skills.jsonl`: {fc.get('legacy_coverage_reproduced'):.6f} "
        f"over {fc.get('n_pred_spans')} predicted spans (= linked / total). Same for the other stored files: " +
        ", ".join(f"{k} {v['legacy_coverage_reproduced']:.6f}" for k, v in c["artifacts"].items() if k != "full_cag_graph") + ".",
        f"- Method mix of the linked spans: {', '.join(f'{k} {pct(v)}' for k, v in fc.get('link_method_share_of_linked', {}).items())}; "
        f"the BM25 fallback accepts any string sharing a token with an entry, so it is not a correctness signal.",
        f"- {f['dictionary_is_gold_derived']}.",
        f"- Test contamination: {f['test_leak']} (alias rows `mistral_high_confidence` mined from anchor predictions on this very test "
        f"split); {pct(at['test_gold_mentions_with_surface_in_table'])} of test gold mentions have a surface in the table "
        f"({at['test_gold_mentions_in_table_by_source']}).",
        f"- No discrimination: coverage is {pct(fc.get('coverage_on_gold_matching_spans', 0))} on predicted spans that exactly match a gold span "
        f"(n={fc.get('n_gold_matching_spans')}) vs {pct(fc.get('coverage_on_non_gold_spans', 0))} on predicted spans that do not (n={fc.get('n_non_gold_spans')}); "
        f"{pct(fc.get('bm25_share_of_linked_on_non_gold_spans', 0))} of the latter's links are BM25 fallbacks.",
        f"- Null control (strings with no specific ESCO skill: random non-gold n-grams, Decorte UNDERSPECIFIED mentions): the shipped linker links {pct(nc['random_non_gold_ngrams']['shipped_linker_coverage'])} of "
        f"{nc['random_non_gold_ngrams']['n']} random non-gold 1-3-grams of the test sentences "
        f"(method mix {nc['random_non_gold_ngrams']['method_mix']}); Decorte UNDERSPECIFIED mentions: n={nc['decorte_underspecified_mentions']['n']}, "
        f"coverage {pct(nc['decorte_underspecified_mentions']['shipped_linker_coverage'])}.",
        "- All of the above are DIAGNOSTICS on predicted spans; none is a linking accuracy. Legitimate numbers are those in section 3 (Decorte gold).",
    ]


def _report(summary_rows, comps, audit, tuned, test_info, val_info, bridge_info, shipped_diag, consistent, n_chk, diag, ann,
            n_test, n_val, n_linkable=0) -> str:
    cols = ["system", "top1_acc_linkable", "top1_acc_linkable_ci", "recall@3", "recall@3_ci", "recall@5", "recall@5_ci",
            "mrr@10", "mrr@10_ci", "nil_aware_acc", "abstain_rate"]
    nil_cols = ["system", "nil_precision", "nil_precision_ci", "nil_recall", "nil_recall_ci", "nil_f1", "nil_f1_ci"]
    nil_strat_cols = ["system", "nil_f1[tech]", "nil_f1[house]", "nil_f1[HARD]", "nil_f1[SOFT]", "nil_f1[seen]", "nil_f1[unseen]"]
    strat_cols = ["system", "top1[tech]", "top1[house]", "top1[HARD]", "top1[SOFT]", "top1[seen]", "top1[unseen]",
                  "top1[HARD_seen]", "top1[HARD_unseen]", "top1[SOFT_seen]", "top1[SOFT_unseen]"]
    alias_cols = ["system", "top1[pref_label_surface]", "n[pref_label_surface]", "top1[alias_listed]", "n[alias_listed]",
                  "top1[variant_surface]", "n[variant_surface]"]
    bucket_cols = ["system", "top1[S1]", "n[S1]", "top1[S2]", "n[S2]", "top1[S3]", "n[S3]", "top1[S4]", "n[S4]"]
    comp_rows = [{"comparison": f"{c['a']} − {c['b']}", "metric": c["metric"], "delta": c["delta"],
                  "CI": f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]", "p_holm": c.get("p_holm")} for c in comps]
    tc = test_info["counts"]
    res_counts = {k: v for k, v in tc.items() if k.startswith("label_resolution")}
    lines = [
        "# L06 — Taxonomy linking against public ESCO gold",
        "",
        "## 1. Audit of the legacy \"Linking Accuracy\" (~0.890)",
        "",
        f"**Verdict:** {audit['verdict']}",
        "",
        f"- Computed as: {audit['computed_as']}.",
        f"- Alias table: {audit['alias_table_rows']} rows, sources {audit['alias_table_sources']}.",
        f"- `linker_train/dev.jsonl`: {audit['linker_train_rows']} toy rows each; reranker: {audit['taxonomy_reranker']}.",
        "- Consequence: the 0.890 must not be cited as linking accuracy anywhere (integrity flag raised in Progress.md).",
        "",
        "## 2. Gold and protocol",
        "",
        f"- Gold: {DATASET}; HF `TechWolf/skill-extraction-{{tech,house}}`, CC-BY-4.0. Decorte *validation* "
        f"({n_val} mentions) is used for every threshold; *test* ({n_test} mentions: {n_linkable} with an ESCO "
        f"label, {n_test - n_linkable} NIL = LABEL NOT PRESENT / UNDERSPECIFIED) is reporting-only.",
        f"- All Decorte sentences come from the SkillSpan **test** split; {tc.get('aligned', 0)}/{tc.get('mentions')} test mentions "
        f"aligned to a repo SkillSpan test id (`test-<i>`), cluster = SkillSpan posting (`source:idx`).",
        f"- Gold label -> URI resolution in ESCO {ESCO_VERSION}: {res_counts} "
        f"-> unresolved {tc.get('gold_label_unresolved', 0)} mentions (excluded from the linkable set); "
        f"unresolved labels: {test_info.get('unresolved_labels')}.",
        "- Mention = (subset, sentence, span); multi-label spans count a hit if top-k contains any gold concept.",
        "- Linking is context-free on GOLD spans, so extraction errors are excluded (upper bound for the pipeline).",
        "- seen = L02 buckets S1+S2 (surface, or its normalisation, is a SkillSpan TRAIN gold span), unseen = L02 S4 (same "
        "`generalization.Bucketer`); HARD / SOFT = SkillSpan tag of the aligned gold span. variant_surface = surface differs "
        "from every gold preferred label; alias_listed = surface is an ESCO alternative label of a gold concept "
        "(dictionary-style lookups trivially succeed here; it is the alias-resolution accuracy, not generalisation).",
        "- `cascade_esco@tau_val`: the validation-tuned threshold is " + str(tuned.get("cascade_esco", {}).get("tau")) +
        " (only an exact alias/normalised hit survives); it is effectively a lookup that abstains otherwise.",
        "",
        "## 3. Results (test; 95% posting-cluster bootstrap CIs)",
        "",
        md_table(summary_rows, cols, fmt="{:.3f}"),
        "",
        "NIL detection (positive class = NIL: gold has no ESCO label; predicted NIL = abstain; posting-cluster CIs). "
        "Systems that never abstain (`random`, `*@never`) score NIL recall 0 by construction except when they have no candidate. "
        "`always_nil` shows the floor: NIL prevalence in test makes predict-everything-NIL already reach the F1 printed on its row, "
        "so `exact` / `cascade_esco@tau_val` (NIL F1 ~0.58) add little NIL skill beyond abstaining:",
        "",
        md_table(summary_rows, nil_cols, fmt="{:.3f}"),
        "",
        md_table(summary_rows, nil_strat_cols, fmt="{:.3f}"),
        "",
        "Strata (top-1 accuracy on linkable mentions; seen = L02 S1+S2, unseen = L02 S4):",
        "",
        md_table(summary_rows, strat_cols, fmt="{:.3f}"),
        "",
        "L02 buckets (S1 exact seen, S2 normalised seen, S3 canonical seen / surface unseen, S4 fully unseen; same `Bucketer` as L02):",
        "",
        md_table(summary_rows, bucket_cols, fmt="{:.3f}"),
        "",
        "Alias-resolution accuracy (top-1 when the surface is an ESCO *alternative* label of the gold concept = `alias_listed`; "
        "`pref_label_surface` = surface equals the gold preferred label; `variant_surface` = differs from every gold preferred label):",
        "",
        md_table(summary_rows, alias_cols, fmt="{:.3f}"),
        "",
        f"Thresholds (validation only): " + "; ".join(f"{k}: tau={v['tau']}, val acc {v['val_nil_aware_acc']:.3f} (never-abstain {v['val_nil_aware_acc_never_abstain']:.3f})" for k, v in tuned.items()),
        "",
        "Paired comparisons vs `cascade_esco@tau_val` (Holm within metric family):",
        "",
        md_table(comp_rows, ["comparison", "metric", "delta", "CI", "p_holm"], fmt="{:.4f}"),
        "",
        "## 4. The shipped linker",
        "",
        f"- Bridge repo canonical entry -> ESCO (exact normalised labels, no gold): {bridge_info}.",
        f"- On gold surfaces: {shipped_diag}.",
        f"- Replayed cascade rank-1 equals `TaxonomyLinker.link` on {consistent}/{n_chk} sampled surfaces.",
        "",
        "## 5. Circularity audit of the legacy number and label-free diagnostics (NOT accuracy)",
        "",
        *_circ_lines(audit.get("circularity") or {}),
        "",
        "Cascade-on-ESCO coverage of the predicted spans (diagnostic only):",
        "",
        f"{diag if diag else 'n/a'}",
        "",
        "## 6. Human annotation sample (for A6 / L11)",
        "",
        f"`annotation_sample.csv` (blinded: candidate order shuffled per item, no scores, no system pick) and the separate "
        f"`annotation_answer_key.jsonl`. {ann if ann else 'not built (smoke)'}",
        "",
        "## 7. Caveats",
        "",
        "- Decorte et al. report no inter-annotator agreement; gold is a single annotation pass.",
        "- 65 SkillSpan test postings -> cluster bootstrap over few clusters; intervals are wide by design.",
        "- The system has no trained taxonomy reranker (placeholder), so no learned linker is evaluated.",
        "- `exact` and `shipped_bridge` abstain intrinsically; their top-k lists are short, so recall@k ~ top-1.",
        "- Decorte validation/test are both drawn from SkillSpan test; the repo alias table contains 76 aliases mined from "
        "anchor predictions on that same split (affects `shipped_bridge` only; no ESCO ids leak).",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--embed", action="store_true", help="compute the MiniLM embedding cache (torch venv) and exit")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.embed:
        embed_main(device=args.device)
        return
    out = run(args)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
