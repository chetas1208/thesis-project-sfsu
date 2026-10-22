"""Linking evals (module=link).

What exists offline, and therefore what this module can honestly measure:

* The taxonomy is ``data/skillllm_v2/taxonomy/skill_aliases.jsonl`` (696 canonical entries).  No ESCO / O*NET /
  Lightcast export is present, and SkillSpan annotates spans only (no taxonomy ids).  **There are therefore no gold
  links for SkillSpan**: recall@k / MRR / link precision against SkillSpan cannot be computed and that slice is
  reported ``skipped``.  Most of the table is the verbatim surface forms of SkillSpan train/dev gold spans plus
  high-confidence Mistral predictions *on this same test split*, so coverage on test predictions is partly circular;
  the provenance of every alias used is reported.
* Gold-labelled linking data we can use:
  (a) ``inputs/evals/linking_probe.jsonl`` - a small hand-written probe set (``source: hand-written``), supplementary
      and NOT drawn from any benchmark;
  (b) leave-one-alias-out probes derived from the curated (``manual_curated``) aliases in the table itself: hold one
      alias out of its entry, link the alias surface, and check that the entry is still found.
* ``TaxonomyLinker.link`` only returns the top-1 link.  Ranked lists for recall@k / MRR are reconstructed here by
  replaying its cascade (exact alias -> normalised name -> fuzzy >= threshold -> BM25 > 0) and de-duplicating by
  canonical id; ``rank_consistency`` checks that rank 1 always equals ``link()``.
* Dense retrieval needs sentence-transformers + model weights, which are absent: reported ``skipped``.
"""

from __future__ import annotations

import json
import random
import tempfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np

from skillllm_v2.evals.contract import EvalContext, EvalResult
from skillllm_v2.evals.extraction_evals import _div, boot_ci, load_gold
from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker, _terms
from skillllm_v2.schema import normalize_text

MODULE = "link"
ALIAS_TABLE = "data/skillllm_v2/taxonomy/skill_aliases.jsonl"
PROBE_PATH = "inputs/evals/linking_probe.jsonl"
PRED_PATH = "artifacts/skillllm_v2/safe_wrapper/linked_skills.jsonl"
DS_PROBE = "linking_probe.jsonl (hand-written supplementary probe set; not a benchmark)"
DS_LOO = "skill_aliases.jsonl leave-one-alias-out (curated aliases; derived from the table, not independent gold)"
DS_PRED = "SkillSpan(HF) test 974-doc subset, safe_wrapper linked_skills (no gold taxonomy ids exist)"
DS_TABLE = "skill_aliases.jsonl (taxonomy table itself)"
METHODS = ("alias", "normalized", "fuzzy", "bm25", "dense")
KS = (1, 5, 10)
AMBIGUITY_MARGIN = 0.05

BASE_CAVEATS = [
    "No gold taxonomy ids exist for SkillSpan and no external taxonomy (ESCO/O*NET/Lightcast) is present offline.",
    "The alias table is built from SkillSpan train/dev gold surfaces and Mistral predictions on this test split, "
    "so exact-alias coverage is partly circular and says little about generalisation to an external taxonomy.",
]
PROBE_CAVEATS = [
    "Hand-written probe set (source: hand-written): small, authored by the project, mix of categories chosen by hand; "
    "supplementary evidence only, absolute numbers are not comparable across papers.",
    "Items are not independent samples of any population; CIs are bootstrap-over-items and only indicate item-sampling noise.",
]


# ----------------------------------------------------------------------------- linker + ranking
def make_linker(ctx: EvalContext, fuzzy_threshold: float = 0.88, table: Optional[Path] = None) -> TaxonomyLinker:
    return TaxonomyLinker(alias_table_paths=[str(table or ctx.path(ALIAS_TABLE))], fuzzy_threshold=fuzzy_threshold)


def load_table(ctx: EvalContext) -> List[Dict[str, Any]]:
    path = ctx.path(ALIAS_TABLE)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bm25_confidence(score: float) -> float:
    return min(0.84, 0.55 + score / (score + 3.0))


def _bm25_scores(linker: TaxonomyLinker, surface: str) -> List[float]:
    """Per-entry BM25 scores replicating TaxonomyLinker._bm25 (reads its private index)."""
    query = _terms(surface)
    docs = linker._bm25_docs
    n_docs = len(docs)
    avg = linker._bm25_avg_len
    scores = [0.0] * n_docs
    if not query:
        return scores
    for i, terms in enumerate(docs):
        score = 0.0
        for term in query:
            df = linker._bm25_doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            freq = terms.count(term)
            denom = freq + 1.5 * (1 - 0.75 + 0.75 * len(terms) / max(1.0, avg))
            score += idf * (freq * 2.5 / denom) if denom else 0.0
        scores[i] = score
    return scores


def rank_candidates(linker: TaxonomyLinker, surface: str) -> List[Dict[str, Any]]:
    """Ranked, de-duplicated candidate list obtained by replaying the linker cascade stage by stage.

    Each item: canonical_skill_id, confidence (the linker's own confidence for that stage), method.
    """
    norm = normalize_text(surface)
    out: List[Dict[str, Any]] = []
    seen: set = set()

    def push(entry: Dict[str, Any], conf: float, method: str) -> None:
        cid = entry["canonical_skill_id"]
        if cid not in seen:
            seen.add(cid)
            out.append({"canonical_skill_id": cid, "confidence": float(conf), "method": method})

    for entry in linker.entries:
        if norm in entry["aliases_norm"]:
            push(entry, 1.0, "alias")
    for entry in linker.entries:
        if norm == normalize_text(entry["canonical_skill_name"]):
            push(entry, 0.96, "normalized")
    fuzzy: List[Tuple[float, int]] = []
    for idx, entry in enumerate(linker.entries):
        best = 0.0
        for alias in entry["aliases_norm"]:
            sm = SequenceMatcher(None, norm, alias)
            if sm.real_quick_ratio() < linker.fuzzy_threshold or sm.quick_ratio() < linker.fuzzy_threshold:
                continue
            best = max(best, sm.ratio())
        if best >= linker.fuzzy_threshold:
            fuzzy.append((best, idx))
    for score, idx in sorted(fuzzy, key=lambda t: (-t[0], t[1])):
        push(linker.entries[idx], score, "fuzzy")
    scores = _bm25_scores(linker, surface)
    for idx in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
        if scores[idx] > 0:
            push(linker.entries[idx], bm25_confidence(scores[idx]), "bm25")
    return out


def margin_of(ranked: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Confidence gap between rank 1 and rank 2 (None when fewer than two candidates)."""
    if len(ranked) < 2:
        return None
    return ranked[0]["confidence"] - ranked[1]["confidence"]


def is_ambiguous(ranked: Sequence[Dict[str, Any]]) -> bool:
    m = margin_of(ranked)
    return m is not None and m < AMBIGUITY_MARGIN


class _Cache:
    """Per-linker cache of (link result, ranked list) by raw surface."""

    def __init__(self, linker: TaxonomyLinker) -> None:
        self.linker = linker
        self._link: Dict[str, Any] = {}
        self._rank: Dict[str, Any] = {}

    def link(self, surface: str) -> Optional[Dict[str, Any]]:
        if surface not in self._link:
            self._link[surface] = self.linker.link(surface)
        return self._link[surface]

    def rank(self, surface: str) -> List[Dict[str, Any]]:
        if surface not in self._rank:
            self._rank[surface] = rank_candidates(self.linker, surface)
        return self._rank[surface]


# ----------------------------------------------------------------------------- metrics on labelled items
def item_scores(gold_id: Optional[str], ranked: Sequence[Dict[str, Any]], top1: Optional[Dict[str, Any]]) -> Dict[str, float]:
    ids = [c["canonical_skill_id"] for c in ranked]
    out: Dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = float(gold_id is not None and gold_id in ids[:k])
    out["rr"] = 1.0 / (ids.index(gold_id) + 1) if gold_id is not None and gold_id in ids else 0.0
    out["linked"] = float(top1 is not None)
    out["correct_top1"] = float(top1 is not None and gold_id is not None and top1["canonical_skill_id"] == gold_id)
    return out


def _mean_ci(values: Sequence[float], ctx: EvalContext, seed_off: int = 0):
    n_boot = min(ctx.n_boot, 200) if ctx.smoke else ctx.n_boot
    arr = np.column_stack([np.asarray(values, dtype=float), np.ones(len(values))]) if len(values) else np.zeros((0, 2))
    return boot_ci(arr, lambda s: _div(s[..., 0], s[..., 1]), n_boot, ctx.seed + seed_off)


def _ratio_ci(num: Sequence[float], den: Sequence[float], ctx: EvalContext, seed_off: int = 0):
    n_boot = min(ctx.n_boot, 200) if ctx.smoke else ctx.n_boot
    arr = np.column_stack([np.asarray(num, float), np.asarray(den, float)]) if len(num) else np.zeros((0, 2))
    return boot_ci(arr, lambda s: _div(s[..., 0], s[..., 1]), n_boot, ctx.seed + seed_off)


def _res(name: str, metric: str, value: Optional[float], n: int, dataset: str, ci=None, details=None, caveats=(),
         status: str = "ok", reason: str = "") -> EvalResult:
    return EvalResult(eval_name=name, module=MODULE, metric=metric, value=value, n=n, dataset=dataset,
                      ci=tuple(ci) if ci else None, details=details or {}, caveats=list(caveats),
                      status=status, reason=reason)


def _skip(name: str, metric: str, dataset: str, reason: str) -> EvalResult:
    return _res(name, metric, None, 0, dataset, caveats=BASE_CAVEATS, status="skipped", reason=reason)


# ----------------------------------------------------------------------------- probe + leave-one-out
def load_probe(ctx: EvalContext) -> Tuple[List[Dict[str, Any]], List[str]]:
    path = ctx.path(PROBE_PATH)
    if not path.exists():
        return [], [f"missing {PROBE_PATH}"]
    table_ids = {r["canonical_skill_id"] for r in load_table(ctx)}
    rows, problems = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") != "hand-written":
            problems.append(f"{row.get('id')}: source must be 'hand-written'")
        if row.get("gold_id") is not None and row["gold_id"] not in table_ids:
            problems.append(f"{row['id']}: gold_id {row['gold_id']!r} not in taxonomy table")
            continue
        rows.append(row)
    return rows, problems


def score_items(items: Sequence[Dict[str, Any]], cache: _Cache) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        top1, ranked = cache.link(it["surface"]), cache.rank(it["surface"])
        sc = item_scores(it.get("gold_id"), ranked, top1)
        consistent = (top1 is None and not ranked) or (top1 is not None and ranked
                                                        and ranked[0]["canonical_skill_id"] == top1["canonical_skill_id"]
                                                        and ranked[0]["method"] == top1["link_method"])
        out.append({**it, **sc, "top1": top1, "ranked_top5": ranked[:5], "ambiguous": is_ambiguous(ranked),
                    "margin": margin_of(ranked), "rank_consistent": consistent})
    return out


def loo_items(ctx: EvalContext) -> List[Dict[str, Any]]:
    """Curated aliases whose surface differs from the canonical name (so holding it out is meaningful)."""
    rows = load_table(ctx)
    out = []
    for i, r in enumerate(rows):
        if r.get("source") != "manual_curated":
            continue
        if r["normalized_alias"] == normalize_text(r["canonical_skill_name"]):
            continue
        out.append({"id": f"loo-{i}", "row_index": i, "surface": r["alias"], "gold_id": r["canonical_skill_id"],
                    "category": "alias_holdout", "source": "alias_table_loo"})
    return out


def score_loo(ctx: EvalContext, items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = load_table(ctx)
    scored = []
    with tempfile.TemporaryDirectory() as tmp:
        for it in items:
            kept = [r for j, r in enumerate(rows) if j != it["row_index"]]
            path = Path(tmp) / f"table_{it['row_index']}.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in kept) + "\n", encoding="utf-8")
            cache = _Cache(make_linker(ctx, table=path))
            scored.extend(score_items([it], cache))
    return scored


def summarize_items(scored: Sequence[Dict[str, Any]], ctx: EvalContext, prefix: str, dataset: str,
                    caveats: Sequence[str]) -> List[EvalResult]:
    results: List[EvalResult] = []
    gold_items = [s for s in scored if s.get("gold_id") is not None]
    n = len(gold_items)
    if not n:
        return [_skip(f"{prefix}.recall_at_1", "recall@1", dataset, "no gold-labelled items")]
    for j, k in enumerate(KS):
        vals = [s[f"recall@{k}"] for s in gold_items]
        results.append(_res(f"{prefix}.recall_at_{k}", f"recall@{k}", sum(vals) / n, n, dataset, _mean_ci(vals, ctx, j),
                            {"n_items_with_gold": n}, caveats))
    rr = [s["rr"] for s in gold_items]
    results.append(_res(f"{prefix}.mrr", "mrr", sum(rr) / n, n, dataset, _mean_ci(rr, ctx, 5), {}, caveats))
    return results


def probe_results(ctx: EvalContext, cache: _Cache) -> List[EvalResult]:
    items, problems = load_probe(ctx)
    if not items:
        return [_skip("link.probe", "all", DS_PROBE, "; ".join(problems) or "probe set empty")]
    scored = score_items(items, cache)
    results = summarize_items(scored, ctx, "link.probe", DS_PROBE, PROBE_CAVEATS)
    gold_items = [s for s in scored if s["gold_id"] is not None]
    null_items = [s for s in scored if s["gold_id"] is None]
    # coverage / precision / false-link rate
    linked = [s for s in scored if s["linked"]]
    results.append(_res("link.probe.coverage", "coverage", len(linked) / len(scored), len(scored), DS_PROBE,
                        _mean_ci([s["linked"] for s in scored], ctx, 6), {"includes_no_match_items": True}, PROBE_CAVEATS))
    correct = [float(s["gold_id"] is not None and s["top1"]["canonical_skill_id"] == s["gold_id"]) for s in linked]
    results.append(_res("link.probe.link_precision", "link_precision", sum(correct) / len(linked) if linked else None,
                        len(linked), DS_PROBE, _mean_ci(correct, ctx, 7) if linked else None,
                        {"definition": "correct links / links emitted; a link on a no-match item counts as wrong",
                         "n_problems_loading": len(problems), "load_problems": problems[:5]}, PROBE_CAVEATS))
    if null_items:
        fl = [s["linked"] for s in null_items]
        results.append(_res("link.probe.no_match_false_link_rate", "false_link_rate", sum(fl) / len(fl), len(fl), DS_PROBE,
                            _mean_ci(fl, ctx, 8),
                            {"false_links": [{"surface": s["surface"], "linked_to": s["top1"]["canonical_skill_id"],
                                              "method": s["top1"]["link_method"], "conf": round(s["top1"]["link_confidence"], 3)}
                                             for s in null_items if s["linked"]],
                             "interpretation": "share of absent skills that still received a link (lower is better); "
                                               "the linker has no abstention threshold, BM25 links anything sharing a token"},
                            PROBE_CAVEATS))
    # per category
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in scored:
        by_cat[s["category"]].append(s)
    for cat, group in sorted(by_cat.items()):
        g = [s for s in group if s["gold_id"] is not None]
        if g:
            v1 = [s["recall@1"] for s in g]
            results.append(_res(f"link.probe.category.{cat}.recall_at_1", "recall@1", sum(v1) / len(g), len(g), DS_PROBE,
                                _mean_ci(v1, ctx, 9),
                                {"category": cat, "recall@5": sum(s["recall@5"] for s in g) / len(g),
                                 "recall@10": sum(s["recall@10"] for s in g) / len(g),
                                 "mrr": sum(s["rr"] for s in g) / len(g),
                                 "errors": [{"surface": s["surface"], "gold": s["gold_id"],
                                             "got": s["top1"]["canonical_skill_id"] if s["top1"] else None}
                                            for s in g if not s["recall@1"]][:8]},
                                PROBE_CAVEATS + ["Per-category n is small; treat as diagnostic."]))
        else:
            fl = [s["linked"] for s in group]
            results.append(_res(f"link.probe.category.{cat}.false_link_rate", "false_link_rate", sum(fl) / len(fl),
                                len(fl), DS_PROBE, _mean_ci(fl, ctx, 9), {"category": cat}, PROBE_CAVEATS))
    # per link method (source type)
    by_method: Dict[str, List[float]] = defaultdict(list)
    for s in linked:
        by_method[s["top1"]["link_method"]].append(float(s["gold_id"] is not None and s["top1"]["canonical_skill_id"] == s["gold_id"]))
    for m in METHODS:
        vals = by_method.get(m, [])
        if m == "dense":
            results.append(_skip("link.probe.method.dense.precision", "link_precision", DS_PROBE,
                                 "dense linker needs sentence-transformers + model weights (not installed / no weights offline)"))
        elif vals:
            results.append(_res(f"link.probe.method.{m}.precision", "link_precision", sum(vals) / len(vals), len(vals),
                                DS_PROBE, _mean_ci(vals, ctx, 10), {"method": m}, PROBE_CAVEATS))
        else:
            results.append(_skip(f"link.probe.method.{m}.precision", "link_precision", DS_PROBE, f"no probe item resolved via {m}"))
    # ambiguity slice (margin between rank 1 and rank 2)
    for name, flag in (("ambiguous_margin_lt_0.05", True), ("unambiguous", False)):
        g = [s for s in gold_items if s["ambiguous"] == flag]
        if g:
            v = [s["recall@1"] for s in g]
            results.append(_res(f"link.probe.slice.{name}.recall_at_1", "recall@1", sum(v) / len(g), len(g), DS_PROBE,
                                _mean_ci(v, ctx, 11), {"slice": name, "margin_threshold": AMBIGUITY_MARGIN},
                                PROBE_CAVEATS))
    # abstention curve: dropping LINKS (never skills) below a confidence threshold
    curve = []
    for tau in (0.0, 0.6, 0.7, 0.8, 0.9, 0.96, 1.0):
        kept = [s for s in linked if s["top1"]["link_confidence"] >= tau]
        ok = sum(1 for s in kept if s["gold_id"] is not None and s["top1"]["canonical_skill_id"] == s["gold_id"])
        curve.append({"tau": tau, "links_kept": len(kept), "link_precision": ok / len(kept) if kept else None,
                      "coverage_of_probe": len(kept) / len(scored), "recall@1_of_gold_items":
                          sum(1 for s in gold_items if s["linked"] and s["top1"]["link_confidence"] >= tau and s["recall@1"]) / max(1, len(gold_items))})
    at = next(c for c in curve if c["tau"] == 0.9)
    results.append(_res("link.abstention.precision_coverage_curve", "link_precision_at_tau_0.9", at["link_precision"],
                        len(linked), DS_PROBE, None,
                        {"curve": curve, "note": "simulation only: the shipped linker has no confidence threshold; abstaining "
                                                 "would drop the LINK (canonical id), never the extracted skill"},
                        PROBE_CAVEATS))
    results.append(_res("link.probe.rank_consistency", "share_rank1_equals_link", sum(s["rank_consistent"] for s in scored) / len(scored),
                        len(scored), DS_PROBE, None,
                        {"inconsistent": [s["surface"] for s in scored if not s["rank_consistent"]][:10]},
                        ["Validates the reconstructed ranked list against TaxonomyLinker.link (should be 1.0)."],
                        status="ok" if all(s["rank_consistent"] for s in scored) else "failed",
                        reason="" if all(s["rank_consistent"] for s in scored) else "reconstructed rank 1 differs from linker.link"))
    # negative controls on the labelled probe
    results.append(negative_control_result(gold_items, ctx, load_table(ctx)))
    return results


def negative_control_result(gold_items: Sequence[Dict[str, Any]], ctx: EvalContext, table_rows: Sequence[Dict[str, Any]],
                            n_draws: int = 200) -> EvalResult:
    """Random taxonomy ids and across-item shuffled links must score ~0 against the same gold."""
    ids = sorted({r["canonical_skill_id"] for r in table_rows})
    rng = random.Random(ctx.seed)
    gold = [s["gold_id"] for s in gold_items]
    top1 = [s["top1"]["canonical_skill_id"] if s["top1"] else None for s in gold_items]
    acc_random = []
    r10_random = []
    for _ in range(n_draws):
        acc_random.append(sum(rng.choice(ids) == g for g in gold) / len(gold))
        r10_random.append(sum(g in rng.sample(ids, 10) for g in gold) / len(gold))
    acc_shuf = []
    for _ in range(n_draws):
        order = list(range(len(gold)))
        rng.shuffle(order)
        shifted = [top1[order[(i + 1) % len(order)]] for i in range(len(order))]
        perm = [None] * len(order)
        for i in range(len(order)):
            perm[order[i]] = shifted[i]
        acc_shuf.append(sum(p == g for p, g in zip(perm, gold)) / len(gold))
    mean_r, mean_s = sum(acc_random) / n_draws, sum(acc_shuf) / n_draws
    ok = mean_r < 0.02 and mean_s < 0.05
    return _res("link.negative_control", "random_id_top1_accuracy", mean_r, len(gold), DS_PROBE, None,
                {"random_ids_top1_mean": mean_r, "random_ids_top1_max": max(acc_random),
                 "random_ids_recall@10_mean": sum(r10_random) / n_draws, "expected_chance_top1": 1 / len(ids),
                 "shuffled_links_across_items_top1_mean": mean_s, "shuffled_links_across_items_top1_max": max(acc_shuf),
                 "n_draws": n_draws, "taxonomy_size": len(ids)},
                PROBE_CAVEATS + ["Shuffled-link control is non-zero only because several probe items share the same gold id."],
                status="ok" if ok else "failed", reason="" if ok else "controls did not collapse to chance")


# ----------------------------------------------------------------------------- pipeline predictions
def _strip(item: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(item)
    s["canonical_skill_id"] = None
    s["canonical_skill_name"] = None
    s["metadata"] = {k: v for k, v in (item.get("metadata") or {}).items() if k != "taxonomy_link"}
    return s


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


SPAN_FIELDS = ("surface_form", "label", "start", "end", "source", "confidence")


def check_never_removes(linker: TaxonomyLinker, docs: Sequence[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Run link_candidates over stripped candidates: nothing may be dropped, reordered or edited (attach-only)."""
    original, memo = linker.link, {}

    def cached(surface: str) -> Optional[Dict[str, Any]]:  # identical surfaces -> identical result; just faster
        if surface not in memo:
            memo[surface] = original(surface)
        return memo[surface]

    linker.link = cached  # type: ignore[method-assign]
    total = removed = edited = unlinked_kept = low_conf_kept = 0
    for cands in docs:
        out = linker.link_candidates([_strip(c) for c in cands])
        total += len(cands)
        if len(out) != len(cands):
            removed += abs(len(cands) - len(out))
            continue
        for a, b in zip(cands, out):
            if any(a.get(f) != b.get(f) for f in SPAN_FIELDS):
                edited += 1
            link = (b.get("metadata") or {}).get("taxonomy_link")
            if b.get("canonical_skill_id") is None:
                unlinked_kept += 1
            elif link and link["link_confidence"] < 0.7:
                low_conf_kept += 1
    return {"fuzzy_threshold": linker.fuzzy_threshold, "skills_in": total, "skills_removed": removed,
            "skills_edited": edited, "kept_without_link": unlinked_kept, "kept_with_low_conf_link": low_conf_kept}


def prediction_results(ctx: EvalContext, cache: _Cache) -> List[EvalResult]:
    path = ctx.path(PRED_PATH)
    if not path.exists():
        return [_skip("link.coverage", "coverage", DS_PRED, f"{PRED_PATH} missing")]
    rows = _load_jsonl(path)
    gold = load_gold(ctx)
    if ctx.smoke:
        rows = rows[: int(ctx.extra.get("smoke_docs", 40))]
    n_docs = len(rows)
    table = load_table(ctx)
    prov: Dict[str, set] = defaultdict(set)
    alias_count: Counter = Counter()
    for r in table:
        prov[r["normalized_alias"]].add(r["source"])
        alias_count[r["canonical_skill_id"]] += 1
    results: List[EvalResult] = []

    per_doc = []  # per doc stats
    method_counts: Counter = Counter()
    method_conf: Dict[str, List[float]] = defaultdict(list)
    tp_flags: List[Tuple[bool, bool]] = []  # (is_tp, linked)
    label_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    prov_counts: Counter = Counter()
    ambiguous_n = ambiguous_linked = linked_n = multi_alias = 0
    agree = disagree = 0
    for row in rows:
        gold_keys = {(g["start"], g["end"], g["label"]) for g in gold.get(row["id"], {}).get("gold_spans", [])}
        linked_d = tot_d = 0
        for sk in row["linked_skills"]:
            tot_d += 1
            link = (sk.get("metadata") or {}).get("taxonomy_link")
            is_linked = sk.get("canonical_skill_id") is not None
            method = link["link_method"] if is_linked and link else "none"
            method_counts[method] += 1
            if is_linked and link:
                method_conf[method].append(link["link_confidence"])
            linked_d += is_linked
            is_tp = (sk.get("start"), sk.get("end"), sk.get("label")) in gold_keys
            tp_flags.append((is_tp, is_linked))
            label_stats[sk["label"]][0] += is_linked
            label_stats[sk["label"]][1] += 1
            if is_linked:
                linked_n += 1
                if method in ("alias", "normalized"):
                    for src in prov.get(normalize_text(sk["surface_form"]), {"unknown"}):
                        prov_counts[src] += 1
                multi_alias += alias_count.get(sk["canonical_skill_id"], 0) >= 2
                live = cache.link(sk["surface_form"])
                if live and live["canonical_skill_id"] == sk["canonical_skill_id"]:
                    agree += 1
                else:
                    disagree += 1
            if cache.rank(sk["surface_form"]) and is_ambiguous(cache.rank(sk["surface_form"])):
                ambiguous_n += 1
                ambiguous_linked += is_linked
        per_doc.append((linked_d, tot_d))
    total = sum(t for _, t in per_doc)
    linked_total = sum(l for l, _ in per_doc)
    cov_caveats = BASE_CAVEATS + ["Coverage counts any non-null link, including low-confidence BM25 links of unknown correctness."]
    results.append(_res("link.coverage", "coverage", linked_total / total if total else None, n_docs, DS_PRED,
                        _ratio_ci([l for l, _ in per_doc], [t for _, t in per_doc], ctx, 20),
                        {"n_spans": total, "n_linked": linked_total, "stored_links": True,
                         "reference": "reports/skillllm_v2/full_cag_graph_metrics.json linking_accuracy is this same quantity"},
                        cov_caveats))
    for lab, (l, t) in sorted(label_stats.items()):
        results.append(_res(f"link.coverage.label.{lab}", "coverage", l / t if t else None, t, DS_PRED, None,
                            {"n_linked": l, "n_spans": t, "ci_note": "n is spans; no doc-level CI computed"}, cov_caveats))
    for name, flag in (("tp_spans", True), ("fp_spans", False)):
        sel = [lk for tp, lk in tp_flags if tp == flag]
        if sel:
            results.append(_res(f"link.coverage.{name}", "coverage", sum(sel) / len(sel), len(sel), DS_PRED, None,
                                {"interpretation": "if linking were discriminative, wrong (FP) spans would be linked less often than correct (TP) spans",
                                 "ci_note": "n is spans"}, cov_caveats))
    for m in METHODS:
        if m == "dense":
            results.append(_skip("link.method.dense.share", "share_of_spans", DS_PRED,
                                 "dense linker disabled in shipped pipeline (dense_model_name=None); no sentence-transformers/model weights offline"))
            continue
        results.append(_res(f"link.method.{m}.share", "share_of_spans", method_counts.get(m, 0) / total if total else None,
                            total, DS_PRED, None,
                            {"method": m, "count": method_counts.get(m, 0),
                             "mean_link_confidence": (sum(method_conf[m]) / len(method_conf[m])) if method_conf.get(m) else None,
                             "note": "'alias' and 'normalized' are the exact-match stages"}, cov_caveats))
    results.append(_res("link.method.none.share", "share_of_spans", method_counts.get("none", 0) / total if total else None,
                        total, DS_PRED, None, {"count": method_counts.get("none", 0)}, cov_caveats))
    exact_total = sum(prov_counts.values())
    results.append(_res("link.alias_provenance", "share_exact_links_via_mistral_test_aliases",
                        prov_counts.get("mistral_high_confidence", 0) / exact_total if exact_total else None,
                        exact_total, DS_PRED, None,
                        {"alias_source_counts": dict(prov_counts),
                         "interpretation": "exact links resolved through aliases mined from Mistral predictions on this test split are circular; "
                                           "skillspan_gold_train_dev aliases are train-derived (legitimate but self-referential)"},
                        BASE_CAVEATS))
    results.append(_res("link.relink_agreement", "agreement_with_stored_links", agree / (agree + disagree) if agree + disagree else None,
                        agree + disagree, DS_PRED, None,
                        {"agree": agree, "disagree": disagree,
                         "note": "live TaxonomyLinker vs links stored in the artifact; <1.0 means the alias table changed since the run"},
                        BASE_CAVEATS))
    # slices without gold: prevalence only
    results.append(_res("link.slice.ambiguity", "share_ambiguous", ambiguous_n / total if total else None, total, DS_PRED, None,
                        {"ambiguous_spans": ambiguous_n, "ambiguous_linked": ambiguous_linked,
                         "margin_threshold": AMBIGUITY_MARGIN,
                         "definition": "gap between rank-1 and rank-2 confidence of the reconstructed ranked list is below the threshold",
                         "accuracy": "not computable: no gold links on SkillSpan (see probe-set ambiguity slice for labelled data)"},
                        BASE_CAVEATS))
    results.append(_res("link.slice.multi_alias_entries", "share_links_to_multi_alias_entries",
                        multi_alias / linked_n if linked_n else None, linked_n, DS_PRED, None,
                        {"definition": "linked spans whose canonical entry has >= 2 alias rows (curated abbreviation/variant groups)"},
                        BASE_CAVEATS))
    results.append(_skip("link.skillspan_gold_links", "recall@k/mrr/precision", "SkillSpan(HF) test",
                         "SkillSpan has span annotations only (no taxonomy ids) and no ESCO/O*NET/Lightcast export is "
                         "present offline, so recall@1/5/10, MRR and link precision against gold ids cannot be measured on it; "
                         "see link.probe.* (hand-written) and link.alias_holdout.* (derived) for labelled supplementary data."))

    # gold-surface coverage: what could be linked if extraction were perfect
    gl = gt = 0
    gold_methods: Counter = Counter()
    gold_by = {r["id"]: gold[r["id"]] for r in rows if r["id"] in gold}
    gdoc = []
    for doc in gold_by.values():
        l = t = 0
        for g in doc["gold_spans"]:
            surf = doc["text"][g["start"]:g["end"]]
            lk = cache.link(surf)
            t += 1
            l += lk is not None
            gold_methods[lk["link_method"] if lk else "none"] += 1
        gdoc.append((l, t))
    gl, gt = sum(l for l, _ in gdoc), sum(t for _, t in gdoc)
    results.append(_res("link.gold_surface_coverage", "coverage", gl / gt if gt else None, len(gdoc), DS_PRED,
                        _ratio_ci([l for l, _ in gdoc], [t for _, t in gdoc], ctx, 21),
                        {"n_gold_spans": gt, "method_counts": dict(gold_methods),
                         "note": "links gold span surfaces (not predictions): upper bound on coverage under perfect extraction"},
                        BASE_CAVEATS + ["Gold train/dev surfaces seeded the table, so test-gold exact coverage is optimistic."]))

    # abstention: link-stage must never remove a skill
    cands = [r["linked_skills"] for r in rows]
    art = check_never_removes(make_linker(ctx), cands)
    pred_rows = {r["id"]: r for r in _load_jsonl(ctx.path("artifacts/skillllm_v2/safe_wrapper/predictions.jsonl"))} \
        if ctx.path("artifacts/skillllm_v2/safe_wrapper/predictions.jsonl").exists() else {}
    art_missing = 0
    art_checked = 0
    for r in rows:
        pr = pred_rows.get(r["id"])
        if pr is None:
            continue
        art_checked += 1
        ka = sorted((s["start"], s["end"], s["label"]) for s in pr["predictions"])
        kb = sorted((s["start"], s["end"], s["label"]) for s in r["linked_skills"])
        art_missing += ka != kb
    sweep = []
    sample = cands[: (30 if ctx.smoke else 250)]
    for thr in (0.0, 0.7, 0.88, 1.01):
        sweep.append(check_never_removes(make_linker(ctx, fuzzy_threshold=thr), sample))
    unlinked_kept = sum(1 for c in cands for s in c if s.get("canonical_skill_id") is None)
    ok = (art["skills_removed"] == 0 and art["skills_edited"] == 0 and art_missing == 0
          and all(s["skills_removed"] == 0 and s["skills_edited"] == 0 for s in sweep))
    results.append(_res("link.abstention.never_removes_skill", "share_of_skills_retained",
                        1.0 - (art["skills_removed"] / art["skills_in"] if art["skills_in"] else 0.0), art["skills_in"], DS_PRED,
                        None, {"live_relink": art, "threshold_sweep_on_sample": sweep,
                               "artifact_pre_vs_post_link_docs_checked": art_checked,
                               "artifact_docs_with_span_set_mismatch": art_missing,
                               "stored_unlinked_skills_still_present": unlinked_kept,
                               "assertion": "link_candidates() is attach-only: no skill is removed, reordered or edited at any "
                                            "fuzzy threshold, and unlinked / low-confidence-linked skills stay in the output"},
                        ["Checks the linking stage only; the verifier/calibrator stages upstream may legitimately drop spans."],
                        status="ok" if ok else "failed", reason="" if ok else "linking stage removed or edited skills"))
    return results


def table_results(ctx: EvalContext) -> List[EvalResult]:
    rows = load_table(ctx)
    cache = _Cache(make_linker(ctx))
    wrong = [(r["alias"], r["canonical_skill_id"], (cache.link(r["alias"]) or {}).get("canonical_skill_id")) for r in rows
             if (cache.link(r["alias"]) or {}).get("canonical_skill_id") != r["canonical_skill_id"]]
    return [_res("link.table_identity_sanity", "top1_accuracy_on_own_aliases", 1 - len(wrong) / len(rows), len(rows), DS_TABLE,
                 None, {"mismatches": wrong[:10], "n_entries": len({r["canonical_skill_id"] for r in rows}),
                        "alias_sources": dict(Counter(r["source"] for r in rows))},
                 ["Sanity check only: every alias must resolve to its own entry (detects alias collisions). Trivially high by construction."],
                 status="ok" if not wrong else "failed", reason="" if not wrong else "alias collisions in the table")]


def run(ctx: EvalContext) -> List[EvalResult]:
    if not ctx.path(ALIAS_TABLE).exists():
        return [_skip("link.all", "all", DS_TABLE, f"{ALIAS_TABLE} missing")]
    cache = _Cache(make_linker(ctx))
    results: List[EvalResult] = []
    results.extend(probe_results(ctx, cache))
    loo = loo_items(ctx)
    if loo:
        scored = score_loo(ctx, loo)
        results.extend(summarize_items(scored, ctx, "link.alias_holdout", DS_LOO,
                                       ["Derived from curated aliases in the table: an abbreviation held out of its entry must be recovered "
                                        "from the rest of the entry. Not independent gold; very small n."]))
        results.append(_res("link.alias_holdout.errors", "n_not_recovered_at_rank1", float(sum(1 for s in scored if not s["recall@1"])),
                            len(scored), DS_LOO, None,
                            {"not_recovered": [{"surface": s["surface"], "gold": s["gold_id"],
                                                "got": s["top1"]["canonical_skill_id"] if s["top1"] else None} for s in scored if not s["recall@1"]]},
                            ["Abbreviations (AWS, NLP, ML, SQL ...) cannot be matched by fuzzy/BM25 once their alias row is removed."]))
    else:
        results.append(_skip("link.alias_holdout", "recall@k", DS_LOO, "no curated multi-form aliases in table"))
    results.extend(table_results(ctx))
    results.extend(prediction_results(ctx, cache))
    return results
