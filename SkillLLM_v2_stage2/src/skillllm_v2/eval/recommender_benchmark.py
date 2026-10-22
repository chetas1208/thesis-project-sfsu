"""Course-recommendation benchmark (v2: independent labels, graded metrics, bootstrap statistics).

    PYTHONPATH=src python -m skillllm_v2.eval.recommender_benchmark \
        [--catalog inputs/courses/course_catalog_large.jsonl] [--n-boot 10000]

What changed relative to the first version
  * Relevance labels are NOT derived from the recommender's own gap output.  They come from
    inputs/courses/relevance_labels.jsonl, produced by skillllm_v2.eval.build_relevance_labels from
    hand-authored per-scenario gold needs (see that module for the procedure and its limits).
  * 48 (user, job) scenarios, a 171-course SYNTHETIC catalogue.
  * Metrics: nDCG@10 (graded gain), Precision@5, Recall@10, MAP@10 (relevant = grade >= 2) plus the
    original Weighted Gap Coverage@5, Prerequisite Violation Rate, Redundancy@5, Explanation
    Completeness.  Gap coverage / redundancy are now computed against the GOLD open needs for every
    system (not each system's own gap list) so systems are scored on the same yardstick.
  * Mean + 95% percentile-bootstrap CI over scenarios; paired bootstrap of Full-vs-baseline with
    Holm-adjusted p-values per metric.
  * Fair baselines: every text/overlap baseline also has a "+ prereq filter" variant that applies the
    same chained prerequisite feasibility rule used by the violation metric, and a user-aware,
    graph-free "Gap-Overlap greedy" baseline isolates what the knowledge graph adds beyond knowing the
    user's gap.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.eval import ranking_metrics as rm
from skillllm_v2.eval.course_recommendation_eval import (
    explanation_completeness_rate,
    prerequisite_violation_rate,
    redundancy_at_k,
    weighted_gap_coverage_at_k,
)
from skillllm_v2.schema import normalize_text

ROOT = Path(__file__).resolve().parents[3]
K_SMALL = 5
K_RANK = 10
REL_THRESHOLD = 2.0

SYSTEM_FULL = "SkillLLM_v2 Full Graph Recommender"
SYSTEMS = [
    "BM25 Keyword",
    "BM25 + prereq filter",
    "Dense Semantic",
    "Dense + prereq filter",
    "Skill Overlap (user-agnostic)",
    "Gap-Overlap Greedy (no graph)",
    "Gap-Overlap Greedy + prereq filter",
    SYSTEM_FULL,
]

# (metric key, display name, higher_is_better)
METRICS = [
    ("ndcg10", "nDCG@10", True),
    ("p5", "Precision@5", True),
    ("recall10", "Recall@10", True),
    ("map10", "MAP@10", True),
    ("gap_cov5", "Weighted Gap Coverage@5", True),
    ("prereq_viol5", "Prerequisite Violation Rate@5", False),
    ("redundancy5", "Redundancy@5", False),
    ("expl5", "Explanation Completeness@5", True),
]
HEADLINE = ["ndcg10", "p5", "recall10", "map10"]

_STOP = set("a an and are as at be by for from has have in into is it of on or our that the their this to we will with you your who "
            "can able not more all any such per etc using use work working team role experience years year strong".split())


def _tok(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#/.]+", text.lower()) if len(t) > 1 and t not in _STOP]


def course_text(c: CourseRecord) -> str:
    return f"{c.title}. {c.description} {' '.join(c.learning_outcomes)} {' '.join(c.teaches_skills)}"


# ------------------------------------------------------------------------------------------------
# prerequisite logic shared by metric + filter variants
# ------------------------------------------------------------------------------------------------
def prereqs_ok(course: CourseRecord, known: set[str]) -> bool:
    return all(normalize_text(p) in known for p in course.prerequisites if normalize_text(p))


def apply_prereq_filter(ranked: list[CourseRecord], user_known: set[str], k: int) -> list[CourseRecord]:
    """Walk the ranked list; keep a course only if its prerequisites are known or taught by an already
    kept course (same chained rule as prerequisite_violation_rate(allow_chained_learning=True))."""
    known = {normalize_text(s) for s in user_known}
    kept: list[CourseRecord] = []
    for c in ranked:
        if prereqs_ok(c, known):
            kept.append(c)
            known |= {normalize_text(s) for s in c.teaches_skills}
            if len(kept) >= k:
                break
    return kept


# ------------------------------------------------------------------------------------------------
# baselines
# ------------------------------------------------------------------------------------------------
class Bm25Index:
    def __init__(self, catalog: CourseCatalog, k1: float = 1.5, b: float = 0.75) -> None:
        self.courses = catalog.list_courses()
        self.docs = [Counter(_tok(course_text(c))) for c in self.courses]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(d.keys())
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list[tuple[float, CourseRecord]]:
        n = len(self.courses)
        q = _tok(query)
        scored = []
        for c, d, L in zip(self.courses, self.docs, self.lens):
            s = 0.0
            for t in set(q):
                tf = d.get(t, 0)
                if not tf:
                    continue
                idf = math.log((n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0)
                s += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * L / self.avg))
            scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], -(x[1].rating or 0.0), x[1].course_id))
        return scored


def _mask_broken_optional_modules() -> None:
    """transformers imports torchvision/torchaudio opportunistically; in this environment those wheels are
    built for a different torch/CUDA and raise at import time, which would break sentence-transformers.
    Mask them (only if they are actually broken) - text embedding does not need them."""
    import importlib
    import sys

    for name in ("torchvision", "torchaudio"):
        try:
            importlib.import_module(name)
        except Exception:
            for m in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
                sys.modules.pop(m, None)
            sys.modules[name] = None  # type: ignore[assignment]


class DenseIndex:
    """Sentence-embedding retrieval (all-MiniLM-L6-v2, loaded offline from the local HF cache).
    Falls back to TF-IDF cosine if sentence-transformers / the model are unavailable (and says so)."""

    def __init__(self, catalog: CourseCatalog) -> None:
        self.courses = catalog.list_courses()
        self.backend = "all-MiniLM-L6-v2"
        self.model = None
        try:
            _mask_broken_optional_modules()
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            self.doc_emb = self.model.encode([course_text(c) for c in self.courses], normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.backend = f"tfidf-cosine-fallback ({type(exc).__name__})"
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.tfidf = TfidfVectorizer(tokenizer=_tok, lowercase=False, token_pattern=None)
            self.doc_emb = self.tfidf.fit_transform([course_text(c) for c in self.courses])

    def rank(self, query: str) -> list[tuple[float, CourseRecord]]:
        if self.model is not None:
            q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            sims = self.doc_emb @ q
        else:
            sims = (self.doc_emb @ self.tfidf.transform([query]).T).toarray().ravel()
        scored = [(float(s), c) for s, c in zip(sims, self.courses)]
        scored.sort(key=lambda x: (-x[0], x[1].course_id))
        return scored


def skill_overlap_rank(catalog: CourseCatalog, job: dict[str, Any]) -> list[tuple[float, CourseRecord]]:
    """User-agnostic: overlap of course skills with ALL job skills (what the original baseline did)."""
    job_skills = {normalize_text(s) for s in job.get("required_skills", []) + job.get("preferred_skills", [])}
    scored = [(float(sum(1 for s in c.teaches_skills if normalize_text(s) in job_skills)), c) for c in catalog]
    scored.sort(key=lambda x: (-x[0], -(x[1].rating or 0.0), x[1].course_id))
    return scored


def user_gap_weights(user: dict[str, Any], job: dict[str, Any]) -> dict[str, float]:
    """Structured-input gap for the user-aware baseline: required=1.0, preferred=0.7, minus skills the user
    has at non-beginner mastery.  Uses ONLY the system-visible inputs (not the gold needs)."""
    have = set()
    for s in user.get("skills", []):
        if str(s.get("mastery_level", "proficient")).lower() in {"beginner"}:
            continue
        have.add(normalize_text(s.get("canonical_skill_name") or s.get("surface_form") or ""))
    gaps: dict[str, float] = {}
    for s in job.get("preferred_skills", []):
        gaps[normalize_text(s)] = 0.7
    for s in job.get("required_skills", []):
        gaps[normalize_text(s)] = 1.0
    return {k: v for k, v in gaps.items() if k not in have}


def gap_overlap_greedy(catalog: CourseCatalog, gaps: dict[str, float], user_known: set[str], prereq_filter: bool, k: int) -> list[CourseRecord]:
    """Greedy weighted set-cover of the user's visible gap; optional chained prerequisite feasibility.
    No knowledge graph, no text matching.  This is the strongest simple non-graph baseline."""
    known = {normalize_text(s) for s in user_known}
    remaining = list(catalog.list_courses())
    covered: set[str] = set()
    out: list[CourseRecord] = []
    while remaining and len(out) < k:
        best, best_key = None, None
        for c in remaining:
            if prereq_filter and not prereqs_ok(c, known):
                continue
            taught = {normalize_text(s) for s in c.teaches_skills}
            new_gain = sum(w for g, w in gaps.items() if g in taught and g not in covered)
            total = sum(w for g, w in gaps.items() if g in taught)
            key = (new_gain, total, c.rating or 0.0, c.course_id)
            # deterministic: larger gain first, then larger total overlap, then rating, then id (lexicographically smaller wins ties)
            if best_key is None or key[:3] > best_key[:3] or (key[:3] == best_key[:3] and key[3] < best_key[3]):
                best, best_key = c, key
        if best is None:
            break
        out.append(best)
        remaining.remove(best)
        taught = {normalize_text(s) for s in best.teaches_skills}
        covered |= taught
        known |= taught
    return out


# ------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def as_rec(c: CourseRecord, explanation: Any = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    d = {"course_id": c.course_id, "title": c.title, "teaches_skills": list(c.teaches_skills), "prerequisites": list(c.prerequisites)}
    if explanation is not None:
        d["explanation"] = explanation
    if extra:
        d.update(extra)
    return d


def evaluate_scenario_lists(
    recs: list[dict[str, Any]],
    grades: dict[str, float],
    grades_content: dict[str, float],
    gold_gaps: list[dict[str, Any]],
    user_known: set[str],
    catalog_by_id: dict[str, CourseRecord],
) -> dict[str, float | None]:
    ids = [r["course_id"] for r in recs]
    norm = [as_rec(catalog_by_id[i]) for i in ids]  # same catalogue view for every system
    out: dict[str, float | None] = {
        "ndcg10": rm.ndcg_at_k(ids, grades, K_RANK),
        "p5": rm.precision_at_k(ids, grades, K_SMALL, REL_THRESHOLD),
        "recall10": rm.recall_at_k(ids, grades, K_RANK, REL_THRESHOLD),
        "map10": rm.average_precision_at_k(ids, grades, K_RANK, REL_THRESHOLD),
        "gap_cov5": weighted_gap_coverage_at_k(norm, gold_gaps, k=K_SMALL),
        "prereq_viol5": prerequisite_violation_rate(norm, user_known, k=K_SMALL, allow_chained_learning=True),
        "redundancy5": redundancy_at_k(norm, gold_gaps, k=K_SMALL),
        "expl5": explanation_completeness_rate(recs, k=K_SMALL),
        "n_returned": float(len(ids)),
        # sensitivity (labels that ignore the prerequisite-feasibility rule)
        "ndcg10_content_only": rm.ndcg_at_k(ids, grades_content, K_RANK),
        "p5_content_only": rm.precision_at_k(ids, grades_content, K_SMALL, REL_THRESHOLD),
    }
    return out


def run_recommender_benchmark(
    catalog_path: str | Path = ROOT / "inputs/courses/course_catalog_large.jsonl",
    scenarios_path: str | Path = ROOT / "inputs/courses/benchmark_scenarios.jsonl",
    labels_path: str | Path = ROOT / "inputs/courses/relevance_labels.jsonl",
    n_boot: int = 10000,
    seed: int = 20260101,
    include_full: bool = True,
    pipeline: Any = None,
) -> dict[str, Any]:
    catalog = CourseCatalog.load_from_file(catalog_path)
    by_id = {c.course_id: c for c in catalog}
    scenarios = load_jsonl(Path(scenarios_path))
    labels = {r["scenario_id"]: r for r in load_jsonl(Path(labels_path))}

    bm25 = Bm25Index(catalog)
    dense = DenseIndex(catalog)
    full = None
    base_graph = None
    if include_full:
        if pipeline is None:
            from skillllm_v2.pipeline.course_recommendation_pipeline import CourseRecommendationPipeline

            pipeline = CourseRecommendationPipeline(catalog_path=str(catalog_path))
        full = pipeline
        base_graph = full.graph.graph.copy()  # networkx MultiDiGraph snapshot (builder itself is not deep-copyable)

    systems = [s for s in SYSTEMS if include_full or s != SYSTEM_FULL]
    per: dict[str, dict[str, list[float | None]]] = {s: {} for s in systems}
    detail: dict[str, dict[str, list[str]]] = {s: {} for s in systems}

    for sc in scenarios:
        sid = sc["scenario_id"]
        user, job = sc["user_profile"], sc["target_job"]
        lab = labels[sid]
        grades = {c.course_id: float(lab["labels"].get(c.course_id, 0)) for c in catalog}
        grades_c = {c.course_id: float(lab["labels_content_only"].get(c.course_id, 0)) for c in catalog}
        needs = sc["gold"]["needs"]
        known_all = set(sc["gold"]["known"]) | set(sc["gold"]["weak"])
        gold_gaps = [{"skill_name": s, "importance": float(w)} for s, w in needs.items() if s not in set(sc["gold"]["known"])]
        user_known = {normalize_text(s) for s in known_all}
        query = f"{job['title']}. {job['text']}"

        lists: dict[str, list[dict[str, Any]]] = {}
        bm_rank = [c for _, c in bm25.rank(query)]
        lists["BM25 Keyword"] = [as_rec(c, "BM25 keyword match") for c in bm_rank[:K_RANK]]
        lists["BM25 + prereq filter"] = [as_rec(c, "BM25 keyword match + prerequisite filter") for c in apply_prereq_filter(bm_rank, user_known, K_RANK)]
        de_rank = [c for _, c in dense.rank(query)]
        lists["Dense Semantic"] = [as_rec(c, "embedding cosine similarity") for c in de_rank[:K_RANK]]
        lists["Dense + prereq filter"] = [as_rec(c, "embedding cosine similarity + prerequisite filter") for c in apply_prereq_filter(de_rank, user_known, K_RANK)]
        so_rank = [c for _, c in skill_overlap_rank(catalog, job)]
        lists["Skill Overlap (user-agnostic)"] = [as_rec(c, "job-skill overlap count") for c in so_rank[:K_RANK]]
        gaps = user_gap_weights(user, job)
        lists["Gap-Overlap Greedy (no graph)"] = [as_rec(c, "greedy weighted gap cover") for c in gap_overlap_greedy(catalog, gaps, user_known, False, K_RANK)]
        lists["Gap-Overlap Greedy + prereq filter"] = [as_rec(c, "greedy weighted gap cover + prerequisite filter") for c in gap_overlap_greedy(catalog, gaps, user_known, True, K_RANK)]
        if full is not None:
            full.graph.graph = base_graph.copy()  # isolate scenarios: no state leaks between users/jobs
            res = full.run(user_input=copy.deepcopy(user), job_input=copy.deepcopy(job), top_k=K_RANK)
            lists[SYSTEM_FULL] = res["recommendations"]

        for s in systems:
            m = evaluate_scenario_lists(lists[s], grades, grades_c, gold_gaps, user_known, by_id)
            for k, v in m.items():
                per[s].setdefault(k, []).append(v)
            detail[s][sid] = [r["course_id"] for r in lists[s]]

    # ----------------------------------------------------------------------- aggregation
    all_keys = [m[0] for m in METRICS] + ["n_returned", "ndcg10_content_only", "p5_content_only"]
    n_sc = len(scenarios)
    valid_idx = {k: [i for i in range(n_sc) if all(per[s][k][i] is not None for s in systems)] for k in all_keys}

    def arr(s: str, k: str) -> list[float]:
        return [float(per[s][k][i]) for i in valid_idx[k]]

    summary: dict[str, dict[str, Any]] = {}
    for s in systems:
        summary[s] = {}
        for k in all_keys:
            mean, lo, hi = rm.bootstrap_mean_ci(arr(s, k), n_boot=n_boot, seed=seed)
            summary[s][k] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(valid_idx[k])}

    paired: dict[str, dict[str, Any]] = {}
    if include_full:
        baselines = [s for s in systems if s != SYSTEM_FULL]
        for k in all_keys:
            raw = {b: rm.paired_bootstrap(arr(SYSTEM_FULL, k), arr(b, k), n_boot=n_boot, seed=seed + 1) for b in baselines}
            adj = rm.holm_adjust([raw[b]["p_value"] for b in baselines])
            for b, a in zip(baselines, adj):
                raw[b]["p_holm"] = a
            for b in baselines:
                paired.setdefault(b, {})[k] = raw[b]

    label_stats = {
        "n_scenarios": n_sc,
        "n_courses": len(catalog),
        "mean_relevant_grade_ge2_per_scenario": sum(sum(1 for g in labels[sc["scenario_id"]]["labels"].values() if g >= 2) for sc in scenarios) / n_sc,
        "mean_relevant_grade_ge1_per_scenario": sum(len(labels[sc["scenario_id"]]["labels"]) for sc in scenarios) / n_sc,
        "grade_totals": dict(sum((Counter({int(g): n for g, n in labels[sc["scenario_id"]]["grade_counts"].items()}) for sc in scenarios), Counter())),
        "label_version": next(iter(labels.values()))["label_version"],
        "scenario_sources": dict(Counter(sc["source"] for sc in scenarios)),
    }
    return {
        "config": {"k_small": K_SMALL, "k_rank": K_RANK, "relevant_if_grade_at_least": REL_THRESHOLD, "n_boot": n_boot, "seed": seed,
                   "dense_backend": dense.backend, "catalog_path": str(catalog_path)},
        "label_stats": label_stats,
        "systems": summary,
        "paired_vs_full": paired,
        "per_scenario": {s: {k: per[s][k] for k in all_keys} for s in systems},
        "scenario_ids": [sc["scenario_id"] for sc in scenarios],
        "rankings": detail,
    }


# ------------------------------------------------------------------------------------------------
# reporting
# ------------------------------------------------------------------------------------------------
def _fmt(m: dict[str, float], digits: int = 3) -> str:
    return f"{m['mean']:.{digits}f} [{m['ci_lo']:.{digits}f}, {m['ci_hi']:.{digits}f}]"


def _verdict(p: dict[str, Any], higher_better: bool) -> str:
    """Return 'better' / 'worse' / 'indistinguishable' for Full vs baseline."""
    d = p["mean_diff"] if higher_better else -p["mean_diff"]
    sig = p["p_holm"] < 0.05 and (p["ci_lo"] > 0 or p["ci_hi"] < 0)
    if not sig:
        return "indistinguishable"
    return "better" if d > 0 else "worse"


def key_findings(res: dict[str, Any]) -> list[str]:
    S = res["systems"]
    P = res["paired_vs_full"]
    names = {k: n for k, n, _ in METRICS}
    hb = {k: h for k, _, h in METRICS}
    out: list[str] = []
    baselines = [s for s in S if s != SYSTEM_FULL]
    # relevance metrics
    for k in HEADLINE:
        better = [b for b in baselines if _verdict(P[b][k], hb[k]) == "better"]
        worse = [b for b in baselines if _verdict(P[b][k], hb[k]) == "worse"]
        ind = [b for b in baselines if _verdict(P[b][k], hb[k]) == "indistinguishable"]
        line = f"- **{names[k]}**: Full = {_fmt(S[SYSTEM_FULL][k])}."
        if better:
            line += f" Significantly higher than: {', '.join(better)}."
        if ind:
            line += f" Not distinguishable (Holm p >= 0.05 or CI spans 0) from: {', '.join(ind)}."
        if worse:
            line += f" Significantly LOWER than: {', '.join(worse)}."
        out.append(line)
    # best baseline on ndcg
    bb = max(baselines, key=lambda b: S[b]["ndcg10"]["mean"])
    out.append(f"- Strongest baseline by nDCG@10 is **{bb}** ({_fmt(S[bb]['ndcg10'])}); Full minus it = {P[bb]['ndcg10']['mean_diff']:+.3f} "
               f"(95% CI [{P[bb]['ndcg10']['ci_lo']:+.3f}, {P[bb]['ndcg10']['ci_hi']:+.3f}], Holm p = {P[bb]['ndcg10']['p_holm']:.3f}).")
    # graph-vs-no-graph isolation
    iso = "Gap-Overlap Greedy + prereq filter"
    if iso in P:
        parts = []
        for k in ["ndcg10", "p5", "recall10", "map10", "gap_cov5", "redundancy5"]:
            p = P[iso][k]
            parts.append(f"{names[k]} {p['mean_diff']:+.3f} (p_holm={p['p_holm']:.3f}, {_verdict(p, hb[k])})")
        out.append(f"- **What the graph adds beyond a user-aware, prerequisite-aware, graph-free baseline** ({iso}): Full minus baseline: " + "; ".join(parts) + ".")
    # prerequisite safety
    pv = {b: S[b]["prereq_viol5"]["mean"] for b in baselines}
    unf = [b for b in baselines if "prereq" not in b and "Gap-Overlap Greedy (no" not in b]
    out.append("- **Prerequisite violations**: Full = " + _fmt(S[SYSTEM_FULL]["prereq_viol5"]) + "; unfiltered text/overlap baselines = " +
               ", ".join(f"{b} {pv[b]:.3f}" for b in unf) + f", Gap-Overlap Greedy (no graph) {pv['Gap-Overlap Greedy (no graph)']:.3f}. Filtered variants: " +
               ", ".join(f"{b} {pv[b]:.3f}" for b in baselines if "prereq filter" in b) +
               ". Because the filtered baselines apply the same rule the metric checks, a 0 violation rate for them is by construction, not a finding; "
               "the finding is that Full does not reach 0.")
    # losses
    losses = []
    for k, nm, h in METRICS:
        for b in baselines:
            p = P[b][k]
            if _verdict(p, h) == "worse":
                losses.append(f"{nm} vs {b} (diff {p['mean_diff']:+.3f}, Holm p={p['p_holm']:.3f})")
    out.append("- **Where Full loses (significant after Holm correction)**: " + ("; ".join(losses) if losses else "none at the 0.05 level") + ".")
    nominal = []
    for k, nm, h in METRICS:
        for b in baselines:
            if (S[b][k]["mean"] > S[SYSTEM_FULL][k]["mean"]) if h else (S[b][k]["mean"] < S[SYSTEM_FULL][k]["mean"]):
                if _verdict(P[b][k], h) != "worse":
                    nominal.append(f"{nm}: {b} ({S[b][k]['mean']:.3f} vs {S[SYSTEM_FULL][k]['mean']:.3f})")
    out.append("- **Baselines with a nominally better mean than Full but not significantly** (do not read as ties in favour of Full): " + ("; ".join(nominal) if nominal else "none") + ".")
    rb = min(baselines, key=lambda b: S[b]["redundancy5"]["mean"])
    pr = P[rb]["redundancy5"]
    worse_red = [b for b in baselines if _verdict(P[b]["redundancy5"], False) == "worse"]
    better_red = [b for b in baselines if _verdict(P[b]["redundancy5"], False) == "better"]
    out.append(f"- **Redundancy@5** (lower is better): Full {_fmt(S[SYSTEM_FULL]['redundancy5'])}; lowest baseline mean is {rb} {_fmt(S[rb]['redundancy5'])} "
               f"(Full minus it {pr['mean_diff']:+.3f}, Holm p={pr['p_holm']:.3f}). Full is significantly WORSE (more redundant) than: "
               f"{', '.join(worse_red) if worse_red else 'none'}; significantly better than: {', '.join(better_red) if better_red else 'none'}. "
               "Absolute values are small (a few hundredths) for every system, so the practical difference is modest; but Full does not 'win' on redundancy.")
    out.append("- **Explanation completeness** is 1.0 for Full and 0.0 for all baselines essentially by construction (baselines emit no evidence/provenance fields); it measures output format, not recommendation quality.")
    nret = S[SYSTEM_FULL]["n_returned"]["mean"]
    if nret < K_RANK:
        out.append(f"- **List length**: Full returned {nret:.1f} courses on average (baselines always return {K_RANK}); it stops when no further candidate covers a gap, "
                   "which caps its Recall@10 and nDCG@10 relative to systems that always fill the list.")
    # sensitivity to the label rule
    sens = []
    for b in baselines:
        p = P[b]["ndcg10_content_only"]
        sens.append(f"{b} {S[b]['ndcg10_content_only']['mean']:.3f} (Full minus it {p['mean_diff']:+.3f}, Holm p={p['p_holm']:.3f}, {_verdict(p, True)})")
    out.append("- **Sensitivity to the prerequisite rule in the labels** (nDCG@10 against content-only labels with the feasibility rule removed): Full "
               f"{S[SYSTEM_FULL]['ndcg10_content_only']['mean']:.3f}; " + "; ".join(sens) +
               ".")
    pairs = [("BM25 Keyword", "BM25 + prereq filter"), ("Dense Semantic", "Dense + prereq filter"),
             ("Gap-Overlap Greedy (no graph)", "Gap-Overlap Greedy + prereq filter")]
    desc = []
    for u, f in pairs:
        if u in S and f in S:
            desc.append(f"{f} minus {u}: {S[f]['ndcg10']['mean'] - S[u]['ndcg10']['mean']:+.3f} with the rule, "
                        f"{S[f]['ndcg10_content_only']['mean'] - S[u]['ndcg10_content_only']['mean']:+.3f} without")
    out.append("- **Effect of adding the prerequisite filter to a baseline, by label variant (nDCG@10)**: " + "; ".join(desc) +
               ". A sign change means the benefit of prerequisite filtering is (at least partly) induced by the label rule rewarding it.")
    return out


CAVEATS = [
    "**Synthetic catalogue.** All 171 courses are synthetic (hand-authored structure, templated outcomes, fictional providers); results say nothing about real MOOCs. Skill names come from a vocabulary checked against the project taxonomy and 1,887 real job postings, but course content, prerequisite chains and durations are author inventions.",
    "**Scenario provenance.** 41 of 48 scenarios are synthetic archetypes; 7 use excerpts of real job postings as job text, but their user profiles and skill annotations are authored. No real learner data.",
    "**Label procedure limits.** Labels come from a deterministic rule over hand-authored gold needs and catalogue skill/prerequisite fields (see build_relevance_labels.py). They do not use any system's output, but they share the catalogue's skill/prerequisite metadata with the skill-aware systems, so skill-aware systems (Skill Overlap, Gap-Overlap, Graph) are structurally favoured over text-only retrieval. The thresholds, the grade cap for bridge-needed courses, and the lenient treatment of weak skills are author choices. Human validation of the stratified sample (reports/skillllm_v2/human_review_sample.csv) has NOT been done yet; until then the absolute numbers are unvalidated.",
    "**Metrics on near-duplicate courses.** The catalogue deliberately contains overlapping courses (e.g. a crash course and a full course on the same skill). Graded nDCG does not penalise recommending several equally relevant duplicates; Redundancy@5 is reported separately for that reason.",
    "**Statistics.** n = 48 scenarios, not independent (shared catalogue, overlapping archetypes). Bootstrap CIs resample scenarios only; they ignore label uncertainty. Holm correction is applied across the 7 baseline comparisons within each metric, not across metrics.",
    "**Not an ablation of the system's components.** Full is compared as a black box; its unexplained behaviours (e.g. how many items it returns, taxonomy-linker canonicalisation quirks such as 'Scikit-Learn' -> 'learn') are not isolated here.",
    "**Dense baseline** uses all-MiniLM-L6-v2 off the shelf with no tuning; BM25 uses default k1=1.5, b=0.75 on job title + posting text. Text baselines are user-agnostic by design (they see only the job), which disadvantages them on personalised labels; Gap-Overlap Greedy is the user-aware baseline.",
]


def write_reports(res: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    S, P = res["systems"], res["paired_vs_full"]
    cfg, ls = res["config"], res["label_stats"]

    slim = {k: v for k, v in res.items() if k not in ("per_scenario", "rankings")}
    full_json = dict(slim)
    full_json["per_scenario"] = res["per_scenario"]
    full_json["rankings_top10"] = res["rankings"]
    (out_dir / "course_recommendation_benchmark.json").write_text(json.dumps(full_json, indent=2), encoding="utf-8")

    with (out_dir / "course_recommendation_eval.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "metric", "mean", "ci95_lo", "ci95_hi", "n_scenarios", "full_minus_system_mean_diff", "paired_p", "paired_p_holm"])
        for s in S:
            for k, nm, _ in METRICS + [("n_returned", "Avg #courses returned", True), ("ndcg10_content_only", "nDCG@10 (content-only labels)", True), ("p5_content_only", "Precision@5 (content-only labels)", True)]:
                m = S[s][k]
                p = P.get(s, {}).get(k)
                w.writerow([s, nm, f"{m['mean']:.6f}", f"{m['ci_lo']:.6f}", f"{m['ci_hi']:.6f}", m["n"],
                            f"{p['mean_diff']:.6f}" if p else "", f"{p['p_value']:.4f}" if p else "", f"{p['p_holm']:.4f}" if p else ""])

    L: list[str] = []
    L += ["# SkillLLM_v2 — Course Recommendation Benchmark (independent labels, bootstrap CIs)", ""]
    L += [f"- Scenarios: **{ls['n_scenarios']}** (user, job) pairs; sources: {ls['scenario_sources']}",
          f"- Catalogue: **{ls['n_courses']} SYNTHETIC courses** (`inputs/courses/course_catalog_large.jsonl`)",
          f"- Labels: graded 0-3 per (scenario, course), `inputs/courses/relevance_labels.jsonl` ({ls['label_version']}); grade totals over all pairs: {ls['grade_totals']}; "
          f"mean #relevant (grade>=2) per scenario = {ls['mean_relevant_grade_ge2_per_scenario']:.1f}",
          f"- Relevant for Precision/Recall/MAP = grade >= {cfg['relevant_if_grade_at_least']:.0f}; nDCG@10 uses gain 2^grade - 1 with ideal over all judged courses.",
          f"- Values are mean over scenarios with 95% percentile-bootstrap CI ({cfg['n_boot']} resamples). Dense baseline backend: {cfg['dense_backend']}.",
          "- Gap coverage and redundancy are computed against the GOLD open needs for every system (not each system's own gap output).", ""]
    L += ["## Main results (mean [95% CI])", ""]
    hdr = "| System | " + " | ".join(f"{n}{' ↑' if h else ' ↓'}" for _, n, h in METRICS) + " | #returned |"
    L += [hdr, "| --- |" + " --- |" * (len(METRICS) + 1)]
    for s in S:
        row = f"| {'**' + s + '**' if s == SYSTEM_FULL else s} | " + " | ".join(_fmt(S[s][k]) for k, _, _ in METRICS) + f" | {S[s]['n_returned']['mean']:.1f} |"
        L.append(row)
    L += ["", "## Paired bootstrap: Full minus baseline (mean diff [95% CI], Holm-adjusted p)", "",
          "Positive = Full higher. For the two lower-is-better metrics a NEGATIVE diff favours Full.", ""]
    show = HEADLINE + ["gap_cov5", "prereq_viol5", "redundancy5"]
    L += ["| Baseline | " + " | ".join(dict((k, n) for k, n, _ in METRICS)[k] for k in show) + " |", "| --- |" + " --- |" * len(show)]
    for b in P:
        cells = []
        for k in show:
            p = P[b][k]
            cells.append(f"{p['mean_diff']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] p={p['p_holm']:.3f} ({p['wins']}W/{p['losses']}L/{p['ties']}T)")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    L += ["", "W/L/T = number of scenarios where Full is better / worse / tied (direction ignores whether higher is better).", ""]
    L += ["## Sensitivity: labels without the prerequisite-feasibility rule", "",
          "Same systems scored against content-only labels (a course is graded purely on how much open-need mass it teaches). If Full's advantage shrinks here, part of its advantage in the main table comes from the label rule rewarding prerequisite awareness.", "",
          "| System | nDCG@10 (content-only) | Precision@5 (content-only) |", "| --- | --- | --- |"]
    for s in S:
        L.append(f"| {s} | {_fmt(S[s]['ndcg10_content_only'])} | {_fmt(S[s]['p5_content_only'])} |")
    L += ["", "## Key Findings (generated from the numbers above)", ""] + key_findings(res)
    L += ["", "## Caveats and limits", ""] + [f"- {c}" for c in CAVEATS]
    L += ["", "## Reproduction", "",
          "```", "PYTHONPATH=src python scripts/build_course_catalog_large.py            # synthetic catalogue",
          "PYTHONPATH=src python -m skillllm_v2.eval.build_relevance_labels         # scenarios + labels + human-review sample",
          "PYTHONPATH=src python -m skillllm_v2.eval.recommender_benchmark          # this report", "```", ""]
    (out_dir / "course_recommendation_benchmark.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(ROOT / "inputs/courses/course_catalog_large.jsonl"))
    ap.add_argument("--scenarios", default=str(ROOT / "inputs/courses/benchmark_scenarios.jsonl"))
    ap.add_argument("--labels", default=str(ROOT / "inputs/courses/relevance_labels.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports/skillllm_v2"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args()
    res = run_recommender_benchmark(args.catalog, args.scenarios, args.labels, n_boot=args.n_boot, seed=args.seed)
    write_reports(res, Path(args.out_dir))
    S = res["systems"]
    print(f"{'System':<38} " + " ".join(f"{n[:14]:>16}" for _, n, _ in METRICS))
    for s in S:
        print(f"{s:<38} " + " ".join(f"{S[s][k]['mean']:>16.3f}" for k, _, _ in METRICS))
    print("wrote", Path(args.out_dir) / "course_recommendation_benchmark.md")


if __name__ == "__main__":
    main()
