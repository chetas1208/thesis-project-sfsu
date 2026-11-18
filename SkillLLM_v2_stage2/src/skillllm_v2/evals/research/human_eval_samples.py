"""L11 human evaluation: blinded annotation packets, separate answer keys, and scoring of returned annotations.

    PYTHONPATH=src python -m skillllm_v2.evals.research.human_eval_samples [--smoke] [--seed 42] [--n-boot 2000]
        [--recs-cache PATH] [--run-recommender]          # build packets + write L11 results.json (pending_human until filled)
    PYTHONPATH=src python -m skillllm_v2.evals.research.human_eval_samples --score   # re-score after annotators return files

Packets (``reports/skillllm_v2/evals/L11_human_eval/packets/<P>__<annotator>.csv``; one file per annotator, same items,
independently shuffled order). Annotator-visible columns never contain system names, gold labels, automatic grades or
strata. Those live in ``inputs/evals/research/L11_human_eval/answer_keys/`` (a different directory tree, never handed out).
  P1 extraction candidate spans: anchor false positives, anchor misses, boundary disagreements, spans added by the
     wrapper or baselines, and agreed true positives (control). SkillSpan test, 974-document subset.
  P2 full-sentence span annotation from scratch: gold-bearing and gold-free SkillSpan test sentences.
  P3 normalization/linking: A4's ``L06_linking/annotation_sample.csv`` if present, else safe-wrapper taxonomy links.
  P4a prerequisite edges of the (synthetic) course catalogue; P4b learning paths (bridge -> primary) from the held-out
     recommender scenarios (or A5's ``L08_graph_retrieval/paths_for_human_eval.jsonl`` if present).
  P5 course recommendations: pooled top-5 of the tuned Full recommender and the pre-specified strongest baseline on the
     24 held-out benchmark-v2 scenarios, plus the 60 rows of ``reports/skillllm_v2/human_review_sample.csv``, re-blinded.
  P6 explanation faithfulness of the Full recommender's explanations (only built when explanations are available).
  P7 perturbation validity (A3's L03 sample, when present).
  T0 training round: synthetic items with an answer key and rationale, done before P1-P7 (see the protocol).

Every human metric is a ``pending_human`` record until two complete annotation files exist. Agreement (Cohen/weighted
kappa, Krippendorff alpha, span F1) is then computed with bootstrap CIs. Substantive metrics additionally need the
adjudication file. LLM judges are not used here. They may only be added as a secondary signal (see the protocol).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from skillllm_v2.evals.research import agreement as ag
from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, md_table, record, rel, sha256_file, write_layer

LAYER = "L11_human_eval"
OUT = EVALS_ROOT / LAYER
KEY_ROOT = REPO_ROOT / "inputs" / "evals" / "research" / LAYER
ANNOTATORS = ("A", "B")
ORDER_SEEDS = {"A": 1001, "B": 2002}
SKILLSPAN = REPO_ROOT / "data" / "source" / "skillspan_hf"
PRED = {
    "anchor": "artifacts/skillllm_v2/mistral_lora/predictions.jsonl",
    "safe_wrapper": "artifacts/skillllm_v2/safe_wrapper/predictions.jsonl",
    "legacy_full_cag_graph": "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl",
    "legacy_gliner_taxonomy": "artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl",
    "gliner_zero_shot": "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl",
    "bert_lookup": "artifacts/skillllm_v2/bert_baseline/predictions.jsonl",
}
REC_SCENARIOS = REPO_ROOT / "inputs/courses/benchmark_scenarios.jsonl"
REC_CATALOG = REPO_ROOT / "inputs/courses/course_catalog_large.jsonl"
REC_LABELS = REPO_ROOT / "inputs/courses/relevance_labels.jsonl"
REC_SPLIT = REPO_ROOT / "reports/skillllm_v2/recommender_split.json"
REC_BENCH = REPO_ROOT / "reports/skillllm_v2/course_recommendation_benchmark_v2.json"
LEGACY_HUMAN_SAMPLE = REPO_ROOT / "reports/skillllm_v2/human_review_sample.csv"
A4_LINKING_SAMPLE = EVALS_ROOT / "L06_linking" / "annotation_sample.csv"
P3_A4_MAX = 150  # cap on the A4/B3 linking sample (300 rows) shown to annotators
A5_PATHS = EVALS_ROOT / "L08_graph_retrieval" / "paths_for_human_eval.jsonl"
L03_DIR = EVALS_ROOT / "L03_robustness"
PRESPEC_BASELINE = "Gap-Overlap Greedy + prereq filter"
MARK_L, MARK_R = "⟦", "⟧"  # mathematical white square brackets mark the candidate span

# Targets (protocol section "Sample size"): +-0.10 half-width for proportions within a stratum (n >= 97 per pooled
# estimate, ~50 per stratum gives +-0.14) and +-0.10 for kappa at an expected kappa of 0.6 (n ~ 150-200).
P1_PER_STRATUM = {"anchor_fp": 50, "anchor_fn": 50, "boundary": 50, "added_by_wrapper": 30, "baseline_only": 20,
                  "agreed_tp_control": 20}
WRAPPER_SYSTEMS = {"safe_wrapper", "legacy_full_cag_graph", "legacy_gliner_taxonomy"}
P2_N = {"gold_bearing": 60, "gold_free": 40}
P3_N = 150
P4A_N = 80
P4B_MAX = 60
P5_TOPK = 5
SMOKE_SCALE = 0.1

# question specs: (column, scale, categories or None)   scale in nominal|ordinal|interval|text|spans
QUESTIONS: Dict[str, List[Tuple[str, str, Optional[List[str]]]]] = {
    "P1": [("q1_is_skill", "nominal", ["yes", "no"]), ("q2_type", "nominal", ["SKILL", "KNOWLEDGE"]),
           ("q3_boundary", "nominal", ["exact", "too_long", "too_short", "shifted"]), ("q4_corrected_span", "text", None)],
    "P2": [("spans", "spans", None), ("q_headline", "nominal", ["yes", "no"])],
    "P3": [("q1_link", "nominal", ["correct", "acceptable_alternative", "wrong", "nil"]), ("q2_better_concept", "text", None)],
    "P4a": [("q1_edge", "ordinal", ["invalid", "questionable", "valid"])],
    "P4b": [("q1_all_edges_valid", "nominal", ["yes", "no"]), ("q2_pedagogical", "ordinal", ["0", "1", "2"]),
            ("q3_missing_step", "text", None)],
    "P5": [("q1_relevance", "ordinal", ["0", "1", "2", "3"]), ("q2_gap_closure", "ordinal", ["none", "partial", "substantial"]),
           ("q3_prereq_ready", "ordinal", ["no", "needs_bridge", "yes"])],
    "P6": [("q1_faithful", "ordinal", ["unfaithful", "partially", "faithful"]), ("q2_unsupported_claims", "text", None),
           ("q3_helpful", "interval", ["1", "2", "3", "4", "5"])],
    "P7": [("q1_meaning_preserved", "ordinal", ["no", "partially", "yes"]), ("q2_skills_recoverable", "nominal", ["yes", "no"]),
           ("q3_fluency", "ordinal", ["1", "2", "3"])],
}
P3_CANDIDATE_MODE = A4_LINKING_SAMPLE.exists()  # A4/B3 ESCO candidate-choice sample replaces the proposed-concept fallback
P3_CHOICES = ["A", "B", "C", "D", "E", "NONE_OF_THESE", "NOT_A_SKILL"]
if P3_CANDIDATE_MODE:
    # single best candidate (annotators are told: if two candidates are equally correct pick the more specific one and name the other in notes)
    QUESTIONS["P3"] = [("q1_choice", "nominal", P3_CHOICES), ("q2_note", "text", None)]

PACKET_TITLES = {
    "P1": "Extraction: candidate span validity, type and boundary", "P2": "Extraction: full-sentence span annotation",
    "P3": "Normalization / taxonomy linking", "P4a": "Graph: prerequisite edge validity",
    "P4b": "Graph: learning-path pedagogical validity", "P5": "Course recommendation: relevance, gap closure, readiness",
    "P6": "Course recommendation: explanation faithfulness", "P7": "Robustness: perturbation validity",
}


# --------------------------------------------------------------------------------------------------------------- io
def _jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("; ".join(map(str, r[c])) if isinstance(r.get(c), (list, tuple)) else r.get(c, "")) for c in cols})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _mark(text: str, start: int, end: int) -> str:
    return f"{text[:start]}{MARK_L}{text[start:end]}{MARK_R}{text[end:]}"


def load_test_docs() -> Dict[str, Dict[str, Any]]:
    from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
    return {d["id"]: d for d in load_skillspan_split(SKILLSPAN, "test", limit=None)}


def load_preds(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for row in _jsonl(path):
        out[row["id"]] = [p for p in row.get("predictions", []) if p.get("start") is not None and p.get("end") is not None]
    return out


def _assign_ids(items: List[Dict[str, Any]], prefix: str, rng: random.Random) -> List[Dict[str, Any]]:
    """Shuffle canonically, then number: ids carry no information about stratum or source system."""
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["item_id"] = f"{prefix}-{i:04d}"
    return items


# ------------------------------------------------------------------------------------------------------------- P1/P2
def build_p1(docs: Dict[str, Dict[str, Any]], rng: random.Random, scale: float) -> Tuple[List[Dict], List[Dict], Dict]:
    preds = {s: load_preds(REPO_ROOT / p) for s, p in PRED.items() if (REPO_ROOT / p).exists()}
    subset = sorted(preds["anchor"])  # the 974 gold-bearing documents with cached anchor predictions
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for did in subset:
        doc = docs[did]
        gold = {(g["start"], g["end"]): g["label"] for g in doc["gold_spans"]}
        by_pos: Dict[Tuple[int, int], set] = defaultdict(set)
        for s, pd in preds.items():
            for p in pd.get(did, []):
                if 0 <= p["start"] < p["end"] <= len(doc["text"]):
                    by_pos[(p["start"], p["end"])].add(s)
        anchor_pos = {k for k, v in by_pos.items() if "anchor" in v}

        def overl(a, b):
            return max(a[0], b[0]) < min(a[1], b[1])

        for pos, systems in by_pos.items():
            base = {"doc_id": did, "source": doc["source"], "start": pos[0], "end": pos[1], "systems": sorted(systems),
                    "in_gold": pos in gold, "gold_label": gold.get(pos, "")}
            if "anchor" in systems:
                if pos in gold:
                    pools["agreed_tp_control"].append(base)
                elif any(overl(pos, g) for g in gold):
                    pools["boundary"].append({**base, "gold_overlap": [list(g) for g in gold if overl(pos, g)]})
                else:
                    pools["anchor_fp"].append(base)
            elif pos not in gold:
                pools["added_by_wrapper" if systems & WRAPPER_SYSTEMS else "baseline_only"].append(base)
        for pos, lab in gold.items():
            if pos not in anchor_pos:
                pools["anchor_fn"].append({"doc_id": did, "source": doc["source"], "start": pos[0], "end": pos[1],
                                           "systems": sorted(by_pos.get(pos, set())), "in_gold": True, "gold_label": lab})
    items, pool_sizes = [], {}
    for stratum, n in P1_PER_STRATUM.items():
        pool = sorted(pools[stratum], key=lambda r: (r["doc_id"], r["start"], r["end"]))
        pool_sizes[stratum] = len(pool)
        k = min(len(pool), max(2, int(round(n * scale))))
        # stratified by data source (tech / house) proportionally to the pool
        by_src = defaultdict(list)
        for r in pool:
            by_src[r["source"]].append(r)
        chosen = []
        for src, rows in sorted(by_src.items()):
            take = min(len(rows), int(round(k * len(rows) / len(pool))))
            chosen += rng.sample(rows, take)
        rest = [r for r in pool if r not in chosen]
        if len(chosen) < k and rest:
            chosen += rng.sample(rest, min(k - len(chosen), len(rest)))
        for r in chosen[:k]:
            items.append({**r, "stratum": stratum})
    items = _assign_ids(items, "P1", rng)
    visible, key = [], []
    for it in items:
        text = docs[it["doc_id"]]["text"]
        visible.append({"item_id": it["item_id"], "sentence": _mark(text, it["start"], it["end"]),
                        "candidate_text": text[it["start"]:it["end"]]})
        key.append({"item_id": it["item_id"], "doc_id": it["doc_id"], "source": it["source"], "start": it["start"],
                    "end": it["end"], "stratum": it["stratum"], "systems_predicting": it["systems"], "in_gold": it["in_gold"],
                    "gold_label": it["gold_label"], "gold_overlap": json.dumps(it.get("gold_overlap", []))})
    return visible, key, {"pool_sizes": pool_sizes, "systems": sorted(preds)}


def build_p2(docs: Dict[str, Dict[str, Any]], rng: random.Random, scale: float) -> Tuple[List[Dict], List[Dict], Dict]:
    anchor_ids = set(load_preds(REPO_ROOT / PRED["anchor"]))
    bearing = sorted(d for d in anchor_ids if docs[d]["gold_spans"] and len(docs[d]["tokens"]) >= 4)
    free = sorted(d for d, doc in docs.items() if not doc["gold_spans"] and len(doc["tokens"]) >= 6
                  and "<" not in doc["text"])  # skip anonymisation-only lines such as <ORGANIZATION>
    chosen = []
    for name, pool in (("gold_bearing", bearing), ("gold_free", free)):
        k = min(len(pool), max(2, int(round(P2_N[name] * scale))))
        chosen += [(name, d) for d in rng.sample(pool, k)]
    items = _assign_ids([{"stratum": s, "doc_id": d} for s, d in chosen], "P2", rng)
    visible, key = [], []
    for it in items:
        doc = docs[it["doc_id"]]
        visible.append({"item_id": it["item_id"], "sentence": doc["text"]})
        key.append({"item_id": it["item_id"], "doc_id": it["doc_id"], "source": doc["source"], "stratum": it["stratum"],
                    "gold_spans": json.dumps([[g["start"], g["end"], g["label"]] for g in doc["gold_spans"]]),
                    "in_anchor_subset": it["doc_id"] in anchor_ids})
    return visible, key, {"pool_sizes": {"gold_bearing": len(bearing), "gold_free": len(free)}}


# ---------------------------------------------------------------------------------------------------------------- P3
def build_p3(docs: Dict[str, Dict[str, Any]], rng: random.Random, scale: float) -> Tuple[List[Dict], List[Dict], Dict]:
    if A4_LINKING_SAMPLE.exists():
        rows = _read_csv(A4_LINKING_SAMPLE)
        hidden_pat = re.compile(r"system|method|score|gold|correct|label_true|answer|key|confidence|source|rank|is_|esco_uri|stratum",
                                re.I)
        cols = list(rows[0].keys()) if rows else []
        shown = [c for c in cols if not hidden_pat.search(c)]
        n_source = len(rows)
        cap = max(4, int(round(P3_A4_MAX * scale)))
        if len(rows) > cap:  # annotator workload: seeded random subsample, drawn from a separate stream so no other packet changes
            sub = random.Random(4242).sample(range(len(rows)), cap)
            rows = [rows[i] for i in sorted(sub)]
        items = _assign_ids([dict(r) for r in rows], "P3", rng)
        visible = [{"item_id": r["item_id"], **{c: r[c] for c in shown}} for r in items]
        key = [{"item_id": r["item_id"], **{c: r[c] for c in cols}} for r in items]
        return visible, key, {"origin": rel(A4_LINKING_SAMPLE), "sha256": sha256_file(A4_LINKING_SAMPLE),
                              "source_rows": n_source, "subsampled_to": len(rows) if n_source > len(rows) else None,
                              "shown_columns": shown, "hidden_columns": [c for c in cols if c not in shown]}
    # fallback: taxonomy links attached by the safe wrapper on the test subset (the anchor-subset documents)
    from skillllm_v2.schema import normalize_text
    test_mined = {normalize_text(r.get("alias", "")) for r in _jsonl(REPO_ROOT / "data/skillllm_v2/taxonomy/skill_aliases.jsonl")
                  if r.get("source") == "mistral_high_confidence"}  # 76 alias rows mined from TEST predictions (I12)
    rows = _jsonl(REPO_ROOT / PRED["safe_wrapper"])
    pool = defaultdict(list)
    for row in rows:
        text = docs[row["id"]]["text"]
        for p in row.get("predictions", []):
            if p.get("start") is None or p.get("end") is None:
                continue
            link = (p.get("metadata") or {}).get("taxonomy_link") or {}
            method = link.get("link_method") or ("none" if not p.get("canonical_skill_name") else "unknown")
            pool[method].append({"doc_id": row["id"], "start": p["start"], "end": p["end"], "mention": text[p["start"]:p["end"]],
                                 "proposed": p.get("canonical_skill_name") or "", "method": method,
                                 "link_confidence": link.get("link_confidence"),
                                 "alias_row_test_mined": method == "alias" and normalize_text(text[p["start"]:p["end"]]) in test_mined})
    n = max(4, int(round(P3_N * scale)))
    total = sum(len(v) for v in pool.values())
    chosen = []
    for m, rows_m in sorted(pool.items()):
        k = max(1, int(round(n * len(rows_m) / total))) if m != "none" else max(1, int(round(n * 0.15)))
        chosen += rng.sample(rows_m, min(k, len(rows_m)))
    items = _assign_ids(chosen, "P3", rng)
    visible = [{"item_id": it["item_id"], "sentence": _mark(docs[it["doc_id"]]["text"], it["start"], it["end"]),
                "mention": it["mention"], "proposed_concept": it["proposed"] or "(no concept proposed)"} for it in items]
    key = [{"item_id": it["item_id"], "doc_id": it["doc_id"], "start": it["start"], "end": it["end"], "method": it["method"],
            "link_confidence": it["link_confidence"], "proposed": it["proposed"],
            "alias_row_test_mined": it["alias_row_test_mined"]} for it in items]
    return visible, key, {"origin": "fallback: safe_wrapper taxonomy links (A4 annotation_sample.csv absent at build time)",
                          "pool_by_method": {k: len(v) for k, v in pool.items()}}


def taxonomy_reference() -> List[Dict[str, Any]]:
    rows = _jsonl(REPO_ROOT / "data/skillllm_v2/taxonomy/skill_aliases.jsonl")
    by = defaultdict(set)
    for r in rows:
        by[r["canonical_skill_name"]].add(r["alias"])
    return [{"canonical_concept": k, "aliases": sorted(v)} for k, v in sorted(by.items(), key=lambda kv: kv[0].lower())]


# ------------------------------------------------------------------------------------------------------- P4 / P5 / P6
def _rec_inputs():
    scen = {s["scenario_id"]: s for s in _jsonl(REC_SCENARIOS)}
    cat = {c["course_id"]: c for c in _jsonl(REC_CATALOG)}
    labels = {r["scenario_id"]: r for r in _jsonl(REC_LABELS)}
    split = json.loads(REC_SPLIT.read_text())
    return scen, cat, labels, split


def load_full_recs(cache: Path, run_recommender: bool, test_ids: List[str]) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], str]:
    """Tuned Full recommender + pre-specified baseline on held-out scenarios (with explanations).  Cached because the
    pipeline is slow to initialise; falls back to the stored benchmark-v2 rankings (no explanations)."""
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["full"], d["baseline"], (f"cached re-run of the frozen DEFAULT_CONFIG recommender {rel(cache)} "
                                          f"(sha256 {sha256_file(cache)[:12]}); top-10 identical to stored benchmark-v2 "
                                          f"rankings for {d.get('rerun_matches_stored_rankings', '?')} held-out scenarios")
    if run_recommender:
        from skillllm_v2.recommend.config import DEFAULT_CONFIG
        from skillllm_v2.recommend.loop_eval import Harness
        H = Harness(with_baselines=True)
        full = H.run_full(DEFAULT_CONFIG, test_ids)
        base = {sid: H.baseline_lists[sid][PRESPEC_BASELINE] for sid in test_ids}
        stored = json.loads(REC_BENCH.read_text())["rankings"]["full"]
        match = sum([r["course_id"] for r in full[sid]] == stored.get(sid) for sid in test_ids)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"full": full, "baseline": base, "rerun_matches_stored_rankings": f"{match}/{len(test_ids)}"},
                                    default=str))
        return full, base, (f"re-run of the frozen DEFAULT_CONFIG recommender (loop_eval.Harness); top-10 identical to the "
                            f"stored benchmark-v2 rankings for {match}/{len(test_ids)} held-out scenarios")
    bench = json.loads(REC_BENCH.read_text())
    full = {sid: [{"course_id": c} for c in bench["rankings"]["full"][sid]] for sid in test_ids}
    return full, {}, "stored rankings in course_recommendation_benchmark_v2.json (no explanations, no baseline list)"


def _course_view(c: Dict[str, Any]) -> Dict[str, Any]:
    return {"course_title": c.get("title", ""), "course_level": c.get("level", ""), "course_hours": c.get("duration_hours", ""),
            "course_description": c.get("description", ""), "course_outcomes": c.get("learning_outcomes", []),
            "course_teaches": c.get("teaches_skills", []), "course_prerequisites": c.get("prerequisites", [])}


def _scenario_view(s: Dict[str, Any]) -> Dict[str, Any]:
    job = s["target_job"]
    return {"learner_background": s.get("background", ""),
            "learner_known_skills": [k.get("surface_form") or k.get("canonical_skill_name") for k in s["user_profile"]["skills"]],
            "target_job_title": job.get("title", ""), "target_job_text": job.get("text", ""),
            "job_required_skills": job.get("required_skills", []), "job_preferred_skills": job.get("preferred_skills", [])}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", (s or "").lower())


def build_p4(rng: random.Random, scale: float, full: Dict[str, List[Dict]]) -> Dict[str, Tuple[List[Dict], List[Dict], Dict]]:
    scen, cat, labels, split = _rec_inputs()
    edges = sorted({(cid, p) for cid, c in cat.items() for p in c.get("prerequisites", [])})
    k = min(len(edges), max(4, int(round(P4A_N * scale))))
    e_items = _assign_ids([{"course_id": c, "prereq": p} for c, p in rng.sample(edges, k)], "P4a", rng)
    vis_a = [{"item_id": e["item_id"], **{x: y for x, y in _course_view(cat[e["course_id"]]).items()
                                         if x in ("course_title", "course_level", "course_description", "course_teaches")},
              "claimed_prerequisite_skill": e["prereq"]} for e in e_items]
    key_a = [{"item_id": e["item_id"], "course_id": e["course_id"], "prereq": e["prereq"]} for e in e_items]
    meta_a = {"pool": len(edges), "origin": rel(REC_CATALOG) + " (synthetic, project-authored catalogue)"}
    # P4b paths
    paths: List[Dict[str, Any]] = []
    origin = ""
    if A5_PATHS.exists():
        origin = rel(A5_PATHS)
        for r in _jsonl(A5_PATHS):
            paths.append({"query_id": r.get("query_id"), "path": r.get("path"), "system": r.get("system", ""),
                          "context": r.get("context", "")})
    else:
        origin = "fallback: bridge->primary course pairs in the tuned Full recommender's held-out top-10 lists"
        for sid in split["test"]:
            recs = full.get(sid, [])
            ids = [r["course_id"] for r in recs]
            known_disp = [k.get("surface_form") or k.get("canonical_skill_name") or "" for k in scen[sid]["user_profile"]["skills"]]
            known = {_norm(k) for k in known_disp}
            for j, pid in enumerate(ids):
                prim = cat.get(pid)
                if not prim:
                    continue
                for i, bid in enumerate(ids):
                    if i == j or bid not in cat:
                        continue
                    bridge_skills = [p for p in prim.get("prerequisites", []) if _norm(p) not in known and
                                     _norm(p) in {_norm(t) for t in cat[bid].get("teaches_skills", [])}]
                    if bridge_skills:
                        paths.append({"query_id": sid, "system": "full", "rank_bridge": i + 1, "rank_primary": j + 1,
                                      "path": [{"src": "learner", "rel": "knows", "dst": known_disp},
                                               {"src": cat[bid]["title"], "rel": "teaches", "dst": bridge_skills},
                                               {"src": bridge_skills, "rel": "prerequisite_of", "dst": prim["title"]},
                                               {"src": prim["title"], "rel": "teaches", "dst": prim.get("teaches_skills", [])}],
                                      "bridge_id": bid, "primary_id": pid})
    kb = min(len(paths), max(2, int(round(P4B_MAX * scale))))
    p_items = _assign_ids(rng.sample(paths, kb) if len(paths) > kb else list(paths), "P4b", rng)
    vis_b, key_b = [], []
    for p in p_items:
        sv = _scenario_view(scen[p["query_id"]]) if p.get("query_id") in scen else {}
        steps = " -> ".join(f"[{_fmt_node(e['src'])}] --{e['rel']}--> [{_fmt_node(e['dst'])}]" for e in (p.get("path") or []))
        vis_b.append({"item_id": p["item_id"], "target_job_title": sv.get("target_job_title", ""),
                      "learner_known_skills": sv.get("learner_known_skills", []), "path": steps})
        key_b.append({"item_id": p["item_id"], **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                                  for k, v in p.items() if k != "item_id"}})
    return {"P4a": (vis_a, key_a, meta_a), "P4b": (vis_b, key_b, {"origin": origin, "pool": len(paths)})}


def _fmt_node(x: Any) -> str:
    return ", ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)


def _explanation_text(rec: Dict[str, Any]) -> str:
    """The user-facing explanation of one Full-recommender item: its explanation_path steps + limitations (the fields the
    CourseExplanationGenerator emits; they may sit at top level or under an 'explanation' key)."""
    ex = rec.get("explanation") if isinstance(rec.get("explanation"), dict) else rec
    if isinstance(rec.get("explanation"), str):
        return rec["explanation"]
    parts: List[str] = []
    for k in ("explanation_path", "explanation_steps"):
        v = ex.get(k)
        if v:
            parts += [str(x) for x in (v if isinstance(v, list) else [v])]
    lim = ex.get("limitations")
    if lim:
        parts.append("Limitations: " + ("; ".join(map(str, lim)) if isinstance(lim, list) else str(lim)))
    return "\n".join(f"- {p}" for p in parts)


def build_p5_p6(rng: random.Random, scale: float, full: Dict[str, List[Dict]], base: Dict[str, List[Dict]],
                rec_origin: str) -> Dict[str, Tuple[List[Dict], List[Dict], Dict]]:
    scen, cat, labels, split = _rec_inputs()
    test = split["test"]
    if scale < 1:
        test = test[:max(2, int(round(len(test) * scale)))]
    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for sid in test:
        for sysname, lists in (("full_tuned", full), (PRESPEC_BASELINE, base)):
            for rank, r in enumerate(lists.get(sid, [])[:P5_TOPK], 1):
                key = (sid, r["course_id"])
                e = pairs.setdefault(key, {"scenario_id": sid, "course_id": r["course_id"], "systems": {}, "strata": set()})
                e["systems"][sysname] = rank
                e["strata"].add("pooled_top5_heldout")
    legacy_rows = _read_csv(LEGACY_HUMAN_SAMPLE) if LEGACY_HUMAN_SAMPLE.exists() else []
    if scale < 1:
        legacy_rows = legacy_rows[:max(2, int(round(len(legacy_rows) * scale)))]
    for r in legacy_rows:
        key = (r["scenario_id"], r["course_id"])
        e = pairs.setdefault(key, {"scenario_id": r["scenario_id"], "course_id": r["course_id"], "systems": {}, "strata": set()})
        e["strata"].add("legacy_human_review_sample")
        e["legacy_row_id"] = r["row_id"]
        e["legacy_auto_grade"] = r["auto_grade_0to3"]
        e["legacy_auto_stratum"] = r["auto_stratum"]
    items = _assign_ids([v for _, v in sorted(pairs.items())], "P5", rng)
    vis, key = [], []
    for it in items:
        s, c = scen[it["scenario_id"]], cat[it["course_id"]]
        vis.append({"item_id": it["item_id"], **_scenario_view(s), **_course_view(c)})
        lab = labels.get(it["scenario_id"], {})
        key.append({"item_id": it["item_id"], "scenario_id": it["scenario_id"], "course_id": it["course_id"],
                    "split": "test" if it["scenario_id"] in split["test"] else "dev", "strata": sorted(it["strata"]),
                    "ranks_by_system": json.dumps(it["systems"]), "rule_label_v1": lab.get("labels", {}).get(it["course_id"], 0),
                    "rule_label_content_only": lab.get("labels_content_only", {}).get(it["course_id"], 0),
                    "gold_needs": json.dumps(s["gold"]["needs"]), "legacy_row_id": it.get("legacy_row_id", ""),
                    "legacy_auto_grade": it.get("legacy_auto_grade", ""), "legacy_auto_stratum": it.get("legacy_auto_stratum", "")})
    meta5 = {"origin": rec_origin, "n_heldout_scenarios": len(test), "n_legacy_rows": len(legacy_rows),
             "legacy_sample": rel(LEGACY_HUMAN_SAMPLE) if legacy_rows else None}
    out = {"P5": (vis, key, meta5)}
    # P6: explanations exist only for the Full system -> separate packet (cannot be system-blind by construction)
    ex_items = []
    for sid in test:
        for rank, r in enumerate(full.get(sid, [])[:P5_TOPK], 1):
            txt = _explanation_text(r)
            if txt:
                ex_items.append({"scenario_id": sid, "course_id": r["course_id"], "rank": rank, "explanation": txt})
    if ex_items:
        ex_items = _assign_ids(ex_items, "P6", rng)
        vis6 = [{"item_id": e["item_id"], **_scenario_view(scen[e["scenario_id"]]), **_course_view(cat[e["course_id"]]),
                 "explanation_shown_to_learner": e["explanation"]} for e in ex_items]
        key6 = [{"item_id": e["item_id"], "scenario_id": e["scenario_id"], "course_id": e["course_id"], "rank": e["rank"]}
                for e in ex_items]
        out["P6"] = (vis6, key6, {"origin": rec_origin, "note": "Full recommender only; not system-blind by construction"})
    return out


def build_p7(rng: random.Random, scale: float) -> Optional[Tuple[List[Dict], List[Dict], Dict]]:
    cands = sorted(list(L03_DIR.glob("*validity*.csv")) + list(L03_DIR.glob("*validity*.jsonl"))) if L03_DIR.exists() else []
    if not cands:
        return None
    src = cands[0]
    rows = _read_csv(src) if src.suffix == ".csv" else _jsonl(src)
    if scale < 1:
        rows = rows[:max(2, int(round(len(rows) * scale)))]
    items = _assign_ids([dict(r) for r in rows], "P7", rng)
    shown = [c for c in ("original_text", "perturbed_text", "original", "perturbed") if rows and c in rows[0]]
    vis = [{"item_id": r["item_id"], **{c: r.get(c, "") for c in shown}} for r in items]
    key = [{"item_id": r["item_id"], **{k: v for k, v in r.items() if k != "item_id"}} for r in items]
    return vis, key, {"origin": rel(src), "sha256": sha256_file(src), "shown_columns": shown}


# ------------------------------------------------------------------------------------------------------- training T0
TRAINING: Dict[str, List[Dict[str, Any]]] = {
    # Rule numbers refer to the SkillSpan annotation guidelines (Zhang et al., NAACL 2022, Appendix B); examples are synthetic.
    "P1": [
        {"sentence": f"Experience with {MARK_L}Kubernetes{MARK_R} and Terraform is required.", "candidate_text": "Kubernetes",
         "answer": {"q1_is_skill": "yes", "q2_type": "KNOWLEDGE", "q3_boundary": "exact"},
         "why": "Something one possesses rather than executes -> KNOWLEDGE (B.2.1); the trigger 'Experience with' is excluded (B.3.6)."},
        {"sentence": f"You will {MARK_L}mentor{MARK_R} junior developers.", "candidate_text": "mentor",
         "answer": {"q1_is_skill": "yes", "q2_type": "SKILL", "q3_boundary": "too_short", "q4_corrected_span": "mentor junior developers"},
         "why": "A skill starts with a verb and includes its object (B.1.1); the bare verb is too short; 'will' is not tagged (B.1.1.1)."},
        {"sentence": f"We are a {MARK_L}fast-growing{MARK_R} team based in a modern office.", "candidate_text": "fast-growing",
         "answer": {"q1_is_skill": "no"}, "why": "Describes the company, not a competence asked of the candidate."},
        {"sentence": f"Strong {MARK_L}knowledge of SQL{MARK_R} is a plus.", "candidate_text": "knowledge of SQL",
         "answer": {"q1_is_skill": "yes", "q2_type": "KNOWLEDGE", "q3_boundary": "too_long", "q4_corrected_span": "SQL"},
         "why": "Fluff and triggers such as 'knowledge of' are excluded (B.3.6)."},
        {"sentence": f"Ability to {MARK_L}work independently{MARK_R} under tight deadlines.", "candidate_text": "work independently",
         "answer": {"q1_is_skill": "yes", "q2_type": "SKILL", "q3_boundary": "exact"},
         "why": "Manner adverbials are part of the skill (B.1.6); the trailing circumstance is excluded (B.3.5)."},
        {"sentence": f"A {MARK_L}Bachelor's degree{MARK_R} in Computer Science or equivalent.", "candidate_text": "Bachelor's degree",
         "answer": {"q1_is_skill": "yes", "q2_type": "KNOWLEDGE", "q3_boundary": "exact"},
         "why": "SkillSpan tags degrees and their fields as separate KNOWLEDGE components (B.2.8); 'Computer Science' would be a second span."},
        {"sentence": f"{MARK_L}Can prepare monthly reports{MARK_R} for management.", "candidate_text": "Can prepare monthly reports",
         "answer": {"q1_is_skill": "yes", "q2_type": "SKILL", "q3_boundary": "too_long", "q4_corrected_span": "prepare monthly reports"},
         "why": "Modal verbs are not tagged (B.1.1.1)."},
        {"sentence": f"{MARK_L}Passion for automation{MARK_R} is essential.", "candidate_text": "Passion for automation",
         "answer": {"q1_is_skill": "yes", "q2_type": "KNOWLEDGE", "q3_boundary": "too_long", "q4_corrected_span": "automation"},
         "why": "An attitude that contains a knowledge component: only the component is tagged (B.1.8)."},
    ],
    "P2": [
        {"sentence": "Proficient in Java and German .", "answer": {"spans": "Java|KNOWLEDGE; German|KNOWLEDGE", "q_headline": "no"},
         "why": "Listed knowledge components are tagged separately, triggers excluded (B.2.8, B.3.6)."},
        {"sentence": "Be curious and self-driven .", "answer": {"spans": "curious|SKILL; self-driven|SKILL", "q_headline": "no"},
         "why": "Coordinated adjectives without a verb are split; attitudes are annotated as SKILL (B.1.2.3, B.1.7)."},
        {"sentence": "Senior Backend Engineer - Go / PostgreSQL", "answer": {"spans": "", "q_headline": "yes"},
         "why": "Components in the posting's top headline/title are not tagged (B.3.4): leave spans empty and mark headline."},
        {"sentence": "Our office is located near the central station .", "answer": {"spans": "", "q_headline": "no"},
         "why": "No skill: leave the cell empty and put 'y' in completed (an empty cell on a completed row means 'no spans')."},
    ],
    "P3": [
        {"sentence": f"Hands-on {MARK_L}k8s{MARK_R} experience.", "mention": "k8s", "proposed_concept": "Kubernetes",
         "answer": {"q1_link": "correct"}, "why": "Common abbreviation of the concept."},
        {"sentence": f"Good {MARK_L}Excel modelling{MARK_R} skills.", "mention": "Excel modelling",
         "proposed_concept": "Microsoft Excel", "answer": {"q1_link": "acceptable_alternative",
                                                           "q2_better_concept": "financial modelling (if in taxonomy)"},
         "why": "The concept is related and defensible but a more specific concept would be better."},
        {"sentence": f"Experience with {MARK_L}Java{MARK_R} 17.", "mention": "Java", "proposed_concept": "JavaScript",
         "answer": {"q1_link": "wrong"}, "why": "Different language."},
        {"sentence": f"Knowledge of {MARK_L}maritime law{MARK_R}.", "mention": "maritime law",
         "proposed_concept": "(no concept proposed)", "answer": {"q1_link": "nil"},
         "why": "No suitable concept in the reference taxonomy: NIL is the correct output."},
    ],
    "P4a": [
        {"course_title": "Deep Learning with PyTorch", "claimed_prerequisite_skill": "Python", "answer": {"q1_edge": "valid"},
         "why": "Standard prerequisite."},
        {"course_title": "Introduction to Spreadsheets", "claimed_prerequisite_skill": "Kubernetes",
         "answer": {"q1_edge": "invalid"}, "why": "Unrelated; would block learners for no reason."},
        {"course_title": "Advanced SQL Optimisation", "claimed_prerequisite_skill": "Statistics",
         "answer": {"q1_edge": "questionable"}, "why": "Helpful for some content but not required."},
    ],
    "P5": [
        {"target_job_title": "Data Engineer", "learner_known_skills": "Python; SQL", "course_title": "Apache Spark Fundamentals",
         "answer": {"q1_relevance": "3", "q2_gap_closure": "substantial", "q3_prereq_ready": "yes"},
         "why": "Directly teaches a central missing requirement; prerequisites met."},
        {"target_job_title": "Data Engineer", "learner_known_skills": "Python; SQL", "course_title": "Introduction to Python",
         "answer": {"q1_relevance": "0", "q2_gap_closure": "none", "q3_prereq_ready": "yes"},
         "why": "Teaches only what the learner already knows."},
    ],
}
if P3_CANDIDATE_MODE:
    TRAINING["P3"] = [
        {"sentence": f"Hands-on {MARK_L}k8s{MARK_R} experience.", "mention": "k8s",
         "candidate_A": "use Kubernetes (synthetic) - container orchestration", "candidate_B": "use Kafka (synthetic) - event streaming",
         "candidate_C": "use Docker (synthetic) - containers", "candidate_D": "use Ansible (synthetic) - configuration management",
         "candidate_E": "use Terraform (synthetic) - infrastructure as code", "options": "A-E | NONE_OF_THESE | NOT_A_SKILL",
         "answer": {"q1_choice": "A"}, "why": "k8s is the common abbreviation of Kubernetes."},
        {"sentence": f"Good {MARK_L}Excel modelling{MARK_R} skills.", "mention": "Excel modelling",
         "candidate_A": "use spreadsheets software (synthetic)", "candidate_B": "use Microsoft Excel (synthetic)",
         "candidate_C": "use Microsoft Word (synthetic)", "candidate_D": "perform financial analysis (synthetic)",
         "candidate_E": "use Power BI (synthetic)", "options": "A-E | NONE_OF_THESE | NOT_A_SKILL",
         "answer": {"q1_choice": "B", "q2_note": "A is also acceptable but broader"},
         "why": "Two defensible candidates: choose the more specific one and mention the other."},
        {"sentence": f"Knowledge of {MARK_L}maritime law{MARK_R}.", "mention": "maritime law",
         "candidate_A": "tax law (synthetic)", "candidate_B": "labour law (synthetic)", "candidate_C": "contract law (synthetic)",
         "candidate_D": "customs regulations (synthetic)", "candidate_E": "environmental law (synthetic)",
         "options": "A-E | NONE_OF_THESE | NOT_A_SKILL", "answer": {"q1_choice": "NONE_OF_THESE"},
         "why": "A real skill, but no listed concept denotes it: NONE_OF_THESE."},
        {"sentence": f"We are a {MARK_L}fast-growing{MARK_R} team in a modern office.", "mention": "fast-growing",
         "candidate_A": "manage growth strategy (synthetic)", "candidate_B": "work in teams (synthetic)",
         "candidate_C": "use agile project management (synthetic)", "candidate_D": "manage a team (synthetic)",
         "candidate_E": "adapt to change (synthetic)", "options": "A-E | NONE_OF_THESE | NOT_A_SKILL",
         "answer": {"q1_choice": "NOT_A_SKILL"}, "why": "Describes the company, not a competence asked of the candidate."},
    ]


def write_training(out_dir: Path, key_dir: Path) -> Dict[str, int]:
    counts = {}
    for p, items in TRAINING.items():
        qcols = [q for q, _, _ in QUESTIONS[p]]
        vis, key = [], []
        for i, it in enumerate(items, 1):
            iid = f"T0-{p}-{i:02d}"
            vis.append({"item_id": iid, **{k: v for k, v in it.items() if k not in ("answer", "why")}})
            key.append({"item_id": iid, **it["answer"], "rationale": it["why"]})
        cols = ["item_id"] + [c for c in vis[0] if c != "item_id"] + qcols + ["notes"]
        _write_csv(out_dir / f"T0_{p}.csv", vis, cols)
        _write_csv(key_dir / f"T0_{p}_key.csv", key, ["item_id"] + qcols + ["rationale"])
        counts[p] = len(items)
    return counts


# ----------------------------------------------------------------------------------------------------------- scoring
def _cell(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    return None if v == "" or v.lower() in ("unsure", "na", "n/a", "?") else v


def parse_span_cell(text: str, cell: str) -> Tuple[List[Tuple[int, int, str]], List[str]]:
    """'surface|TYPE; surface#2|TYPE' -> character spans (k-th occurrence with #k); returns (spans, problems)."""
    spans, problems = [], []
    for part in [p.strip() for p in (cell or "").split(";") if p.strip()]:
        if "|" not in part:
            problems.append(f"no |TYPE in {part!r}")
            continue
        surf, typ = part.rsplit("|", 1)
        occ = 1
        m = re.match(r"^(.*)#(\d+)$", surf.strip())
        if m:
            surf, occ = m.group(1), int(m.group(2))
        surf = surf.strip()
        starts = [x.start() for x in re.finditer(re.escape(surf), text)] if surf else []
        if len(starts) < occ:
            problems.append(f"{surf!r} not found (occurrence {occ})")
            continue
        spans.append((starts[occ - 1], starts[occ - 1] + len(surf), typ.strip().upper()))
    return spans, problems


def _annotation_files(packet: str, ann_dir: Path) -> Dict[str, Path]:
    return {a: ann_dir / f"{packet}__{a}.csv" for a in (*ANNOTATORS, "ADJ") if (ann_dir / f"{packet}__{a}.csv").exists()}


def score_packet(packet: str, ann_dir: Path, n_items: int, n_boot: int, seed: int,
                 sentences: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Agreement records for one packet: measured when both annotators completed >= 90% of items, else pending_human."""
    files = _annotation_files(packet, ann_dir)
    recs: List[Dict[str, Any]] = []
    common = dict(n_units=n_items, unit="annotated item", split="L11 packet " + packet, dataset=PACKET_TITLES[packet])
    rows = {a: {r["item_id"]: r for r in _read_csv(p)} for a, p in files.items() if a in ANNOTATORS}
    qspecs = [q for q in QUESTIONS[packet] if q[1] != "text"]
    done = {}
    for a in ANNOTATORS:
        r = rows.get(a, {})
        first = qspecs[0][0]
        done[a] = sum(1 for x in r.values() if _cell(x.get(first)) is not None or (packet == "P2" and x.get("completed", "").strip()))
    complete = all(done.get(a, 0) >= 0.9 * n_items for a in ANNOTATORS) and n_items > 0
    for col, scale, cats in qspecs:
        if not complete:
            reason = (f"awaiting human annotation: annotator completion A={done.get('A', 0)}/{n_items}, B={done.get('B', 0)}/{n_items} "
                      f"(need >= 90% from both); drop files into {rel(ann_dir)}/{packet}__A.csv and __B.csv")
            metrics = ["span_iaa_f1_exact", "span_iaa_f1_overlap"] if scale == "spans" else \
                [f"{col}_cohen_kappa" + ("_quadratic" if scale in ("ordinal", "interval") else ""), f"{col}_krippendorff_alpha_{scale}"]
            for m in metrics:
                recs.append(record(LAYER, "human_annotators", m, None, status="pending_human", reason=reason, direction="higher",
                                   details={"packet": packet}, **common))
            continue
        ids = sorted(set(rows["A"]) & set(rows["B"]))
        if scale == "spans":
            sa, sb = {}, {}
            for i in ids:
                txt = (sentences or {}).get(i, rows["A"][i].get("sentence", ""))
                sa[i], _ = parse_span_cell(txt, rows["A"][i].get(col, ""))
                sb[i], _ = parse_span_cell(txt, rows["B"][i].get(col, ""))
            for mode in ("exact", "overlap"):
                r = ag.span_f1_with_ci(sa, sb, mode, True, n_boot, seed)
                recs.append(record(LAYER, "human_annotators", f"span_iaa_f1_{mode}", r["value"], ci=r["ci"], ci_method=r["method"],
                                   direction="higher", details={"packet": packet, "n_a": r["n_a"], "n_b": r["n_b"]},
                                   **{**common, "n_units": r["n_items"]}))
            continue
        a_vals = [_cell(rows["A"][i].get(col)) for i in ids]
        b_vals = [_cell(rows["B"][i].get(col)) for i in ids]
        weights = "quadratic" if scale in ("ordinal", "interval") else None
        k = ag.kappa_with_ci(a_vals, b_vals, cats, weights, n_boot, seed)
        recs.append(record(LAYER, "human_annotators", f"{col}_cohen_kappa" + ("_quadratic" if weights else ""), k["value"],
                           ci=k["ci"], ci_method=k["method"], direction="higher",
                           details={"packet": packet, "categories": cats, "percent_agreement": ag.percent_agreement(a_vals, b_vals),
                                    "gwet_ac1": ag.gwet_ac1(a_vals, b_vals, cats) if scale == "nominal" else None},
                           **{**common, "n_units": k["n_items"]}))
        level = scale if scale in ag.LEVELS else "nominal"
        conv = (lambda v: None if v is None else (cats.index(v) if v in cats else None))
        data = [[conv(v) for v in a_vals], [conv(v) for v in b_vals]]
        al = ag.alpha_with_ci(data, level, list(range(len(cats))), n_boot, seed)
        recs.append(record(LAYER, "human_annotators", f"{col}_krippendorff_alpha_{level}", al["value"], ci=al["ci"],
                           ci_method=al["method"], direction="higher", details={"packet": packet},
                           **{**common, "n_units": al["n_items"]}))
    return recs


SUBSTANTIVE = [
    ("P1", "anchor_fp_judged_valid_rate", "share of anchor 'false positives' (vs SkillSpan gold) that adjudicated humans judge to be valid skill spans (gold incompleteness)", "none"),
    ("P1", "wrapper_added_span_valid_rate", "share of spans added by wrapper stages (not in gold, not by the anchor) judged valid", "higher"),
    ("P1", "baseline_only_span_valid_rate", "share of spans proposed only by the token-lookup / zero-shot GLiNER baselines (not in gold) judged valid", "none"),
    ("P1", "anchor_fn_confirmed_rate", "share of gold spans missed by the anchor that humans confirm as skills", "none"),
    ("P1", "control_tp_valid_rate", "agreed true positives judged valid (annotator calibration check; expected near 1)", "higher"),
    ("P2", "human_vs_gold_exact_f1", "adjudicated human spans vs SkillSpan gold, exact span F1 (human ceiling)", "higher"),
    ("P2", "gold_free_sentences_with_skill_rate", "share of gold-free test sentences in which the adjudicated annotation has >= 1 skill span", "none"),
    ("P3", "linking_accuracy_correct_or_acceptable", "adjudicated: proposed concept correct or acceptable alternative", "higher"),
    ("P3", "linking_accuracy_strict", "adjudicated: proposed concept correct", "higher"),
    ("P3", "human_choice_matches_decorte_gold_rate", "candidate-choice mode: share of control items (Decorte ESCO gold present among the candidates) where the adjudicated human choice is the gold concept", "higher"),
    ("P3", "linker_top1_matches_adjudicated_human_rate", "candidate-choice mode: per linker, share of items whose top-ranked candidate equals the adjudicated human choice (independent ESCO linking gold; the key maps letters back to linker ranks)", "higher"),
    ("P4a", "prerequisite_edge_validity_rate", "adjudicated catalogue prerequisite edges judged valid", "higher"),
    ("P4b", "path_pedagogical_validity_mean", "mean adjudicated pedagogical validity (0-2) of learning paths", "higher"),
    ("P5", "human_vs_rule_label_weighted_kappa", "quadratic-weighted kappa between adjudicated human relevance and rule labels v1 (label validity)", "higher"),
    ("P5", "human_precision_at_5_full_minus_baseline", "paired difference in human-judged P@5 (relevance >= 2) between the tuned Full recommender and the pre-specified baseline", "higher"),
    ("P6", "explanation_faithfulness_rate", "share of explanations adjudicated faithful (every claim supported by course record / profile / job)", "higher"),
    ("P7", "perturbation_meaning_preserved_rate", "share of perturbed sentences adjudicated meaning-preserving", "higher"),
]


# -------------------------------------------------------------------------------------------------------------- main
def sample_size_table() -> List[Dict[str, Any]]:
    rows = []
    for name, hw in (("proportion within one P1 stratum (n=50)", None), ("pooled proportion", 0.10), ("pooled proportion", 0.07),
                     ("pooled proportion", 0.05)):
        if hw is None:
            rows.append({"target": name, "halfwidth": round(ag.Z95 * math.sqrt(0.25 / 50), 3), "n_required": 50,
                         "basis": "Wald, p=0.5 worst case"})
        else:
            rows.append({"target": name, "halfwidth": hw, "n_required": ag.n_for_proportion(hw), "basis": "Wald, p=0.5 worst case"})
    po, pe = ag.expected_kappa_po_pe(0.6, 0.5)
    rows.append({"target": "Cohen kappa (expected 0.6, prevalence 0.5)", "halfwidth": 0.10, "n_required": ag.n_for_kappa(0.10, po, pe),
                 "basis": "Cohen (1960) large-sample SE"})
    for n in (100, 150, 200):
        rows.append({"target": f"Cohen kappa simulated at n={n} (kappa 0.6, prev 0.5)", "halfwidth": round(ag.simulate_kappa_halfwidth(n, 0.6, 0.5, 2000, 42), 3),
                     "n_required": n, "basis": "Monte-Carlo, 2000 draws, seed 42"})
    rows.append({"target": "mean relevance 0-3 (sd 1.0)", "halfwidth": 0.15, "n_required": ag.n_for_mean(0.15, 1.0), "basis": "normal approx."})
    return rows


def build(seed: int, smoke: bool, recs_cache: Path, run_recommender: bool) -> Dict[str, Any]:
    scale = SMOKE_SCALE if smoke else 1.0
    out = OUT / ("smoke" if smoke else "")
    key_dir = KEY_ROOT / ("smoke" if smoke else "") / "answer_keys"
    pk_dir = out / "packets"
    docs = load_test_docs()
    scen, cat, labels, split = _rec_inputs()
    full, base, rec_origin = load_full_recs(recs_cache, run_recommender, split["test"])
    built: Dict[str, Tuple[List[Dict], List[Dict], Dict]] = {}
    built["P1"] = build_p1(docs, random.Random(seed + 1), scale)
    built["P2"] = build_p2(docs, random.Random(seed + 2), scale)
    built["P3"] = build_p3(docs, random.Random(seed + 3), scale)
    built.update(build_p4(random.Random(seed + 4), scale, full))
    built.update(build_p5_p6(random.Random(seed + 5), scale, full, base, rec_origin))
    p7 = build_p7(random.Random(seed + 7), scale)
    if p7:
        built["P7"] = p7
    manifest = {}
    for p, (vis, key, meta) in built.items():
        qcols = [q for q, _, _ in QUESTIONS[p]]
        if p == "P2":
            qcols = qcols + ["completed"]
        disp = ["item_id"] + [c for c in (vis[0] if vis else {}) if c != "item_id"]
        for a in ANNOTATORS:
            order = list(vis)
            random.Random(ORDER_SEEDS[a] + seed).shuffle(order)
            _write_csv(pk_dir / f"{p}__{a}.csv", order, disp + qcols + ["notes"])
        kcols = sorted({c for r in key for c in r}, key=lambda c: (c != "item_id", c))
        _write_csv(key_dir / f"{p}_key.csv", key, kcols)
        manifest[p] = {"title": PACKET_TITLES[p], "n_items": len(vis), "questions": QUESTIONS[p],
                       "packet_files": [rel(pk_dir / f"{p}__{a}.csv") for a in ANNOTATORS],
                       "answer_key": rel(key_dir / f"{p}_key.csv"), **{k: v for k, v in meta.items()}}
    tax = taxonomy_reference()
    _write_csv(pk_dir / "reference_taxonomy.csv", tax, ["canonical_concept", "aliases"])
    manifest["training"] = write_training(pk_dir / "training", key_dir / "training")
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "annotations" / "README.md").write_text(
        "Annotators return their filled packet here as `<packet>__A.csv` / `<packet>__B.csv` (unchanged item_id column).\n"
        "The adjudicator returns `<packet>__ADJ.csv` containing only the items listed in `<packet>__ADJ_todo.csv`.\n"
        "Then run `PYTHONPATH=src python -m skillllm_v2.evals.research.human_eval_samples --score`.\n")
    (out / "packet_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return {"manifest": manifest, "out": out, "built": built, "rec_origin": rec_origin, "key_dir": key_dir}


def adjudication_todo(packet: str, ann_dir: Path) -> Optional[Path]:
    files = _annotation_files(packet, ann_dir)
    if not all(a in files for a in ANNOTATORS):
        return None
    ra = {r["item_id"]: r for r in _read_csv(files["A"])}
    rb = {r["item_id"]: r for r in _read_csv(files["B"])}
    cols = [q for q, s, _ in QUESTIONS[packet] if s != "text"]
    todo = []
    for i in sorted(set(ra) & set(rb)):
        if any((_cell(ra[i].get(c)) or "").lower() != (_cell(rb[i].get(c)) or "").lower() for c in cols):
            todo.append({"item_id": i, **{f"A_{c}": ra[i].get(c, "") for c in cols}, **{f"B_{c}": rb[i].get(c, "") for c in cols}})
    path = ann_dir / f"{packet}__ADJ_todo.csv"
    _write_csv(path, todo, ["item_id"] + [f"{s}_{c}" for c in cols for s in ("A", "B")] + cols + ["notes"])
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="L11 human-evaluation packets and agreement scoring")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--recs-cache", type=Path, default=KEY_ROOT / "cache" / "full_tuned_heldout_recs.json")
    ap.add_argument("--run-recommender", action="store_true", help="re-run the frozen recommender for explanations (slow)")
    ap.add_argument("--score", action="store_true", help="only re-score existing packets against returned annotations")
    a = ap.parse_args(argv)
    if a.score:
        out = OUT / ("smoke" if a.smoke else "")
        manifest = json.loads((out / "packet_manifest.json").read_text())
        built = None
    else:
        res = build(a.seed, a.smoke, a.recs_cache, a.run_recommender)
        manifest, out, built = res["manifest"], res["out"], res["built"]
    ann_dir = out / "annotations"
    records: List[Dict[str, Any]] = []
    packets = [p for p in manifest if p != "training"]
    for p in packets:
        n = manifest[p]["n_items"]
        records.append(record(LAYER, "annotation_design", "items_in_packet", float(n), n_units=n, unit="item", split=f"L11 packet {p}",
                              dataset=PACKET_TITLES[p], direction="none", provenance=[manifest[p]["answer_key"]],
                              details={"origin": manifest[p].get("origin", ""), "packet_files": manifest[p]["packet_files"]}))
        sentences = None
        if p == "P2" and (KEY_ROOT / ("smoke" if a.smoke else "") / "answer_keys" / "P2_key.csv").exists():
            docs = load_test_docs()
            keyrows = _read_csv(KEY_ROOT / ("smoke" if a.smoke else "") / "answer_keys" / "P2_key.csv")
            sentences = {r["item_id"]: docs[r["doc_id"]]["text"] for r in keyrows}
        records += score_packet(p, ann_dir, n, a.n_boot, a.seed, sentences)
        adjudication_todo(p, ann_dir)
    for p in ("P7",):
        if p not in packets:
            records.append(record(LAYER, "human_annotators", "perturbation_meaning_preserved_rate", None, n_units=0, unit="item",
                                  split="L11 packet P7", dataset=PACKET_TITLES["P7"], direction="higher", status="pending_human",
                                  reason="A3's perturbation validity sample (L03_robustness/*validity*.csv|jsonl) not present at build time; "
                                         "re-run this module after it appears"))
    cand_only = {"human_choice_matches_decorte_gold_rate", "linker_top1_matches_adjudicated_human_rate"}
    fallback_only = {"linking_accuracy_correct_or_acceptable", "linking_accuracy_strict"}
    for p, metric, definition, direction in SUBSTANTIVE:
        if p not in packets:
            continue
        if p == "P3" and metric in (fallback_only if P3_CANDIDATE_MODE else cand_only):
            continue  # metric belongs to the other P3 mode (proposed-concept fallback vs ESCO candidate choice)
        files = _annotation_files(p, ann_dir)
        why = ("no annotation files returned yet" if not files else
               "adjudicated labels required (" + ", ".join(sorted(files)) + " present; ADJ missing)" if "ADJ" not in files else
               "adjudicated labels present; substantive scoring is run by the coordinator with the answer key (not automated "
               "in this build)")
        records.append(record(LAYER, "adjudicated_humans", metric, None, n_units=manifest[p]["n_items"], unit="item",
                              split=f"L11 packet {p}", dataset=PACKET_TITLES[p], direction=direction, status="pending_human",
                              reason=why, details={"definition": definition, "packet": p}))
    ss = sample_size_table()
    report = render_report(manifest, records, ss, a)
    write_layer(LAYER, records, report, seed=a.seed, smoke=a.smoke,
                extra_meta={"packet_manifest": rel(out / "packet_manifest.json"), "sample_size_table": ss})
    print(f"L11: {len(packets)} packets; " + ", ".join(f"{p}={manifest[p]['n_items']}" for p in packets))
    return 0


def render_report(manifest: Dict[str, Any], records: List[Dict[str, Any]], ss: List[Dict[str, Any]], a: argparse.Namespace) -> str:
    pk = [{"packet": p, "title": m["title"], "items": m["n_items"], "origin": str(m.get("origin", "SkillSpan test"))[:90]}
          for p, m in manifest.items() if p != "training"]
    status = Counter(r["status"] for r in records)
    lines = [
        "# L11 — Human evaluation (packets built; human metrics pending)", "",
        "Protocol: `docs/thesis/evaluation/human_eval_protocol.md`. Two independent annotators (A, B) + one adjudicator; "
        "system identity hidden; item order randomised separately per annotator; answer keys live in "
        "`inputs/evals/research/L11_human_eval/answer_keys/` (never handed out). LLM judges are not used.", "",
        f"Records: {dict(status)}. Every human metric is `pending_human` until both annotators return >= 90% of a packet.", "",
        "## Packets", "", md_table(pk, ["packet", "title", "items", "origin"]), "",
        f"Training round T0 (synthetic items with answer key and rationale): {manifest.get('training')}.", "",
        "## Sample-size justification", "", md_table(ss, ["target", "halfwidth", "n_required", "basis"], fmt="{:.3f}"), "",
        "## Design notes", "",
        "- P1 strata come from the SkillSpan test gold and stored predictions (974-document anchor subset). The anchor_fp "
        "stratum measures gold incompleteness: an anchor 'false positive' that humans accept as a skill means the reported "
        "precision is pessimistic. The control stratum (agreed true positives) checks annotator calibration.",
        "- P2 includes gold-free test sentences, the population excluded from all stored extraction predictions "
        "(974/3,569 caveat), so the human rate of skills in 'skill-free' sentences can be set against A3's spurious-prediction rate.",
        "- P5 annotators see the job text and the listed required/preferred skills (the systems' inputs), NOT the rule-label "
        "`gold.needs` (which include implicit needs and drive the v1 rule labels); the legacy 60-row sample showed its "
        "automatic grade and needed-skill weights and is re-blinded here.",
        "- P6 cannot be system-blind (only the Full recommender produces explanations) and is reported as such.",
        "- Agreement: Cohen's kappa (quadratic-weighted for ordinal scales), Gwet's AC1 for nominal scales, Krippendorff's "
        "alpha at the scale's level, span-level F1 for free span annotation; 95% percentile bootstrap over items. "
        "`agreement.py` reproduces the published Krippendorff (2011) example (nominal 0.743, ordinal 0.815, interval 0.849, "
        "ratio 0.797) and matches the `krippendorff` package and scikit-learn to <= 2.3e-16 (see `agreement_oracle_check.json`).",
        "", f"Command: `PYTHONPATH=src python -m skillllm_v2.evals.research.human_eval_samples --seed {a.seed} --n-boot {a.n_boot}`"
        " (build); add `--score` after annotations are returned.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
