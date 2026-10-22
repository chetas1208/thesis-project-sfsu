"""Dev/test tuning harness for the course recommender (reuses the benchmark's baselines, labels and metrics unchanged).

    PYTHONPATH=src python -m skillllm_v2.recommend.loop_eval split            # (re)write the fixed dev/test split
    PYTHONPATH=src python -m skillllm_v2.recommend.loop_eval dev ROUND ...    # evaluate named configs on the dev half
    PYTHONPATH=src python -m skillllm_v2.recommend.loop_eval test             # frozen config, test half, ONCE

Protocol (see reports/skillllm_v2/recommender_loop_log.md)
  * The 48 benchmark scenarios are split ONCE into dev (24) / test (24) by sorting sha256("<salt>:<scenario_id>")
    (no scenario id is special-cased anywhere; the split is recorded in reports/skillllm_v2/recommender_split.json).
  * All tuning reads dev scenarios only.  ``test`` is run once with the frozen DEFAULT_CONFIG.
  * Baselines, labels and metric code are imported from skillllm_v2.eval.recommender_benchmark / ranking_metrics
    without modification.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from skillllm_v2.course.ingest_catalog import CourseCatalog
from skillllm_v2.eval import ranking_metrics as rm
from skillllm_v2.eval import recommender_benchmark as rb
from skillllm_v2.recommend.config import RecommenderConfig
from skillllm_v2.schema import normalize_text

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "reports/skillllm_v2/recommender_split.json"
SPLIT_SALT = "skillllm-recsys-loop-v1"
N_DEV = 24


def split_ids(scenario_ids: list[str], salt: str = SPLIT_SALT, n_dev: int = N_DEV) -> tuple[list[str], list[str]]:
    keyed = sorted(scenario_ids, key=lambda s: hashlib.sha256(f"{salt}:{s}".encode()).hexdigest())
    return sorted(keyed[:n_dev]), sorted(keyed[n_dev:])


def write_split(scenarios_path: Path = ROOT / "inputs/courses/benchmark_scenarios.jsonl", path: Path = SPLIT_PATH) -> dict[str, Any]:
    scs = rb.load_jsonl(scenarios_path)
    ids = [s["scenario_id"] for s in scs]
    dev, test = split_ids(ids)
    by = {s["scenario_id"]: s for s in scs}
    info = {
        "method": f"sort scenario ids by sha256('{SPLIT_SALT}:<scenario_id>'); first {N_DEV} = dev, remaining = test",
        "salt": SPLIT_SALT,
        "n_dev": len(dev),
        "n_test": len(test),
        "dev": dev,
        "test": test,
        "composition": {
            half: {
                "experience_level": {lv: sum(1 for i in idx if by[i]["experience_level"] == lv) for lv in ("junior", "mid", "senior")},
                "source": {src: sum(1 for i in idx if by[i]["source"] == src) for src in sorted({s["source"] for s in scs})},
            }
            for half, idx in (("dev", dev), ("test", test))
        },
        "note": "Fixed before any tuning. Tune on dev only; test is evaluated once with the frozen config.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def load_split(path: Path = SPLIT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class Harness:
    """Holds the catalogue / scenarios / labels / baselines / a reusable pipeline; evaluates configs on id subsets."""

    def __init__(
        self,
        catalog_path: Path = ROOT / "inputs/courses/course_catalog_large.jsonl",
        scenarios_path: Path = ROOT / "inputs/courses/benchmark_scenarios.jsonl",
        labels_path: Path = ROOT / "inputs/courses/relevance_labels.jsonl",
        with_baselines: bool = True,
    ) -> None:
        self.catalog = CourseCatalog.load_from_file(catalog_path)
        self.by_id = {c.course_id: c for c in self.catalog}
        self.scenarios = {s["scenario_id"]: s for s in rb.load_jsonl(scenarios_path)}
        self.labels = {r["scenario_id"]: r for r in rb.load_jsonl(labels_path)}
        self.catalog_path = catalog_path
        from skillllm_v2.pipeline.course_recommendation_pipeline import CourseRecommendationPipeline

        self.pipe = CourseRecommendationPipeline(catalog_path=str(catalog_path))
        self.base_graph = self.pipe.graph.graph.copy()
        self.dense_backend = ""
        self.baseline_lists: dict[str, dict[str, list[dict[str, Any]]]] = {}
        if with_baselines:
            self._build_baselines()

    # ---- scenario plumbing ----------------------------------------------------------------------------------
    def _scenario_context(self, sid: str):
        sc = self.scenarios[sid]
        lab = self.labels[sid]
        grades = {c.course_id: float(lab["labels"].get(c.course_id, 0)) for c in self.catalog}
        grades_c = {c.course_id: float(lab["labels_content_only"].get(c.course_id, 0)) for c in self.catalog}
        needs = sc["gold"]["needs"]
        known_all = set(sc["gold"]["known"]) | set(sc["gold"]["weak"])
        gold_gaps = [{"skill_name": s, "importance": float(w)} for s, w in needs.items() if s not in set(sc["gold"]["known"])]
        user_known = {normalize_text(s) for s in known_all}
        return sc, grades, grades_c, gold_gaps, user_known

    def _build_baselines(self) -> None:
        bm25 = rb.Bm25Index(self.catalog)
        dense = rb.DenseIndex(self.catalog)
        self.dense_backend = dense.backend
        K = rb.K_RANK
        for sid, sc in self.scenarios.items():
            _, _, _, _, user_known = self._scenario_context(sid)
            user, job = sc["user_profile"], sc["target_job"]
            query = f"{job['title']}. {job['text']}"
            L: dict[str, list[dict[str, Any]]] = {}
            bm = [c for _, c in bm25.rank(query)]
            L["BM25 Keyword"] = [rb.as_rec(c) for c in bm[:K]]
            L["BM25 + prereq filter"] = [rb.as_rec(c) for c in rb.apply_prereq_filter(bm, user_known, K)]
            de = [c for _, c in dense.rank(query)]
            L["Dense Semantic"] = [rb.as_rec(c) for c in de[:K]]
            L["Dense + prereq filter"] = [rb.as_rec(c) for c in rb.apply_prereq_filter(de, user_known, K)]
            so = [c for _, c in rb.skill_overlap_rank(self.catalog, job)]
            L["Skill Overlap (user-agnostic)"] = [rb.as_rec(c) for c in so[:K]]
            gaps = rb.user_gap_weights(user, job)
            L["Gap-Overlap Greedy (no graph)"] = [rb.as_rec(c) for c in rb.gap_overlap_greedy(self.catalog, gaps, user_known, False, K)]
            L["Gap-Overlap Greedy + prereq filter"] = [rb.as_rec(c) for c in rb.gap_overlap_greedy(self.catalog, gaps, user_known, True, K)]
            self.baseline_lists[sid] = L

    # ---- evaluation -----------------------------------------------------------------------------------------
    def run_full(self, config: RecommenderConfig, ids: list[str], top_k: int = rb.K_RANK) -> dict[str, list[dict[str, Any]]]:
        """Run the Full recommender with ``config`` on each scenario (graph state reset per scenario, as in the benchmark)."""
        self.pipe.set_config(config)
        out: dict[str, list[dict[str, Any]]] = {}
        for sid in ids:
            sc = self.scenarios[sid]
            self.pipe.graph.graph = self.base_graph.copy()
            res = self.pipe.run(user_input=copy.deepcopy(sc["user_profile"]), job_input=copy.deepcopy(sc["target_job"]), top_k=top_k)
            out[sid] = res["recommendations"]
        return out

    def score_lists(self, lists: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float | None]]:
        per: dict[str, dict[str, float | None]] = {}
        for sid, recs in lists.items():
            _, grades, grades_c, gold_gaps, user_known = self._scenario_context(sid)
            per[sid] = rb.evaluate_scenario_lists(recs, grades, grades_c, gold_gaps, user_known, self.by_id)
        return per

    def eval_config(self, config: RecommenderConfig, ids: list[str]) -> dict[str, dict[str, float | None]]:
        return self.score_lists(self.run_full(config, ids))

    def eval_baselines(self, ids: list[str]) -> dict[str, dict[str, dict[str, float | None]]]:
        out: dict[str, dict[str, dict[str, float | None]]] = {}
        for name in rb.SYSTEMS:
            if name == rb.SYSTEM_FULL:
                continue
            out[name] = self.score_lists({sid: self.baseline_lists[sid][name] for sid in ids})
        return out


def mean_of(per: dict[str, dict[str, float | None]], key: str, ids: list[str] | None = None) -> float:
    ids = ids or list(per)
    vals = [per[i][key] for i in ids if per[i][key] is not None]
    return sum(vals) / len(vals) if vals else float("nan")


TABLE_KEYS = ["ndcg10", "p5", "recall10", "map10", "gap_cov5", "prereq_viol5", "redundancy5", "expl5", "n_returned"]


def fmt_row(name: str, per: dict[str, dict[str, float | None]]) -> str:
    return f"{name:<44}" + " ".join(f"{mean_of(per, k):>9.3f}" for k in TABLE_KEYS)


def header() -> str:
    return f"{'config':<44}" + " ".join(f"{k:>9}" for k in TABLE_KEYS)
