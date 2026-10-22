"""The recommender tuning loop (dev half ONLY): five rounds, one change per round, before/after logged.

    PYTHONPATH=src python -m skillllm_v2.recommend.loop_rounds      # writes reports/skillllm_v2/recommender_loop_log.md + .json

Selection protocol (fixed before running; applies to every round)
  * Objective: dev nDCG@10 (the primary metric of the benchmark).  Dev = the 24 scenarios in recommender_split.json.
  * Hard constraints for any accepted configuration: Prerequisite Violation Rate@5 not above the previous round's,
    Explanation Completeness@5 == 1.0, dev Redundancy@5 <= previous round + 0.001.
  * Round 1 is a constraint round (drive the violation rate to 0): accept the best arm with violation 0 whose dev nDCG@10
    is within 0.005 of the previous round.  Rounds 2-5 are performance rounds: accept the best constraint-satisfying arm
    only if it improves dev nDCG@10 by at least 0.002; otherwise the round is REJECTED and the config is unchanged.
    Arms within 0.001 nDCG@10 of the best are treated as tied; the tie goes to the arm returning the longest list.
  * Nothing here reads the test half; the test half is evaluated once by recommender_final.py with the frozen config.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillllm_v2.recommend.config import LEGACY_CONFIG, RecommenderConfig
from skillllm_v2.recommend.loop_eval import ROOT, TABLE_KEYS, Harness, load_split, mean_of

LOG_MD = ROOT / "reports/skillllm_v2/recommender_loop_log.md"
LOG_JSON = ROOT / "reports/skillllm_v2/recommender_loop_log.json"
MIN_GAIN = 0.002
R1_TOLERANCE = 0.005
RED_TOL = 0.001
TIE_TOL = 0.001

R1_FEATURE = {"prereq_mode": "strict_bridge", "compat_basis": "learner", "w_unlock": 0.2}
R2_FEATURE = {"fill_related": True, "related_match": "token", "retrieval_pool": 40, "related_discount": 0.8}
R3_FEATURE = {"redundancy_mode": "mmr", "w_redundancy": 0.10}


def _round_specs() -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "title": "Strict prerequisite handling with bridge sequencing",
            "kind": "constraint",
            "what": "A course is eligible only if every prerequisite is known by the learner or taught by a course placed earlier in the list; "
                    "a blocked course can be unlocked by a startable bridge course (placed before it) chosen by unlock gain. "
                    "Prerequisite compatibility in the score counts only the learner's own skills (prefers courses startable now).",
            "arms": {
                "strict_drop, compat=chain": {"prereq_mode": "strict_drop", "compat_basis": "chain"},
                "strict_drop, compat=learner": {"prereq_mode": "strict_drop", "compat_basis": "learner"},
                "strict_bridge, compat=chain, w_unlock=0.4": {"prereq_mode": "strict_bridge", "compat_basis": "chain", "w_unlock": 0.4},
                "strict_bridge, compat=learner, w_unlock=0.0": {"prereq_mode": "strict_bridge", "compat_basis": "learner", "w_unlock": 0.0},
                "strict_bridge, compat=learner, w_unlock=0.2": R1_FEATURE,
                "strict_bridge, compat=learner, w_unlock=0.4": {"prereq_mode": "strict_bridge", "compat_basis": "learner", "w_unlock": 0.4},
            },
            "chosen": "strict_bridge, compat=learner, w_unlock=0.2",
        },
        {
            "id": 2,
            "title": "Fill the list towards top-K with next-best feasible related courses",
            "kind": "performance",
            "what": "When no gap-covering or bridge candidate is left, append feasible courses whose text mentions an open gap skill or that teach a "
                    "skill sharing a content word with it (flagged role='related_text', with an explicit text-only limitation), best score first. "
                    "Candidate pool 15 -> 40.",
            "arms": {
                "fill phrase-match only, pool=40": {**R2_FEATURE, "related_match": "phrase"},
                "fill token-match, pool=15": {**R2_FEATURE, "retrieval_pool": 15},
                "fill token-match, pool=40, discount=0.8": R2_FEATURE,
                "fill token-match, pool=25, discount=0.8": {**R2_FEATURE, "retrieval_pool": 25},
            },
            "chosen": "fill token-match, pool=40, discount=0.8",
        },
        {
            "id": 3,
            "title": "Redundancy / diversity penalty (MMR-style) tuned on dev",
            "kind": "performance",
            "what": "Replace the share-of-own-skills redundancy penalty by an MMR-style penalty on the already-covered gap MASS and tune its weight; "
                    "constraint: dev Redundancy@5 must not exceed the previous round's.",
            "arms": {
                **{f"redundancy=fraction, w={w}": {"redundancy_mode": "fraction", "w_redundancy": w} for w in (0.0, 0.05, 0.10, 0.25, 0.40)},
                **{f"redundancy=mmr, w={w}" + ("" if w == 0.10 else ""): {"redundancy_mode": "mmr", "w_redundancy": w} for w in (0.0, 0.05, 0.10, 0.25, 0.40, 0.60, 1.0)},
            },
            "chosen": "redundancy=mmr, w=0.1",
        },
        {
            "id": 4,
            "title": "Learner-level inference (user-level suitability)",
            "kind": "performance",
            "what": "The benchmark's user profiles carry no experience level, so the level-suitability term used a constant 'intermediate'. "
                    "Infer the level from the profile: (a) number of declared skills, (b) depth of the learner's skills (mean highest catalogue level at which each is taught).",
            "arms": {
                **{f"infer by skill count, jr<={a} sr>={b}": {"infer_user_level": True, "level_method": "skill_count", "level_junior_skills": a, "level_senior_skills": b}
                   for a, b in ((3, 6), (4, 7), (5, 8))},
                **{f"infer by skill depth, jr<={a} sr>={b}": {"infer_user_level": True, "level_method": "catalog_depth", "level_depth_junior": a, "level_depth_senior": b}
                   for a in (1.4, 1.6, 1.7) for b in (2.0, 2.2)},
            },
            "chosen": None,
        },
        {
            "id": 5,
            "title": "Weights of the interpretable score terms (coordinate search on dev)",
            "kind": "performance",
            "what": "One-at-a-time coordinate search (two passes max) over a coarse grid per weight; a move is taken only if it improves dev nDCG@10 by >= 0.002 "
                    "and satisfies the hard constraints.",
            "arms": {},
            "chosen": "coordinate",
        },
    ]


GRIDS = {
    "w_gap": [0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    "w_semantic": [0.0, 0.05, 0.15, 0.25, 0.35],
    "w_prereq": [0.0, 0.1, 0.2, 0.3],
    "w_level": [0.0, 0.05, 0.15, 0.3],
    "w_goal": [0.0, 0.1, 0.2],
    "w_workload": [0.05, 0.2],
    "w_unlock": [0.0, 0.1, 0.2, 0.4, 0.6],
}


def _pick(pool):
    """Best arm by dev nDCG@10; arms within TIE_TOL of the best are treated as tied and the one returning the longest list wins."""
    if not pool:
        return None
    top = max(a[3]["ndcg10"] for a in pool)
    tied = [a for a in pool if a[3]["ndcg10"] >= top - TIE_TOL]
    return max(tied, key=lambda a: (round(a[3]["n_returned"], 6), a[3]["ndcg10"]))


def _row(name: str, m: dict[str, float]) -> str:
    return f"| {name} | " + " | ".join(f"{m[k]:.3f}" for k in TABLE_KEYS) + " |"


def _hdr() -> list[str]:
    return ["| config | " + " | ".join(TABLE_KEYS) + " |", "| --- |" + " --- |" * len(TABLE_KEYS)]


def run_loop() -> dict[str, Any]:
    H = Harness(with_baselines=False)
    dev = load_split()["dev"]
    cache: dict[str, Any] = {}

    def evaluate(cfg: RecommenderConfig):
        key = json.dumps(cfg.to_dict(), sort_keys=True)
        if key not in cache:
            per = H.eval_config(cfg, dev)
            cache[key] = (per, {k: mean_of(per, k) for k in TABLE_KEYS})
        return cache[key]

    def ok(m: dict[str, float], prev: dict[str, float]) -> bool:
        return m["prereq_viol5"] <= prev["prereq_viol5"] + 1e-9 and m["expl5"] >= 1.0 - 1e-9 and m["redundancy5"] <= prev["redundancy5"] + RED_TOL

    cur = LEGACY_CONFIG
    cur_per, cur_m = evaluate(cur)
    md: list[str] = []
    js: dict[str, Any] = {"split": "dev", "n_dev": len(dev), "rounds": [], "round0": {"config": cur.to_dict(), "dev": cur_m}}
    md += ["# Recommender improvement loop - dev-half log", "",
           "Protocol: 48 benchmark scenarios were split ONCE (sha256-sorted ids; `recommender_split.json`) into 24 dev / 24 test. Every number in this file is "
           "from the 24 **dev** scenarios; the test half is not touched until `recommender_final.py` runs the frozen config once (see "
           "`course_recommendation_benchmark_v2.md`). Primary objective: dev nDCG@10. Hard constraints for an accepted config: Prerequisite Violation "
           "Rate@5 not above the previous round, Explanation Completeness@5 = 1.0, dev Redundancy@5 <= previous round + 0.001. Round 1 is a constraint round "
           "(violation -> 0 within 0.005 nDCG); rounds 2-5 need >= +0.002 dev nDCG@10 to be accepted, else they are REJECTED and the config is unchanged. "
           "(Arms within 0.001 nDCG@10 of the best count as tied; the tie goes to the arm that returns the longest list.) "
           "Labels, baselines and metric code are untouched; no gold/relevance label is an input of the recommender.", "",
           "Caveat that applies to every round: with 24 dev scenarios a +0.002..0.01 nDCG change is well inside sampling noise; the rules above only stop "
           "us from accepting changes that hurt. Dev-set tuning of any kind can overfit - only the test half can say whether a gain is real.", "",
           "## Round 0 - starting point (legacy recommender, soft prerequisite score)", ""] + _hdr() + [_row("R0 legacy", cur_m), ""]

    for spec in _round_specs():
        rid = spec["id"]
        before_cfg, before_per, before_m = cur, cur_per, cur_m
        arms_out: list[tuple[str, RecommenderConfig, dict[str, Any], dict[str, float]]] = []
        notes = ""
        if spec["chosen"] == "coordinate":
            best_cfg, best_m = before_cfg, before_m
            trail: list[str] = []
            n_evals = 0
            for pas in (1, 2):
                moved = False
                for name, vals in GRIDS.items():
                    cand_best = (getattr(best_cfg, name), best_m)
                    for v in vals:
                        if v == getattr(best_cfg, name):
                            continue
                        c = best_cfg.with_(**{name: v})
                        per, m = evaluate(c)
                        n_evals += 1
                        arms_out.append((f"pass {pas}: {name}={v}", c, per, m))
                        if ok(m, before_m) and m["ndcg10"] > cand_best[1]["ndcg10"] + MIN_GAIN:
                            cand_best = (v, m)
                    if cand_best[0] != getattr(best_cfg, name):
                        trail.append(f"pass {pas}: {name} {getattr(best_cfg, name)} -> {cand_best[0]} (dev nDCG@10 {best_m['ndcg10']:.4f} -> {cand_best[1]['ndcg10']:.4f})")
                        best_cfg = best_cfg.with_(**{name: cand_best[0]})
                        best_m = cand_best[1]
                        moved = True
                if not moved:
                    break
            chosen_cfg, chosen_label = best_cfg, ("weights moved: " + "; ".join(trail)) if trail else "no weight move cleared the bar"
            notes = f"{n_evals} configurations evaluated. " + (" ".join(trail) if trail else "")
            accepted = chosen_cfg != before_cfg
        else:
            for label, delta in spec["arms"].items():
                c = before_cfg.with_(**delta)
                per, m = evaluate(c)
                arms_out.append((label, c, per, m))
            pool = [a for a in arms_out if ok(a[3], before_m)]
            if spec["kind"] == "constraint":
                pool = [a for a in pool if a[3]["prereq_viol5"] == 0.0]
                best = _pick(pool)
                accepted = best is not None and best[3]["ndcg10"] >= before_m["ndcg10"] - R1_TOLERANCE
            else:
                best = _pick(pool)
                accepted = best is not None and best[3]["ndcg10"] >= before_m["ndcg10"] + MIN_GAIN
            chosen_cfg = best[1] if (accepted and best) else before_cfg
            chosen_label = best[0] if best else "none"
        after_per, after_m = evaluate(chosen_cfg)
        diffs = [after_per[s]["ndcg10"] - before_per[s]["ndcg10"] for s in dev]
        wl = (sum(d > 1e-9 for d in diffs), sum(d < -1e-9 for d in diffs), sum(abs(d) <= 1e-9 for d in diffs))
        status = "ACCEPTED" if accepted else "REJECTED (config unchanged)"
        md += [f"## Round {rid} - {spec['title']} [{status}]", "", f"**Change:** {spec['what']}", ""]
        md += _hdr() + [_row("before (previous round)", before_m)]
        for label, _, _, m in arms_out:
            md.append(_row(label + (" **<- adopted**" if accepted and chosen_label == label else ""), m))
        md += [_row("**after (carried forward)**", after_m), ""]
        md += [f"Dev nDCG@10: {before_m['ndcg10']:.3f} -> {after_m['ndcg10']:.3f} ({after_m['ndcg10'] - before_m['ndcg10']:+.3f}); per-scenario W/L/T = {wl[0]}/{wl[1]}/{wl[2]}. "
               f"Prereq violation {before_m['prereq_viol5']:.3f} -> {after_m['prereq_viol5']:.3f}; avg list length {before_m['n_returned']:.2f} -> {after_m['n_returned']:.2f}; "
               f"Redundancy@5 {before_m['redundancy5']:.3f} -> {after_m['redundancy5']:.3f}; Explanation Completeness {after_m['expl5']:.3f}.", ""]
        if notes:
            md += [notes, ""]
        if not accepted:
            md += [f"Best constraint-satisfying arm: {chosen_label}; it did not reach the +{MIN_GAIN} dev nDCG@10 bar, so this change is **not** adopted.", ""]
        if accepted:
            js_changes = {k: v for k, v in chosen_cfg.to_dict().items() if v != before_cfg.to_dict()[k]}
        else:
            js_changes = {}
        js["rounds"].append({"round": rid, "title": spec["title"], "accepted": accepted, "changes": js_changes,
                             "before": before_m, "after": after_m, "arms": {a[0]: a[3] for a in arms_out}, "wlt_ndcg10": wl})
        cur, cur_per, cur_m = chosen_cfg, after_per, after_m

    md += ["## Final dev-tuned configuration (frozen for the test run)", "", "```json", json.dumps(cur.to_dict(), indent=2), "```", ""]
    delta = {k: v for k, v in cur.to_dict().items() if v != LEGACY_CONFIG.to_dict()[k]}
    md += ["Differences from the legacy config: `" + json.dumps(delta) + "`", "",
           f"Dev summary: nDCG@10 {js['round0']['dev']['ndcg10']:.3f} -> {cur_m['ndcg10']:.3f}, prereq violation {js['round0']['dev']['prereq_viol5']:.3f} -> {cur_m['prereq_viol5']:.3f}, "
           f"avg list length {js['round0']['dev']['n_returned']:.2f} -> {cur_m['n_returned']:.2f}.", ""]
    js["final_config"] = cur.to_dict()
    js["final_dev"] = cur_m
    LOG_MD.write_text("\n".join(md), encoding="utf-8")
    LOG_JSON.write_text(json.dumps(js, indent=2), encoding="utf-8")
    return js


if __name__ == "__main__":
    out = run_loop()
    print(json.dumps(out["final_config"], indent=2))
    print({k: round(v, 4) for k, v in out["final_dev"].items()})
