"""Frozen-config evaluation of the tuned Full recommender on the HELD-OUT test half (run once).

    PYTHONPATH=src python -m skillllm_v2.recommend.recommender_final [--force]

Writes reports/skillllm_v2/course_recommendation_benchmark_v2.{md,json}.  Baselines, labels and metric code come unchanged from
skillllm_v2.eval.recommender_benchmark.  The Full config is DEFAULT_CONFIG (must equal the final config of recommender_loop_log.json).
Also reported: the pre-tuning (legacy) Full on the same halves, the dev half (what was tuned on) and the full 48 (partly dev-tuned).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from skillllm_v2.eval import ranking_metrics as rm
from skillllm_v2.eval import recommender_benchmark as rb
from skillllm_v2.recommend.config import DEFAULT_CONFIG, LEGACY_CONFIG
from skillllm_v2.recommend.loop_eval import ROOT, Harness, load_split

OUT_MD = ROOT / "reports/skillllm_v2/course_recommendation_benchmark_v2.md"
OUT_JSON = ROOT / "reports/skillllm_v2/course_recommendation_benchmark_v2.json"
LOOP_JSON = ROOT / "reports/skillllm_v2/recommender_loop_log.json"
SYS_LEGACY = "SkillLLM_v2 Full (legacy, pre-tuning)"
PRESPEC_BASELINE = "Gap-Overlap Greedy + prereq filter"  # strongest baseline in the earlier 48-scenario benchmark and on dev
KEYS = [m[0] for m in rb.METRICS] + ["n_returned", "ndcg10_content_only", "p5_content_only"]
NAMES = {k: n for k, n, _ in rb.METRICS}
NAMES.update({"n_returned": "Avg #returned", "ndcg10_content_only": "nDCG@10 (content-only labels)", "p5_content_only": "Precision@5 (content-only labels)"})
HIB = {k: h for k, _, h in rb.METRICS}


def _summ(per: dict[str, dict[str, list], ], ids: list[str], all_ids: list[str], n_boot: int, seed: int, systems: list[str]):
    idx = [all_ids.index(i) for i in ids]
    valid = {k: [i for i in idx if all(per[s][k][i] is not None for s in systems)] for k in KEYS}
    arr = lambda s, k: [float(per[s][k][i]) for i in valid[k]]
    S = {s: {k: dict(zip(("mean", "ci_lo", "ci_hi"), rm.bootstrap_mean_ci(arr(s, k), n_boot=n_boot, seed=seed)), n=len(valid[k])) for k in KEYS} for s in systems}
    baselines = [s for s in systems if s not in (rb.SYSTEM_FULL, SYS_LEGACY)]
    P: dict[str, dict[str, Any]] = {}
    for k in KEYS:
        raw = {b: rm.paired_bootstrap(arr(rb.SYSTEM_FULL, k), arr(b, k), n_boot=n_boot, seed=seed + 1) for b in baselines}
        for b, a in zip(baselines, rm.holm_adjust([raw[b]["p_value"] for b in baselines])):
            raw[b]["p_holm"] = a
        for b in baselines:
            P.setdefault(b, {})[k] = raw[b]
    L = {k: rm.paired_bootstrap(arr(rb.SYSTEM_FULL, k), arr(SYS_LEGACY, k), n_boot=n_boot, seed=seed + 2) for k in KEYS}
    return {"systems": S, "paired_vs_full": P, "tuned_vs_legacy": L, "n": len(idx)}


def _fmt(m: dict) -> str:
    return rb._fmt(m)


def _tables(title: str, R: dict[str, Any], systems: list[str]) -> list[str]:
    S, P = R["systems"], R["paired_vs_full"]
    out = [f"### {title} (n = {R['n']} scenarios)", ""]
    out += ["| System | " + " | ".join(f"{NAMES[k]}{' ↑' if HIB[k] else ' ↓'}" for k, _, _ in rb.METRICS) + " | #returned |", "| --- |" + " --- |" * (len(rb.METRICS) + 1)]
    for s in systems:
        nm = f"**{s}**" if s == rb.SYSTEM_FULL else s
        out.append(f"| {nm} | " + " | ".join(_fmt(S[s][k]) for k, _, _ in rb.METRICS) + f" | {S[s]['n_returned']['mean']:.1f} |")
    show = rb.HEADLINE + ["gap_cov5", "prereq_viol5", "redundancy5"]
    out += ["", f"Paired bootstrap, tuned Full minus baseline (mean diff [95% CI], Holm p over the 7 baselines, W/L/T of scenarios). For the two lower-is-better metrics a negative diff favours Full.", "",
            "| Baseline | " + " | ".join(NAMES[k] for k in show) + " |", "| --- |" + " --- |" * len(show)]
    for b in P:
        out.append(f"| {b} | " + " | ".join(f"{P[b][k]['mean_diff']:+.3f} [{P[b][k]['ci_lo']:+.3f}, {P[b][k]['ci_hi']:+.3f}] p={P[b][k]['p_holm']:.3f} ({P[b][k]['wins']}W/{P[b][k]['losses']}L/{P[b][k]['ties']}T)" for k in show) + " |")
    L = R["tuned_vs_legacy"]
    out += ["", "Tuned Full minus legacy (pre-tuning) Full, paired (raw p, no multiplicity correction): " + "; ".join(
        f"{NAMES[k]} {L[k]['mean_diff']:+.3f} [{L[k]['ci_lo']:+.3f}, {L[k]['ci_hi']:+.3f}] p={L[k]['p_value']:.3f}" for k in show + ["n_returned"]), ""]
    return out


def _findings(R: dict[str, Any], label: str) -> list[str]:
    S, P = R["systems"], R["paired_vs_full"]
    bl = list(P)
    out = []
    for k in rb.HEADLINE:
        better = [b for b in bl if rb._verdict(P[b][k], HIB[k]) == "better"]
        ind = [b for b in bl if rb._verdict(P[b][k], HIB[k]) == "indistinguishable"]
        worse = [b for b in bl if rb._verdict(P[b][k], HIB[k]) == "worse"]
        line = f"- **{NAMES[k]}** ({label}): tuned Full = {_fmt(S[rb.SYSTEM_FULL][k])}. Significantly higher than: {', '.join(better) or 'none'}. Not distinguishable from: {', '.join(ind) or 'none'}."
        if worse:
            line += f" Significantly LOWER than: {', '.join(worse)}."
        out.append(line)
    top = max(bl, key=lambda b: S[b]["ndcg10"]["mean"])
    for nm in dict.fromkeys([PRESPEC_BASELINE, top]):
        p = P[nm]["ndcg10"]
        out.append(f"- Strongest-baseline comparison ({'pre-specified' if nm == PRESPEC_BASELINE else 'strongest on this half by mean nDCG@10'}) **{nm}** ({_fmt(S[nm]['ndcg10'])}): "
                   f"Full minus it = {p['mean_diff']:+.3f} (95% CI [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}], Holm p = {p['p_holm']:.3f}) -> "
                   f"{'Full significantly better' if rb._verdict(p, True) == 'better' else 'NOT a significant difference' if rb._verdict(p, True) == 'indistinguishable' else 'Full significantly worse'}.")
    losses = [f"{NAMES[k]} vs {b} ({P[b][k]['mean_diff']:+.3f}, Holm p={P[b][k]['p_holm']:.3f})" for k, _, h in rb.METRICS for b in bl if rb._verdict(P[b][k], h) == "worse"]
    out.append(f"- **Where Full is significantly worse** ({label}): " + ("; ".join(losses) if losses else "none at the 0.05 level") + ".")
    nominal = [f"{NAMES[k]}: {b} ({S[b][k]['mean']:.3f} vs {S[rb.SYSTEM_FULL][k]['mean']:.3f})" for k, _, h in rb.METRICS for b in bl
               if ((S[b][k]["mean"] > S[rb.SYSTEM_FULL][k]["mean"]) if h else (S[b][k]["mean"] < S[rb.SYSTEM_FULL][k]["mean"])) and rb._verdict(P[b][k], h) != "worse"]
    out.append("- Baselines with a nominally better mean than Full but not significantly so: " + ("; ".join(nominal) if nominal else "none") + ".")
    out.append(f"- Prerequisite Violation Rate@5 = {_fmt(S[rb.SYSTEM_FULL]['prereq_viol5'])}; average list length = {S[rb.SYSTEM_FULL]['n_returned']['mean']:.2f}; "
               f"Explanation Completeness@5 = {S[rb.SYSTEM_FULL]['expl5']['mean']:.3f} (1.0 for Full and 0.0 for baselines by construction: baselines emit no explanation fields; a 0 violation rate for the filtered baselines is also by construction).")
    out.append("- Content-only-label sensitivity (nDCG@10): " + "; ".join(
        f"vs {b} {P[b]['ndcg10_content_only']['mean_diff']:+.3f} (p_holm={P[b]['ndcg10_content_only']['p_holm']:.3f}, {rb._verdict(P[b]['ndcg10_content_only'], True)})" for b in bl) + ".")
    return out


def run_final(n_boot: int = 10000, seed: int = 20260101, force: bool = False) -> dict[str, Any]:
    if OUT_JSON.exists() and not force:
        raise SystemExit(f"{OUT_JSON} exists: the held-out test half is evaluated ONCE. Use --force only to regenerate the same frozen run.")
    loop = json.loads(LOOP_JSON.read_text(encoding="utf-8"))
    assert DEFAULT_CONFIG.to_dict() == loop["final_config"], "DEFAULT_CONFIG differs from the config frozen by the dev loop"
    sp = load_split()
    dev, test = sp["dev"], sp["test"]
    H = Harness(with_baselines=True)
    all_ids = [i for i in H.scenarios]
    assert set(all_ids) == set(dev) | set(test)
    systems = [s for s in rb.SYSTEMS if s != rb.SYSTEM_FULL] + [rb.SYSTEM_FULL, SYS_LEGACY]
    per: dict[str, dict[str, list]] = {s: {k: [] for k in KEYS} for s in systems}
    lists_full = H.run_full(DEFAULT_CONFIG, all_ids)
    lists_old = H.run_full(LEGACY_CONFIG, all_ids)
    scored = {rb.SYSTEM_FULL: H.score_lists(lists_full), SYS_LEGACY: H.score_lists(lists_old)}
    for b, d in H.eval_baselines(all_ids).items():
        scored[b] = d
    for s in systems:
        for sid in all_ids:
            for k in KEYS:
                per[s][k].append(scored[s][sid][k])
    halves = {"test": test, "dev": dev, "all48": all_ids}
    R = {h: _summ(per, ids, all_ids, n_boot, seed, systems) for h, ids in halves.items()}
    ls = {}
    for h, ids in halves.items():
        ls[h] = {"mean_relevant_ge2": sum(sum(1 for g in H.labels[i]["labels"].values() if g >= 2) for i in ids) / len(ids)}

    L: list[str] = ["# SkillLLM_v2 - Course Recommendation Benchmark v2: tuned Full recommender on the held-out test half", ""]
    _t = R["test"]; _p = _t["paired_vs_full"][PRESPEC_BASELINE]["ndcg10"]
    _verd = {"better": "significantly better than", "worse": "significantly worse than", "indistinguishable": "NOT significantly different from"}[rb._verdict(_p, True)]
    L += [f"**Headline (held-out test half, 24 scenarios, config frozen before this run):** tuned Full nDCG@10 = {_fmt(_t['systems'][rb.SYSTEM_FULL]['ndcg10'])}; it is {_verd} the strongest "
          f"baseline ({PRESPEC_BASELINE}, {_fmt(_t['systems'][PRESPEC_BASELINE]['ndcg10'])}): diff {_p['mean_diff']:+.3f} [{_p['ci_lo']:+.3f}, {_p['ci_hi']:+.3f}], Holm p = {_p['p_holm']:.3f}. "
          f"Prerequisite violation rate {_t['systems'][rb.SYSTEM_FULL]['prereq_viol5']['mean']:.3f} (legacy {_t['systems'][SYS_LEGACY]['prereq_viol5']['mean']:.3f}), avg list length {_t['systems'][rb.SYSTEM_FULL]['n_returned']['mean']:.2f}. "
          "Everything in the 'dev' and 'all 48' sections is "
          "partly or wholly tuned-on data and is shown only for continuity with the earlier benchmark.", "",
          "- Split: 24 dev / 24 test, fixed by sha256-sorted scenario ids before any tuning (`reports/skillllm_v2/recommender_split.json`). Tuning used dev only (`recommender_loop_log.md`; 5 rounds, one change each; round 4 rejected).",
          f"- Frozen Full config = `DEFAULT_CONFIG` == `recommender_loop_log.json` final_config. Differences from the legacy config: `{json.dumps({k: v for k, v in DEFAULT_CONFIG.to_dict().items() if v != LEGACY_CONFIG.to_dict()[k]})}`.",
          "- Catalogue: 171 SYNTHETIC courses; labels `inputs/courses/relevance_labels.jsonl` (v1-rule), relevant = grade >= 2, nDCG@10 gain 2^g-1; baselines, labels and metric code unchanged. "
          f"{n_boot} bootstrap resamples, 95% percentile CIs over scenarios; Holm correction over the 7 baseline comparisons per metric. Dense backend: {H.dense_backend}.",
          "- No gold/relevance label is an input of the recommender (the tuned system reads user profile, job requirements, catalogue fields and the graph only); no scenario id is special-cased.", ""]
    L += ["## 1. Held-out TEST half (primary result)", ""] + _tables("Test half", R["test"], systems[:-1] + [SYS_LEGACY])
    L += ["#### Findings (test half)", ""] + _findings(R["test"], "test half") + [""]
    L += ["## 2. DEV half (the half that was tuned on - optimistic by construction)", ""] + _tables("Dev half", R["dev"], systems[:-1] + [SYS_LEGACY])
    L += ["## 3. All 48 scenarios (continuity with course_recommendation_benchmark.md; PARTLY DEV-TUNED, do not read as a held-out result)", ""] + _tables("All 48", R["all48"], systems[:-1] + [SYS_LEGACY])
    L += ["#### Findings (all 48, partly dev-tuned)", ""] + _findings(R["all48"], "all 48, partly dev-tuned") + [""]
    L += ["## Caveats", "",
          "- n = 24 test scenarios is small; CIs are wide and scenarios share a catalogue/archetypes, so they are not independent. Bootstrap ignores label uncertainty.",
          "- Synthetic catalogue and scenarios; labels come from a deterministic rule over the same catalogue fields the skill-aware systems read, and the rule caps courses that need a bridge at grade 1 (this is what makes strict, startable-now prerequisite handling pay off; see the content-only sensitivity rows). Human validation of the labels (human_review_sample.csv) has NOT been done.",
          "- The graph-free baseline 'Gap-Overlap Greedy + prereq filter' implements the same chained prerequisite rule as the violation metric, so its 0.000 violation rate is by construction; Full now reaches 0.000 through the same catalogue-field rule (not through the graph).",
          "- Explanation Completeness is a format check (baselines emit no explanation fields), not a quality measure. Round 2 adds 'related_text' fill courses whose explanation states they are text-related only.",
          "- Learner-level inference (round 4) did not help on dev and is disabled; the benchmark profiles carry no experience level, so level suitability is effectively constant here and untested.",
          "- Tuning touched 24 scenarios with ~100 configurations in total; dev gains of a few thousandths are inside noise.", "",
          "## Reproduction", "", "```", "PYTHONPATH=src python -m skillllm_v2.recommend.loop_eval        # (library) split + harness",
          "PYTHONPATH=src python -m skillllm_v2.recommend.loop_rounds      # dev-only loop -> recommender_loop_log.{md,json}",
          "PYTHONPATH=src python -m skillllm_v2.recommend.recommender_final  # this report (test half, run once)", "```", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "config": {"full": DEFAULT_CONFIG.to_dict(), "legacy": LEGACY_CONFIG.to_dict(), "n_boot": n_boot, "seed": seed, "dense_backend": H.dense_backend},
        "split": {"dev": dev, "test": test}, "results": R, "label_stats": ls,
        "per_scenario": {s: {k: per[s][k] for k in KEYS} for s in systems}, "scenario_ids": all_ids,
        "rankings": {"full": {i: [r["course_id"] for r in lists_full[i]] for i in all_ids}, "legacy": {i: [r["course_id"] for r in lists_old[i]] for i in all_ids}},
        "recommendation_roles": {i: [r.get("role") for r in lists_full[i]] for i in all_ids},
    }, indent=2), encoding="utf-8")
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    R = run_final(n_boot=a.n_boot, force=a.force)
    t = R["test"]["systems"]
    for s, v in t.items():
        print(f"{s:<42} ndcg10 {v['ndcg10']['mean']:.3f} p5 {v['p5']['mean']:.3f} viol {v['prereq_viol5']['mean']:.3f} n {v['n_returned']['mean']:.2f}")


if __name__ == "__main__":
    main()
