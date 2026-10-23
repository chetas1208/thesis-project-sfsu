"""Independent graded-relevance labels for the course-recommendation benchmark.

    PYTHONPATH=src python -m skillllm_v2.eval.build_relevance_labels \
        [--catalog inputs/courses/course_catalog_large.jsonl]

The builder is fully reproducible from the repository alone: every scenario's job text is a plain constant
in benchmark_scenarios_spec.py (no private corpus is read).

Writes
  inputs/courses/benchmark_scenarios.jsonl      scenario inputs (user profile + job) AND the gold needs
  inputs/courses/relevance_labels.jsonl         one line per scenario: {course_id: grade 0..3} (non-zero only)
  reports/skillllm_v2/human_review_sample.csv   stratified sample for manual grading

WHY THESE LABELS ARE (MORE) INDEPENDENT
---------------------------------------
* The gold need set of a scenario is AUTHORED BY HAND (benchmark_scenarios_spec.py) from the job text and
  the user's background.  It is NOT computed from GapEngine / CourseRecommendationPipeline output, nor
  from any compared system's ranking.  It also contains "implicit" needs that are not given to any
  system as structured input.
* Labels are computed by a small deterministic rule procedure over (gold needs, user skills, catalogue
  fields).  No retrieval model, embedding or ranker is involved.

THE PROCEDURE (per scenario s, per course c)
--------------------------------------------
  N   = {skill: w}  gold needs, w = 3 (required), 2 (implicit), 1 (preferred)
  K   = user's proficient skills, W = user's weak skills (beginner)
  N'  = N minus K          (weak skills stay open: the user still has to improve them)
  T   = c.teaches_skills,  P = c.prerequisites
  useful = T ∩ N';  m = sum of w over useful;  focus f = |useful| / |T|

  content grade g0:
      0  if m == 0
      3  if (m >= 3 and f >= 0.5) or (m >= 5 and f >= 0.4)
      2  if m >= 3, or (m >= 2 and f >= 0.5)
      1  otherwise (m > 0)
  prerequisite feasibility (applied to g0):
      R0 = K ∪ W (weak skills are treated as sufficient prerequisite exposure, a lenient choice)
      R1 = R0 ∪ skills taught by catalogue courses whose prerequisites ⊆ R0    (one "bridge" course)
      P ⊆ R0           -> grade = g0                         (can start now)
      P ⊆ R1           -> grade = min(g0, 1)                 (useful, but only after a bridge course)
      otherwise        -> grade = 0                          (not reachable within one bridge)
  enabler rule: a course that would otherwise be 0 but teaches a skill that is a missing prerequisite of
      some content-relevant (g0 >= 2) course, and is itself directly startable, gets grade 1.
  The sensitivity labels ("content_only") are simply g0 without the feasibility rule/enabler rule.

KNOWN LIMITS (also stated in the benchmark report)
  * The rule uses the same catalogue fields (teaches_skills, prerequisites) the systems read, so it
    rewards systems that reason over those fields (Skill-Overlap, Gap-weighted, Graph) more than a
    purely text-based system.  This is a property of any skill-set-based gold standard.
  * Thresholds (3/2/1, focus 0.5) and the "weak counts as prerequisite-satisfying" convention are
    author choices, not validated against learners.  human_review_sample.csv exists so the author can
    grade a stratified sample by hand and report agreement (Cohen's kappa / Spearman) before trusting
    the numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from skillllm_v2.course.ingest_catalog import CourseCatalog
from skillllm_v2.eval.benchmark_scenarios_spec import S, SCENARIOS

ROOT = Path(__file__).resolve().parents[3]
LABEL_VERSION = "v1-rule"
W_REQ, W_IMPL, W_PREF = 3, 2, 1


def gold_needs(sc: S) -> dict[str, int]:
    needs: dict[str, int] = {}
    for s in [x.strip() for x in sc.pref.split(",") if x.strip()]:
        needs[s] = max(needs.get(s, 0), W_PREF)
    for s in [x.strip() for x in sc.implicit.split(",") if x.strip()]:
        needs[s] = max(needs.get(s, 0), W_IMPL)
    for s in [x.strip() for x in sc.req.split(",") if x.strip()]:
        needs[s] = max(needs.get(s, 0), W_REQ)
    return needs


def content_grade(taught: list[str], open_needs: dict[str, int]) -> tuple[int, float, float]:
    useful = [s for s in taught if s in open_needs]
    m = float(sum(open_needs[s] for s in useful))
    f = len(useful) / max(1, len(taught))
    if m == 0:
        return 0, m, f
    if (m >= 3 and f >= 0.5) or (m >= 5 and f >= 0.4):
        return 3, m, f
    if m >= 3 or (m >= 2 and f >= 0.5):
        return 2, m, f
    return 1, m, f


def label_scenario(sc: S, catalog: CourseCatalog) -> dict:
    needs = gold_needs(sc)
    K = set(sc.known_list())
    W = set(sc.weak_list())
    open_needs = {s: w for s, w in needs.items() if s not in K}
    R0 = K | W
    courses = catalog.list_courses()
    direct = {c.course_id: set(c.prerequisites) <= R0 for c in courses}
    R1 = set(R0)
    for c in courses:
        if direct[c.course_id]:
            R1 |= set(c.teaches_skills)

    g0: dict[str, int] = {}
    info: dict[str, dict] = {}
    for c in courses:
        g, m, f = content_grade(c.teaches_skills, open_needs)
        g0[c.course_id] = g
        info[c.course_id] = {"m": m, "f": round(f, 3)}

    final: dict[str, int] = {}
    reason: dict[str, str] = {}
    missing_by_course: dict[str, list[str]] = {}
    for c in courses:
        P = set(c.prerequisites)
        g = g0[c.course_id]
        miss = sorted(P - R0)
        missing_by_course[c.course_id] = miss
        if g == 0:
            final[c.course_id] = 0
            reason[c.course_id] = "teaches no open need"
        elif not miss:
            final[c.course_id] = g
            reason[c.course_id] = f"covers open need mass {info[c.course_id]['m']:.0f}, focus {info[c.course_id]['f']}; startable now"
        elif P <= R1:
            final[c.course_id] = min(g, 1)
            reason[c.course_id] = f"relevant (content grade {g}) but needs bridge for prerequisite(s) {miss}"
        else:
            final[c.course_id] = 0
            reason[c.course_id] = f"relevant (content grade {g}) but prerequisite(s) {miss} not reachable within one bridge"

    # enabler rule
    enablers: dict[str, list[str]] = {}
    for c in courses:
        if g0[c.course_id] >= 2 and missing_by_course[c.course_id] and set(c.prerequisites) <= R1:
            for s in missing_by_course[c.course_id]:
                enablers.setdefault(s, []).append(c.course_id)
    for c in courses:
        if final[c.course_id] == 0 and direct[c.course_id]:
            hit = [s for s in c.teaches_skills if s in enablers]
            if hit:
                final[c.course_id] = 1
                reason[c.course_id] = f"enabler: teaches prerequisite {hit} of relevant course(s) {sorted({x for s in hit for x in enablers[s]})[:3]}"

    return {
        "needs": needs,
        "open_needs": open_needs,
        "final": final,
        "content_only": g0,
        "reason": reason,
        "missing": missing_by_course,
    }


def build(catalog_path: Path, out_dir: Path, report_dir: Path, seed: int = 13) -> None:
    catalog = CourseCatalog.load_from_file(catalog_path)
    vocab = {s for c in catalog for s in c.teaches_skills}
    scen_rows, label_rows, review_pool = [], [], []
    problems = []
    for sc in SCENARIOS:
        if sc.source == "real_posting_excerpt":
            raise SystemExit(f"{sc.sid}: real-posting text must not be used in the benchmark")
        text = sc.text
        posting_meta = sc.provenance
        lab = label_scenario(sc, catalog)
        needs = lab["needs"]
        for s in list(needs) + sc.known_list() + sc.weak_list():
            if s not in vocab:
                problems.append((sc.sid, s))
        req = [x.strip() for x in sc.req.split(",") if x.strip()]
        pref = [x.strip() for x in sc.pref.split(",") if x.strip()]
        skills = []
        for s in sc.known_list():
            skills.append({"surface_form": s, "canonical_skill_name": s, "confidence": 1.0})
        for s in sc.weak_list():
            skills = [x for x in skills if x["surface_form"] != s]
            skills.append({"surface_form": s, "canonical_skill_name": s, "confidence": 1.0, "mastery_level": "beginner"})
        scen_rows.append({
            "scenario_id": sc.sid,
            "title": f"{sc.background} -> {sc.job_title}",
            "background": sc.background,
            "experience_level": sc.level,
            "source": sc.source,
            "posting_provenance": posting_meta,
            "user_profile": {"user_id": f"user-{sc.sid}", "skills": skills, "target_roles": [sc.job_title]},
            "target_job": {"job_id": f"job-{sc.sid}", "title": sc.job_title, "text": text,
                           "required_skills": req, "preferred_skills": pref},
            "gold": {"needs": needs, "implicit_not_given_to_systems": [x.strip() for x in sc.implicit.split(",") if x.strip()],
                     "known": sc.known_list(), "weak": sc.weak_list()},
        })
        nz = {cid: g for cid, g in lab["final"].items() if g > 0}
        nz_c = {cid: g for cid, g in lab["content_only"].items() if g > 0}
        label_rows.append({
            "scenario_id": sc.sid,
            "label_version": LABEL_VERSION,
            "n_courses": len(catalog),
            "labels": nz,
            "labels_content_only": nz_c,
            "grade_counts": dict(Counter(lab["final"].values())),
        })
        for c in catalog:
            review_pool.append((sc, lab, c))

    if problems:
        raise SystemExit(f"skills not in catalogue vocabulary: {problems}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_scenarios.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in scen_rows) + "\n", encoding="utf-8")
    (out_dir / "relevance_labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in label_rows) + "\n", encoding="utf-8")

    # --- human review sample: stratified by auto grade, independent of any system's output ---
    rng = random.Random(seed)
    by_grade: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    hard_neg = []
    for sc, lab, c in review_pool:
        g = lab["final"][c.course_id]
        by_grade[g].append((sc, lab, c))
        if g == 0 and lab["content_only"][c.course_id] >= 1:
            hard_neg.append((sc, lab, c))  # relevant by content, zeroed by the prerequisite rule
    picks = []
    for g in (3, 2, 1):
        picks += [(g, x) for x in rng.sample(by_grade[g], min(15, len(by_grade[g])))]
    picks += [(0, x) for x in rng.sample(hard_neg, min(8, len(hard_neg)))]
    picks += [(0, x) for x in rng.sample(by_grade[0], 7)]
    rng.shuffle(picks)
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "human_review_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "scenario_id", "user_background", "user_known_skills", "user_weak_skills", "target_job_title",
                    "job_text_excerpt", "needed_skills(weight)", "course_id", "course_title", "course_level", "course_hours",
                    "course_teaches", "course_prerequisites", "auto_grade_0to3", "auto_stratum", "auto_reason",
                    "human_grade_0to3", "human_notes"])
        for n, (g, (sc, lab, c)) in enumerate(picks, start=1):
            stratum = f"grade{g}" + ("_hardneg" if (g == 0 and (sc, lab, c) in hard_neg) else "")
            w.writerow([n, sc.sid, sc.background, "; ".join(sc.known_list()), "; ".join(sc.weak_list()), sc.job_title,
                        next(r["target_job"]["text"] for r in scen_rows if r["scenario_id"] == sc.sid)[:300],
                        "; ".join(f"{k}({v})" for k, v in lab["needs"].items()),
                        c.course_id, c.title, c.level, c.duration_hours, "; ".join(c.teaches_skills), "; ".join(c.prerequisites),
                        g, stratum, lab["reason"][c.course_id], "", ""])

    tot = Counter()
    for r in label_rows:
        for g, n in r["grade_counts"].items():
            tot[int(g)] += n
    nrel = [sum(1 for g in r["labels"].values() if g >= 2) for r in label_rows]
    print(f"scenarios={len(scen_rows)} courses={len(catalog)} grade totals={dict(sorted(tot.items()))}")
    print(f"relevant (grade>=2) per scenario: min={min(nrel)} mean={sum(nrel)/len(nrel):.1f} max={max(nrel)}")
    print("scenarios with zero grade>=2:", [r["scenario_id"] for r, n in zip(label_rows, nrel) if n == 0])
    print(f"human review rows: {len(picks)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(ROOT / "inputs/courses/course_catalog_large.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "inputs/courses"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports/skillllm_v2"))
    args = ap.parse_args()
    build(Path(args.catalog), Path(args.out_dir), Path(args.report_dir))


if __name__ == "__main__":
    main()
