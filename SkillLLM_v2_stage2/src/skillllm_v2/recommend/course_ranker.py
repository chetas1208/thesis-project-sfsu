from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from skillllm_v2.course.ingest_catalog import CourseRecord
from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.recommend.config import LEGACY_CONFIG, RecommenderConfig
from skillllm_v2.recommend.gap_vector import GapVector
from skillllm_v2.schema import normalize_text

LEVEL_ORDINAL = {
    "beginner": 1,
    "introductory": 1,
    "intermediate": 2,
    "advanced": 3,
    "mastery": 3,
    "junior": 1,
    "mid": 2,
    "senior": 3,
}


@dataclass(slots=True)
class ScoredCourse:
    course_id: str
    title: str
    rank: int
    score: float
    covered_missing_skills: list[str]
    covered_canonical_ids: list[str]
    prerequisite_status: dict[str, Any]
    score_breakdown: dict[str, float]
    course_record: CourseRecord
    metadata: dict[str, Any] = field(default_factory=dict)
    role: str = "primary"  # "primary" (teaches gap skills) | "bridge" (teaches a prerequisite of a blocked course)

    def to_dict(self) -> dict[str, Any]:
        return {
            "course_id": self.course_id,
            "title": self.title,
            "rank": self.rank,
            "score": round(self.score, 4),
            "role": self.role,
            "covered_missing_skills": self.covered_missing_skills,
            "covered_canonical_ids": self.covered_canonical_ids,
            "prerequisite_status": self.prerequisite_status,
            "score_breakdown": {k: round(v, 4) for k, v in self.score_breakdown.items()},
        }


def _raw_taught(course: CourseRecord) -> set[str]:
    return {normalize_text(s) for s in course.teaches_skills if normalize_text(s)}


def _raw_prereqs(course: CourseRecord) -> set[str]:
    return {normalize_text(p) for p in course.prerequisites if normalize_text(p)}


class CourseRanker:
    """Interpretable rule-weighted course ranker implementing weighted gap coverage and prerequisite reasoning.

    The behaviour is controlled by a :class:`RecommenderConfig`.  With the default (legacy) config it is the
    original soft-prerequisite greedy ranker.  In the strict modes a course is only eligible when every
    prerequisite is already known by the learner or taught by a course selected earlier in the list (so the
    returned list is a valid learning order), and in ``strict_bridge`` mode a blocked course can be unlocked
    by recommending a startable *bridge* course that teaches the missing prerequisite.
    """

    LEVEL_ORDINAL = LEVEL_ORDINAL

    def __init__(
        self,
        w_gap: float | None = None,
        w_semantic: float | None = None,
        w_prereq: float | None = None,
        w_level: float | None = None,
        w_goal: float | None = None,
        w_redundancy: float | None = None,
        w_workload: float | None = None,
        filter_strict_prereq_violations: bool = False,
        config: RecommenderConfig | None = None,
    ) -> None:
        cfg = config or LEGACY_CONFIG
        overrides = {
            k: v
            for k, v in dict(
                w_gap=w_gap, w_semantic=w_semantic, w_prereq=w_prereq, w_level=w_level,
                w_goal=w_goal, w_redundancy=w_redundancy, w_workload=w_workload,
            ).items()
            if v is not None
        }
        self.config = cfg.with_(**overrides) if overrides else cfg
        self.filter_strict_prereq_violations = filter_strict_prereq_violations
        # kept as attributes for backwards compatibility
        for name in ("w_gap", "w_semantic", "w_prereq", "w_level", "w_goal", "w_redundancy", "w_workload"):
            setattr(self, name, getattr(self.config, name))

    # ------------------------------------------------------------------ learner level
    def learner_level(self, user_profile: dict[str, Any], catalog_courses: Sequence[CourseRecord] | None = None) -> tuple[int, str]:
        """Return (ordinal level 1..3, provenance).  Uses the declared level when given; otherwise, if enabled in the
        config, infers it from the learner's own skills (never from gold labels)."""
        declared = user_profile.get("experience_level")
        if declared:
            return LEVEL_ORDINAL.get(str(declared).lower(), 2), "declared"
        cfg = self.config
        if cfg.infer_user_level and cfg.level_method == "skill_count":
            n = len(user_profile.get("skills", []))
            if n <= cfg.level_junior_skills:
                return 1, "inferred_from_skill_count"
            if n >= cfg.level_senior_skills:
                return 3, "inferred_from_skill_count"
            return 2, "inferred_from_skill_count"
        if cfg.infer_user_level and cfg.level_method == "catalog_depth" and catalog_courses:
            # depth of what the learner already knows: for each known skill, the highest catalogue level at which it is taught
            top_level: dict[str, int] = {}
            for c in catalog_courses:
                o = LEVEL_ORDINAL.get(c.level.lower(), 2)
                for sk in c.teaches_skills:
                    k = normalize_text(sk)
                    top_level[k] = max(top_level.get(k, 0), o)
            vals = [top_level[k] for k in self._user_skill_ids(user_profile) if k in top_level]
            if vals:
                depth = sum(vals) / len(vals)
                if depth <= cfg.level_depth_junior:
                    return 1, "inferred_from_skill_depth"
                if depth >= cfg.level_depth_senior:
                    return 3, "inferred_from_skill_depth"
                return 2, "inferred_from_skill_depth"
        return 2, "default"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _user_skill_ids(user_profile: dict[str, Any]) -> set[str]:
        return {
            normalize_text(s.get("canonical_skill_id") or s.get("surface_form") or s.get("name") or "")
            if isinstance(s, dict)
            else normalize_text(s)
            for s in user_profile.get("skills", [])
        }

    @staticmethod
    def _teaches_ids(course: CourseRecord, graph: CourseGraphBuilder | None) -> set[str]:
        ids = {normalize_text(s) for s in course.teaches_skills + course.canonical_skills}
        if graph and graph.graph.has_node(course.course_id):
            for _, sid, data in graph.graph.out_edges(course.course_id, data=True):
                if data.get("type") == "TEACHES":
                    ids.add(sid)
        return ids

    def _prereq_eval(
        self,
        course: CourseRecord,
        known: set[str],
        provided_by: dict[str, str],
        graph: CourseGraphBuilder | None,
        user_skills: set[str],
    ) -> dict[str, Any]:
        if self.config.prereq_mode == "soft":
            if graph:
                return graph.evaluate_prerequisites(course.course_id, user_skills)
            course_prereqs = {normalize_text(p) for p in course.prerequisites + course.canonical_prerequisites}
            satisfied = sorted(course_prereqs & user_skills)
            missing = sorted(course_prereqs - user_skills)
            compat = len(satisfied) / max(1, len(course_prereqs)) if course_prereqs else 1.0
            return {
                "course_id": course.course_id,
                "is_satisfied": len(missing) == 0,
                "compatibility_score": compat,
                "satisfied_prerequisites": satisfied,
                "missing_prerequisites": missing,
                "violation_count": len(missing),
            }
        # strict modes: catalogue prerequisites vs the learner's skills plus skills taught by earlier picks
        prereqs = sorted(_raw_prereqs(course))
        satisfied = []
        missing = []
        for p in prereqs:
            if p in known:
                src = provided_by.get(p, "learner_profile") if p not in user_skills else "learner_profile"
                satisfied.append({"skill_id": p, "canonical_skill_name": p, "satisfied_by": src})
            else:
                missing.append({"skill_id": p, "canonical_skill_name": p})
        if self.config.compat_basis == "learner":
            # prefer courses the learner can start NOW: compatibility counts only the learner's own skills
            compat = sum(1 for p in prereqs if p in user_skills) / max(1, len(prereqs)) if prereqs else 1.0
        else:
            compat = len(satisfied) / max(1, len(prereqs)) if prereqs else 1.0
        return {
            "course_id": course.course_id,
            "is_satisfied": not missing,
            "compatibility_score": compat,
            "satisfied_prerequisites": satisfied,
            "missing_prerequisites": missing,
            "violation_count": len(missing),
        }

    # ------------------------------------------------------------------ main
    def rank(
        self,
        candidates: Sequence[CourseRecord],
        user_profile: dict[str, Any],
        target_job: dict[str, Any],
        gap_vector: GapVector,
        graph: CourseGraphBuilder | None = None,
        top_k: int = 5,
        catalog_courses: Sequence[CourseRecord] | None = None,
    ) -> list[ScoredCourse]:
        cfg = self.config
        strict = cfg.prereq_mode in ("strict_drop", "strict_bridge")
        bridging = cfg.prereq_mode == "strict_bridge"
        user_skills = self._user_skill_ids(user_profile)
        user_level_val, level_source = self.learner_level(user_profile, catalog_courses)
        target_roles = [r.lower() for r in user_profile.get("target_roles", [])]
        job_title = str(target_job.get("title") or target_job.get("job_id") or "").lower()
        max_hours = float(user_profile.get("preferences", {}).get("max_course_hours", 60.0))
        gap_ids = set(gap_vector.vector.keys())
        gap_name_map = {g.canonical_skill_id: g.skill_name for g in gap_vector.gaps}
        total_mass = max(1e-6, gap_vector.total_gap_mass)
        job_words = set(normalize_text(f"{job_title} {' '.join(g.skill_name for g in gap_vector.gaps)}").split())

        remaining = list(candidates)
        remaining_ids = {c.course_id for c in remaining}
        all_courses = list(catalog_courses) if catalog_courses is not None else list(candidates)
        selected: list[ScoredCourse] = []
        covered_gap_ids: set[str] = set()
        known: set[str] = set(user_skills)  # learner skills + skills taught by earlier picks (strict modes)
        provided_by: dict[str, str] = {}
        taught_cache: dict[str, set[str]] = {}

        def teaches(course: CourseRecord) -> set[str]:
            if course.course_id not in taught_cache:
                taught_cache[course.course_id] = self._teaches_ids(course, graph)
            return taught_cache[course.course_id]

        def level_suitability(course: CourseRecord) -> float:
            diff = LEVEL_ORDINAL.get(course.level.lower(), 2) - user_level_val
            if diff in (0, 1):
                return 1.0  # perfect fit or slightly challenging
            if diff == -1:
                return 0.8  # slight review
            if diff >= 2:
                return 0.4  # too advanced for current level
            return 0.6  # too basic

        def semantic_relevance(course: CourseRecord) -> float:
            if course.learning_outcomes:
                outcome_words = set(normalize_text(" ".join(course.learning_outcomes)).split())
                if outcome_words and job_words:
                    return min(1.0, len(outcome_words & job_words) / max(5, len(job_words) * 0.3))
            return 0.5

        def goal_alignment(course: CourseRecord) -> float:
            if target_roles:
                title_norm = course.title.lower()
                if any(role in title_norm or title_norm in role for role in target_roles):
                    return 1.0
                if job_title and (job_title in title_norm or title_norm in job_title):
                    return 0.9
            return 0.7

        def score_course(course: CourseRecord, prereq_eval: dict[str, Any], unlock_gain: float = 0.0, require_gap: bool = True):
            """Return (score, breakdown, matching_gap_ids) or None when the course is not eligible."""
            taught_ids = teaches(course)
            matching = taught_ids & gap_ids
            if require_gap and not matching:
                return None
            newly = matching - covered_gap_ids
            redundant = matching & covered_gap_ids
            gap_coverage = sum(gap_vector.get_gap_weight(s) for s in newly) / total_mass
            if cfg.redundancy_mode == "mmr":
                redundancy = min(1.0, sum(gap_vector.get_gap_weight(s) for s in redundant) / total_mass)
            else:
                redundancy = len(redundant) / max(1, len(taught_ids))
            sem = semantic_relevance(course)
            lvl = level_suitability(course)
            goal = goal_alignment(course)
            workload = max(0.0, (course.duration_hours - max_hours) / 100.0)
            compat = prereq_eval["compatibility_score"]
            raw = (
                cfg.w_gap * gap_coverage
                + cfg.w_semantic * sem
                + cfg.w_prereq * compat
                + cfg.w_level * lvl
                + cfg.w_goal * goal
                + cfg.w_unlock * unlock_gain
                - cfg.w_redundancy * redundancy
                - cfg.w_workload * workload
            )
            score = max(0.0, min(1.0, raw))
            breakdown = {
                "gap_coverage": gap_coverage,
                "semantic_relevance": sem,
                "prerequisite_compatibility": compat,
                "level_suitability": lvl,
                "goal_alignment": goal,
                "unlock_gain": unlock_gain,
                "redundancy_penalty": redundancy,
                "workload_penalty": workload,
                "final_score": score,
            }
            return score, breakdown, matching

        def find_bridge(missing: str, course_pool: Sequence[CourseRecord], exclude: set[str]) -> CourseRecord | None:
            """Best startable course (all its prerequisites known now) that teaches the missing prerequisite."""
            best, best_key = None, None
            for b in course_pool:
                if b.course_id in exclude:
                    continue
                if missing not in _raw_taught(b) or not _raw_prereqs(b) <= known:
                    continue
                own_new = sum(gap_vector.get_gap_weight(s) for s in (teaches(b) & gap_ids) - covered_gap_ids)
                key = (-own_new, LEVEL_ORDINAL.get(b.level.lower(), 2), b.duration_hours, -(b.rating or 0.0), b.course_id)
                if best_key is None or key < best_key:
                    best, best_key = b, key
            return best

        selected_ids: set[str] = set()
        while len(selected) < top_k:
            best: dict[str, Any] | None = None
            blocked: list[tuple[CourseRecord, dict[str, Any], float]] = []

            for course in remaining:
                if course.course_id in selected_ids:
                    continue
                pe = self._prereq_eval(course, known, provided_by, graph, user_skills)
                if self.filter_strict_prereq_violations and not strict and not pe["is_satisfied"]:
                    continue
                if strict and not pe["is_satisfied"]:
                    if bridging:
                        res_blocked = score_course(course, {**pe, "compatibility_score": 1.0})
                        if res_blocked is not None:
                            blocked.append((course, pe, res_blocked[1]["gap_coverage"]))
                    continue
                res = score_course(course, pe)
                if res is None:
                    continue
                score, breakdown, matching = res
                if best is None or score > best["score"]:
                    best = {"course": course, "score": score, "breakdown": breakdown, "matching": matching, "pe": pe,
                            "role": "primary", "unlocks": []}

            if bridging and blocked:
                # candidate bridges: startable courses teaching a prerequisite that blocks an otherwise useful course
                bridge_map: dict[str, dict[str, Any]] = {}
                pool_for_bridges = all_courses
                for x_course, x_pe, x_gap in blocked:
                    miss_ids = [m["skill_id"] for m in x_pe["missing_prerequisites"]]
                    for m in miss_ids:
                        b = find_bridge(m, pool_for_bridges, selected_ids | {x_course.course_id})
                        if b is None:
                            continue
                        entry = bridge_map.setdefault(b.course_id, {"course": b, "unlocks": []})
                        entry["unlocks"].append({
                            "course_id": x_course.course_id,
                            "title": x_course.title,
                            "prerequisite_skill": m,
                            "share": 1.0 / max(1, len(miss_ids)),
                            "gap_coverage": x_gap,
                            "gap_skill_ids": sorted(teaches(x_course) & gap_ids),
                        })
                for entry in bridge_map.values():
                    b = entry["course"]
                    pe_b = self._prereq_eval(b, known, provided_by, graph, user_skills)
                    unlock_gain = min(1.0, cfg.bridge_discount * sum(u["share"] * u["gap_coverage"] for u in entry["unlocks"]))
                    res = score_course(b, pe_b, unlock_gain=unlock_gain, require_gap=False)
                    if res is None:
                        continue
                    score, breakdown, matching = res
                    if best is None or score > best["score"]:
                        best = {"course": b, "score": score, "breakdown": breakdown, "matching": matching, "pe": pe_b,
                                "role": "bridge", "unlocks": entry["unlocks"]}

            if best is None or best["score"] <= max(0.0, cfg.min_score):
                break

            course = best["course"]
            covered_ids = sorted(best["matching"])
            covered_names = [gap_name_map.get(sid, sid) for sid in covered_ids]
            meta: dict[str, Any] = {"learner_level_source": level_source}
            role = "primary" if covered_ids else "bridge"  # a pure bridge covers no gap skill itself
            if best["unlocks"]:
                meta["unlocks"] = best["unlocks"]
            if role == "bridge":
                for g in sorted({g for u in best["unlocks"] for g in u["gap_skill_ids"]}):
                    name = gap_name_map.get(g, g)
                    if name not in covered_names:
                        covered_names.append(name)
            selected.append(ScoredCourse(
                course_id=course.course_id,
                title=course.title,
                rank=len(selected) + 1,
                score=best["score"],
                covered_missing_skills=covered_names,
                covered_canonical_ids=covered_ids,
                prerequisite_status=best["pe"],
                score_breakdown=best["breakdown"],
                course_record=course,
                metadata=meta,
                role=role,
            ))
            selected_ids.add(course.course_id)
            covered_gap_ids.update(covered_ids)
            if course.course_id in remaining_ids:
                remaining = [c for c in remaining if c.course_id != course.course_id]
            for s in _raw_taught(course):
                known.add(s)
                provided_by.setdefault(s, f"earlier_course:{course.course_id}")

        # ---- tier 2: related-text fill (only after every gap-covering / bridge candidate is exhausted) ------------------
        if cfg.fill_related and len(selected) < top_k:
            gap_norm_names = {gid: normalize_text(gap_name_map.get(gid, gid)) for gid in gap_ids}
            while len(selected) < top_k:
                best_rel = None
                for course in remaining:
                    if course.course_id in selected_ids:
                        continue
                    pe = self._prereq_eval(course, known, provided_by, graph, user_skills)
                    if strict and not pe["is_satisfied"]:
                        continue
                    text = " " + normalize_text(" ".join([course.title, course.description] + list(course.learning_outcomes))) + " "
                    taught_tokens = {t for sk in course.teaches_skills for t in normalize_text(sk).split() if len(t) >= 4}
                    mentioned = sorted(
                        gid for gid, nm in gap_norm_names.items()
                        if nm
                        and (f" {nm} " in text or (cfg.related_match == "token" and any(t in taught_tokens for t in nm.split() if len(t) >= 4)))
                    )
                    if not mentioned:
                        continue
                    res = score_course(course, pe, require_gap=False)
                    if res is None:
                        continue
                    score, breakdown, _ = res
                    score *= cfg.related_discount
                    breakdown = {**breakdown, "final_score": score}
                    if score > max(0.0, cfg.min_score) and (best_rel is None or score > best_rel["score"]):
                        best_rel = {"course": course, "score": score, "breakdown": breakdown, "pe": pe, "mentioned": mentioned}
                if best_rel is None:
                    break
                course = best_rel["course"]
                selected.append(ScoredCourse(
                    course_id=course.course_id,
                    title=course.title,
                    rank=len(selected) + 1,
                    score=best_rel["score"],
                    covered_missing_skills=[gap_name_map.get(g, g) for g in best_rel["mentioned"]],
                    covered_canonical_ids=best_rel["mentioned"],
                    prerequisite_status=best_rel["pe"],
                    score_breakdown=best_rel["breakdown"],
                    course_record=course,
                    metadata={"learner_level_source": level_source, "evidence_type": "text_mention_only"},
                    role="related_text",
                ))
                selected_ids.add(course.course_id)
                for s_ in _raw_taught(course):
                    known.add(s_)
                    provided_by.setdefault(s_, f"earlier_course:{course.course_id}")

        # annotate bridge/unlock links with whether (and where) the unlocked course made it into the final list
        rank_of = {sc.course_id: sc.rank for sc in selected}
        for sc in selected:
            for u in sc.metadata.get("unlocks", []):
                u["in_list"] = u["course_id"] in rank_of and rank_of[u["course_id"]] > sc.rank
                u["unlocked_course_rank"] = rank_of.get(u["course_id"])
        return selected
