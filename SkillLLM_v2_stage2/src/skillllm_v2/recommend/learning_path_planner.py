from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from skillllm_v2.course.ingest_catalog import CourseRecord
from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.recommend.course_ranker import ScoredCourse
from skillllm_v2.recommend.gap_vector import GapVector
from skillllm_v2.schema import normalize_text


@dataclass(slots=True)
class LearningStage:
    stage_number: int
    stage_name: str
    courses: list[dict[str, Any]]
    skills_acquired: list[str]
    estimated_hours: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LearningPath:
    user_id: str
    target_role: str
    total_courses: int
    total_duration_hours: float
    stages: list[LearningStage]
    ordered_courses: list[dict[str, Any]]
    cumulative_gap_coverage: list[dict[str, Any]]
    unresolved_prerequisites: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "target_role": self.target_role,
            "total_courses": self.total_courses,
            "total_duration_hours": round(self.total_duration_hours, 1),
            "stages": [s.to_dict() for s in self.stages],
            "ordered_courses": self.ordered_courses,
            "cumulative_gap_coverage": self.cumulative_gap_coverage,
            "unresolved_prerequisites": self.unresolved_prerequisites,
        }


class LearningPathPlanner:
    """Plans a prerequisite-consistent multi-course learning path to close target skill gaps."""

    LEVEL_ORDER = {"beginner": 1, "introductory": 1, "intermediate": 2, "advanced": 3, "mastery": 3}

    def plan(
        self,
        recommended_courses: Sequence[ScoredCourse],
        user_profile: dict[str, Any],
        target_job: dict[str, Any],
        gap_vector: GapVector,
        graph: CourseGraphBuilder | None = None,
    ) -> LearningPath:
        user_id = str(user_profile.get("user_id") or "user")
        job_title = str(target_job.get("title") or target_job.get("job_id") or "Target Role")

        if not recommended_courses:
            return LearningPath(
                user_id=user_id,
                target_role=job_title,
                total_courses=0,
                total_duration_hours=0.0,
                stages=[],
                ordered_courses=[],
                cumulative_gap_coverage=[],
                unresolved_prerequisites=[],
            )

        # Collect user existing skills
        user_skills: set[str] = {
            normalize_text(s.get("canonical_skill_id") or s.get("surface_form") or s.get("name") or "")
            if isinstance(s, dict)
            else normalize_text(s)
            for s in user_profile.get("skills", [])
        }

        # Build map of course_id -> ScoredCourse
        course_map = {rc.course_id: rc for rc in recommended_courses}

        # Determine skills taught by each recommended course
        course_teaches: dict[str, set[str]] = {}
        course_prereqs: dict[str, set[str]] = {}
        for cid, rc in course_map.items():
            crec = rc.course_record
            t_skills = {normalize_text(s) for s in crec.teaches_skills + crec.canonical_skills}
            p_skills = {normalize_text(s) for s in crec.prerequisites + crec.canonical_prerequisites}
            if graph and graph.graph.has_node(cid):
                for _, sid, data in graph.graph.out_edges(cid, data=True):
                    if data.get("type") == "TEACHES":
                        t_skills.add(sid)
                    elif data.get("type") == "REQUIRES_PREREQUISITE":
                        p_skills.add(sid)
            course_teaches[cid] = t_skills
            course_prereqs[cid] = p_skills

        # Dependency graph between courses: Course A depends on Course B if B teaches a prereq of A
        dependency_graph: dict[str, set[str]] = {cid: set() for cid in course_map}
        for cid_a, prereqs_a in course_prereqs.items():
            for p in prereqs_a:
                if p not in user_skills:
                    for cid_b, teaches_b in course_teaches.items():
                        if cid_a != cid_b and p in teaches_b:
                            dependency_graph[cid_a].add(cid_b)

        # Topological sort with tie-breaking by level and rank
        in_degree = {cid: len(deps) for cid, deps in dependency_graph.items()}
        queue = [cid for cid, deg in in_degree.items() if deg == 0]

        def sort_key(cid: str) -> tuple[int, int]:
            lvl = self.LEVEL_ORDER.get(course_map[cid].course_record.level.lower(), 2)
            rnk = course_map[cid].rank
            return (lvl, rnk)

        queue.sort(key=sort_key)
        ordered_ids: list[str] = []

        while queue:
            curr_id = queue.pop(0)
            ordered_ids.append(curr_id)

            # For courses depending on curr_id, decrement in-degree
            for next_id, deps in dependency_graph.items():
                if curr_id in deps:
                    in_degree[next_id] -= 1
                    if in_degree[next_id] == 0:
                        queue.append(next_id)
            queue.sort(key=sort_key)

        # Add any cycles / remaining courses
        for cid in course_map:
            if cid not in ordered_ids:
                ordered_ids.append(cid)

        # Simulate path progression and cumulative coverage
        known_skills_sim = set(user_skills)
        ordered_course_info: list[dict[str, Any]] = []
        cumulative_coverage: list[dict[str, Any]] = []
        unresolved_prereqs: list[dict[str, Any]] = []
        total_duration = 0.0

        for step, cid in enumerate(ordered_ids, 1):
            rc = course_map[cid]
            crec = rc.course_record
            total_duration += crec.duration_hours

            # Check prerequisites at time of taking
            prereqs = course_prereqs[cid]
            missing_at_step = prereqs - known_skills_sim
            for mp in missing_at_step:
                unresolved_prereqs.append({
                    "course_id": cid,
                    "course_title": crec.title,
                    "missing_prerequisite": mp,
                    "step": step,
                })

            # Skills acquired
            new_skills = course_teaches[cid]
            known_skills_sim.update(new_skills)

            # Calculate cumulative gap coverage
            covered_gap_mass = sum(
                gap_vector.get_gap_weight(s)
                for s in (known_skills_sim & set(gap_vector.vector.keys()))
            )
            pct = covered_gap_mass / max(1e-6, gap_vector.total_gap_mass)

            info = {
                "step": step,
                "course_id": cid,
                "title": crec.title,
                "level": crec.level,
                "duration_hours": crec.duration_hours,
                "provider": crec.provider,
                "covers_skills": rc.covered_missing_skills,
                "prerequisites_satisfied_at_step": len(missing_at_step) == 0,
            }
            ordered_course_info.append(info)
            cumulative_coverage.append({
                "step": step,
                "course_id": cid,
                "cumulative_gap_coverage_ratio": round(pct, 4),
                "cumulative_gap_coverage_pct": f"{pct * 100:.1f}%",
            })

        # Group into stages
        stages: list[LearningStage] = []
        if len(ordered_course_info) <= 2:
            stages.append(
                LearningStage(
                    stage_number=1,
                    stage_name="Target Role Skill Closing",
                    courses=ordered_course_info,
                    skills_acquired=sorted(list({s for c in ordered_ids for s in course_teaches[c]})),
                    estimated_hours=total_duration,
                )
            )
        else:
            mid = (len(ordered_course_info) + 1) // 2
            stage1_courses = ordered_course_info[:mid]
            stage2_courses = ordered_course_info[mid:]
            stages.append(
                LearningStage(
                    stage_number=1,
                    stage_name="Core Foundations & Immediate Gaps",
                    courses=stage1_courses,
                    skills_acquired=sorted(list({s for c in [x['course_id'] for x in stage1_courses] for s in course_teaches[c]})),
                    estimated_hours=sum(c["duration_hours"] for c in stage1_courses),
                )
            )
            stages.append(
                LearningStage(
                    stage_number=2,
                    stage_name="Advanced Specialization & Role Readiness",
                    courses=stage2_courses,
                    skills_acquired=sorted(list({s for c in [x['course_id'] for x in stage2_courses] for s in course_teaches[c]})),
                    estimated_hours=sum(c["duration_hours"] for c in stage2_courses),
                )
            )

        return LearningPath(
            user_id=user_id,
            target_role=job_title,
            total_courses=len(ordered_ids),
            total_duration_hours=total_duration,
            stages=stages,
            ordered_courses=ordered_course_info,
            cumulative_gap_coverage=cumulative_coverage,
            unresolved_prerequisites=unresolved_prereqs,
        )
