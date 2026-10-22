from __future__ import annotations

from typing import Any, Sequence

from skillllm_v2.course.ingest_catalog import CourseRecord
from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.recommend.course_ranker import ScoredCourse
from skillllm_v2.recommend.gap_vector import GapVector, SkillGap
from skillllm_v2.schema import normalize_text


class CourseExplanationGenerator:
    """Generates evidence-backed, graph-traceable explanation packets for course recommendations.

    Fulfills the Minimum JSON Output contract:
    course_id, title, rank, score, covered_missing_skills, prerequisite_status,
    evidence, explanation_path, limitations, next_profile_update_action.
    """

    def generate_explanation(
        self,
        scored_course: ScoredCourse,
        user_profile: dict[str, Any],
        target_job: dict[str, Any],
        gap_vector: GapVector,
        graph: CourseGraphBuilder | None = None,
    ) -> dict[str, Any]:
        crec: CourseRecord = scored_course.course_record
        job_title = str(target_job.get("title") or target_job.get("job_id") or "Target Role")
        user_id = str(user_profile.get("user_id") or "user")

        # Map gap IDs to SkillGap objects
        gap_map: dict[str, SkillGap] = {g.canonical_skill_id: g for g in gap_vector.gaps}

        # 1. Build evidence list
        evidence_items: list[dict[str, Any]] = []
        explanation_steps: list[str] = []

        text_only = scored_course.role == "related_text"
        for cid in scored_course.covered_canonical_ids:
            gap = gap_map.get(cid)
            skill_name = gap.skill_name if gap else cid

            # Job requirement evidence
            job_sentence = gap.source_sentence if (gap and gap.source_sentence) else f"Required by role {job_title}"
            req_type = gap.gap_type if gap else "missing_critical"

            # Course learning outcome evidence
            course_outcome = ""
            if crec.learning_outcomes:
                # Find outcome mentioning the skill
                for out in crec.learning_outcomes:
                    if skill_name.lower() in out.lower() or cid in normalize_text(out):
                        course_outcome = out
                        break
                if not course_outcome:
                    course_outcome = crec.learning_outcomes[0]
            elif crec.description:
                course_outcome = crec.description

            evidence_entry = {
                "skill": skill_name,
                "canonical_skill_id": cid,
                "job_requirement_evidence": job_sentence,
                "gap_classification": req_type,
                "user_evidence": f"User has strength {gap.user_strength:.2f} (gap weight: {gap.gap_weight:.2f})" if gap else "User missing verified skill",
                "course_learning_outcome_evidence": course_outcome,
            }
            evidence_items.append(evidence_entry)

            if text_only:
                evidence_entry["evidence_type"] = "text_mention_only"
                explanation_steps.append(
                    f"Target role '{job_title}' requires '{skill_name}' -> User is {req_type.replace('_', ' ')} -> "
                    f"Course '{crec.title}' is textually related to '{skill_name}' (name mentioned in its text, or a shared content word with a skill it teaches); "
                    f"'{skill_name}' itself is not listed among the skills it teaches"
                )
                continue
            # Traceable explanation path:
            # target role requires skill -> user is missing/weak -> course teaches the skill
            step_str = (
                f"Target role '{job_title}' requires '{skill_name}' -> "
                f"User is {req_type.replace('_', ' ')} -> "
                f"Course '{crec.title}' teaches '{skill_name}'"
            )
            if course_outcome:
                step_str += f" via outcome: \"{course_outcome}\""
            explanation_steps.append(step_str)

        # 1b. Courses that make a blocked, more relevant course startable (every pure bridge, and some primaries).
        unlocks = scored_course.metadata.get("unlocks", [])
        for u in unlocks:
            where = (f"recommended later at rank {u['unlocked_course_rank']}" if u.get("in_list")
                     else "not in this top-K list (it would stay blocked until this prerequisite is met)")
            step = (f"Course '{u['title']}' requires prerequisite '{u['prerequisite_skill']}' -> "
                    f"'{crec.title}' teaches '{u['prerequisite_skill']}' and therefore comes first ({where})")
            if scored_course.role == "bridge":
                for gid in u.get("gap_skill_ids", []):
                    gap = gap_map.get(gid)
                    skill_name = gap.skill_name if gap else gid
                    if any(e["canonical_skill_id"] == gid and e.get("via_course_id") == u["course_id"] for e in evidence_items):
                        continue
                    evidence_items.append({
                        "skill": skill_name,
                        "canonical_skill_id": gid,
                        "job_requirement_evidence": (gap.source_sentence if (gap and gap.source_sentence) else f"Required by role {job_title}"),
                        "gap_classification": gap.gap_type if gap else "missing_critical",
                        "user_evidence": (f"User has strength {gap.user_strength:.2f} (gap weight: {gap.gap_weight:.2f})" if gap else "User missing verified skill"),
                        "course_learning_outcome_evidence": (crec.learning_outcomes[0] if crec.learning_outcomes else crec.description),
                        "evidence_type": "enabling_prerequisite",
                        "via_course_id": u["course_id"],
                        "via_prerequisite_skill": u["prerequisite_skill"],
                    })
                    explanation_steps.append(
                        f"Target role '{job_title}' requires '{skill_name}' -> Course '{u['title']}' teaches '{skill_name}' -> " + step
                    )
            else:
                explanation_steps.append("Also enables: " + step)

        # 2. Prerequisite status statement
        prereq_status = scored_course.prerequisite_status
        is_prereq_satisfied = prereq_status.get("is_satisfied", True)
        sat_list = prereq_status.get("satisfied_prerequisites", [])
        mis_list = prereq_status.get("missing_prerequisites", [])

        if not crec.prerequisites and not sat_list and not mis_list:
            prereq_explanation = "Course has no prerequisite requirements."
            prereq_summary = "satisfied_none_required"
        elif is_prereq_satisfied:
            sat_names = []
            for p in sat_list:
                if isinstance(p, dict):
                    nm = p.get("canonical_skill_name") or p.get("skill_name") or p.get("skill_id", str(p))
                    by = str(p.get("satisfied_by") or "")
                    if by.startswith("earlier_course:"):
                        nm = f"{nm} (taught by earlier recommended course {by.split(':', 1)[1]})"
                    sat_names.append(nm)
                else:
                    sat_names.append(str(p))
            prereq_explanation = f"Prerequisite requirements satisfied: {', '.join(sat_names)}."
            prereq_summary = "all_satisfied"
        else:
            mis_names = [
                p.get("canonical_skill_name") or p.get("skill_name") or p.get("skill_id", str(p))
                if isinstance(p, dict) else str(p)
                for p in mis_list
            ]
            prereq_explanation = f"Caution: Missing prerequisite skill(s): {', '.join(mis_names)}."
            prereq_summary = f"flagged_missing_{len(mis_names)}_prerequisites"

        explanation_steps.append(prereq_explanation)

        # 3. Limitations
        limitations: list[str] = []
        if not is_prereq_satisfied:
            mis_names = [
                p.get("canonical_skill_name") or p.get("skill_name") or p.get("skill_id", str(p))
                if isinstance(p, dict) else str(p)
                for p in mis_list
            ]
            limitations.append(f"Prerequisite gap: Learner should first acquire {', '.join(mis_names)}.")
        if scored_course.role == "related_text":
            limitations.append("Text-level relatedness only: the course does not list the open gap skill(s) among the skills it teaches.")
        if scored_course.role == "bridge":
            limitations.append("Bridge course: recommended because it teaches a prerequisite of a more relevant course, not for its own gap coverage.")
        if crec.duration_hours > 40.0:
            limitations.append(f"High workload commitment: Course estimated at {crec.duration_hours} hours.")
        if scored_course.score_breakdown.get("redundancy_penalty", 0.0) > 0.0:
            limitations.append("Contains partial skill overlap with higher-ranked recommendations in this session.")
        if not limitations:
            limitations.append("No blocking constraints detected.")

        # 4. Next profile update action
        skills_to_add = []
        for s in crec.teaches_skills:
            skills_to_add.append({
                "surface_form": s,
                "label": "HARD_SKILL",
                "canonical_skill_id": normalize_text(s),
                "canonical_skill_name": s,
                "confidence": 0.95,
                "evidence": f"Completed course {crec.course_id}: {crec.title}",
                "source": "completed_course",
            })

        next_action = {
            "trigger": "on_course_completion",
            "course_id": crec.course_id,
            "skills_to_add": skills_to_add,
            "target_mastery": "proficient",
            "profile_update_schema": {
                "user_id": user_id,
                "completed_courses": [crec.course_id],
                "skills": skills_to_add,
            },
        }

        return {
            "course_id": crec.course_id,
            "title": crec.title,
            "rank": scored_course.rank,
            "score": scored_course.score,
            "role": scored_course.role,
            "unlocks": unlocks,
            "covered_missing_skills": scored_course.covered_missing_skills,
            "prerequisite_status": {
                "is_satisfied": is_prereq_satisfied,
                "status_code": prereq_summary,
                "satisfied": sat_list,
                "missing": mis_list,
                "explanation": prereq_explanation,
            },
            "evidence": evidence_items,
            "explanation_path": explanation_steps,
            "limitations": limitations,
            "next_profile_update_action": next_action,
        }

    def generate_recommendations_payload(
        self,
        ranked_courses: Sequence[ScoredCourse],
        user_profile: dict[str, Any],
        target_job: dict[str, Any],
        gap_vector: GapVector,
        graph: CourseGraphBuilder | None = None,
    ) -> list[dict[str, Any]]:
        return [
            self.generate_explanation(rc, user_profile, target_job, gap_vector, graph)
            for rc in ranked_courses
        ]
