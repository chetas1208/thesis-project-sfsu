from __future__ import annotations

from typing import Any

from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.recommend.course_ranker import CourseRanker
from skillllm_v2.recommend.explanation_generator import CourseExplanationGenerator
from skillllm_v2.recommend.gap_vector import GapEngine, GapVector, SkillGap
from skillllm_v2.schema import normalize_text


class CourseRecommender:
    """Recommends courses that close measured skill gaps using graph evidence and interpretable ranking."""

    def __init__(self, ranker: CourseRanker | None = None) -> None:
        self.ranker = ranker or CourseRanker()
        self.explainer = CourseExplanationGenerator()
        self.gap_engine = GapEngine()

    def recommend(
        self,
        gaps: list[dict[str, Any]] | GapVector,
        courses: list[dict[str, Any]] | list[CourseRecord],
        user_profile: dict[str, Any] | None = None,
        target_job: dict[str, Any] | None = None,
        graph: Any = None,
        return_explanation_packets: bool = False,
    ) -> list[dict[str, Any]]:
        user_profile = user_profile or {}
        target_job = target_job or {"job_id": "target_role", "title": "Target Role"}

        # Normalize gaps into GapVector if needed
        if isinstance(gaps, GapVector):
            gap_vector = gaps
        else:
            # Convert list of gap dicts to GapVector
            skill_gaps: list[SkillGap] = []
            vector: dict[str, float] = {}
            for item in gaps:
                sname = str(item.get("skill") or item.get("skill_name") or "")
                sid = item.get("canonical_skill_id") or normalize_text(sname)
                gtype = item.get("gap_type", "missing_critical")
                imp = float(item.get("importance", 1.0))
                weight = imp if gtype == "missing_critical" else (imp * 0.7)
                sg = SkillGap(
                    canonical_skill_id=sid,
                    skill_name=sname,
                    gap_type=gtype,
                    importance=imp,
                    requirement_strength="required" if "critical" in gtype else "preferred",
                    user_strength=0.0,
                    gap_weight=weight,
                    source_sentence=item.get("source_sentence", ""),
                )
                skill_gaps.append(sg)
                vector[sid] = weight

            total_mass = sum(vector.values())
            gap_vector = GapVector(
                user_id=str(user_profile.get("user_id") or "user"),
                job_id=str(target_job.get("job_id") or "target_job"),
                gaps=skill_gaps,
                vector=vector,
                total_gap_mass=total_mass,
                critical_count=sum(1 for g in skill_gaps if "critical" in g.gap_type),
                preferred_count=sum(1 for g in skill_gaps if "preferred" in g.gap_type),
                weak_count=sum(1 for g in skill_gaps if "weak" in g.gap_type),
            )

        # Standardize courses into CourseRecords
        course_records: list[CourseRecord] = []
        for c in courses:
            if isinstance(c, CourseRecord):
                course_records.append(c)
            else:
                course_records.append(CourseCatalog._dict_to_record(c))

        # Rank courses
        ranked = self.ranker.rank(
            candidates=course_records,
            user_profile=user_profile,
            target_job=target_job,
            gap_vector=gap_vector,
            graph=graph,
            top_k=len(course_records),
        )

        if return_explanation_packets:
            return self.explainer.generate_recommendations_payload(
                ranked, user_profile, target_job, gap_vector, graph
            )

        # Backward-compatible formatted list
        results: list[dict[str, Any]] = []
        for sc in ranked:
            crec = sc.course_record
            why = (
                f"Covers {len(sc.covered_missing_skills)} missing skill(s): "
                f"{', '.join(sc.covered_missing_skills)}"
            )
            explanation_payload = self.explainer.generate_explanation(
                sc, user_profile, target_job, gap_vector, graph
            )
            entry = {
                "course_id": sc.course_id,
                "course_name": crec.title,
                "title": crec.title,
                "rank": sc.rank,
                "score": sc.score,
                "match_score": sc.score,
                "covers_skills": sc.covered_missing_skills,
                "covered_missing_skills": sc.covered_missing_skills,
                "gap_coverage": sc.score_breakdown.get("gap_coverage", 0.0),
                "why_recommended": why,
                "prerequisite_status": explanation_payload["prerequisite_status"],
                "evidence": explanation_payload["evidence"],
                "explanation_path": explanation_payload["explanation_path"],
                "limitations": explanation_payload["limitations"],
                "next_profile_update_action": explanation_payload["next_profile_update_action"],
            }
            results.append(entry)

        return results
