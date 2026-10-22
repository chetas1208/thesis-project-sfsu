from __future__ import annotations

from typing import Any, Sequence

from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.course.prerequisite_parser import PrerequisiteParser, PrerequisiteRequirement
from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker
from skillllm_v2.schema import normalize_text
from skillllm_v2.wrapper import SkillLLMv2Pipeline


class CourseSkillExtractor:
    """Extracts, verifies, and canonicalizes skills and prerequisites from course descriptions and learning outcomes."""

    def __init__(
        self,
        pipeline: SkillLLMv2Pipeline | None = None,
        linker: TaxonomyLinker | None = None,
        prerequisite_parser: PrerequisiteParser | None = None,
    ) -> None:
        self.linker = linker or TaxonomyLinker()
        self.pipeline = pipeline or SkillLLMv2Pipeline(linker=self.linker)
        self.prereq_parser = prerequisite_parser or PrerequisiteParser()

    def extract_from_course(self, course: CourseRecord | dict[str, Any]) -> dict[str, Any]:
        record = course if isinstance(course, CourseRecord) else CourseCatalog._dict_to_record(course)
        course_dict = record.to_dict()

        taught_skills_evidence: list[dict[str, Any]] = []
        canonical_skills_map: dict[str, dict[str, Any]] = {}

        # 1. Process declared teaches_skills
        for skill_name in record.teaches_skills:
            norm_name = normalize_text(skill_name)
            link_info = self.linker.link(skill_name)
            cid = link_info["canonical_skill_id"] if link_info else norm_name
            cname = link_info["canonical_skill_name"] if link_info else skill_name
            taxonomy = link_info["taxonomy"] if link_info else "CUSTOM"

            entry = {
                "surface_form": skill_name,
                "canonical_skill_id": cid,
                "canonical_skill_name": cname,
                "taxonomy": taxonomy,
                "confidence": 1.0,
                "source": "declared",
                "evidence": f"Declared course outcome in {record.title}",
                "depth": self._map_level_to_depth(record.level),
            }
            taught_skills_evidence.append(entry)
            canonical_skills_map[cid] = entry

        # 2. Extract from learning outcomes
        for idx, outcome in enumerate(record.learning_outcomes):
            if not outcome.strip():
                continue
            extracted = self.pipeline.extract(outcome, include_gliner=False, include_dictionary=True, normalize=True)
            for cand in extracted:
                cid = cand.get("canonical_skill_id") or normalize_text(cand["surface_form"])
                cname = cand.get("canonical_skill_name") or cand["surface_form"]
                entry = {
                    "surface_form": cand["surface_form"],
                    "canonical_skill_id": cid,
                    "canonical_skill_name": cname,
                    "taxonomy": cand.get("metadata", {}).get("taxonomy_link", {}).get("taxonomy", "CUSTOM"),
                    "confidence": float(cand.get("confidence", 0.9)),
                    "source": "learning_outcome",
                    "evidence": outcome,
                    "depth": self._map_level_to_depth(record.level),
                    "outcome_index": idx,
                }
                taught_skills_evidence.append(entry)
                if cid not in canonical_skills_map:
                    canonical_skills_map[cid] = entry

        # 3. Extract from description if few or no skills extracted
        if len(canonical_skills_map) == 0 and record.description:
            extracted_desc = self.pipeline.extract(record.description, include_gliner=False, include_dictionary=True, normalize=True)
            for cand in extracted_desc:
                cid = cand.get("canonical_skill_id") or normalize_text(cand["surface_form"])
                cname = cand.get("canonical_skill_name") or cand["surface_form"]
                entry = {
                    "surface_form": cand["surface_form"],
                    "canonical_skill_id": cid,
                    "canonical_skill_name": cname,
                    "taxonomy": cand.get("metadata", {}).get("taxonomy_link", {}).get("taxonomy", "CUSTOM"),
                    "confidence": float(cand.get("confidence", 0.85)),
                    "source": "description",
                    "evidence": record.description,
                    "depth": self._map_level_to_depth(record.level),
                }
                taught_skills_evidence.append(entry)
                if cid not in canonical_skills_map:
                    canonical_skills_map[cid] = entry

        # 4. Process prerequisites
        parsed_prereqs = self.prereq_parser.parse(record.prerequisites)
        prereq_records: list[dict[str, Any]] = []
        canonical_prereqs_map: dict[str, dict[str, Any]] = {}

        for pr in parsed_prereqs:
            link_info = self.linker.link(pr.skill_name)
            cid = link_info["canonical_skill_id"] if link_info else normalize_text(pr.skill_name)
            cname = link_info["canonical_skill_name"] if link_info else pr.skill_name
            pr_dict = {
                "skill_name": pr.skill_name,
                "canonical_skill_id": cid,
                "canonical_skill_name": cname,
                "strictness": pr.strictness,
                "level": pr.level,
                "evidence": pr.evidence,
            }
            prereq_records.append(pr_dict)
            canonical_prereqs_map[cid] = pr_dict

        course_dict["taught_skills_evidence"] = taught_skills_evidence
        course_dict["canonical_skills"] = sorted(list(canonical_skills_map.keys()))
        course_dict["canonical_skill_names"] = [item["canonical_skill_name"] for item in canonical_skills_map.values()]
        course_dict["parsed_prerequisites"] = prereq_records
        course_dict["canonical_prerequisites"] = sorted(list(canonical_prereqs_map.keys()))
        course_dict["canonical_prerequisite_names"] = [item["canonical_skill_name"] for item in canonical_prereqs_map.values()]

        return course_dict

    def extract_from_catalog(self, catalog: CourseCatalog | Sequence[CourseRecord | dict[str, Any]]) -> list[dict[str, Any]]:
        items = catalog.list_courses() if isinstance(catalog, CourseCatalog) else catalog
        return [self.extract_from_course(item) for item in items]

    @staticmethod
    def _map_level_to_depth(level: str) -> str:
        lvl = level.lower().strip()
        if lvl in {"beginner", "introductory"}:
            return "introductory"
        if lvl in {"advanced", "mastery"}:
            return "advanced"
        return "intermediate"
