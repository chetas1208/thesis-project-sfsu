from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.graph.schema import EdgeType, NodeType
from skillllm_v2.schema import normalize_text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class CourseGraphBuilder:
    """Knowledge graph builder connecting Users, Jobs, Courses, and Canonical Skills."""

    def __init__(self) -> None:
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError("networkx is required for the Knowledge Graph layer") from exc
        self.nx = nx
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, node_type: NodeType, **properties: Any) -> None:
        self.graph.add_node(node_id, type=node_type.value, **properties)

    def add_edge(self, u: str, v: str, edge_type: EdgeType, **properties: Any) -> None:
        self.graph.add_edge(u, v, type=edge_type.value, **properties)

    def add_skill(
        self,
        skill_id: str,
        name: str | None = None,
        taxonomy: str = "CUSTOM",
        aliases: list[str] | None = None,
        **properties: Any,
    ) -> None:
        name = name or skill_id
        self.add_node(
            skill_id,
            NodeType.SKILL,
            canonical_skill_id=skill_id,
            canonical_skill_name=name,
            taxonomy=taxonomy,
            aliases=aliases or [name],
            **properties,
        )

    def add_user_skill(
        self,
        user_id: str,
        skill_id: str,
        confidence: float = 1.0,
        evidence_text: str = "",
        source: str = "profile",
        last_seen: str | None = None,
        mastery_level: str = "proficient",
        **properties: Any,
    ) -> None:
        self.add_node(user_id, NodeType.USER)
        if not self.graph.has_node(skill_id):
            self.add_skill(skill_id)
        self.add_edge(
            user_id,
            skill_id,
            EdgeType.HAS_SKILL,
            confidence=float(confidence),
            evidence_text=evidence_text,
            source=source,
            last_seen=last_seen or _utc_now_iso(),
            mastery_level=mastery_level,
            **properties,
        )

    def add_job_requirement(
        self,
        job_id: str,
        skill_id: str,
        importance: float = 1.0,
        requirement_strength: str = "required",
        source_sentence: str = "",
        **properties: Any,
    ) -> None:
        self.add_node(job_id, NodeType.JOB)
        if not self.graph.has_node(skill_id):
            self.add_skill(skill_id)
        edge_type = EdgeType.REQUIRES if requirement_strength == "required" else EdgeType.PREFERS
        self.add_edge(
            job_id,
            skill_id,
            edge_type,
            importance=float(importance),
            requirement_strength=requirement_strength,
            source_sentence=source_sentence,
            **properties,
        )

    def add_course_skill(
        self,
        course_id: str,
        skill_id: str,
        depth: str = "intermediate",
        confidence: float = 1.0,
        learning_outcome_evidence: str = "",
        outcome_index: int = 0,
        **properties: Any,
    ) -> None:
        self.add_node(course_id, NodeType.COURSE)
        if not self.graph.has_node(skill_id):
            self.add_skill(skill_id)
        self.add_edge(
            course_id,
            skill_id,
            EdgeType.TEACHES,
            depth=depth,
            confidence=float(confidence),
            learning_outcome_evidence=learning_outcome_evidence,
            outcome_index=outcome_index,
            **properties,
        )

    def add_course_prerequisite(
        self,
        course_id: str,
        skill_id: str,
        prerequisite_evidence: str = "",
        level: str = "beginner",
        strictness: str = "strict",
        **properties: Any,
    ) -> None:
        self.add_node(course_id, NodeType.COURSE)
        if not self.graph.has_node(skill_id):
            self.add_skill(skill_id)
        self.add_edge(
            course_id,
            skill_id,
            EdgeType.REQUIRES_PREREQUISITE,
            prerequisite_evidence=prerequisite_evidence,
            level=level,
            strictness=strictness,
            **properties,
        )

    def add_skill_relation(
        self,
        skill_a: str,
        skill_b: str,
        relation_type: str = "alias",
        weight: float = 1.0,
        **properties: Any,
    ) -> None:
        if not self.graph.has_node(skill_a):
            self.add_skill(skill_a)
        if not self.graph.has_node(skill_b):
            self.add_skill(skill_b)
        self.add_edge(
            skill_a,
            skill_b,
            EdgeType.RELATED_TO,
            relation_type=relation_type,
            weight=float(weight),
            **properties,
        )

    def ingest_user_profile(self, profile: dict[str, Any]) -> str:
        user_id = str(profile.get("user_id") or "default_user")
        self.add_node(user_id, NodeType.USER, target_roles=profile.get("target_roles", []))
        for item in profile.get("skills", []):
            if isinstance(item, str):
                sid = normalize_text(item)
                self.add_skill(sid, name=item)
                self.add_user_skill(user_id, sid, confidence=1.0)
            elif isinstance(item, dict):
                sid = item.get("canonical_skill_id") or normalize_text(item.get("surface_form") or item.get("name") or "")
                sname = item.get("canonical_skill_name") or item.get("surface_form") or item.get("name") or sid
                conf = float(item.get("confidence", 1.0))
                evidence = item.get("evidence", "")
                if isinstance(evidence, list):
                    evidence = evidence[0]["text"] if evidence and isinstance(evidence[0], dict) else ", ".join(map(str, evidence))
                self.add_skill(sid, name=sname, taxonomy=item.get("taxonomy", "CUSTOM"))
                self.add_user_skill(user_id, sid, confidence=conf, evidence_text=str(evidence), last_seen=item.get("last_seen"))
        return user_id

    def ingest_job(self, job: dict[str, Any]) -> str:
        job_id = str(job.get("job_id") or "target_job")
        title = str(job.get("title") or job_id)
        self.add_node(job_id, NodeType.JOB, title=title)

        for item in job.get("required_skills", []):
            if isinstance(item, str):
                sid = normalize_text(item)
                self.add_skill(sid, name=item)
                self.add_job_requirement(job_id, sid, importance=1.0, requirement_strength="required")
            elif isinstance(item, dict):
                sid = item.get("canonical_skill_id") or normalize_text(item.get("skill_name") or item.get("name") or "")
                sname = item.get("canonical_skill_name") or item.get("skill_name") or item.get("name") or sid
                imp = float(item.get("importance", 1.0))
                src = str(item.get("source_sentence", ""))
                self.add_skill(sid, name=sname)
                self.add_job_requirement(job_id, sid, importance=imp, requirement_strength="required", source_sentence=src)

        for item in job.get("preferred_skills", []):
            if isinstance(item, str):
                sid = normalize_text(item)
                self.add_skill(sid, name=item)
                self.add_job_requirement(job_id, sid, importance=0.7, requirement_strength="preferred")
            elif isinstance(item, dict):
                sid = item.get("canonical_skill_id") or normalize_text(item.get("skill_name") or item.get("name") or "")
                sname = item.get("canonical_skill_name") or item.get("skill_name") or item.get("name") or sid
                imp = float(item.get("importance", 0.7))
                src = str(item.get("source_sentence", ""))
                self.add_skill(sid, name=sname)
                self.add_job_requirement(job_id, sid, importance=imp, requirement_strength="preferred", source_sentence=src)

        return job_id

    def ingest_course(self, course: CourseRecord | dict[str, Any]) -> str:
        record = course if isinstance(course, CourseRecord) else CourseCatalog._dict_to_record(course)
        cid = record.course_id
        self.add_node(
            cid,
            NodeType.COURSE,
            title=record.title,
            description=record.description,
            level=record.level,
            duration_hours=record.duration_hours,
            provider=record.provider,
            rating=record.rating,
            cost_usd=record.cost_usd,
            learning_outcomes=record.learning_outcomes,
        )

        # Teaches skills
        taught_items = record.metadata.get("taught_skills_evidence") if record.metadata else None
        if taught_items:
            for item in taught_items:
                sid = item.get("canonical_skill_id") or normalize_text(item.get("surface_form") or "")
                sname = item.get("canonical_skill_name") or item.get("surface_form") or sid
                self.add_skill(sid, name=sname, taxonomy=item.get("taxonomy", "CUSTOM"))
                self.add_course_skill(
                    cid,
                    sid,
                    depth=item.get("depth", record.level),
                    confidence=float(item.get("confidence", 1.0)),
                    learning_outcome_evidence=item.get("evidence", ""),
                    outcome_index=int(item.get("outcome_index", 0)),
                )
        else:
            for skill_name in record.teaches_skills:
                sid = normalize_text(skill_name)
                self.add_skill(sid, name=skill_name)
                self.add_course_skill(cid, sid, depth=record.level, confidence=1.0, learning_outcome_evidence=record.title)

        # Prerequisites
        parsed_prereqs = record.metadata.get("parsed_prerequisites") if record.metadata else None
        if parsed_prereqs:
            for item in parsed_prereqs:
                sid = item.get("canonical_skill_id") or normalize_text(item.get("skill_name") or "")
                sname = item.get("canonical_skill_name") or item.get("skill_name") or sid
                self.add_skill(sid, name=sname)
                self.add_course_prerequisite(
                    cid,
                    sid,
                    prerequisite_evidence=item.get("evidence", ""),
                    level=item.get("level", "beginner"),
                    strictness=item.get("strictness", "strict"),
                )
        else:
            for prereq_name in record.prerequisites:
                sid = normalize_text(prereq_name)
                self.add_skill(sid, name=prereq_name)
                self.add_course_prerequisite(cid, sid, prerequisite_evidence=prereq_name, level="beginner", strictness="strict")

        return cid

    def ingest_catalog(self, catalog: CourseCatalog | Sequence[CourseRecord | dict[str, Any]]) -> list[str]:
        items = catalog.list_courses() if isinstance(catalog, CourseCatalog) else catalog
        return [self.ingest_course(item) for item in items]

    def get_user_skills(self, user_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not self.graph.has_node(user_id):
            return results
        for _, skill_id, data in self.graph.out_edges(user_id, data=True):
            if data.get("type") == EdgeType.HAS_SKILL.value:
                skill_node = self.graph.nodes.get(skill_id, {})
                results.append(
                    {
                        "skill_id": skill_id,
                        "canonical_skill_id": skill_id,
                        "canonical_skill_name": skill_node.get("canonical_skill_name", skill_id),
                        "confidence": data.get("confidence", 1.0),
                        "evidence_text": data.get("evidence_text", ""),
                        "last_seen": data.get("last_seen"),
                        "mastery_level": data.get("mastery_level", "proficient"),
                    }
                )
        return results

    def get_job_requirements(self, job_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not self.graph.has_node(job_id):
            return results
        for _, skill_id, data in self.graph.out_edges(job_id, data=True):
            edge_type = data.get("type")
            if edge_type in {EdgeType.REQUIRES.value, EdgeType.PREFERS.value}:
                skill_node = self.graph.nodes.get(skill_id, {})
                results.append(
                    {
                        "skill_id": skill_id,
                        "canonical_skill_id": skill_id,
                        "canonical_skill_name": skill_node.get("canonical_skill_name", skill_id),
                        "requirement_strength": data.get("requirement_strength", "required"),
                        "importance": data.get("importance", 1.0),
                        "source_sentence": data.get("source_sentence", ""),
                    }
                )
        return results

    def get_course_taught_skills(self, course_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not self.graph.has_node(course_id):
            return results
        for _, skill_id, data in self.graph.out_edges(course_id, data=True):
            if data.get("type") == EdgeType.TEACHES.value:
                skill_node = self.graph.nodes.get(skill_id, {})
                results.append(
                    {
                        "skill_id": skill_id,
                        "canonical_skill_id": skill_id,
                        "canonical_skill_name": skill_node.get("canonical_skill_name", skill_id),
                        "depth": data.get("depth", "intermediate"),
                        "confidence": data.get("confidence", 1.0),
                        "learning_outcome_evidence": data.get("learning_outcome_evidence", ""),
                    }
                )
        return results

    def get_course_prerequisites(self, course_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not self.graph.has_node(course_id):
            return results
        for _, skill_id, data in self.graph.out_edges(course_id, data=True):
            if data.get("type") == EdgeType.REQUIRES_PREREQUISITE.value:
                skill_node = self.graph.nodes.get(skill_id, {})
                results.append(
                    {
                        "skill_id": skill_id,
                        "canonical_skill_id": skill_id,
                        "canonical_skill_name": skill_node.get("canonical_skill_name", skill_id),
                        "strictness": data.get("strictness", "strict"),
                        "level": data.get("level", "beginner"),
                        "prerequisite_evidence": data.get("prerequisite_evidence", ""),
                    }
                )
        return results

    def evaluate_prerequisites(self, course_id: str, user_skill_ids: set[str]) -> dict[str, Any]:
        prereqs = self.get_course_prerequisites(course_id)
        if not prereqs:
            return {
                "course_id": course_id,
                "is_satisfied": True,
                "compatibility_score": 1.0,
                "satisfied_prerequisites": [],
                "missing_prerequisites": [],
                "violation_count": 0,
            }

        satisfied = []
        missing = []
        for p in prereqs:
            sid = p["skill_id"]
            if sid in user_skill_ids:
                satisfied.append(p)
            else:
                missing.append(p)

        is_sat = len(missing) == 0
        compat = len(satisfied) / max(1, len(prereqs))
        return {
            "course_id": course_id,
            "is_satisfied": is_sat,
            "compatibility_score": compat,
            "satisfied_prerequisites": satisfied,
            "missing_prerequisites": missing,
            "violation_count": len(missing),
        }

    def trace_explanation_path(
        self,
        user_id: str,
        job_id: str,
        course_id: str,
        target_skill_id: str,
    ) -> dict[str, Any]:
        """Constructs an explainable graph path: Job REQUIRES Skill <- User MISSING Skill; Course TEACHES Skill."""
        skill_node = self.graph.nodes.get(target_skill_id, {})
        skill_name = skill_node.get("canonical_skill_name", target_skill_id)

        # 1. Job requirement edge
        job_edge = None
        if self.graph.has_node(job_id):
            for _, tgt, data in self.graph.out_edges(job_id, data=True):
                if tgt == target_skill_id:
                    job_edge = data
                    break

        # 2. User possession edge
        user_has_edge = None
        if self.graph.has_node(user_id):
            for _, tgt, data in self.graph.out_edges(user_id, data=True):
                if tgt == target_skill_id:
                    user_has_edge = data
                    break

        # 3. Course teaches edge
        course_teaches_edge = None
        if self.graph.has_node(course_id):
            for _, tgt, data in self.graph.out_edges(course_id, data=True):
                if tgt == target_skill_id and data.get("type") == EdgeType.TEACHES.value:
                    course_teaches_edge = data
                    break

        # Build readable human trace
        job_title = self.graph.nodes.get(job_id, {}).get("title", job_id)
        course_title = self.graph.nodes.get(course_id, {}).get("title", course_id)

        req_str = job_edge.get("requirement_strength", "required") if job_edge else "required"
        user_status = "user has documented proficiency" if user_has_edge else "user currently lacks or is weak in this skill"
        teaches_evidence = course_teaches_edge.get("learning_outcome_evidence", "") if course_teaches_edge else ""

        narrative = (
            f"Target role '{job_title}' {req_str} skill '{skill_name}' -> "
            f"{user_status} -> "
            f"Course '{course_title}' teaches '{skill_name}'"
        )
        if teaches_evidence:
            narrative += f" ('{teaches_evidence}')"

        return {
            "target_skill_id": target_skill_id,
            "skill_name": skill_name,
            "job_requirement": job_edge,
            "user_possession": user_has_edge,
            "course_evidence": course_teaches_edge,
            "narrative_path": narrative,
        }
