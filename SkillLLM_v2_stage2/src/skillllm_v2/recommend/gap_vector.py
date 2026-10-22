from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.graph.schema import EdgeType
from skillllm_v2.schema import normalize_text


@dataclass(slots=True)
class SkillGap:
    canonical_skill_id: str
    skill_name: str
    gap_type: str  # missing_critical | missing_preferred | weak_skill | outdated_skill | uncertain_skill | covered_by_transferable_skill
    importance: float
    requirement_strength: str  # "required" | "preferred"
    user_strength: float  # [0.0, 1.0]
    gap_weight: float  # [0.0, 1.0] prioritized deficiency weight
    source_sentence: str = ""
    evidence: str = ""
    transferable_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GapVector:
    user_id: str
    job_id: str
    gaps: list[SkillGap]
    vector: dict[str, float]  # canonical_skill_id -> gap_weight
    total_gap_mass: float
    critical_count: int
    preferred_count: int
    weak_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "job_id": self.job_id,
            "gaps": [g.to_dict() for g in self.gaps],
            "vector": self.vector,
            "total_gap_mass": round(self.total_gap_mass, 4),
            "critical_count": self.critical_count,
            "preferred_count": self.preferred_count,
            "weak_count": self.weak_count,
        }

    def get_gap_weight(self, skill_id: str) -> float:
        return self.vector.get(skill_id, self.vector.get(normalize_text(skill_id), 0.0))


class GapEngine:
    """Computes quantitative skill gaps and weighted priority vectors from user profile and job requirements."""

    STRENGTH_MULTIPLIER = {
        "required": 1.0,
        "preferred": 0.7,
    }

    MASTERY_WEIGHTS = {
        "beginner": 0.35,
        "intermediate": 0.70,
        "proficient": 1.0,
        "advanced": 1.0,
        "master": 1.0,
    }

    def __init__(
        self,
        weak_threshold: float = 0.5,
        uncertain_threshold: float = 0.6,
        obsolescence_days: int = 730,  # 2 years
    ) -> None:
        self.weak_threshold = weak_threshold
        self.uncertain_threshold = uncertain_threshold
        self.obsolescence_days = obsolescence_days

    def compute(
        self,
        user_profile: dict[str, Any],
        job: dict[str, Any],
        graph: CourseGraphBuilder | None = None,
    ) -> GapVector:
        user_id = str(user_profile.get("user_id") or "user")
        job_id = str(job.get("job_id") or "job")

        # 1. Parse user skills into lookup structure
        user_skills_map: dict[str, dict[str, Any]] = {}
        for item in user_profile.get("skills", []):
            if isinstance(item, str):
                sid = normalize_text(item)
                user_skills_map[sid] = {
                    "canonical_skill_id": sid,
                    "name": item,
                    "confidence": 1.0,
                    "mastery": "proficient",
                    "user_strength": 1.0,
                }
            elif isinstance(item, dict):
                sid = item.get("canonical_skill_id") or normalize_text(item.get("surface_form") or item.get("name") or "")
                sname = item.get("canonical_skill_name") or item.get("surface_form") or item.get("name") or sid
                conf = float(item.get("confidence", 1.0))
                mastery_str = str(item.get("mastery_level") or item.get("mastery") or "proficient").lower()
                mastery_factor = self.MASTERY_WEIGHTS.get(mastery_str, 0.7)
                effective_strength = conf * mastery_factor
                user_skills_map[sid] = {
                    "canonical_skill_id": sid,
                    "name": sname,
                    "confidence": conf,
                    "mastery": mastery_str,
                    "user_strength": effective_strength,
                    "last_seen": item.get("last_seen"),
                    "evidence": item.get("evidence"),
                }

        weak_skills = {normalize_text(s) for s in user_profile.get("weak_skills", [])}
        outdated_skills = {normalize_text(s) for s in user_profile.get("outdated_skills", [])}
        uncertain_skills = {normalize_text(s) for s in user_profile.get("uncertain_skills", [])}
        transferable_map = {normalize_text(k): v for k, v in user_profile.get("transferable_skills", {}).items()}

        # 2. Collect job requirements
        job_requirements: list[dict[str, Any]] = []

        # Required skills
        for item in job.get("required_skills", []):
            if isinstance(item, str):
                sid = normalize_text(item)
                job_requirements.append({
                    "skill_id": sid,
                    "name": item,
                    "importance": 1.0,
                    "strength": "required",
                    "source_sentence": "",
                })
            elif isinstance(item, dict):
                sid = item.get("canonical_skill_id") or normalize_text(item.get("skill_name") or item.get("name") or "")
                sname = item.get("canonical_skill_name") or item.get("skill_name") or item.get("name") or sid
                job_requirements.append({
                    "skill_id": sid,
                    "name": sname,
                    "importance": float(item.get("importance", 1.0)),
                    "strength": "required",
                    "source_sentence": item.get("source_sentence", ""),
                })

        # Preferred skills
        for item in job.get("preferred_skills", []):
            if isinstance(item, str):
                sid = normalize_text(item)
                job_requirements.append({
                    "skill_id": sid,
                    "name": item,
                    "importance": 0.7,
                    "strength": "preferred",
                    "source_sentence": "",
                })
            elif isinstance(item, dict):
                sid = item.get("canonical_skill_id") or normalize_text(item.get("skill_name") or item.get("name") or "")
                sname = item.get("canonical_skill_name") or item.get("skill_name") or item.get("name") or sid
                job_requirements.append({
                    "skill_id": sid,
                    "name": sname,
                    "importance": float(item.get("importance", 0.7)),
                    "strength": "preferred",
                    "source_sentence": item.get("source_sentence", ""),
                })

        # 3. Determine gaps
        gaps: list[SkillGap] = []
        vector: dict[str, float] = {}

        critical_count = 0
        preferred_count = 0
        weak_count = 0

        now = datetime.now(timezone.utc)

        for req in job_requirements:
            sid = req["skill_id"]
            sname = req["name"]
            imp = req["importance"]
            strength = req["strength"]
            str_mult = self.STRENGTH_MULTIPLIER.get(strength, 1.0)
            src_sent = req["source_sentence"]

            user_entry = user_skills_map.get(sid)

            if user_entry is None:
                # User does not have skill directly
                # Check graph or transferable map for transferable skills
                transferable_from = transferable_map.get(sid)
                if transferable_from is None and graph is not None and graph.graph.has_node(sid):
                    # Check related edges in graph
                    for u_sid in user_skills_map:
                        if graph.graph.has_edge(u_sid, sid):
                            transferable_from = user_skills_map[u_sid]["name"]
                            break

                if transferable_from:
                    gap_type = "covered_by_transferable_skill"
                    user_str = 0.4
                    weight = imp * str_mult * (1.0 - user_str)
                    gaps.append(
                        SkillGap(
                            canonical_skill_id=sid,
                            skill_name=sname,
                            gap_type=gap_type,
                            importance=imp,
                            requirement_strength=strength,
                            user_strength=user_str,
                            gap_weight=weight,
                            source_sentence=src_sent,
                            transferable_from=transferable_from,
                        )
                    )
                    vector[sid] = weight
                else:
                    if strength == "required":
                        gap_type = "missing_critical"
                        critical_count += 1
                    else:
                        gap_type = "missing_preferred"
                        preferred_count += 1
                    user_str = 0.0
                    weight = imp * str_mult
                    gaps.append(
                        SkillGap(
                            canonical_skill_id=sid,
                            skill_name=sname,
                            gap_type=gap_type,
                            importance=imp,
                            requirement_strength=strength,
                            user_strength=user_str,
                            gap_weight=weight,
                            source_sentence=src_sent,
                        )
                    )
                    vector[sid] = weight
            else:
                # User possesses skill: check if weak, outdated, or uncertain
                user_str = user_entry["user_strength"]
                gap_type = None

                # Outdated check
                if sid in outdated_skills:
                    gap_type = "outdated_skill"
                elif user_entry.get("last_seen"):
                    try:
                        seen_dt = datetime.fromisoformat(user_entry["last_seen"].replace("Z", "+00:00"))
                        if (now - seen_dt).days > self.obsolescence_days:
                            gap_type = "outdated_skill"
                    except Exception:
                        pass

                # Weak check
                if gap_type is None and (sid in weak_skills or user_str < self.weak_threshold):
                    gap_type = "weak_skill"
                    weak_count += 1

                # Uncertain check
                if gap_type is None and (sid in uncertain_skills or user_entry.get("confidence", 1.0) < self.uncertain_threshold):
                    gap_type = "uncertain_skill"

                if gap_type:
                    weight = imp * str_mult * (1.0 - user_str)
                    gaps.append(
                        SkillGap(
                            canonical_skill_id=sid,
                            skill_name=sname,
                            gap_type=gap_type,
                            importance=imp,
                            requirement_strength=strength,
                            user_strength=user_str,
                            gap_weight=weight,
                            source_sentence=src_sent,
                            evidence=str(user_entry.get("evidence", "")),
                        )
                    )
                    vector[sid] = weight

        total_mass = sum(vector.values())
        return GapVector(
            user_id=user_id,
            job_id=job_id,
            gaps=gaps,
            vector=vector,
            total_gap_mass=total_mass,
            critical_count=critical_count,
            preferred_count=preferred_count,
            weak_count=weak_count,
        )
