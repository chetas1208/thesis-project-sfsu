from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SkillLabel = Literal["HARD_SKILL", "SOFT_SKILL"]
SkillSource = Literal["bert", "gliner", "mistral", "dictionary"]
TaxonomyName = Literal["ESCO", "ONET", "LIGHTCAST", "CUSTOM"]
LinkMethod = Literal["alias", "fuzzy", "bm25", "dense", "reranker"]


@dataclass(slots=True)
class SkillCandidate:
    surface_form: str
    label: SkillLabel
    start: int | None
    end: int | None
    source: SkillSource
    confidence: float
    evidence: str
    canonical_skill_id: str | None = None
    canonical_skill_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MergedSkillCandidate:
    surface_form: str
    label: SkillLabel
    start: int | None
    end: int | None
    sources: list[SkillSource]
    confidence: float
    evidence: list[str]
    canonical_skill_id: str | None = None
    canonical_skill_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerifiedSkill:
    surface_form: str
    label: SkillLabel
    start: int | None
    end: int | None
    confidence: float
    is_valid_skill: bool
    verification_reason: str
    sources: list[SkillSource]
    canonical_skill_id: str | None = None
    canonical_skill_name: str | None = None
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaxonomyLink:
    surface_form: str
    canonical_skill_id: str
    canonical_skill_name: str
    taxonomy: TaxonomyName
    link_confidence: float
    link_method: LinkMethod
    top_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SystemBenchmarkResult:
    system_name: str
    exact_span_precision: float | None = None
    exact_span_recall: float | None = None
    exact_span_f1: float | None = None
    partial_span_f1: float | None = None
    token_f1: float | None = None
    hard_skill_f1: float | None = None
    soft_skill_f1: float | None = None
    linking_accuracy: float | None = None
    job_match_ndcg_at_10: float | None = None
    course_rec_precision_at_5: float | None = None
    latency_per_doc_seconds: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JobMatchResult:
    job_id: str
    match_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    critical_missing_skills: list[str]
    evidence: list[str]
    recommendation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CourseRecommendation:
    course_id: str
    course_name: str
    covers_skills: list[str]
    gap_coverage: float
    why_recommended: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_skill_candidate(candidate: dict[str, Any]) -> None:
    required = {
        "surface_form",
        "label",
        "start",
        "end",
        "source",
        "confidence",
        "evidence",
        "canonical_skill_id",
        "canonical_skill_name",
        "metadata",
    }
    missing = sorted(required.difference(candidate))
    if missing:
        raise ValueError(f"SkillCandidate missing fields: {missing}")
    if candidate["label"] not in {"HARD_SKILL", "SOFT_SKILL"}:
        raise ValueError(f"Invalid skill label: {candidate['label']}")
    if candidate["source"] not in {"bert", "gliner", "mistral", "dictionary"}:
        raise ValueError(f"Invalid skill source: {candidate['source']}")
    start, end = candidate["start"], candidate["end"]
    if (start is None) != (end is None):
        raise ValueError("start and end must both be set or both be null")
    if start is not None and (not isinstance(start, int) or not isinstance(end, int) or start >= end):
        raise ValueError(f"Invalid offsets: {start}, {end}")
    confidence = float(candidate["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Invalid confidence: {confidence}")


def validate_system_result(result: dict[str, Any]) -> None:
    if not result.get("system_name"):
        raise ValueError("system_name is required")
    for key, value in result.items():
        if key.endswith("_f1") or key.endswith("_accuracy") or key.endswith("_10") or key.endswith("_5"):
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric or null")
