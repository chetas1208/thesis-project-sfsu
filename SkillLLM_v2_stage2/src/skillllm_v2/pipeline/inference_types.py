"""Typed stage artifacts, trace and result objects for the SkillLLM_v2 inference pipeline.

Every stage of :class:`skillllm_v2.pipeline.inference.SkillLLMV2Pipeline`
returns one of the artifact dataclasses below. Each carries a
:class:`StageRecord` (status, component, timing, counts, provenance) and is
JSON-serializable through ``to_dict()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

PIPELINE_VERSION = "skillllm_v2.inference/1"

#: Stage names in execution order (the numbering matches docs/INFERENCE_PIPELINE.md).
STAGE_ORDER: tuple[str, ...] = (
    "extract",  # 1  Mistral anchor + optional GLiNER + dictionary
    "merge",  # 2  candidate merger (source-aware, anchor-protected, add-only)
    "refine",  # 3  boundary refine / select
    "verify",  # 4  span verifier
    "calibrate",  # 5  confidence calibrator
    "link",  # 6  taxonomy link (attach-only)
    "gate",  # 7  safe-wrapper non-regression gate / fallback to anchor
    "profile",  # 8  profile summary, skill memory, compaction, context + contextual retrieval
    "graph",  # 9  graph build + retriever + explainer
    "match",  # 10 job match + gap detection
    "courses",  # 11 course recommendation, through CourseRecommenderPort only
    "explain",  # 12 final explanation
)
#: Sub-stage recorded in the trace only when the job is given as raw text.
JOB_EXTRACT_STAGE = "job_extract"
#: Stages that cannot be switched off (everything else passes data through when skipped).
REQUIRED_STAGES: frozenset[str] = frozenset({"extract"})

StageStatus = Literal["ok", "skipped", "degraded", "failed"]
AnchorMode = Literal["model", "stub", "none"]
GateDecision = Literal["accept", "fallback_to_anchor"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class StageRecord:
    name: str
    status: StageStatus = "ok"
    component: str = ""
    duration_seconds: float = 0.0
    input_count: int | None = None
    output_count: int | None = None
    notes: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "component": self.component,
            "duration_seconds": round(self.duration_seconds, 6),
            "input_count": self.input_count,
            "output_count": self.output_count,
            "notes": list(self.notes),
            "provenance": self.provenance,
            "error": self.error,
        }


@dataclass
class PipelineTrace:
    run_id: str
    pipeline_version: str = PIPELINE_VERSION
    started_at: str = field(default_factory=utc_now_iso)
    config_source: str | None = None
    config_hash: str | None = None
    anchor_mode: AnchorMode = "none"
    degraded: bool = False
    #: Components that produced deterministic stub/fallback output instead of model output.
    stub_components: list[str] = field(default_factory=list)
    stages: list[StageRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_seconds: float = 0.0

    def add(self, record: StageRecord) -> StageRecord:
        self.stages.append(record)
        if record.status in {"degraded", "failed"}:
            self.degraded = True
        return record

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def flag_stub(self, component: str) -> None:
        if component not in self.stub_components:
            self.stub_components.append(component)
        self.degraded = True

    def get(self, name: str) -> StageRecord | None:
        for record in reversed(self.stages):
            if record.name == name:
                return record
        return None

    @property
    def stage_names(self) -> list[str]:
        return [record.name for record in self.stages]

    @property
    def timings(self) -> dict[str, float]:
        return {record.name: round(record.duration_seconds, 6) for record in self.stages}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline_version": self.pipeline_version,
            "started_at": self.started_at,
            "config_source": self.config_source,
            "config_hash": self.config_hash,
            "anchor_mode": self.anchor_mode,
            "degraded": self.degraded,
            "stub_components": list(self.stub_components),
            "warnings": list(self.warnings),
            "total_seconds": round(self.total_seconds, 6),
            "timings": self.timings,
            "stages": [record.to_dict() for record in self.stages],
        }


# --------------------------------------------------------------------------- artifacts
@dataclass
class ExtractArtifact:
    """Stage 1: raw candidates from each extractor (before any merging)."""

    candidates: list[dict[str, Any]]
    anchor: list[dict[str, Any]]
    gliner: list[dict[str, Any]]
    dictionary: list[dict[str, Any]]
    anchor_mode: AnchorMode
    record: StageRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_mode": self.anchor_mode,
            "counts": {"anchor": len(self.anchor), "gliner": len(self.gliner), "dictionary": len(self.dictionary), "total": len(self.candidates)},
            "anchor": self.anchor,
            "gliner": self.gliner,
            "dictionary": self.dictionary,
            "record": self.record.to_dict(),
        }


@dataclass
class CandidateArtifact:
    """Stages 2-6: a candidate list plus what the stage dropped (verify/calibrate)."""

    stage: str
    candidates: list[dict[str, Any]]
    record: StageRecord
    dropped: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "candidates": self.candidates, "dropped": self.dropped, "record": self.record.to_dict()}


@dataclass
class GateArtifact:
    """Stage 7: non-regression decision and the three candidate sets it chooses between."""

    decision: GateDecision
    reason: str
    basis: str  # offline_artifact | metrics | forced_wrapper | forced_anchor | structural_only | not_applicable | missing_status
    selected: list[dict[str, Any]]  # extraction output: wrapper candidates, or the untouched anchor on fallback
    anchor: list[dict[str, Any]]  # untouched anchor candidates
    enriched: list[dict[str, Any]]  # wrapper (merged/refined/verified/calibrated/linked) candidates
    anchor_retention: float | None
    anchor_mode: AnchorMode
    record: StageRecord
    gate_detail: dict[str, Any] = field(default_factory=dict)

    @property
    def fell_back(self) -> bool:
        return self.decision == "fallback_to_anchor"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "basis": self.basis,
            "fell_back_to_anchor": self.fell_back,
            "anchor_mode": self.anchor_mode,
            "anchor_retention": self.anchor_retention,
            "gate_detail": self.gate_detail,
            "counts": {"selected": len(self.selected), "anchor": len(self.anchor), "enriched": len(self.enriched)},
            "record": self.record.to_dict(),
        }


@dataclass
class ProfileArtifact:
    """Stage 8: summary, skill memory, compacted/cached context packet, and retrieval hits."""

    profile_summary: dict[str, Any]
    skill_memory: dict[str, Any]
    context_packet: dict[str, Any]
    retrieved_skills: list[dict[str, Any]]
    retrieved_jobs: list[dict[str, Any]]
    record: StageRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_summary": self.profile_summary,
            "skill_memory": self.skill_memory,
            "context_packet": {"packet_type": self.context_packet.get("packet_type"), "content_hash": self.context_packet.get("content_hash"), "cache_policy": self.context_packet.get("cache_policy")},
            "retrieved_skills": self.retrieved_skills,
            "retrieved_jobs": [{"job_id": item["record"].get("job_id"), "score": item["score"]} for item in self.retrieved_jobs],
            "record": self.record.to_dict(),
        }


@dataclass
class GraphArtifact:
    """Stage 9: user/skill/job graph, graph retrieval results and path explanations."""

    user_node: str
    job_node: str | None
    user_skills: list[str]
    job_required_skills: list[str]
    graph_missing_skills: list[str]
    explanations: list[dict[str, Any]]
    node_count: int
    edge_count: int
    record: StageRecord
    graph: Any = field(default=None, repr=False)  # networkx.MultiDiGraph; not serialized
    #: skill -> recommended-course paths, added after stage 11 (TEACHES edges retrieved with SkillGraphRetriever)
    course_paths: list[dict[str, Any]] = field(default_factory=list)
    #: missing skill -> catalogue courses teaching it, read from the graph store (only when ``graph.backend`` != "local")
    catalog_courses: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        extra = {"catalog_courses": self.catalog_courses} if self.catalog_courses else {}
        return {
            **extra,
            "user_node": self.user_node,
            "job_node": self.job_node,
            "user_skills": self.user_skills,
            "job_required_skills": self.job_required_skills,
            "graph_missing_skills": self.graph_missing_skills,
            "explanations": self.explanations,
            "course_paths": self.course_paths,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "record": self.record.to_dict(),
        }


@dataclass
class MatchArtifact:
    """Stage 10: job matches (all supplied jobs), target job, gaps and gap vector."""

    matches: list[dict[str, Any]]
    target_job: dict[str, Any] | None
    job_match: dict[str, Any] | None
    gaps: list[dict[str, Any]]
    gap_vector: dict[str, Any] | None
    user_profile: dict[str, Any]
    record: StageRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_job_id": (self.target_job or {}).get("job_id"),
            "matches": self.matches,
            "job_match": self.job_match,
            "gaps": self.gaps,
            "gap_vector": self.gap_vector,
            "user_profile": self.user_profile,
            "record": self.record.to_dict(),
        }


@dataclass
class CourseArtifact:
    """Stage 11: course recommendations obtained through the CourseRecommenderPort only."""

    port: str
    available: bool
    recommendations: list[dict[str, Any]]
    learning_path: dict[str, Any] | None
    note: str
    record: StageRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "available": self.available,
            "note": self.note,
            "recommendations": self.recommendations,
            "learning_path": self.learning_path,
            "record": self.record.to_dict(),
        }


@dataclass
class ExplanationArtifact:
    """Stage 12: the final, evidence-backed recommendation explanation."""

    explanation: dict[str, Any]
    record: StageRecord

    def to_dict(self) -> dict[str, Any]:
        return {"explanation": self.explanation, "record": self.record.to_dict()}


# --------------------------------------------------------------------------- results
@dataclass
class ExtractionResult:
    """Output of stages 1-7 (``run_extraction_only``) or 1-6 (legacy ``extract``)."""

    text: str
    skills: list[dict[str, Any]]  # selected extraction output (wrapper candidates, or untouched anchor on fallback)
    enriched_skills: list[dict[str, Any]]  # wrapper-linked candidates (what downstream stages use by default)
    extract: ExtractArtifact
    merge: CandidateArtifact
    refine: CandidateArtifact
    verify: CandidateArtifact
    calibrate: CandidateArtifact
    link: CandidateArtifact
    gate: GateArtifact | None
    trace: PipelineTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "skills": self.skills,
            "enriched_skills": self.enriched_skills,
            "stages": {
                "extract": self.extract.to_dict(),
                "merge": self.merge.to_dict(),
                "refine": self.refine.to_dict(),
                "verify": self.verify.to_dict(),
                "calibrate": self.calibrate.to_dict(),
                "link": self.link.to_dict(),
                "gate": self.gate.to_dict() if self.gate else None,
            },
            "trace": self.trace.to_dict(),
        }


@dataclass
class PipelineResult:
    """Output of a full ``run``: every stage artifact (``None`` when skipped) plus the trace."""

    profile_text: str
    user_id: str
    extraction: ExtractionResult
    profile: ProfileArtifact | None
    graph: GraphArtifact | None
    match: MatchArtifact | None
    courses: CourseArtifact | None
    explanation: ExplanationArtifact | None
    trace: PipelineTrace

    @property
    def skills(self) -> list[dict[str, Any]]:
        return self.extraction.skills

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "skills": self.extraction.skills,
            "extraction": self.extraction.to_dict()["stages"],
            "profile": self.profile.to_dict() if self.profile else None,
            "graph": self.graph.to_dict() if self.graph else None,
            "match": self.match.to_dict() if self.match else None,
            "courses": self.courses.to_dict() if self.courses else None,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "trace": self.trace.to_dict(),
        }
