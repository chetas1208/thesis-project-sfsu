"""course_port.py

The seam between the end-to-end inference pipeline and course recommendation.

Course ingestion and ranking (``skillllm_v2.course``, ``skillllm_v2.recommend``,
``skillllm_v2.pipeline.course_recommendation_pipeline``) are owned and evolved
separately. The inference pipeline therefore talks to them ONLY through the
:class:`CourseRecommenderPort` protocol defined here:

* :class:`CourseRecommendationPipelineAdapter` -- delegates to the existing
  ``CourseRecommendationPipeline`` (imported lazily, never modified).
* :class:`NullCourseRecommender` -- an explicitly-labelled no-op used when no
  recommender is configured or the adapter is unavailable. It returns an empty
  list and says so; it never fabricates recommendations.

Anything that satisfies the protocol (including test doubles) can be passed to
``SkillLLMV2Pipeline(course_port=...)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

NULL_NOTE = (
    "NULL IMPLEMENTATION: no course recommender is configured, so 'recommendations' is intentionally empty. "
    "Wire a CourseRecommenderPort (e.g. CourseRecommendationPipelineAdapter) to enable stage 11."
)


@dataclass
class CourseRecommendationRequest:
    user_id: str
    #: ``{"user_id", "skills": [skill dicts with canonical_skill_id/name, confidence, evidence], "target_roles"}``
    user_profile: dict[str, Any]
    #: ``{"job_id", "title", "required_skills": [...], "preferred_skills": [...]}``
    target_job: dict[str, Any]
    #: Output of ``GapDetector.detect`` (informational; the adapter's pipeline recomputes its own gap vector).
    gaps: list[dict[str, Any]] = field(default_factory=list)
    top_k: int = 5


@dataclass
class CourseRecommendationResponse:
    recommendations: list[dict[str, Any]]
    learning_path: dict[str, Any] | None
    available: bool
    implementation: str
    note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation": self.implementation,
            "available": self.available,
            "note": self.note,
            "recommendations": self.recommendations,
            "learning_path": self.learning_path,
            "extras": self.extras,
        }


@runtime_checkable
class CourseRecommenderPort(Protocol):
    """What the inference pipeline requires from a course recommender."""

    name: str

    def is_available(self) -> bool:
        """Cheap check (must not do heavy work such as ingesting a catalog)."""
        ...

    def recommend(self, request: CourseRecommendationRequest) -> CourseRecommendationResponse:
        ...


class NullCourseRecommender:
    """Clearly-labelled no-op implementation of :class:`CourseRecommenderPort`."""

    name = "null"

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason

    def is_available(self) -> bool:
        return False

    def recommend(self, request: CourseRecommendationRequest) -> CourseRecommendationResponse:
        note = NULL_NOTE if not self.reason else f"{NULL_NOTE} Reason: {self.reason}"
        return CourseRecommendationResponse([], None, False, "NullCourseRecommender", note)


class CourseRecommendationPipelineAdapter:
    """Delegates to ``CourseRecommendationPipeline`` without modifying it.

    The wrapped pipeline is built lazily on the first :meth:`recommend` call
    (construction ingests the catalog). ``pipeline`` lets callers (and tests)
    inject an already-built object exposing ``run(user_input, job_input, top_k)``;
    ``pipeline_factory`` replaces the default construction.
    """

    name = "course_recommendation_pipeline"

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        pipeline: Any | None = None,
        pipeline_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self._pipeline = pipeline
        self._factory = pipeline_factory
        self._unavailable_reason: str | None = None

    # -- availability -------------------------------------------------------------
    def is_available(self) -> bool:
        if self._pipeline is not None or self._factory is not None:
            return True
        if self.catalog_path is not None and not self.catalog_path.exists():
            self._unavailable_reason = f"course catalog not found: {self.catalog_path}"
            return False
        try:
            import importlib

            importlib.import_module("skillllm_v2.pipeline.course_recommendation_pipeline")
        except Exception as exc:  # the course side is developed separately; never break inference over it
            self._unavailable_reason = f"CourseRecommendationPipeline import failed: {type(exc).__name__}: {exc}"
            return False
        return True

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            if self._factory is not None:
                self._pipeline = self._factory()
            else:
                from skillllm_v2.pipeline.course_recommendation_pipeline import CourseRecommendationPipeline

                self._pipeline = CourseRecommendationPipeline(catalog_path=self.catalog_path)
        return self._pipeline

    # -- port ---------------------------------------------------------------------
    def recommend(self, request: CourseRecommendationRequest) -> CourseRecommendationResponse:
        if not self.is_available():
            return CourseRecommendationResponse([], None, False, "CourseRecommendationPipelineAdapter", f"UNAVAILABLE: {self._unavailable_reason}")
        pipeline = self._get_pipeline()
        result = pipeline.run(user_input=request.user_profile, job_input=request.target_job, top_k=request.top_k)
        recommendations = [dict(item) for item in result.get("recommendations", [])]
        for item in recommendations:
            # Downstream explanation code reads ``course_name``; the ranker's payload uses ``title``.
            item.setdefault("course_name", item.get("title") or item.get("course_id"))
        return CourseRecommendationResponse(
            recommendations=recommendations,
            learning_path=result.get("learning_path"),
            available=True,
            implementation="CourseRecommendationPipeline",
            note="delegated to CourseRecommendationPipeline.run",
            extras={
                "skill_gap_analysis": result.get("skill_gap_analysis"),
                "pipeline_metadata": result.get("pipeline_metadata"),
            },
        )


def build_course_recommender(
    mode: str = "auto",
    catalog_path: str | Path | None = None,
) -> CourseRecommenderPort:
    """Factory: ``null`` | ``adapter`` | ``auto`` (adapter when importable and the catalog exists, else null)."""
    mode = (mode or "auto").lower()
    if mode == "null":
        return NullCourseRecommender("courses.port=null")
    adapter = CourseRecommendationPipelineAdapter(catalog_path=catalog_path)
    if mode == "adapter":
        return adapter
    if mode == "auto":
        return adapter if adapter.is_available() else NullCourseRecommender(adapter.unavailable_reason)
    raise ValueError(f"Unknown courses.port mode: {mode!r} (expected null | adapter | auto)")
