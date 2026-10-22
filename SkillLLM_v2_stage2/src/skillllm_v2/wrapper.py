"""Backward-compatible extraction wrapper.

``SkillLLMv2Pipeline.extract`` (extract -> merge -> refine -> verify -> calibrate -> link) now
delegates to :class:`skillllm_v2.pipeline.inference.SkillLLMV2Pipeline`, the single end-to-end
inference pipeline, sharing THIS object's components. The public attributes
(``mistral``, ``gliner``, ``dictionary``, ``merger``, ``refiner``, ``verifier``, ``calibrator``,
``linker``) and the return type (a list of candidate dicts) are unchanged.

Use ``SkillLLMV2Pipeline`` directly for the full 12-stage flow (gate, profile, graph, match,
courses, explanation) with a trace; use ``SkillLLMv2Pipeline.pipeline()`` to get that object
configured with this wrapper's components.
"""
from __future__ import annotations

import warnings
from typing import Any

from skillllm_v2.extract import CandidateMerger, DictionaryExtractor, GLiNERExtractor, MistralExtractor
from skillllm_v2.normalize import TaxonomyLinker
from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier


class SkillLLMv2Pipeline:
    def __init__(
        self,
        mistral: MistralExtractor | None = None,
        gliner: GLiNERExtractor | None = None,
        dictionary: DictionaryExtractor | None = None,
        linker: TaxonomyLinker | None = None,
    ) -> None:
        self.mistral = mistral or MistralExtractor()
        self.gliner = gliner or GLiNERExtractor()
        self.dictionary = dictionary or DictionaryExtractor()
        self.merger = CandidateMerger()
        self.refiner = BoundaryRefiner()
        self.verifier = SpanVerifier()
        self.calibrator = ConfidenceCalibrator()
        self.linker = linker or TaxonomyLinker()
        self.last_trace: Any | None = None  # PipelineTrace of the most recent ``extract`` call

    def pipeline(self, **overrides: Any) -> Any:
        """The end-to-end ``SkillLLMV2Pipeline`` sharing this wrapper's current components."""
        from skillllm_v2.pipeline.inference import SkillLLMV2Pipeline  # lazy: avoids an import cycle via skillllm_v2.pipeline

        components = dict(
            mistral=self.mistral,
            gliner=self.gliner,
            dictionary=self.dictionary,
            merger=self.merger,
            refiner=self.refiner,
            verifier=self.verifier,
            calibrator=self.calibrator,
            linker=self.linker,
        )
        components.update(overrides)
        return SkillLLMV2Pipeline(**components)

    def extract(
        self,
        text: str,
        include_gliner: bool = True,
        include_dictionary: bool = True,
        normalize: bool = True,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Extract -> merge -> refine -> verify -> calibrate (-> link); returns the candidate dicts.

        A failing extractor (Mistral / GLiNER / dictionary) does not block the call: the failure is recorded in
        ``self.last_trace`` (``extract`` stage ``status="degraded"``, error text in ``provenance["extractor_failures"]``),
        a ``RuntimeWarning`` names it, and the remaining extractors are used. If every extractor fails the result is an
        empty list and ``self.last_trace.get("extract").status == "failed"``. ``strict=True`` re-raises the extractor's
        exception instead.
        """
        stages = {"extract", "merge", "refine", "verify", "calibrate"} | ({"link"} if normalize else set())
        result = self.pipeline(strict_extractors=True if strict else None).extract_candidates(
            text, include_gliner=include_gliner, include_dictionary=include_dictionary, stages=stages
        )
        self.last_trace = result.trace
        record = result.trace.get("extract")
        failures = (record.provenance.get("extractor_failures") if record is not None else None) or {}
        if failures:
            status = "FAILED (empty extraction)" if record.status == "failed" else "degraded"
            warnings.warn(
                f"SkillLLMv2Pipeline extraction {status}: "
                + "; ".join(f"{name}: {error}" for name, error in failures.items()),
                RuntimeWarning,
                stacklevel=2,
            )
        if result.trace.stub_components and not getattr(self, "_stub_warned", False):
            self._stub_warned = True
            warnings.warn(
                f"SkillLLMv2Pipeline is running on deterministic STUB output ({', '.join(result.trace.stub_components)}), "
                "not Mistral/GLiNER model output. Pass a MistralExtractor with model_path to use the real anchor.",
                RuntimeWarning,
                stacklevel=2,
            )
        return result.enriched_skills
