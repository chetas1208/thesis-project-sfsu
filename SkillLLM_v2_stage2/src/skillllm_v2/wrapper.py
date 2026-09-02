from __future__ import annotations

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

    def extract(self, text: str, include_gliner: bool = True, include_dictionary: bool = True, normalize: bool = True) -> list[dict[str, Any]]:
        candidates = list(self.mistral.extract(text))
        if include_gliner:
            candidates.extend(self.gliner.extract(text))
        if include_dictionary:
            candidates.extend(self.dictionary.extract(text))
        merged = self.merger.merge(candidates)
        refined = self.refiner.refine(text, merged)
        verified = self.verifier.filter(refined)
        calibrated = self.calibrator.calibrate(verified)
        return self.linker.link_candidates(calibrated) if normalize else calibrated
