from __future__ import annotations

from pathlib import Path
from typing import Any

from skillllm_v2.config import load_yaml
from skillllm_v2.schema import validate_skill
from skillllm_v2.source_aware_policy import candidate_sources, is_anchor_candidate

DEFAULT_TUNED_CONFIG_PATH = "artifacts/skillllm_v2/confidence_calibrator/tuned_config.yaml"

# Confidence is always in [0, 1], so a threshold below 0 can never reject a
# candidate -- it is the "always accept" sentinel used both for protected
# anchor sources and for the calibrator's no-op grid point.
ALWAYS_ACCEPT = -1.0

DEFAULT_CONFIG: dict[str, Any] = {
    # Candidates whose metadata.sources intersects this set are the Raw
    # Mistral anchor (or corroborated by it) and are NEVER rejected by
    # confidence calibration -- only brand-new GLiNER/dictionary-only
    # additions (already gated by CandidateMerger's add-only policy) are
    # subject to the thresholds below.
    "protect_anchor_sources": ["mistral"],
    "hard_skill_threshold": 0.45,
    "soft_skill_threshold": 0.42,
    "dictionary_only_threshold": 0.70,
    "multi_source_threshold": 0.40,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_calibrator_config(tuned_config_path: str | None = DEFAULT_TUNED_CONFIG_PATH) -> dict[str, Any]:
    config = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}
    if tuned_config_path and Path(tuned_config_path).exists():
        tuned = load_yaml(tuned_config_path)
        config = _deep_merge(config, tuned.get("confidence_calibrator", tuned))
    return config


class ConfidenceCalibrator:
    def __init__(
        self,
        hard_skill_threshold: float | None = None,
        soft_skill_threshold: float | None = None,
        dictionary_only_threshold: float | None = None,
        multi_source_threshold: float | None = None,
        protect_anchor_sources: list[str] | None = None,
        config: dict[str, Any] | None = None,
        tuned_config_path: str | None = DEFAULT_TUNED_CONFIG_PATH,
    ) -> None:
        self.config = config or load_calibrator_config(tuned_config_path)
        if hard_skill_threshold is not None:
            self.config["hard_skill_threshold"] = hard_skill_threshold
        if soft_skill_threshold is not None:
            self.config["soft_skill_threshold"] = soft_skill_threshold
        if dictionary_only_threshold is not None:
            self.config["dictionary_only_threshold"] = dictionary_only_threshold
        if multi_source_threshold is not None:
            self.config["multi_source_threshold"] = multi_source_threshold
        if protect_anchor_sources is not None:
            self.config["protect_anchor_sources"] = list(protect_anchor_sources)
        self.protect_anchor_sources = frozenset(self.config.get("protect_anchor_sources", ["mistral"]))
        self.hard_skill_threshold = self.config["hard_skill_threshold"]
        self.soft_skill_threshold = self.config["soft_skill_threshold"]
        self.dictionary_only_threshold = self.config["dictionary_only_threshold"]
        self.multi_source_threshold = self.config["multi_source_threshold"]

    def threshold_for(self, candidate: dict[str, Any]) -> float:
        if is_anchor_candidate(candidate, self.protect_anchor_sources):
            return ALWAYS_ACCEPT
        sources = candidate_sources(candidate)
        if len(sources) > 1:
            return self.multi_source_threshold
        if sources == {"dictionary"}:
            return self.dictionary_only_threshold
        if candidate["label"] == "SOFT_SKILL":
            return self.soft_skill_threshold
        return self.hard_skill_threshold

    def accept(self, candidate: dict[str, Any]) -> bool:
        validate_skill(candidate)
        return float(candidate["confidence"]) >= self.threshold_for(candidate)

    def calibrate(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for candidate in candidates:
            threshold = self.threshold_for(candidate)
            item = dict(candidate)
            item["metadata"] = dict(candidate.get("metadata") or {})
            item["metadata"]["calibration_threshold"] = threshold
            item["metadata"]["calibration_accepted"] = float(candidate["confidence"]) >= threshold
            if item["metadata"]["calibration_accepted"]:
                output.append(item)
        return output
