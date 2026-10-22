"""SkillLLM_v2: skill extraction with a Mistral-Small-3.1-24B LoRA system, as an installable library.

The public API lives in :mod:`skillllm_v2.api` and is re-exported here::

    from skillllm_v2 import SkillLLM
    result = SkillLLM.offline().extract("Experienced in Python, SQL and stakeholder management.")

Importing this package is cheap and never imports torch/transformers/peft; those are loaded only when a model is
used (install them with ``pip install 'skillllm[model]'``). The research modules (``extract``, ``verify``,
``normalize``, ``eval`` ...) keep working exactly as before.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .api import (
    AnalysisResult,
    ExtractionMetadata,
    ExtractionResult,
    MistralPeftBackend,
    OfflineBackend,
    Provenance,
    Skill,
    SkillLLM,
    SkillMatch,
    SpanBackend,
)
from .ensemble import (
    VoteCalibration,
    default_vote_calibration,
    fit_vote_calibration,
    majority_threshold,
    vote_documents,
    vote_spans,
)
from .errors import (
    ConfigError,
    DeviceError,
    FeatureUnavailableError,
    GPUMemoryError,
    MissingExtraError,
    ModelNotFoundError,
    SkillLLMError,
)

__all__ = [
    "__version__",
    # API
    "SkillLLM",
    "Skill",
    "Provenance",
    "ExtractionResult",
    "ExtractionMetadata",
    "AnalysisResult",
    "SkillMatch",
    "SpanBackend",
    "OfflineBackend",
    "MistralPeftBackend",
    # ensemble
    "vote_spans",
    "vote_documents",
    "majority_threshold",
    "VoteCalibration",
    "fit_vote_calibration",
    "default_vote_calibration",
    # errors
    "SkillLLMError",
    "MissingExtraError",
    "ModelNotFoundError",
    "DeviceError",
    "GPUMemoryError",
    "FeatureUnavailableError",
    "ConfigError",
]
