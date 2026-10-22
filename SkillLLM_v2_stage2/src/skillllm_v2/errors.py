"""Exception types raised by the public ``skillllm_v2`` library API.

Every error derives from :class:`SkillLLMError` so callers can catch the whole
family, and also from the closest built-in exception so existing ``except``
clauses keep working.
"""
from __future__ import annotations


class SkillLLMError(Exception):
    """Base class for all errors raised deliberately by the SkillLLM library."""


class MissingExtraError(SkillLLMError, ImportError):
    """An optional dependency group is not installed (or is installed but cannot be imported).

    The message names the extra to install."""

    def __init__(self, module: str, extra: str, purpose: str = "", detail: str | None = None) -> None:
        self.module = module
        self.extra = extra
        what = f" ({purpose})" if purpose else ""
        if detail:
            msg = (
                f"Optional dependency '{module}'{what} is installed but failed to import: {detail}. "
                f"This is usually a version mismatch; try:  pip install -U 'skillllm[{extra}]'"
            )
        else:
            msg = (
                f"Optional dependency '{module}' is required{what} but is not installed. "
                f"Install it with:  pip install 'skillllm[{extra}]'"
            )
        super().__init__(msg)


class ModelNotFoundError(SkillLLMError, FileNotFoundError):
    """A base-model or adapter path does not exist or is not a usable model directory."""


class DeviceError(SkillLLMError, RuntimeError):
    """The requested device or quantization cannot be used on this machine (for example no CUDA)."""


class GPUMemoryError(SkillLLMError, RuntimeError):
    """The GPU ran out of memory while loading or running the model; the message gives remedies."""


class FeatureUnavailableError(SkillLLMError, NotImplementedError):
    """A capability that depends on a component that is not importable or not configured."""


class ConfigError(SkillLLMError, ValueError):
    """An invalid or unsupported configuration was supplied."""
