"""Helpers for optional dependencies: availability checks and actionable import errors.

Nothing here imports a heavy package at module import time; availability is
probed with :func:`importlib.util.find_spec`, which does not execute the target.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

from .errors import MissingExtraError

#: import name -> (pip distribution name, extra that provides it)
OPTIONAL_MODULES: dict[str, tuple[str, str]] = {
    "torch": ("torch", "model"),
    "transformers": ("transformers", "model"),
    "peft": ("peft", "model"),
    "bitsandbytes": ("bitsandbytes", "model"),
    "accelerate": ("accelerate", "model"),
    "safetensors": ("safetensors", "model"),
    "gliner": ("gliner", "gliner"),
    "numpy": ("numpy", "eval"),
    "scipy": ("scipy", "eval"),
    "sklearn": ("scikit-learn", "eval"),
    "pytest": ("pytest", "test"),
}

EXTRAS: tuple[str, ...] = ("model", "gliner", "eval", "test")


def is_available(module: str) -> bool:
    """True if ``module`` can be imported (without importing it)."""
    if module in sys.modules:
        return sys.modules[module] is not None
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, extra: str | None = None, purpose: str = "") -> ModuleType:
    """Import ``module`` or raise :class:`MissingExtraError` naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        resolved = extra or OPTIONAL_MODULES.get(module.split(".")[0], (module, "model"))[1]
        installed = not isinstance(exc, ModuleNotFoundError) or getattr(exc, "name", None) not in (module, module.split(".")[0])
        raise MissingExtraError(module, resolved, purpose, detail=str(exc)[:200] if installed else None) from exc


def installed_extras() -> dict[str, dict[str, bool]]:
    """``{extra: {module: importable}}`` for every optional dependency group."""
    report: dict[str, dict[str, bool]] = {extra: {} for extra in EXTRAS}
    for module, (_, extra) in OPTIONAL_MODULES.items():
        report[extra][module] = is_available(module)
    return report


def extra_status() -> dict[str, bool]:
    """``{extra: all modules of that extra importable}``."""
    return {extra: all(mods.values()) for extra, mods in installed_extras().items()}
