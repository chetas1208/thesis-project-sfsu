"""Backward-compatible re-export of :mod:`skillllm_v2.policy.source_aware_policy`.

The implementation lives in the ``skillllm_v2.policy`` package (the canonical
home of the source-aware anchor policy). This module is kept so existing
imports (``from skillllm_v2.source_aware_policy import ...``) keep working.
"""
from skillllm_v2.policy.source_aware_policy import (
    ANCHOR_SOURCES,
    candidate_sources,
    is_anchor_candidate,
    source_key,
)

__all__ = [
    "ANCHOR_SOURCES",
    "candidate_sources",
    "is_anchor_candidate",
    "source_key",
]
