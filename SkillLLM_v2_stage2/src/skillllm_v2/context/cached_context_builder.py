from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

from skillllm_v2.context.budget import json_tokens
from skillllm_v2.context.context_compactor import compact_context

#: wall-clock fields that describe *when* a packet was built, not *what* it contains. They are excluded from
#: ``content_hash`` so the cache key is stable across seconds (it used to change every second).
VOLATILE_KEYS = frozenset({"generated_at", "timestamp", "content_hash"})


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def compute_content_hash(packet: dict[str, Any]) -> str:
    """SHA-256 of the packet with volatile (wall-clock) fields removed, keys sorted."""
    serialized = json.dumps(_strip_volatile({k: v for k, v in packet.items() if k != "content_hash"}), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_cached_context_packet(
    raw_profile_text: str,
    extracted_skills: list[dict[str, Any]],
    job_target: str | None = None,
    course_catalog_summaries: list[dict[str, Any]] | None = None,
    taxonomy_aliases: dict[str, str] | None = None,
    user_id: str = "demo_user",
    budget_tokens: int | None = None,
    *,
    timestamp: str | None = None,
    clock: Callable[[], str] | None = None,
    source_reliability: Mapping[str, float] | None = None,
    label_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the cached (CAG) context packet.

    ``budget_tokens`` bounds the *whole packet* (wrapper, payload and hash) in proxy tokens, see
    :mod:`skillllm_v2.context.budget`; ``None`` keeps everything (the default, as before). ``timestamp`` / ``clock``
    inject the time stamped into ``generated_at`` for deterministic tests; ``content_hash`` ignores it either way.
    """
    kwargs: dict[str, Any] = dict(
        job_target=job_target,
        course_catalog_summaries=course_catalog_summaries,
        taxonomy_aliases=taxonomy_aliases,
        user_id=user_id,
        timestamp=timestamp,
        clock=clock,
        source_reliability=source_reliability,
        label_weights=label_weights,
    )
    cache_policy = {
        "cag": ["profile_summary", "skill_memory", "taxonomy_aliases", "course_catalog_summary"],
        "rag": ["job_postings", "scraped_descriptions"],
        "graph": ["user_skill_job_course_paths"],
    }

    def assemble(compacted: dict[str, Any]) -> dict[str, Any]:
        packet = {"packet_type": "skillllm_v2_cached_context", "payload": compacted}
        if budget_tokens is None:  # static policy boilerplate is identical in every packet: omitted under a budget
            packet["cache_policy"] = cache_policy
        packet["content_hash"] = compute_content_hash(packet)
        return packet

    if budget_tokens is None:
        return assemble(compact_context(raw_profile_text, extracted_skills, **kwargs))

    # Budgeted: reserve the tokens the wrapper itself costs (payload excluded), then verify the whole packet and
    # tighten the reservation if the additive estimate was off.
    wrapper = json_tokens({"packet_type": "skillllm_v2_cached_context", "payload": {}, "content_hash": "0" * 64}) - 2
    reserve = wrapper
    packet: dict[str, Any] = {}
    for _ in range(6):
        compacted = compact_context(raw_profile_text, extracted_skills, budget_tokens=budget_tokens, reserve_tokens=reserve, **kwargs)
        packet = assemble(compacted)
        total = json_tokens(packet)
        if total <= budget_tokens or not compacted["budget"]["kept"]:
            break
        reserve += total - budget_tokens
    compacted = packet["payload"]
    compacted["budget"]["used"] = json_tokens(packet)
    compacted["budget"]["met"] = compacted["budget"]["used"] <= budget_tokens
    packet["content_hash"] = compute_content_hash(packet)
    return packet
