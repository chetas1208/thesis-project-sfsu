from __future__ import annotations

import hashlib
import json
from typing import Any

from skillllm_v2.context.context_compactor import compact_context


def build_cached_context_packet(
    raw_profile_text: str,
    extracted_skills: list[dict[str, Any]],
    job_target: str | None = None,
    course_catalog_summaries: list[dict[str, Any]] | None = None,
    taxonomy_aliases: dict[str, str] | None = None,
    user_id: str = "demo_user",
) -> dict[str, Any]:
    compacted = compact_context(
        raw_profile_text=raw_profile_text,
        extracted_skills=extracted_skills,
        job_target=job_target,
        course_catalog_summaries=course_catalog_summaries,
        taxonomy_aliases=taxonomy_aliases,
        user_id=user_id,
    )
    packet = {
        "packet_type": "skillllm_v2_cached_context",
        "cache_policy": {
            "cag": ["profile_summary", "skill_memory", "taxonomy_aliases", "course_catalog_summary"],
            "rag": ["job_postings", "scraped_descriptions"],
            "graph": ["user_skill_job_course_paths"],
        },
        "payload": compacted,
    }
    serialized = json.dumps(packet, sort_keys=True, ensure_ascii=False)
    packet["content_hash"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return packet
