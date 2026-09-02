from __future__ import annotations

from typing import Any

from skillllm_v2.context.profile_summarizer import summarize_profile
from skillllm_v2.context.skill_memory_builder import build_skill_memory


def compact_context(
    raw_profile_text: str,
    extracted_skills: list[dict[str, Any]],
    job_target: str | None = None,
    course_catalog_summaries: list[dict[str, Any]] | None = None,
    taxonomy_aliases: dict[str, str] | None = None,
    user_id: str = "demo_user",
) -> dict[str, Any]:
    summary = summarize_profile(
        raw_profile_text,
        extracted_skills=extracted_skills,
        user_id=user_id,
        target_roles=[job_target] if job_target else [],
    )
    memory = build_skill_memory(summary)
    stable_context = {
        "profile_summary": summary,
        "skill_memory": memory,
        "taxonomy_aliases": taxonomy_aliases or {},
        "course_catalog_summary": course_catalog_summaries or [],
    }
    dynamic_context_policy = {
        "rag_sources": ["job_postings", "scraped_descriptions"],
        "reason": "Job postings are large and change frequently, so they should be retrieved rather than cached.",
    }
    graph_context_policy = {
        "paths": ["user -> skill -> missing_skill -> job -> course"],
        "reason": "Relational explanation belongs in the graph layer, not in extraction.",
    }
    return {
        "stable_cag_context": stable_context,
        "dynamic_rag_policy": dynamic_context_policy,
        "graph_retrieval_policy": graph_context_policy,
        "conflict_notes": summary.get("conflict_notes", []),
    }
