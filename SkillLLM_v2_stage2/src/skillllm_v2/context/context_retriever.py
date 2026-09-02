from __future__ import annotations

from typing import Any

from skillllm_v2.schema import normalize_text


class ContextRetriever:
    def __init__(self, packet: dict[str, Any]) -> None:
        self.packet = packet

    def retrieve_skills(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        terms = set(normalize_text(query).split())
        skills = (
            self.packet.get("payload", {})
            .get("stable_cag_context", {})
            .get("profile_summary", {})
            .get("skills", [])
        )
        scored = []
        for skill in skills:
            haystack = normalize_text(" ".join([
                str(skill.get("surface_form", "")),
                str(skill.get("canonical_skill_name", "")),
                " ".join(e.get("text", "") for e in skill.get("evidence", [])),
            ]))
            score = len(terms & set(haystack.split()))
            if score:
                scored.append((score, skill))
        return [skill for _, skill in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]
