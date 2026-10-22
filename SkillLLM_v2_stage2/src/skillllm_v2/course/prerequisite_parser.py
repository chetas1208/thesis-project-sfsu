from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Sequence

from skillllm_v2.schema import normalize_text


@dataclass(slots=True)
class PrerequisiteRequirement:
    skill_name: str
    strictness: str = "strict"  # "strict" | "recommended"
    level: str = "beginner"  # "beginner" | "intermediate" | "advanced"
    evidence: str = ""
    canonical_skill_id: str | None = None
    canonical_skill_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrerequisiteParser:
    """Parses structured and unstructured prerequisite declarations for courses."""

    STRICT_PATTERNS = [
        re.compile(r"\b(prerequisite|prerequisites|requirements?|required|must have|mandatory)\b", re.IGNORECASE),
    ]

    RECOMMENDED_PATTERNS = [
        re.compile(r"\b(recommended|prior experience|helpful|suggested|optional|good to have)\b", re.IGNORECASE),
    ]

    NONE_PATTERNS = [
        re.compile(r"\b(none|no prerequisites|no prior knowledge|open to all)\b", re.IGNORECASE),
    ]

    LEVEL_MAP = {
        "basic": "beginner",
        "introductory": "beginner",
        "beginner": "beginner",
        "fundamental": "beginner",
        "intermediate": "intermediate",
        "working knowledge": "intermediate",
        "advanced": "advanced",
        "proficient": "advanced",
    }

    def __init__(self, known_skills: set[str] | None = None) -> None:
        self.known_skills = {normalize_text(s): s for s in known_skills or []}

    def parse(self, raw_prerequisites: Sequence[str] | str) -> list[PrerequisiteRequirement]:
        if not raw_prerequisites:
            return []

        if isinstance(raw_prerequisites, str):
            text = raw_prerequisites.strip()
            if not text:
                return []
            if any(p.search(text) for p in self.NONE_PATTERNS):
                return []
            sentences = [s.strip() for s in re.split(r"[.\n;]", text) if s.strip()]
            requirements: list[PrerequisiteRequirement] = []
            for sent in sentences:
                strictness = "recommended" if any(p.search(sent) for p in self.RECOMMENDED_PATTERNS) else "strict"
                level = self._detect_level(sent)
                extracted_skills = self._extract_skills_from_snippet(sent)
                for skill in extracted_skills:
                    requirements.append(
                        PrerequisiteRequirement(
                            skill_name=skill,
                            strictness=strictness,
                            level=level,
                            evidence=sent,
                        )
                    )
            return requirements

        # List of items
        requirements: list[PrerequisiteRequirement] = []
        for item in raw_prerequisites:
            if not item or not str(item).strip():
                continue
            item_str = str(item).strip()
            if any(p.search(item_str) for p in self.NONE_PATTERNS):
                continue
            strictness = "recommended" if any(p.search(item_str) for p in self.RECOMMENDED_PATTERNS) else "strict"
            level = self._detect_level(item_str)
            cleaned = self._clean_prereq_string(item_str)
            if cleaned:
                requirements.append(
                    PrerequisiteRequirement(
                        skill_name=cleaned,
                        strictness=strictness,
                        level=level,
                        evidence=item_str,
                    )
                )
        return requirements

    def _detect_level(self, text: str) -> str:
        text_lower = text.lower()
        for kw, lvl in self.LEVEL_MAP.items():
            if kw in text_lower:
                return lvl
        return "beginner"

    def _clean_prereq_string(self, text: str) -> str:
        cleaned = re.sub(
            r"^(prerequisites?:?|requirements?:?|recommended:?|prior knowledge in:?)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"\b(basic|intermediate|advanced|introductory)\b", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def _extract_skills_from_snippet(self, text: str) -> list[str]:
        cleaned = self._clean_prereq_string(text)
        # Split by and / commas
        parts = [p.strip() for p in re.split(r",|\band\b|;", cleaned) if p.strip()]
        res = []
        for part in parts:
            if len(part) >= 2 and not any(p.search(part) for p in self.NONE_PATTERNS):
                norm = normalize_text(part)
                if norm in self.known_skills:
                    res.append(self.known_skills[norm])
                else:
                    res.append(part)
        return res
