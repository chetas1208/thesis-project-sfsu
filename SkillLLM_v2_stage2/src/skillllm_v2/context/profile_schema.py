from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class Evidence:
    source: str
    text: str
    confidence: float
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProfileSkillFact:
    surface_form: str
    label: str
    canonical_skill_id: str | None
    canonical_skill_name: str | None
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return (self.canonical_skill_id or self.surface_form).strip().lower()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = [item.to_dict() for item in self.evidence]
        return payload


@dataclass(slots=True)
class ProfileSummary:
    user_id: str
    generated_at: str
    skills: list[ProfileSkillFact]
    target_roles: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    course_history: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    conflict_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["skills"] = [skill.to_dict() for skill in self.skills]
        return payload
