"""Read a user's free-text description of their experience and return profile skills with competence levels.

    "I have 3 years of Java, some Python, and I've never used Docker."
      -> Java / proficient, Python / beginner   (Docker is reported as negated, not counted)

An LLM (OpenAI, same setup as scripts/llm_extract_syllabi.py) does the reading. Each skill must quote the
text it came from; a quote that is not in the text is dropped, so the model cannot add skills the user
never mentioned. The returned skill dicts have the shape GapEngine and CourseGraphBuilder already read.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker
from skillllm_v2.schema import normalize_text

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"  # SkillLLM_v2_stage2/.env
DEFAULT_MODEL = "gpt-4.1-mini"
LEVELS = ("beginner", "intermediate", "proficient")  # GapEngine.MASTERY_WEIGHTS keys, weakest first
EXPLICIT_CONFIDENCE = 1.0
INFERRED_CONFIDENCE = 0.7
APPROX_WORD_SHARE = 0.85  # same threshold as the syllabus extractor

SYSTEM_PROMPT = """You read a person's description of their own technical experience and list their skills.

Rules:
- Only technical skills: programming languages, frameworks, libraries, tools, databases, and CS/engineering
  concepts. Leave out soft skills (communication, teamwork, leadership).
- Name skills concisely as they would appear on a resume (e.g. "Java", "Spring Boot", "JUnit", "REST APIs").
- "explicit": the person names the skill. "inferred": not named but clearly implied by what they did
  (e.g. "built REST APIs" implies HTTP). Give a one-sentence reason for inferred skills.
- level: how competent the person is, judged from their wording:
  - "beginner": "some", "a little", "familiar with", "basic", "took a class", "forgot most of it", "learning"
  - "intermediate": "used it in projects", "comfortable with", "worked with", a single class project
  - "proficient": "years of", "built X with", "led", "expert", "advanced", professional or repeated use
- negated: true when the person says they do NOT have the skill ("never used Docker", "haven't learned React").
- last_used: when they say when they last used it, give "YYYY-MM" or "YYYY", computed from today's date
  (given in the message). Otherwise null.
- evidence: ONE short passage copied word-for-word from the person's text. Do not paraphrase.
- Return only JSON matching this shape:
{"skills": [{"skill": string, "type": "explicit" | "inferred", "level": "beginner" | "intermediate" | "proficient",
             "evidence": string, "last_used": string or null, "negated": boolean, "reason": string or null}]}"""


@dataclass
class TextProfileResult:
    skills: list[dict[str, Any]] = field(default_factory=list)
    negated: list[dict[str, Any]] = field(default_factory=list)
    dropped_unverified: list[dict[str, Any]] = field(default_factory=list)


def load_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "OPENAI_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError(f"No API key found. Set OPENAI_API_KEY or add OPENAI_API_KEY=... to {ENV_FILE}")


def _normalize(text: str) -> str:
    """Lowercase, unify quotes/dashes, and collapse whitespace."""
    text = text.lower().translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", text).strip()


def _quote_found(quote: str | None, norm_text: str, text_words: set[str]) -> bool:
    if not quote:
        return False
    norm_quote = _normalize(quote)
    if re.search(rf"(?<![a-z0-9]){re.escape(norm_quote)}(?![a-z0-9])", norm_text):
        return True
    words = re.findall(r"[a-z0-9+#]+", norm_quote)
    return bool(words) and sum(w in text_words for w in words) / len(words) >= APPROX_WORD_SHARE


def _level(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    return value if value in LEVELS else "intermediate"


def _last_seen(raw: Any) -> str | None:
    match = re.fullmatch(r"\s*(\d{4})(?:-(\d{1,2}))?\s*", str(raw or ""))
    if not match:
        return None
    month = int(match.group(2) or 1)
    if not 1 <= month <= 12:
        return None
    return f"{match.group(1)}-{month:02d}-01T00:00:00+00:00"


class TextProfileExtractor:
    def __init__(self, client: Any = None, linker: TaxonomyLinker | None = None,
                 model: str = DEFAULT_MODEL, today: date | None = None) -> None:
        self._client = client
        self.linker = linker or TaxonomyLinker()
        self.model = model
        self.today = today

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=load_api_key())
        return self._client

    def _ask(self, text: str) -> list[dict[str, Any]]:
        today = (self.today or date.today()).isoformat()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,  # same text -> (nearly) the same answer
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Today is {today}.\n\nText:\n\n{text}"},
            ],
        )
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Model did not return JSON: {content!r:.200}") from exc
        skills = data.get("skills") if isinstance(data, dict) else None
        if not isinstance(skills, list):
            raise ValueError(f"Model JSON has no 'skills' list: {content!r:.200}")
        return [s for s in skills if isinstance(s, dict) and s.get("skill")]

    def _to_profile_skill(self, raw: dict[str, Any]) -> dict[str, Any]:
        name = str(raw["skill"]).strip()
        match = self.linker.link(name)
        return {
            "surface_form": name,
            "canonical_skill_id": match["canonical_skill_id"] if match else normalize_text(name),
            "canonical_skill_name": match["canonical_skill_name"] if match else name,
            "taxonomy": match.get("taxonomy", "CUSTOM") if match else "CUSTOM",
            "confidence": INFERRED_CONFIDENCE if raw.get("type") == "inferred" else EXPLICIT_CONFIDENCE,
            "mastery_level": _level(raw.get("level")),
            "last_seen": _last_seen(raw.get("last_used")),
            "evidence": raw.get("evidence") or "",
            "source": "text",
        }

    def extract(self, text: str) -> TextProfileResult:
        result = TextProfileResult()
        if not text or not text.strip():
            return result
        norm_text = _normalize(text)
        text_words = set(re.findall(r"[a-z0-9+#]+", norm_text))
        kept: dict[str, dict[str, Any]] = {}
        for raw in self._ask(text):
            if raw.get("negated"):
                result.negated.append(raw)
                continue
            if not _quote_found(raw.get("evidence"), norm_text, text_words):
                result.dropped_unverified.append(raw)
                continue
            skill = self._to_profile_skill(raw)
            sid = skill["canonical_skill_id"]
            if sid not in kept or LEVELS.index(skill["mastery_level"]) > LEVELS.index(kept[sid]["mastery_level"]):
                kept[sid] = skill
        result.skills = list(kept.values())
        return result
