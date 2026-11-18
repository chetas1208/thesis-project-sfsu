from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from skillllm_v2.extract import text_profile_extractor as tpe
from skillllm_v2.extract.text_profile_extractor import TextProfileExtractor

TEXT = ("I have experience in Java, using Spring Boot and JUnit to build REST APIs. "
        "I know some Python. I took a class on SQL but forgot most of it. "
        "I've never used Docker. I used C++ two years ago.")
TODAY = date(2026, 11, 17)


class FakeClient:
    """Stands in for openai.OpenAI: records each request and returns canned JSON."""

    def __init__(self, content: str) -> None:
        self.calls: list[dict] = []
        self._content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def sk(skill, level, evidence, type_="explicit", last_used=None, negated=False):
    return {"skill": skill, "type": type_, "level": level, "evidence": evidence,
            "last_used": last_used, "negated": negated, "reason": None}


def make(canned=None, content=None):
    client = FakeClient(content if content is not None else json.dumps({"skills": canned or []}))
    return TextProfileExtractor(client=client, today=TODAY), client


def by_name(result):
    return {s["surface_form"]: s for s in result.skills}


def test_levels_and_fields():
    ex, _ = make([sk("Java", "proficient", "I have experience in Java"),
                  sk("Python", "beginner", "I know some Python")])
    by = by_name(ex.extract(TEXT))
    assert by["Java"]["mastery_level"] == "proficient"
    assert by["Python"]["mastery_level"] == "beginner"
    assert by["Java"]["canonical_skill_id"] == "java"
    assert by["Java"]["source"] == "text"
    assert by["Java"]["confidence"] == 1.0
    assert by["Java"]["evidence"] == "I have experience in Java"


def test_inferred_skill_gets_lower_confidence():
    ex, _ = make([sk("REST APIs", "intermediate", "build REST APIs", type_="inferred")])
    assert by_name(ex.extract(TEXT))["REST APIs"]["confidence"] == 0.7


def test_negated_skill_dropped_but_reported():
    ex, _ = make([sk("Docker", "beginner", "I've never used Docker", negated=True)])
    result = ex.extract(TEXT)
    assert result.skills == []
    assert [n["skill"] for n in result.negated] == ["Docker"]


def test_unverified_evidence_dropped():
    ex, _ = make([sk("Kubernetes", "proficient", "I am a Kubernetes expert")])
    result = ex.extract(TEXT)
    assert result.skills == []
    assert [d["skill"] for d in result.dropped_unverified] == ["Kubernetes"]


def test_last_used_becomes_iso_last_seen():
    ex, _ = make([sk("C++", "intermediate", "I used C++ two years ago", last_used="2024-11")])
    assert by_name(ex.extract(TEXT))["C++"]["last_seen"].startswith("2024-11")


def test_unparseable_last_used_is_none():
    ex, _ = make([sk("C++", "intermediate", "I used C++ two years ago", last_used="a while ago")])
    by = by_name(ex.extract(TEXT))
    assert by["C++"]["last_seen"] is None


def test_unknown_level_normalized():
    ex, _ = make([sk("Java", "PROFICIENT", "I have experience in Java"),
                  sk("Python", "Advanced", "I know some Python")])
    by = by_name(ex.extract(TEXT))
    assert by["Java"]["mastery_level"] == "proficient"
    assert by["Python"]["mastery_level"] == "intermediate"


def test_duplicate_skill_keeps_highest_level():
    ex, _ = make([sk("Python", "beginner", "I know some Python"),
                  sk("python", "proficient", "I know some Python")])
    result = ex.extract(TEXT)
    assert len(result.skills) == 1
    assert result.skills[0]["mastery_level"] == "proficient"


def test_empty_text_makes_no_call():
    ex, client = make([sk("Java", "proficient", "Java")])
    result = ex.extract("   ")
    assert result.skills == []
    assert client.calls == []


@pytest.mark.parametrize("content", ["not json", json.dumps({"skills": "x"})])
def test_malformed_response_raises_value_error(content):
    ex, _ = make(content=content)
    with pytest.raises(ValueError):
        ex.extract(TEXT)


def test_request_uses_temperature_0_and_json_mode():
    ex, client = make([])
    ex.extract(TEXT)
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert client.calls[0]["model"] == "gpt-4.1-mini"


def test_missing_api_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(tpe, "ENV_FILE", tmp_path / ".env")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        tpe.load_api_key()
