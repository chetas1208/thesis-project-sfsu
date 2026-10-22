"""Regression tests for the match / gap / extractor-degradation fixes (docs/BUGFIX_LOG_MATCH.md). CPU only, no weights."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest

from skillllm_v2.match import GapDetector, JobMatcher
from skillllm_v2.match.skill_keys import SkillKeyResolver, skill_key
from skillllm_v2.pipeline.inference import SkillLLMV2Pipeline
from skillllm_v2.wrapper import SkillLLMv2Pipeline

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"SQL": "structured query language", "Postgres": "PostgreSQL", "k8s": "Kubernetes", "AWS": "Amazon Web Services"}


def job(required: list[Any], preferred: list[Any] | None = None, **kw: Any) -> dict[str, Any]:
    return {"job_id": "j", "required_skills": required, "preferred_skills": preferred or [], **kw}


def by_skill(gaps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {g["skill"]: g for g in gaps}


# ---------------------------------------------------------------------------------------------- skill_key
def test_skill_key_is_case_whitespace_and_punctuation_insensitive() -> None:
    assert skill_key("Python") == skill_key("  python ") == skill_key("PYTHON")
    assert skill_key("Machine   Learning") == skill_key("machine learning") == skill_key("Machine-Learning")
    assert skill_key("CI/CD") == skill_key("ci cd")
    assert skill_key("scikit-learn") == skill_key("Scikit Learn")


def test_skill_key_preserves_symbol_bearing_names() -> None:
    keys = {skill_key(s) for s in ("C", "C++", "C#", "F#", ".NET", "NET", "Node.js", "Node JS")}
    assert len(keys) == 8
    assert skill_key("C++") == skill_key("c++") == skill_key("C ++")
    assert skill_key("Node.js") == skill_key("NODE.JS")


# ---------------------------------------------------------------------------------------------- matcher
def test_matcher_is_case_invariant() -> None:
    j = job(["Python", "AWS"], ["Docker"], semantic_role_fit=0.5)
    base = JobMatcher().score({"skills": ["Python", "AWS", "Docker"]}, j)
    low = JobMatcher().score({"skills": ["python", "aws", "docker"]}, j)
    assert low["match_score"] == pytest.approx(base["match_score"])
    assert low["missing_skills"] == [] and low["matched_skills"] == sorted(["Python", "AWS", "Docker"])


def test_matcher_does_not_merge_symbol_distinct_skills() -> None:
    res = JobMatcher().score({"skills": ["C"]}, job(["C++", "C#", ".NET"]))
    assert res["missing_skills"] == sorted(["C++", "C#", ".NET"])
    assert JobMatcher().score({"skills": ["c++", "C#", ".net"]}, job(["C++", "C#", ".NET"]))["missing_skills"] == []


def test_matcher_alias_resolution_with_explicit_table() -> None:
    m = JobMatcher(aliases=ALIASES, use_default_aliases=False)
    res = m.score({"skills": ["structured query language", "postgresql"]}, job(["SQL", "Postgres"]))
    assert res["missing_skills"] == []
    assert JobMatcher(use_default_aliases=False).score({"skills": ["structured query language"]}, job(["SQL"]))["missing_skills"] == ["SQL"]


def test_matcher_default_alias_table_joins_canonical_name_and_surface() -> None:
    # the extractor emits 'structured query language' / 'Amazon Web Services'; jobs say 'SQL' / 'AWS'
    res = JobMatcher().score({"skills": ["structured query language", "Amazon Web Services"]}, job(["SQL", "AWS"]))
    assert res["missing_skills"] == []


def test_matcher_dict_skills_and_canonical_precedence() -> None:
    entry = {"surface_form": "SQL", "canonical_skill_name": "structured query language", "canonical_skill_id": "structured-query-language"}
    assert JobMatcher().score({"skill_entries": [entry]}, job(["SQL"]))["missing_skills"] == []
    assert JobMatcher().score({"skills": [entry]}, job(["structured query language"]))["missing_skills"] == []
    # surface that resolves (via the table) to a DIFFERENT skill than the declared canonical one: canonical wins
    wrong = {"surface_form": "k8s", "canonical_skill_name": "Docker"}
    m = JobMatcher(aliases=ALIASES, use_default_aliases=False)
    assert m.score({"skills": [wrong]}, job(["Kubernetes"]))["missing_skills"] == ["Kubernetes"]
    assert m.score({"skills": [wrong]}, job(["Docker"]))["missing_skills"] == []


def test_matcher_duplicate_job_skills_count_once() -> None:
    res = JobMatcher().score({"skills": []}, job(["Python", "python", "PYTHON"]))
    assert res["missing_skills"] == ["Python"] and res["match_score"] == 0.0


def test_matcher_exact_input_scores_unchanged() -> None:
    res = JobMatcher().score({"skills": ["Python"]}, job(["Python", "AWS"]))
    assert res["missing_skills"] == ["AWS"] and res["critical_missing_skills"] == ["AWS"]
    assert res["match_score"] == pytest.approx(0.45 * 0.5 - 0.25 * 0.5)


# ---------------------------------------------------------------------------------------------- detector
def test_detector_is_case_invariant_and_alias_aware() -> None:
    d = GapDetector()
    assert d.detect({"skills": ["python", "sql"]}, job(["Python", "SQL"])) == []
    assert d.detect({"skills": ["structured query language"]}, job(["SQL"], ["AWS"]))[0]["skill"] == "AWS"
    assert d.detect({"skills": ["structured query language"]}, job(["SQL"])) == []


def test_detector_symbol_distinct_skills_stay_gaps() -> None:
    gaps = GapDetector().detect({"skills": ["C"]}, job(["C++", "C#"]))
    assert {g["skill"] for g in gaps} == {"C++", "C#"}


def test_detector_weak_held_skill_is_not_relabelled_as_transferable() -> None:
    profile = {"skills": ["Python"], "weak_skills": ["Python"], "transferable_skills": {"Python": "Ruby"}}
    gaps = GapDetector().detect(profile, job(["Python"]))
    assert [g["gap_type"] for g in gaps] == ["weak_skill"]
    assert "transferable_from" not in gaps[0]


def test_detector_missing_skill_with_transferable_cover_still_relabelled() -> None:
    gaps = GapDetector().detect({"skills": ["Ruby"], "transferable_skills": {"python": "Ruby"}}, job(["Python"], ["Go"]))
    g = by_skill(gaps)
    assert g["Python"]["gap_type"] == "covered_by_transferable_skill" and g["Python"]["transferable_from"] == "Ruby"
    assert g["Go"]["gap_type"] == "missing_preferred"


def test_detector_skill_in_both_lists_reported_once_as_required() -> None:
    gaps = GapDetector().detect({"skills": []}, job(["Python", "AWS"], ["python", "Docker"]))
    assert [g["skill"] for g in gaps] == ["Python", "AWS", "Docker"]
    py = gaps[0]
    assert py["gap_type"] == "missing_critical" and py["requirement_strength"] == "required" and py["also_listed_as_preferred"] is True
    assert gaps[2]["gap_type"] == "missing_preferred"


def test_detector_preferred_weak_outdated_uncertain_are_reported() -> None:
    profile = {"skills": ["A", "B", "C", "D"], "weak_skills": ["A"], "outdated_skills": ["B"], "uncertain_skills": ["C"]}
    gaps = by_skill(GapDetector().detect(profile, job([], ["A", "B", "C", "D", "E"])))
    assert gaps["A"]["gap_type"] == "weak_skill" and gaps["B"]["gap_type"] == "outdated_skill" and gaps["C"]["gap_type"] == "uncertain_skill"
    assert "D" not in gaps and gaps["E"]["gap_type"] == "missing_preferred"
    assert all(g["requirement_strength"] == "preferred" for g in gaps.values())


def test_detector_required_weak_outdated_uncertain_precedence_unchanged() -> None:
    profile = {"skills": ["A", "B", "C"], "weak_skills": ["A", "B"], "outdated_skills": ["B"], "uncertain_skills": ["C"]}
    gaps = by_skill(GapDetector().detect(profile, job(["A", "B", "C", "D"])))
    assert [gaps[s]["gap_type"] for s in "ABCD"] == ["weak_skill", "weak_skill", "uncertain_skill", "missing_critical"]


def test_detector_legacy_schema_is_kept() -> None:
    gaps = GapDetector().detect({"skills": ["Python"]}, job(["Python", "AWS"], ["SQL"]))
    assert {g["gap_type"] for g in gaps} == {"missing_critical", "missing_preferred"}
    for g in gaps:
        assert {"skill", "gap_type"} <= set(g)


def test_detector_weak_flag_matches_across_spellings() -> None:
    profile = {"skills": ["structured query language"], "weak_skills": ["structured query language"]}
    assert [g["gap_type"] for g in GapDetector().detect(profile, job(["SQL"]))] == ["weak_skill"]


def test_detector_dict_user_skills_supported() -> None:
    profile = {"skills": [{"name": "python", "canonical_skill_id": "python", "confidence": 0.9}]}
    assert GapDetector().detect(profile, job(["Python"])) == []


def test_detect_vector_meets_canonical_name_and_surface() -> None:
    d = GapDetector()
    user = {"user_id": "u", "skills": ["structured query language", "Amazon Web Services"]}
    gv = d.detect_vector(user, job(["SQL", "AWS", "Docker"]))
    assert set(gv.vector) == {"docker"}  # SQL and AWS are held: no gap; ids of the job's skills are unchanged
    gv_low = d.detect_vector({"user_id": "u", "skills": ["python"]}, job(["Python", "Go"]))
    assert set(gv_low.vector) == {"go"}


def test_detect_vector_weak_and_transferable_map_onto_job_ids() -> None:
    user = {"skills": ["structured query language", "Ruby"], "weak_skills": ["structured query language"], "transferable_skills": {"python": "Ruby"}}
    gv = GapDetector().detect_vector(user, job(["SQL", "Python"]))
    types = {g.canonical_skill_id: g.gap_type for g in gv.gaps}
    assert types == {"sql": "weak_skill", "python": "covered_by_transferable_skill"}


def test_detect_vector_without_aliases_matches_legacy_engine_on_exact_input() -> None:
    from skillllm_v2.recommend.gap_vector import GapEngine

    user, j = {"user_id": "u", "skills": ["Python"]}, job(["Python", "AWS"], ["SQL"])
    assert GapDetector().detect_vector(user, j).vector == GapEngine().compute(user, j).vector


# ---------------------------------------------------------------------------------------------- extractor degradation
class Boom:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("extractor unavailable")

    def load(self) -> None:
        return None

    def is_model_loaded(self) -> bool:
        return True

    def extract(self, text: str) -> list[dict[str, Any]]:
        raise self.exc


class FakeAnchor:
    def load(self) -> None:
        return None

    def is_model_loaded(self) -> bool:
        return True

    def extract(self, text: str) -> list[dict[str, Any]]:
        start = text.find("Python")
        return [{"surface_form": "Python", "label": "HARD_SKILL", "start": start, "end": start + 6, "source": "mistral", "confidence": 0.86,
                 "evidence": "Python", "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {}}]


def make(**kw: Any) -> SkillLLMV2Pipeline:
    cfg = {"inference": {"anchor": {"mode": "stub"}, "gliner": {"enabled": False}, "courses": {"port": "null"}}}
    return SkillLLMV2Pipeline(cfg, root=ROOT, **kw)


TEXT = "I know Python and SQL."


def test_dictionary_failure_degrades_and_keeps_anchor_output() -> None:
    result = make(mistral=FakeAnchor(), dictionary=Boom()).extract_candidates(TEXT)
    rec = result.trace.get("extract")
    assert rec.status == "degraded" and "dictionary" in rec.provenance["extractor_failures"]
    assert "RuntimeError: extractor unavailable" in rec.provenance["extractor_failures"]["dictionary"]
    assert any("dictionary failed" in w for w in result.trace.warnings) and result.trace.degraded
    assert [s["surface_form"] for s in result.enriched_skills] == ["Python"]  # anchor output kept, nothing fabricated


def test_anchor_failure_degrades_and_keeps_dictionary_output() -> None:
    result = make(mistral=Boom()).extract_candidates(TEXT)
    rec = result.trace.get("extract")
    assert rec.status == "degraded" and set(rec.provenance["extractor_failures"]) == {"mistral"}
    assert result.trace.anchor_mode == "none"
    assert {s["surface_form"].lower() for s in result.enriched_skills} >= {"python", "sql"}


def test_gliner_failure_is_recorded_per_extractor() -> None:
    cfg = {"inference": {"anchor": {"mode": "none"}, "gliner": {"enabled": True}, "courses": {"port": "null"}}}
    result = SkillLLMV2Pipeline(cfg, root=ROOT, gliner=Boom(ValueError("hub offline"))).extract_candidates(TEXT)
    rec = result.trace.get("extract")
    assert rec.status == "degraded" and "ValueError: hub offline" in rec.provenance["extractor_failures"]["gliner"]
    assert result.enriched_skills  # dictionary still produced candidates


def test_all_extractors_failing_returns_empty_failed_extraction() -> None:
    cfg = {"inference": {"anchor": {"mode": "stub"}, "gliner": {"enabled": True}, "courses": {"port": "null"}}}
    pipe = SkillLLMV2Pipeline(cfg, root=ROOT, mistral=Boom(), gliner=Boom(), dictionary=Boom())
    result = pipe.extract_candidates(TEXT)
    rec = result.trace.get("extract")
    assert result.enriched_skills == [] and result.skills == []
    assert rec.status == "failed" and rec.error and "extractor unavailable" in rec.error
    assert set(rec.provenance["extractor_failures"]) == {"mistral", "gliner", "dictionary"}


def test_strict_mode_reraises_extractor_failures() -> None:
    with pytest.raises(RuntimeError, match="extractor unavailable"):
        make(mistral=FakeAnchor(), dictionary=Boom(), strict_extractors=True).extract_candidates(TEXT)
    with pytest.raises(RuntimeError, match="extractor unavailable"):
        make(mistral=Boom(), strict_extractors=True).extract_candidates(TEXT)


def test_full_run_survives_extractor_failure_and_reports_it() -> None:
    result = make(mistral=FakeAnchor(), dictionary=Boom()).run(TEXT, JOB_FOR_RUN)
    assert result.trace.get("extract").status == "degraded"
    assert result.match is not None and result.match.job_match["missing_skills"] == ["Amazon Web Services"]


JOB_FOR_RUN = {"job_id": "j1", "title": "T", "required_skills": ["Python", "AWS"], "preferred_skills": []}


def test_wrapper_extract_degrades_warns_and_exposes_trace() -> None:
    wrapper = SkillLLMv2Pipeline(dictionary=Boom())
    with pytest.warns(RuntimeWarning, match="dictionary: RuntimeError: extractor unavailable"):
        out = wrapper.extract("I know Python")
    assert isinstance(out, list)
    assert wrapper.last_trace.get("extract").status == "degraded"


def test_wrapper_extract_all_failed_is_empty_and_strict_raises() -> None:
    wrapper = SkillLLMv2Pipeline(mistral=Boom(), gliner=Boom(), dictionary=Boom())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert wrapper.extract("I know Python") == []
    assert any("FAILED" in str(w.message) for w in caught)
    assert wrapper.last_trace.get("extract").status == "failed"
    with pytest.raises(RuntimeError, match="extractor unavailable"):
        wrapper.extract("I know Python", strict=True)


# ---------------------------------------------------------------------------------------------- pipeline wiring
def test_pipeline_match_stage_meets_canonical_name_and_surface() -> None:
    pipe = make(mistral=FakeAnchor())
    skills = [{"surface_form": "SQL", "label": "HARD_SKILL", "canonical_skill_name": "structured query language",
               "canonical_skill_id": "structured-query-language", "confidence": 0.9, "source": "mistral"}]
    artifact = pipe.stage_match("u", skills, [{"job_id": "j", "required_skills": ["SQL", "Docker"], "preferred_skills": []}])
    assert artifact.job_match["missing_skills"] == ["Docker"]
    assert {g["skill"] for g in artifact.gaps} == {"Docker"}
    assert set(artifact.gap_vector["vector"]) == {"docker"}
    assert artifact.user_profile["skill_entries"][0]["surface_form"] == "SQL"
