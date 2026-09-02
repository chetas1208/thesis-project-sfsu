from __future__ import annotations

from pathlib import Path

from skillllm_v2.extract import CandidateMerger, DictionaryExtractor
from skillllm_v2.graph import SkillGraphBuilder, SkillGraphRetriever
from skillllm_v2.graph.schema import EdgeType, NodeType
from skillllm_v2.match import CourseRecommender, GapDetector, JobMatcher
from skillllm_v2.normalize import TaxonomyLinker
from skillllm_v2.pipeline import SkillLLMv2Pipeline
from skillllm_v2.schema import SkillCandidate, validate_skill
from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator


def test_schema_validation_and_offsets() -> None:
    skill = SkillCandidate("Python", "HARD_SKILL", 0, 6, "mistral", 0.9, "Python").to_dict()
    validate_skill(skill)


def test_dictionary_preserves_offsets() -> None:
    text = "We need Python and communication."
    skills = DictionaryExtractor().extract(text)
    python = next(skill for skill in skills if skill["surface_form"] == "Python")
    assert text[python["start"] : python["end"]] == "Python"


def test_candidate_merger_alias_equivalence() -> None:
    candidates = [
        SkillCandidate("AWS", "HARD_SKILL", 0, 3, "mistral", 0.8, "AWS").to_dict(),
        SkillCandidate("Amazon Web Services", "HARD_SKILL", 10, 29, "dictionary", 0.82, "Amazon Web Services").to_dict(),
    ]
    merged = CandidateMerger().merge(candidates)
    assert len(merged) == 1
    assert set(merged[0]["metadata"]["sources"]) == {"mistral", "dictionary"}


def test_boundary_refinement_requires_alias_evidence_for_edits() -> None:
    refiner = BoundaryRefiner()
    text = "Python programming and strong communication skills"
    py = SkillCandidate("Python programming", "HARD_SKILL", 0, 18, "mistral", 0.9, "Python programming").to_dict()
    comm_start = text.index("strong")
    comm = SkillCandidate("strong communication skills", "SOFT_SKILL", comm_start, len(text), "mistral", 0.9, "strong communication skills").to_dict()
    preserved = refiner.refine(text, [py])
    assert [item["surface_form"] for item in preserved] == [py["surface_form"]]
    assert preserved[0]["metadata"]["boundary_edit"]["edit_type"] == "original"

    refined = refiner.refine(text, [comm])
    assert [item["surface_form"] for item in refined] == ["communication"]
    assert refined[0]["metadata"]["boundary_edit"]["edit_type"] == "taxonomy_normalized"


def test_boundary_refinement_preserves_conjunction_without_evidence() -> None:
    text = "team collaboration and leadership"
    candidate = SkillCandidate(text, "SOFT_SKILL", 0, len(text), "gliner", 0.7, text).to_dict()
    refined = BoundaryRefiner().refine(text, [candidate])
    assert [item["surface_form"] for item in refined] == [text]
    assert refined[0]["metadata"]["boundary_edit"]["edit_type"] == "original"


def test_confidence_calibration() -> None:
    candidate = SkillCandidate("Python", "HARD_SKILL", 0, 6, "dictionary", 0.5, "Python").to_dict()
    assert not ConfidenceCalibrator().accept(candidate)
    candidate["metadata"] = {"sources": ["mistral", "dictionary"]}
    assert ConfidenceCalibrator().accept(candidate)


def test_taxonomy_linking() -> None:
    match = TaxonomyLinker().link("AWS")
    assert match is not None
    assert match["canonical_skill_name"] == "Amazon Web Services"


def test_graph_construction_and_retrieval() -> None:
    builder = SkillGraphBuilder()
    builder.add_node("u1", NodeType.USER)
    builder.add_node("python", NodeType.SKILL)
    builder.add_edge("u1", "python", EdgeType.HAS_SKILL)
    retriever = SkillGraphRetriever(builder.graph)
    assert retriever.skills_for_user("u1") == ["python"]


def test_job_gap_and_course_flow() -> None:
    profile = {"skills": ["Python"], "weak_skills": ["Python"]}
    job = {"job_id": "j1", "required_skills": ["Python", "AWS"], "preferred_skills": ["SQL"]}
    match = JobMatcher().score(profile, job)
    gaps = GapDetector().detect(profile, job)
    courses = CourseRecommender().recommend(gaps, [{"course_id": "c1", "course_name": "AWS", "teaches_skills": ["AWS"], "prerequisite_skills": ["Python"]}], profile)
    assert match["job_id"] == "j1"
    assert any(gap["gap_type"] == "missing_critical" for gap in gaps)
    assert courses[0]["course_id"] == "c1"


def test_smoke_pipeline_outputs_traceable_skills() -> None:
    skills = SkillLLMv2Pipeline().extract("Python, SQL, AWS, communication, and leadership.")
    assert skills
    for skill in skills:
        validate_skill(skill)
        assert skill["evidence"]
        assert skill["start"] is not None
        assert skill["end"] is not None
        assert skill["canonical_skill_name"] or skill["canonical_skill_id"] or skill["metadata"]


def test_skillspan_dataset_exists() -> None:
    assert Path("data/source/skillspan_hf/train.json").exists()
