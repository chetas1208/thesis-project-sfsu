"""Wiring tests for skillllm_v2.pipeline.inference.SkillLLMV2Pipeline (CPU only, no model weights)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from skillllm_v2.io_utils import read_json, read_jsonl
from skillllm_v2.pipeline.course_port import (
    CourseRecommendationPipelineAdapter,
    CourseRecommendationRequest,
    CourseRecommenderPort,
    NullCourseRecommender,
    build_course_recommender,
)
from skillllm_v2.pipeline.inference import COMPONENT_SWITCHES, DeterministicStubExtractor, SkillLLMV2Pipeline, main
from skillllm_v2.pipeline.inference_types import (
    STAGE_ORDER,
    CandidateArtifact,
    CourseArtifact,
    ExplanationArtifact,
    ExtractArtifact,
    ExtractionResult,
    GateArtifact,
    GraphArtifact,
    MatchArtifact,
    PipelineResult,
    ProfileArtifact,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_TEXT = "I know Python, SQL, Docker, communication, and machine learning."
JOB = {"job_id": "ml-platform", "title": "ML Platform", "required_skills": ["Python", "AWS"], "preferred_skills": ["Docker", "communication"],
       "semantic_role_fit": 0.8, "experience_level_fit": 0.8, "user_preference_fit": 0.8}


def make_pipeline(anchor: str = "stub", **inference: Any) -> SkillLLMV2Pipeline:
    cfg: dict[str, Any] = {"inference": {"anchor": {"mode": anchor}, "gliner": {"enabled": False}, "courses": {"port": "null"}}}
    for key, value in inference.items():
        cfg["inference"].setdefault(key, {}).update(value) if isinstance(value, dict) else cfg["inference"].__setitem__(key, value)
    return SkillLLMV2Pipeline(cfg, root=ROOT)


class FakeAnchor:
    """Stands in for a loaded Mistral model: returns fixed anchor spans and reports weights as loaded."""

    def __init__(self, surfaces: tuple[str, ...] = ("Python", "SQL")) -> None:
        self.surfaces = surfaces

    def load(self) -> None:
        return None

    def is_model_loaded(self) -> bool:
        return True

    def extract(self, text: str) -> list[dict[str, Any]]:
        out = []
        for surface in self.surfaces:
            start = text.find(surface)
            if start >= 0:
                out.append({"surface_form": surface, "label": "HARD_SKILL", "start": start, "end": start + len(surface), "source": "mistral",
                            "confidence": 0.86, "evidence": surface, "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {}})
        return out


class RecordingPort:
    name = "recording"

    def __init__(self) -> None:
        self.requests: list[CourseRecommendationRequest] = []

    def is_available(self) -> bool:
        return True

    def recommend(self, request: CourseRecommendationRequest):
        from skillllm_v2.pipeline.course_port import CourseRecommendationResponse

        self.requests.append(request)
        recs = [{"course_id": "aws-101", "course_name": "AWS Foundations", "covered_missing_skills": [s["skill_name"] for s in request.target_job["required_skills"]][-1:]}]
        return CourseRecommendationResponse(recs, {"stages": []}, True, "RecordingPort", "test double")


# --------------------------------------------------------------------------- stage wiring
def test_stage_extract_wires_anchor_and_dictionary_with_stub_flag() -> None:
    p = make_pipeline()
    art = p.stage_extract(PROFILE_TEXT)
    assert isinstance(art, ExtractArtifact) and art.anchor_mode == "stub"
    assert art.anchor and art.dictionary and not art.gliner
    assert all(c["metadata"]["extractor_mode"] == "deterministic_stub" and c["metadata"]["stub"] for c in art.anchor)
    assert art.record.status == "degraded"
    assert art.record.provenance["per_source"]["anchor"] == len(art.anchor)


def test_stages_2_to_6_are_individually_callable_and_keep_anchor() -> None:
    p = make_pipeline()
    text = PROFILE_TEXT
    ext = p.stage_extract(text)
    merged = p.stage_merge(text, ext.candidates)
    refined = p.stage_refine(text, merged.candidates)
    verified = p.stage_verify(refined.candidates, text=text)
    calibrated = p.stage_calibrate(verified.candidates)
    linked = p.stage_link(calibrated.candidates)
    for art, name in [(merged, "merge"), (refined, "refine"), (verified, "verify"), (calibrated, "calibrate"), (linked, "link")]:
        assert isinstance(art, CandidateArtifact) and art.stage == name and art.record.name == name
        assert art.record.input_count is not None and art.record.output_count == len(art.candidates)
    assert merged.record.provenance["policy"] == "source_aware_add_only"
    anchor_spans = {(c["start"], c["end"]) for c in ext.anchor}
    assert anchor_spans <= {(c["start"], c["end"]) for c in linked.candidates}
    assert all(c["metadata"]["verification"] for c in verified.candidates)
    assert "thresholds" in calibrated.record.provenance


def test_link_is_attach_only_and_low_confidence_links_never_remove_skills() -> None:
    p = make_pipeline(taxonomy={"attach_min_confidence": 1.01})  # nothing clears the bar
    ext = p.stage_extract(PROFILE_TEXT)
    cands = p.stage_merge(PROFILE_TEXT, ext.candidates).candidates
    linked = p.stage_link(cands)
    assert len(linked.candidates) == len(cands)
    assert linked.record.provenance["links_applied"] == 0
    assert linked.record.provenance["removed_candidates"] == 0
    for before, after in zip(cands, linked.candidates):
        assert after["canonical_skill_id"] == before["canonical_skill_id"]
        link = after["metadata"].get("taxonomy_link")
        if link:
            assert link["applied"] is False
    attached = make_pipeline().stage_link(cands)
    assert len(attached.candidates) == len(cands)
    assert any(c["canonical_skill_name"] for c in attached.candidates)


def test_verify_stage_reports_dropped_non_anchor_candidates() -> None:
    p = make_pipeline()
    cand = {"surface_form": "experience", "label": "HARD_SKILL", "start": 0, "end": 10, "source": "dictionary", "confidence": 0.8, "evidence": "experience",
            "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {}}
    art = p.stage_verify([cand])
    assert art.candidates == [] and art.dropped[0]["reason"] == "generic_phrase"


def test_learned_verifier_absent_degrades_gracefully(tmp_path: Path) -> None:
    p = make_pipeline(verify={"learned": {"enabled": True, "model_path": str(tmp_path / "missing.joblib")}})
    ext = p.stage_extract(PROFILE_TEXT)
    art = p.stage_verify(ext.candidates, text=PROFILE_TEXT)
    assert art.record.status == "degraded"
    assert art.record.provenance["learned_verifier"]["available"] is False
    assert len(art.candidates) >= len(ext.anchor)


def test_learned_verifier_real_artifact_never_rejects_anchor() -> None:
    model = ROOT / "artifacts/skillllm_v2/span_verifier/model/verifier_model.joblib"
    if not model.exists():
        pytest.skip("trained verifier artifact not present")
    p = make_pipeline(verify={"learned": {"enabled": True, "model_path": str(model), "threshold": 0.999}})
    if not p.learned_verifier.available:
        pytest.skip(f"verifier not loadable: {p.learned_verifier.unavailable_reason}")
    ext = p.stage_extract(PROFILE_TEXT)
    art = p.stage_verify(p.stage_merge(PROFILE_TEXT, ext.candidates).candidates, text=PROFILE_TEXT)
    anchor_n = sum(1 for c in art.candidates if "mistral" in (c["metadata"].get("sources") or [c["source"]]))
    assert anchor_n >= 1
    assert all("learned_verifier" in c["metadata"] for c in art.candidates)


# --------------------------------------------------------------------------- degrade paths
def test_anchor_none_is_dictionary_only_and_flagged() -> None:
    res = make_pipeline("none").run_extraction_only(PROFILE_TEXT)
    assert res.trace.anchor_mode == "none" and res.trace.degraded
    assert not res.trace.stub_components
    assert res.extract.anchor == [] and res.skills
    assert res.gate.basis == "not_applicable" and res.gate.decision == "accept"


def test_auto_without_weights_does_not_silently_use_stub() -> None:
    p = SkillLLMV2Pipeline({"inference": {"gliner": {"enabled": False}, "courses": {"port": "null"}}}, root=ROOT)  # anchor auto, allow_stub default
    res = p.run_extraction_only(PROFILE_TEXT)
    assert res.trace.anchor_mode == "none" and not res.trace.stub_components and res.trace.degraded


def test_auto_with_allow_stub_uses_flagged_stub() -> None:
    p = SkillLLMV2Pipeline({"inference": {"anchor": {"allow_stub": True}, "gliner": {"enabled": False}, "courses": {"port": "null"}}}, root=ROOT)
    res = p.run_extraction_only(PROFILE_TEXT)
    assert res.trace.anchor_mode == "stub" and "anchor:deterministic_stub" in res.trace.stub_components


def test_explicit_mistral_mode_without_weights_raises() -> None:
    cfg = {"model": {"model_path": "/nonexistent/mistral"}, "inference": {"anchor": {"mode": "mistral"}, "courses": {"port": "null"}}}
    with pytest.raises(FileNotFoundError):
        SkillLLMV2Pipeline(cfg, root=ROOT).run_extraction_only(PROFILE_TEXT)


def test_missing_gliner_is_reported_not_faked() -> None:
    p = SkillLLMV2Pipeline({"inference": {"anchor": {"mode": "stub"}, "gliner": {"enabled": True}, "courses": {"port": "null"}}}, root=ROOT)
    art = p.stage_extract("Teamwork and Python.", trace=None)
    try:
        import gliner  # noqa: F401
        pytest.skip("gliner installed")
    except ImportError:
        pass
    assert art.gliner == []
    assert any("gliner" in note for note in art.record.notes)


def test_anchor_confidence_is_constant_unless_hook_supplied() -> None:
    p = make_pipeline()
    art = p.stage_extract(PROFILE_TEXT)
    assert any("constant" in n for n in art.record.notes) and art.record.provenance["anchor_confidence"] == "constant"
    hooked = SkillLLMV2Pipeline(
        {"inference": {"anchor": {"mode": "stub"}, "gliner": {"enabled": False}, "courses": {"port": "null"}}},
        root=ROOT,
        confidence_hook=lambda text, cands: [dict(c, confidence=0.5) for c in cands],
    )
    art2 = hooked.stage_extract(PROFILE_TEXT)
    assert {c["confidence"] for c in art2.anchor} == {0.5} and art2.record.provenance["anchor_confidence"] == "hook"
    lp = SkillLLMV2Pipeline({"inference": {"anchor": {"mode": "stub", "confidence": "logprob"}, "gliner": {"enabled": False}, "courses": {"port": "null"}}}, root=ROOT)
    trace = lp._new_trace()
    lp.stage_extract(PROFILE_TEXT, trace=trace)
    assert any("NOT IMPLEMENTED" in w for w in trace.warnings)


# --------------------------------------------------------------------------- stage 7: fallback to anchor
def test_gate_accepts_wrapper_when_structural_guard_passes_for_stub() -> None:
    res = make_pipeline().run_extraction_only(PROFILE_TEXT)
    assert isinstance(res.gate, GateArtifact) and res.gate.decision == "accept"
    assert res.gate.basis == "structural_only" and res.gate.anchor_retention == 1.0
    assert res.skills == res.enriched_skills


def test_gate_forced_anchor_returns_untouched_anchor_but_keeps_enriched() -> None:
    p = SkillLLMV2Pipeline({"inference": {"gliner": {"enabled": False}, "gate": {"mode": "anchor"}, "courses": {"port": "null"}}}, root=ROOT, mistral=FakeAnchor())
    res = p.run_extraction_only(PROFILE_TEXT)
    assert res.trace.anchor_mode == "model" and res.gate.fell_back
    raw = FakeAnchor().extract(PROFILE_TEXT)
    # untouched anchor: same spans/labels/confidences, no link, no boundary edits; only the provenance tag was added at extraction
    assert res.skills == res.extract.anchor == [dict(c, metadata={"extractor_mode": "model"}) for c in raw]
    assert all(not s["canonical_skill_name"] for s in res.skills)
    assert any(s["canonical_skill_name"] for s in res.enriched_skills) and len(res.enriched_skills) >= len(res.skills)
    assert res.gate.selected is not res.gate.anchor  # defensive copy


def test_gate_offline_artifact_decision_is_honoured(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"final_decision": "fallback_to_anchor", "final_variant": "full_cag_graph",
                                  "gate_results": [{"variant": "full_cag_graph", "decision": "fallback_to_anchor", "reason": "exact F1 below floor"}]}))
    cfg = {"inference": {"gliner": {"enabled": False}, "gate": {"status_path": str(status)}, "courses": {"port": "null"}}}
    res = SkillLLMV2Pipeline(cfg, root=ROOT, mistral=FakeAnchor()).run_extraction_only(PROFILE_TEXT)
    assert res.gate.fell_back and res.gate.basis == "offline_artifact" and "below floor" in res.gate.reason
    status.write_text(json.dumps({"final_decision": "accept", "final_variant": "full_cag_graph", "gate_results": []}))
    res2 = SkillLLMV2Pipeline(cfg, root=ROOT, mistral=FakeAnchor()).run_extraction_only(PROFILE_TEXT)
    assert res2.gate.decision == "accept" and res2.gate.basis == "offline_artifact"


def test_gate_missing_offline_status_defaults_to_anchor(tmp_path: Path) -> None:
    cfg = {"inference": {"gliner": {"enabled": False}, "gate": {"status_path": str(tmp_path / "nope.json")}, "courses": {"port": "null"}}}
    res = SkillLLMV2Pipeline(cfg, root=ROOT, mistral=FakeAnchor()).run_extraction_only(PROFILE_TEXT)
    assert res.gate.fell_back and res.gate.basis == "missing_status"


def test_gate_reuses_evaluate_gate_with_explicit_metrics() -> None:
    p = SkillLLMV2Pipeline({"inference": {"gliner": {"enabled": False}, "courses": {"port": "null"}}}, root=ROOT, mistral=FakeAnchor())
    anchor_m = {"exact_span_f1": 0.67, "soft_skill_f1": 0.60}
    bad = p.run_extraction_only(PROFILE_TEXT, candidate_metrics={"exact_span_f1": 0.50, "soft_skill_f1": 0.60}, anchor_metrics=anchor_m)
    assert bad.gate.fell_back and bad.gate.basis == "metrics" and bad.gate.gate_detail["decision"] == "fallback_to_anchor"
    good = p.run_extraction_only(PROFILE_TEXT, candidate_metrics={"exact_span_f1": 0.671, "soft_skill_f1": 0.60}, anchor_metrics=anchor_m)
    assert good.gate.decision == "accept" and good.gate.basis == "metrics"


def test_structural_guard_falls_back_when_wrapper_drops_anchor_spans(tmp_path: Path) -> None:
    class DropAll:
        protect_anchor_sources = frozenset({"mistral"})
        hard_skill_threshold = soft_skill_threshold = dictionary_only_threshold = multi_source_threshold = 0.0

        def calibrate(self, candidates):
            return []

        def threshold_for(self, c):
            return 0.0

    p = SkillLLMV2Pipeline({"inference": {"gliner": {"enabled": False}, "gate": {"mode": "wrapper"}, "courses": {"port": "null"}}}, root=ROOT,
                           mistral=FakeAnchor(), calibrator=DropAll())
    res = p.run_extraction_only(PROFILE_TEXT)
    assert res.enriched_skills == [] and res.gate.fell_back and "structural_guard" in res.gate.basis
    assert res.gate.anchor_retention == 0.0 and len(res.skills) == 2


# --------------------------------------------------------------------------- stages 8-12
def test_profile_graph_match_stages_standalone() -> None:
    p = make_pipeline()
    ext = p.run_extraction_only(PROFILE_TEXT)
    jobs = p._prepare_jobs([JOB], set(STAGE_ORDER), set(), ext.trace)
    target = jobs[0]
    assert target["required_skills"] == ["Python", "Amazon Web Services"]  # canonicalized through AliasResolver
    profile = p.stage_profile(PROFILE_TEXT, ext.enriched_skills, user_id="u1", jobs=jobs, target_job=target)
    assert isinstance(profile, ProfileArtifact) and profile.profile_summary["user_id"] == "u1"
    assert profile.skill_memory["skill_count"] == len(profile.profile_summary["skills"])
    assert profile.context_packet["content_hash"] and profile.retrieved_jobs[0]["record"]["job_id"] == "ml-platform"
    assert profile.record.provenance["retrieved_skills"] == len(profile.retrieved_skills)
    graph = p.stage_graph("u1", ext.enriched_skills, jobs, target)
    assert isinstance(graph, GraphArtifact) and graph.graph_missing_skills == ["Amazon Web Services"]
    kinds = {e["skill"]: e["kind"] for e in graph.explanations}
    assert kinds["Python"] == "has" and kinds["Amazon Web Services"] == "missing"
    assert graph.explanations[0]["path_names"] == ["u1", "Python", "ML Platform"]
    match = p.stage_match("u1", ext.enriched_skills, jobs)
    assert isinstance(match, MatchArtifact) and match.job_match["missing_skills"] == ["Amazon Web Services"]
    assert match.gaps[0]["gap_type"] == "missing_critical" and match.gap_vector["critical_count"] == 1


def test_courses_go_through_port_only_and_receive_canonical_request() -> None:
    port = RecordingPort()
    assert isinstance(port, CourseRecommenderPort)
    p = SkillLLMV2Pipeline({"inference": {"anchor": {"mode": "stub"}, "gliner": {"enabled": False}}}, root=ROOT, course_port=port)
    res = p.run(PROFILE_TEXT, JOB, user_id="u1")
    assert len(port.requests) == 1
    request = port.requests[0]
    assert request.user_id == "u1" and request.target_job["job_id"] == "ml-platform"
    assert [s["skill_name"] for s in request.target_job["required_skills"]] == ["Python", "Amazon Web Services"]
    assert all(s.get("canonical_skill_id") for s in request.target_job["required_skills"])
    assert request.gaps and request.gaps[0]["skill"] == "Amazon Web Services"
    assert isinstance(res.courses, CourseArtifact) and res.courses.recommendations[0]["course_name"] == "AWS Foundations"
    assert res.graph.course_paths[0]["missing_skill"] == "Amazon Web Services"  # TEACHES edges read back via SkillGraphRetriever
    assert any("AWS Foundations" in step for step in res.explanation.explanation["recommended_learning_path"])


def test_pipeline_module_does_not_import_course_or_recommend_directly() -> None:
    src = (ROOT / "src/skillllm_v2/pipeline/inference.py").read_text()
    for banned in ("skillllm_v2.course", "skillllm_v2.recommend", "course_recommendation_pipeline", "CourseRecommender("):
        assert banned not in src.replace("skillllm_v2.pipeline.course_port", "")


def test_null_port_is_clearly_labelled_and_degrades() -> None:
    null = NullCourseRecommender()
    assert isinstance(null, CourseRecommenderPort) and null.is_available() is False
    res = make_pipeline().run(PROFILE_TEXT, JOB)
    assert res.courses.available is False and res.courses.recommendations == [] and "NULL IMPLEMENTATION" in res.courses.note
    assert res.trace.get("courses").status == "degraded"
    assert any("NULL IMPLEMENTATION" in lim for lim in res.explanation.explanation["limitations"])


def test_adapter_delegates_to_pipeline_without_modifying_it() -> None:
    class FakeCourseRecommendationPipeline:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any, int]] = []

        def run(self, user_input, job_input, top_k=5):
            self.calls.append((user_input, job_input, top_k))
            return {"recommendations": [{"course_id": "c1", "title": "Course One"}], "learning_path": {"stages": []}, "skill_gap_analysis": {"gaps": []}}

    fake = FakeCourseRecommendationPipeline()
    adapter = CourseRecommendationPipelineAdapter(pipeline=fake)
    assert isinstance(adapter, CourseRecommenderPort) and adapter.is_available()
    response = adapter.recommend(CourseRecommendationRequest("u", {"user_id": "u", "skills": []}, {"job_id": "j"}, [], top_k=3))
    assert fake.calls[0][2] == 3 and response.available and response.recommendations[0]["course_name"] == "Course One"


def test_adapter_unavailable_returns_labelled_empty_response(tmp_path: Path) -> None:
    adapter = CourseRecommendationPipelineAdapter(catalog_path=tmp_path / "missing.jsonl")
    assert adapter.is_available() is False
    response = adapter.recommend(CourseRecommendationRequest("u", {}, {"job_id": "j"}))
    assert not response.available and response.recommendations == [] and "UNAVAILABLE" in response.note
    assert isinstance(build_course_recommender("auto", tmp_path / "missing.jsonl"), NullCourseRecommender)
    assert isinstance(build_course_recommender("adapter"), CourseRecommendationPipelineAdapter)
    with pytest.raises(ValueError):
        build_course_recommender("bogus")


def test_real_adapter_delegates_to_existing_course_pipeline() -> None:
    catalog = ROOT / "inputs/demo/course_catalog.jsonl"
    p = SkillLLMV2Pipeline({"inference": {"anchor": {"mode": "stub"}, "gliner": {"enabled": False}}}, root=ROOT,
                           course_port=CourseRecommendationPipelineAdapter(catalog_path=catalog))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = p.run(PROFILE_TEXT, JOB)
    assert res.courses.available and res.courses.port == "course_recommendation_pipeline"
    assert res.courses.note == "delegated to CourseRecommendationPipeline.run"
    assert res.trace.get("courses").provenance["implementation"] == "CourseRecommendationPipeline"


# --------------------------------------------------------------------------- end to end on inputs/demo
def test_end_to_end_on_demo_inputs_with_stub_extractor(tmp_path: Path) -> None:
    profile = read_json(ROOT / "inputs/demo/user_profile.json")
    jobs = read_jsonl(ROOT / "inputs/demo/jobs.jsonl")
    res = make_pipeline().run(profile, jobs)
    assert isinstance(res, PipelineResult) and isinstance(res.extraction, ExtractionResult)
    assert res.user_id == "demo-user"
    names = {s["surface_form"].lower() for s in res.skills}
    assert {"python", "sql", "docker", "communication", "machine learning"} <= names
    assert all(isinstance(a, t) for a, t in [(res.profile, ProfileArtifact), (res.graph, GraphArtifact), (res.match, MatchArtifact),
                                              (res.courses, CourseArtifact), (res.explanation, ExplanationArtifact)])
    assert {m["job_id"] for m in res.match.matches} == {"ml-platform", "data-analyst"}
    assert res.match.target_job["job_id"] == "ml-platform"  # best score among the two demo jobs
    assert res.explanation.explanation["missing_skills"] == ["Amazon Web Services"]
    assert res.explanation.explanation["extraction_policy"]["anchor_mode"] == "stub"
    assert any("STUB" in lim for lim in res.explanation.explanation["limitations"])
    out = json.loads(json.dumps(res.to_dict()))
    assert out["trace"]["stub_components"] == ["anchor:deterministic_stub"]


def test_job_given_as_raw_text_is_extracted_and_traced() -> None:
    res = make_pipeline().run(PROFILE_TEXT, "Machine Learning Engineer\nRequires Python, Docker and AWS. Strong communication.")
    assert res.trace.get("job_extract") is not None
    job = res.match.target_job
    assert job["input_form"] == "text_extracted" and "Python" in job["required_skills"]
    assert res.match.record.notes  # no fit fields on text jobs is stated, not hidden


def test_target_job_id_and_multiple_jobs() -> None:
    jobs = [JOB, {"job_id": "data-analyst", "required_skills": ["SQL", "Excel"], "preferred_skills": []}]
    p = make_pipeline()
    assert p.run(PROFILE_TEXT, jobs, target_job_id="data-analyst").match.target_job["job_id"] == "data-analyst"
    with pytest.raises(ValueError):
        p.run(PROFILE_TEXT, jobs, target_job_id="nope")


# --------------------------------------------------------------------------- trace / provenance / switches
def test_trace_is_complete_timed_and_serializable() -> None:
    res = make_pipeline().run(PROFILE_TEXT, JOB)
    names = res.trace.stage_names
    assert [n for n in names if n != "job_extract"] == list(STAGE_ORDER)
    for record in res.trace.stages:
        assert record.component and record.duration_seconds >= 0.0
        assert record.status in {"ok", "degraded", "skipped", "failed"}
        if record.name in {"extract", "merge", "refine", "verify", "calibrate", "link", "gate", "profile", "graph", "match", "courses", "explain"}:
            assert record.provenance, f"{record.name} has no provenance"
    assert res.trace.total_seconds >= sum(r.duration_seconds for r in res.trace.stages) * 0.5
    d = res.trace.to_dict()
    assert d["run_id"] and d["config_hash"] and d["pipeline_version"] and set(d["timings"]) == set(names)
    json.dumps(res.to_dict())


def test_every_stage_can_be_skipped_and_is_still_traced() -> None:
    p = make_pipeline()
    for stage in STAGE_ORDER:
        if stage == "extract":
            with pytest.raises(ValueError):
                p.run(PROFILE_TEXT, JOB, skip=[stage])
            continue
        res = p.run(PROFILE_TEXT, JOB, skip=[stage])
        assert res.trace.get(stage).status == "skipped", stage
        assert res.skills, stage


def test_skipping_extraction_stages_passes_candidates_through() -> None:
    p = make_pipeline()
    full = p.run_extraction_only(PROFILE_TEXT)
    skipped = p.run_extraction_only(PROFILE_TEXT, skip=["refine", "verify", "calibrate", "link"])
    assert all(c["canonical_skill_name"] is None or "taxonomy_link" not in c["metadata"] for c in skipped.skills)
    assert any("taxonomy_link" in c["metadata"] for c in full.skills)
    assert all(skipped.trace.get(n).status == "skipped" for n in ["refine", "verify", "calibrate", "link"])


def test_component_switches_and_stage_subset() -> None:
    assert set(COMPONENT_SWITCHES) == {"mistral", "gliner", "dictionary"}
    p = make_pipeline()
    no_dict = p.run_extraction_only(PROFILE_TEXT, skip=["dictionary"])
    assert no_dict.extract.dictionary == [] and no_dict.extract.anchor
    no_anchor = p.run_extraction_only(PROFILE_TEXT, skip=["mistral"])
    assert no_anchor.extract.anchor == [] and no_anchor.trace.anchor_mode == "none"
    subset = p.run(PROFILE_TEXT, JOB, stages=["extract", "merge", "match"])
    assert subset.match is not None and subset.graph is None and subset.courses is None and subset.explanation is None
    assert subset.trace.get("graph").status == "skipped"
    with pytest.raises(ValueError):
        p.run(PROFILE_TEXT, JOB, stages=["bogus"])
    with pytest.raises(ValueError):
        p.run(PROFILE_TEXT, JOB, skip=["bogus"])


def test_no_job_skips_downstream_job_stages() -> None:
    res = make_pipeline().run(PROFILE_TEXT)
    assert res.profile is not None and res.graph is None and res.match is None and res.courses is None and res.explanation is None
    assert res.trace.get("match").notes == ["no job input"]


def test_downstream_stage_failure_is_recorded_not_fatal() -> None:
    class Boom:
        def score(self, *a, **k):
            raise RuntimeError("boom")

    p = make_pipeline()
    p.job_matcher = Boom()
    p.cfg["stages"]["graph"] = False  # _choose_target needs one job only -> no scoring
    res = p.run(PROFILE_TEXT, JOB)
    assert res.match is None and res.trace.get("match").status == "failed" and "boom" in res.trace.get("match").error
    assert res.trace.degraded and res.skills
    p.cfg["continue_on_error"] = False
    with pytest.raises(RuntimeError):
        p.run(PROFILE_TEXT, JOB)


def test_invalid_candidates_from_injected_extractor_fail_loudly() -> None:
    class Bad(FakeAnchor):
        def extract(self, text):
            return [{"surface_form": "x", "label": "BOGUS", "start": 0, "end": 1, "source": "mistral", "confidence": 0.5, "evidence": "x",
                     "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {}}]

    from skillllm_v2.pipeline.inference import PipelineStageError

    p = SkillLLMV2Pipeline({"inference": {"gliner": {"enabled": False}, "courses": {"port": "null"}}}, root=ROOT, mistral=Bad())
    with pytest.raises(PipelineStageError):
        p.run_extraction_only("x y z")


# --------------------------------------------------------------------------- configuration / wiring of orphans
def test_default_config_file_extends_stage2_config_and_is_used() -> None:
    p = SkillLLMV2Pipeline(root=ROOT, course_port=NullCourseRecommender())
    assert p.config_source and p.config_source.endswith("skillllm_v2_inference.yaml")
    assert p.base_config["model"]["name"].startswith("mistral-small")  # inherited via `extends`
    assert p.cfg["anchor"]["allow_stub"] is False and p.cfg["gate"]["on_missing_status"] == "fallback_to_anchor"
    assert p.base_config["calibration"]["hard_skill_threshold"] == p.calibrator.hard_skill_threshold == 0.45
    with pytest.raises(ValueError):
        SkillLLMV2Pipeline({"inference": {"stages": {"nonsense": True}}}, root=ROOT)


def test_esco_onet_lightcast_loaders_are_wired_through_taxonomy_registry(tmp_path: Path) -> None:
    from skillllm_v2.normalize import ChainedTaxonomyLinker

    esco = tmp_path / "esco.jsonl"
    esco.write_text(json.dumps({"canonical_skill_id": "esco:py", "canonical_skill_name": "Python (computer programming)", "aliases": ["Python"], "taxonomy": "ESCO"}) + "\n")
    p = SkillLLMV2Pipeline({"inference": {"anchor": {"mode": "stub"}, "gliner": {"enabled": False}, "courses": {"port": "null"},
                                          "taxonomy": {"esco_path": str(esco)}}}, root=ROOT)
    assert isinstance(p.linker, ChainedTaxonomyLinker) and p.linker.names[0] == "ESCO"
    link = p.linker.link("Python")
    assert link["canonical_skill_id"] == "esco:py" and link["taxonomy"] == "ESCO"
    res = p.run_extraction_only(PROFILE_TEXT)
    assert any(s["canonical_skill_id"] == "esco:py" for s in res.enriched_skills)


def test_policy_duplication_resolved_both_import_paths_are_the_same_objects() -> None:
    import skillllm_v2.policy as policy
    import skillllm_v2.policy.source_aware_policy as impl
    import skillllm_v2.source_aware_policy as root_shim

    assert root_shim.is_anchor_candidate is policy.is_anchor_candidate is impl.is_anchor_candidate
    assert root_shim.ANCHOR_SOURCES is impl.ANCHOR_SOURCES
    src = (ROOT / "src/skillllm_v2/source_aware_policy.py").read_text()
    assert "def " not in src  # shim only
    from skillllm_v2.schema import SkillCandidate as A
    from skillllm_v2.schemas import SkillCandidate as B

    assert A is B


def test_load_config_resolves_extends(tmp_path: Path) -> None:
    from skillllm_v2.config import load_config

    parent = tmp_path / "parent.yaml"
    parent.write_text("a: {x: 1, y: 2}\nb: 3\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\na: {y: 9}\n")
    assert load_config(child) == {"a": {"x": 1, "y": 9}, "b": 3}


# --------------------------------------------------------------------------- legacy wrapper compatibility
def test_legacy_wrapper_delegates_keeps_api_and_flags_stub() -> None:
    from skillllm_v2.wrapper import SkillLLMv2Pipeline

    legacy = SkillLLMv2Pipeline()
    for attr in ("mistral", "gliner", "dictionary", "merger", "refiner", "verifier", "calibrator", "linker"):
        assert hasattr(legacy, attr)
    with pytest.warns(RuntimeWarning, match="STUB"):
        skills = legacy.extract("Python, SQL, AWS, communication, and leadership.")
    assert isinstance(skills, list) and skills
    assert any(s["metadata"].get("extractor_mode") == "deterministic_stub" for s in skills)
    assert all("canonical_skill_name" in s for s in skills)
    unlinked = legacy.extract("Python and SQL", normalize=False)
    assert all("taxonomy_link" not in s["metadata"] for s in unlinked)
    e2e = legacy.pipeline().run_extraction_only("Python and SQL")
    assert e2e.trace.anchor_mode == "stub"


def test_deterministic_stub_extractor_is_deterministic_and_flagged() -> None:
    a = DeterministicStubExtractor().extract("Python and SQL")
    b = DeterministicStubExtractor().extract("Python and SQL")
    assert [(c["surface_form"], c["start"]) for c in a] == [(c["surface_form"], c["start"]) for c in b]
    assert all(c["metadata"]["stub"] for c in a)


# --------------------------------------------------------------------------- CLI
def test_cli_writes_json_for_demo_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.json"
    code = main(["--profile", str(ROOT / "inputs/demo/user_profile.json"), "--job", str(ROOT / "inputs/demo/jobs.jsonl"), "--out", str(out),
                 "--no-mistral", "--courses", "null", "--skip", "gliner"])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["trace"]["anchor_mode"] == "stub" and payload["trace"]["degraded"]
    assert payload["match"]["target_job_id"] == "ml-platform" and payload["courses"]["available"] is False
    assert "stub" in capsys.readouterr().out
