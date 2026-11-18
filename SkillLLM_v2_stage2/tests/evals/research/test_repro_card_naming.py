"""Evaluation-card helpers and naming-audit screening (offline: no network, no heavy hashing)."""

from skillllm_v2.evals.research import naming_audit as na
from skillllm_v2.evals.research import repro_card as rc


def test_module_command_from_argv():
    argv = ["/x/SkillLLM_v2_stage2/src/skillllm_v2/evals/research/extraction_full.py", "--seed", "42"]
    cmd = rc._module_command(argv)
    assert cmd.endswith("-m skillllm_v2.evals.research.extraction_full --seed 42") and cmd.startswith("PYTHONPATH=src")
    assert rc._module_command(None) == ""


def test_prompts_written_with_hashes(tmp_path):
    rows = rc.section_prompts(tmp_path)
    names = {r["name"] for r in rows}
    assert "extraction/original/system" in names and "extraction/original/user_template" in names
    orig = [r for r in rows if r["name"] == "extraction/original/user_template"][0]
    assert orig["default"] is True and len(orig["sha256"]) == 64
    assert "{text}" in (tmp_path / "prompts" / "extraction_original_user_template.txt").read_text()


def test_seeds_and_commands_tolerate_missing_layers():
    layers = {"L01_extraction": {"meta": {"seed": 42, "argv": ["a/src/skillllm_v2/evals/research/m.py"], "utc": "t",
                                          "git": {"commit": "abcdef1234567", "dirty": True}}}, "L02_generalization": None}
    assert rc.section_seeds(layers)["layer_seeds"] == {"L01_extraction": 42}
    cmds = {c["layer"]: c for c in rc.section_commands(layers)}
    assert cmds["L02_generalization"]["status"] == "missing" and "dirty" in cmds["L01_extraction"]["git"]


def test_naming_match_kind():
    assert na.match_kind("SkillGraphLM", "SkillGraph-LM: a model") == "exact"
    assert na.match_kind("SkillGraph LM", "the skillgraph lm toolkit") == "exact"
    assert na.match_kind("SkillGraph", "SkillGraphLM rocks") == "substring"
    assert na.match_kind("SkillGraphLM", "Skill graphs for agents") == ""
    assert na.match_kind("Skill-LLM", "Skill-LLM: Repurposing General-Purpose LLMs") == "exact"


def test_naming_screen_rules():
    paper = {"status": "ok", "hits": [{"title": "SkillGraph: agents", "title_match": "exact"}]}
    assert na.screen("arxiv", paper)[0] == "high"
    assert na.screen("arxiv", {"status": "ok", "hits": [{"title": "x", "title_match": "substring"}]})[0] == "low"
    assert na.screen("arxiv", {"status": "ok", "hits": []})[0] == "none"
    assert na.screen("pypi", {"status": "ok", "hits": [{"title_match": "exact", "domain": False}]})[0] == "medium"
    assert na.screen("pypi", {"status": "ok", "hits": [{"title_match": "exact", "domain": True}]})[0] == "high"
    assert na.screen("github", {"status": "ok", "hits": [{"title_match": "exact", "stars": 3, "domain": False}]})[0] == "medium"
    assert na.screen("github", {"status": "ok", "hits": [{"title_match": "exact", "stars": 80, "domain": False}]})[0] == "high"
    assert na.screen("dblp", {"status": "blocked: bot challenge"})[0] == "unknown"


def test_naming_summary_takes_max_and_prefers_final_risk():
    audit = {"records": [{"term": "T", "source": "arxiv", "screening_risk": "low"},
                         {"term": "T", "source": "github", "screening_risk": "unknown"}],
             "manual_records": [{"term": "T", "source": "web", "screening_risk": "low", "final_risk": "medium"}]}
    s = na.summarise(audit)["T"]
    assert s["max_risk"] == "medium" and s["unknown_sources"] == ["github"]


def test_naming_finalize_and_report(tmp_path):
    """finalize() merges raw + manual + assessment, refuses a final risk below screening without override_reason, and the
    rendered report carries the verdict, the per-query manual records and the alternatives."""
    import json

    import pytest

    from skillllm_v2.evals.research import naming_audit as na

    rec = {"date_utc": "2026-11-18T00:00:00+00:00", "term": "FooLM", "source": "arxiv", "query_url": "u", "status": "ok",
           "hit_count": 1, "relevant_hits": [{"title": "FooLM paper", "url": "https://x"}], "screening_risk": "high",
           "rationale": "exact", "note": ""}
    (tmp_path / "naming_audit_raw.json").write_text(json.dumps({"meta": {"date_utc": "t", "screening_rule": "rule"}, "records": [rec]}))
    (tmp_path / "manual_queries.json").write_text(json.dumps([
        {"date_utc": "2026-11-18T01:00:00+00:00", "term": "FooLM", "source": "web_search", "query": "q", "status": "ok",
         "hit_count": 0, "screening_risk": "none", "final_risk": "low", "rationale": "nothing", "relevant_hits": []}]))
    bad = {"terms": {"FooLM": {"final_risk": "low", "why": "w"}}, "verdict": {"recommendation": "ADOPT"}}
    (tmp_path / "assessment.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        na.finalize(tmp_path)
    good = {"terms": {"FooLM": {"final_risk": "high", "why": "w"}}, "decision_rule": "rule",
            "verdict": {"recommendation": "ADOPT WITH CAVEAT", "summary": "s", "caveats": ["c1"]}}
    (tmp_path / "assessment.json").write_text(json.dumps(good))
    f = na.finalize(tmp_path)
    assert f["terms"]["FooLM"]["screening_max_risk"] == "high"
    rep = na.render_report(tmp_path)
    assert "ADOPT WITH CAVEAT" in rep and "web_search" in rep and "c1" in rep
