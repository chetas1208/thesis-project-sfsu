"""Regression tests for the graph-module bugs found by the graph evals (docs/BUGFIX_LOG.md, bugs 5-11)."""

from __future__ import annotations

import logging

import networkx as nx
import pytest

from skillllm_v2.graph.graph_builder import SkillGraphBuilder
from skillllm_v2.graph.graph_retriever import SkillGraphRetriever
from skillllm_v2.graph.schema import EdgeType, NodeType
from skillllm_v2.graph.skill_graph_explainer import explain_skill_path, narrate_skill_trace
from skillllm_v2.schema import graph_key, normalize_text


def _world() -> SkillGraphBuilder:
    b = SkillGraphBuilder()
    b.add_user_skill("u1", "python", confidence=0.9, evidence_text="Built ETL in Python", source="mistral")
    b.add_user_skill("u1", "sql")
    b.add_user_skill("u2", "go")
    b.add_job_requirement("j1", "python", required=True, importance=0.8, source_sentence="You must know Python.")
    b.add_job_requirement("j1", "docker", required=False)
    b.add_course_skill("c1", "docker", learning_outcome_evidence="Deploy with Docker")
    return b


# ------------------------------------------------------------------ bug 5: user connects to job / course


def test_user_connects_to_job_through_a_shared_skill_with_true_directions():
    b = _world()
    out = explain_skill_path(b.graph, "u1", "j1")
    assert out["found"] and out["path"] == ["u1", "python", "j1"] and out["hops"] == 2
    assert [(e["relation"], e["direction"]) for e in out["edges"]] == [("HAS_SKILL", "forward"), ("REQUIRES", "reverse")]
    assert out["explanation"] == "u1 -[HAS_SKILL]-> python <-[REQUIRES]- j1"
    assert out["confidence"] == 1.0 and not out["exceeds_max_hops"] and out["endpoint"] == "j1"


def test_user_reaches_course_via_job_when_hops_allow():
    b = _world()
    out = explain_skill_path(b.graph, "u1", "c1", max_hops=6)
    assert out["found"] and out["path"][0] == "u1" and out["path"][-1] == "c1"
    # every reported hop is a real stored edge in the stated direction
    for e in out["edges"]:
        a, z = (e["from"], e["to"]) if e["direction"] == "forward" else (e["to"], e["from"])
        assert b.graph.has_edge(a, z) and any(d["type"] == e["relation"] for d in b.graph[a][z].values())


def test_unrelated_nodes_and_unknown_nodes_give_no_path():
    b = _world()
    assert not explain_skill_path(b.graph, "u2", "j1")["found"]
    out = explain_skill_path(b.graph, "u1", "nobody")
    assert out["path"] == [] and out["reason"] == "unknown node" and out["confidence"] == 0.0
    assert explain_skill_path(b.graph, "ghost", "u1")["path"] == []


def test_reverse_hops_only_across_skill_incidence_edges_not_across_related_to():
    g = nx.MultiDiGraph()
    for n in "abc":
        g.add_node(n, type="Skill")
    g.add_edge("a", "b", type="RELATED_TO")
    g.add_edge("b", "c", type="RELATED_TO")
    assert explain_skill_path(g, "a", "c")["path"] == ["a", "b", "c"]
    assert explain_skill_path(g, "c", "a")["path"] == []  # RELATED_TO is directed: no free reverse hop


def test_undirected_graph_input_still_works():
    g = nx.Graph()
    g.add_edge("u", "s", type="HAS_SKILL")
    g.add_edge("j", "s", type="REQUIRES")
    out = explain_skill_path(g, "u", "j")
    assert out["path"] == ["u", "s", "j"] and {e["direction"] for e in out["edges"]} == {"undirected"}


def test_directed_subgraph_view_like_the_pipeline_uses():
    b = _world()
    sub = b.graph.subgraph(["u1", "python", "j1"])
    assert explain_skill_path(sub, "u1", "j1")["path"] == ["u1", "python", "j1"]


def test_shortest_path_prefers_fewest_hops_and_is_deterministic():
    g = nx.MultiDiGraph()
    for a, z in [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]:
        g.add_edge(a, z, type="RELATED_TO")
    first = explain_skill_path(g, "a", "d")
    assert first["path"] == ["a", "d"]
    assert explain_skill_path(g, "a", "d") == first
    assert explain_skill_path(g, "a", "a")["path"] == ["a"]


# ------------------------------------------------------------------ bug 6: max_hops never mislabels a path


def _chain(n: int) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for i in range(n):
        g.add_edge(f"n{i}", f"n{i + 1}", type="RELATED_TO")
    return g


def test_path_longer_than_max_hops_is_never_truncated_and_ends_at_the_target():
    out = explain_skill_path(_chain(6), "n0", "n6", max_hops=4)
    assert out["path"][0] == "n0" and out["path"][-1] == "n6" == out["target"] == out["endpoint"]
    assert out["hops"] == 6 and out["exceeds_max_hops"] is True and out["truncated"] is False
    assert out["confidence"] < 1.0 and "longer than max_hops" in out["explanation"]


def test_strict_max_hops_returns_no_path_with_a_reason():
    out = explain_skill_path(_chain(6), "n0", "n6", max_hops=4, strict_max_hops=True)
    assert out["path"] == [] and out["found"] is False and "6 hops" in out["reason"] and out["hops"] == 6
    assert explain_skill_path(_chain(4), "n0", "n4", max_hops=4, strict_max_hops=True)["path"][-1] == "n4"


def test_returned_path_always_ends_at_the_requested_target():
    g = _chain(9)
    for hops in (1, 2, 4, 8, 20):
        out = explain_skill_path(g, "n0", "n9", max_hops=hops)
        assert out["path"][-1] == "n9"


# ------------------------------------------------------------------ bug 7: unknown ids do not leak


def test_retriever_unknown_id_returns_empty_instead_of_other_users_skills():
    b = SkillGraphBuilder()
    for d in "0123456789":
        b.add_user_skill(d, f"skill-{d}")
    r = SkillGraphRetriever(b.graph)
    assert r.skills_for_user("14") == []  # networkx would iterate the characters "1" and "4"
    assert r.skills_for_user("1") == ["skill-1"]
    assert r.required_skills_for_job("14") == [] and r.courses_for_skill("skill-1 skill-2") == []
    assert r.skills_for_user("") == [] and r.skills_for_user(["1"]) == []  # type: ignore[arg-type]


def test_retriever_methods_still_work_and_deduplicate():
    b = _world()
    r = SkillGraphRetriever(b.graph)
    assert r.skills_for_user("u1") == ["python", "sql"]
    assert r.required_skills_for_job("j1") == ["python"]
    assert r.courses_for_skill("docker") == ["c1"]
    assert r.skills_for_user("j1") == []  # a Job node has no HAS_SKILL edges


# ------------------------------------------------------------------ bug 8: faithful narrative helper


def test_narrate_skill_trace_states_absent_relations_instead_of_fabricating_them():
    b = _world()
    out = narrate_skill_trace(b.graph, "u1", "j1", "c1", "python")
    text = out["narrative_path"]
    assert "requires skill 'python'" in text and "user has documented proficiency" in text
    assert "no TEACHES edge" in text and " teaches " not in text  # c1 teaches docker, not python
    assert out["relations_present"] == {"job_requires_or_prefers": True, "user_has_skill": True, "user_missing_skill": False, "course_teaches": False}
    none = narrate_skill_trace(b.graph, "u2", "j1", "c1", "sql")
    assert "no REQUIRES or PREFERS edge" in none["narrative_path"] and "no HAS_SKILL or MISSING_SKILL" in none["narrative_path"]
    assert " requires " not in none["narrative_path"] and " teaches " not in none["narrative_path"]
    present = narrate_skill_trace(b.graph, "u1", "j1", "c1", "docker")
    assert "prefers skill 'docker'" in present["narrative_path"] and "teaches 'docker' ('Deploy with Docker')" in present["narrative_path"]
    assert present["job_requirement"]["type"] == "PREFERS" and present["course_evidence"]["type"] == "TEACHES"


def test_narrate_skill_trace_missing_skill_edge_and_unknown_nodes():
    b = _world()
    b.add_edge("u2", "python", EdgeType.MISSING_SKILL)
    assert "MISSING_SKILL" in narrate_skill_trace(b.graph, "u2", "j1", "c1", "python")["narrative_path"]
    out = narrate_skill_trace(b.graph, "nobody", "nojob", "nocourse", "noskill")
    assert out["relations_present"] == {"job_requires_or_prefers": False, "user_has_skill": False, "user_missing_skill": False, "course_teaches": False}


# ------------------------------------------------------------------ bug 9: builder hygiene


def test_reingesting_is_idempotent():
    once, twice = _world(), _world()
    twice.add_user_skill("u1", "python", confidence=0.9, evidence_text="Built ETL in Python", source="mistral")
    twice.add_job_requirement("j1", "python", required=True, importance=0.8, source_sentence="You must know Python.")
    twice.add_course_skill("c1", "docker", learning_outcome_evidence="Deploy with Docker")
    assert twice.graph.number_of_edges() == once.graph.number_of_edges() == 6
    assert sorted(twice.graph.edges(data=True, keys=True), key=str) == sorted(once.graph.edges(data=True, keys=True), key=str)
    assert twice.graph.number_of_nodes() == once.graph.number_of_nodes()


def test_reingest_updates_properties_but_empty_values_do_not_erase_real_ones():
    b = SkillGraphBuilder()
    b.add_user_skill("u", "s", confidence=0.5, evidence_text="real evidence", source="bert")
    b.add_user_skill("u", "s", confidence=0.7)  # no evidence / default source: must not erase the evidence
    data = b.graph["u"]["s"]["HAS_SKILL"]
    assert b.graph.number_of_edges() == 1 and data["confidence"] == 0.7 and data["evidence_text"] == "real evidence"


def test_different_edge_types_between_the_same_nodes_coexist():
    b = SkillGraphBuilder()
    b.add_user_skill("u", "s")
    b.add_edge("u", "s", EdgeType.MISSING_SKILL)
    b.add_edge("u", "s", EdgeType.MISSING_SKILL)
    assert {d["type"] for d in b.graph["u"]["s"].values()} == {"HAS_SKILL", "MISSING_SKILL"} and b.graph.number_of_edges() == 2


def test_add_node_never_silently_retypes_and_flags_it():
    b = SkillGraphBuilder()
    b.add_node("x", NodeType.USER)
    b.add_node("x", NodeType.SKILL)
    assert b.graph.nodes["x"]["type"] == "User" and any("retype" in w for w in b.warnings)
    strict = SkillGraphBuilder(strict=True)
    strict.add_node("x", NodeType.USER)
    with pytest.raises(ValueError, match="retype"):
        strict.add_node("x", NodeType.SKILL)


def test_add_node_same_type_merges_properties():
    b = SkillGraphBuilder()
    b.add_node("s", NodeType.SKILL, name="Python")
    b.add_node("s", NodeType.SKILL, aliases=["py"])
    assert b.graph.nodes["s"]["name"] == "Python" and b.graph.nodes["s"]["aliases"] == ["py"] and b.warnings == []


def test_edge_to_unknown_node_is_flagged_and_strict_mode_rejects_it(caplog):
    b = SkillGraphBuilder()
    with caplog.at_level(logging.WARNING):
        b.add_edge("ghost-course", "ghost-skill", EdgeType.TEACHES)
    assert b.warnings and "unknown node" in b.warnings[0] and "ghost" in caplog.text
    assert b.graph.nodes["ghost-skill"].get("placeholder") is True and "type" not in b.graph.nodes["ghost-skill"]
    b.add_node("ghost-skill", NodeType.SKILL)  # later typing upgrades the placeholder and clears the flag
    assert b.graph.nodes["ghost-skill"]["type"] == "Skill" and "placeholder" not in b.graph.nodes["ghost-skill"]
    strict = SkillGraphBuilder(strict=True)
    with pytest.raises(ValueError, match="unknown node"):
        strict.add_edge("a", "b", EdgeType.TEACHES)
    assert strict.graph.number_of_nodes() == 0 and strict.graph.number_of_edges() == 0


def test_builder_never_reads_the_wall_clock_and_clock_is_injectable():
    default = SkillGraphBuilder()
    default.add_user_skill("u", "s")
    assert "last_seen" not in default.graph["u"]["s"]["HAS_SKILL"]
    ticks = iter(["2026-01-01T00:00:00+00:00", "2030-01-01T00:00:00+00:00"])
    timed = SkillGraphBuilder(clock=lambda: next(ticks))
    timed.add_user_skill("u", "s")
    assert timed.graph["u"]["s"]["HAS_SKILL"]["last_seen"] == "2026-01-01T00:00:00+00:00"
    timed.add_user_skill("u", "t", last_seen="2020-05-05T00:00:00+00:00")  # explicit beats the clock
    assert timed.graph["u"]["t"]["HAS_SKILL"]["last_seen"] == "2020-05-05T00:00:00+00:00"


def test_same_input_gives_identical_graphs_across_builds():
    a, b = _world(), _world()
    assert sorted(a.graph.nodes(data=True), key=str) == sorted(b.graph.nodes(data=True), key=str)
    assert sorted(a.graph.edges(data=True, keys=True), key=str) == sorted(b.graph.edges(data=True, keys=True), key=str)


# ------------------------------------------------------------------ bug 10: symbol-bearing skill keys


@pytest.mark.parametrize(
    "name,key",
    [("C++", "c++"), ("C#", "c#"), ("C", "c"), ("F#", "f#"), (".NET", ".net"), ("Node.js", "node.js"), ("ASP.NET", "asp.net"), ("Vue.js", "vue.js"), ("  Python  ", "python")],
)
def test_graph_key_preserves_symbol_bearing_names(name, key):
    assert graph_key(name) == key


def test_graph_key_separates_what_normalize_text_collapses():
    names = ["C", "C++", "C#", "F", "F#", ".NET", "NET"]
    assert len({graph_key(n) for n in names}) == len(names) - 0 or graph_key(".NET") != graph_key("NET")
    assert len({normalize_text(n) for n in ["C", "C++", "C#"]}) == 1  # the old behaviour that caused the collision
    assert len({graph_key(n) for n in ["C", "C++", "C#"]}) == 3
    assert graph_key(".NET") != graph_key("NET") and graph_key("F#") != graph_key("F")


@pytest.mark.parametrize("name", ["Python", "Machine Learning", "scikit-learn", "A/B testing", "CI/CD", "Power BI", "python 3.9", "R language", "k8s", "résumé writing", "Spring Boot"])
def test_graph_key_equals_normalize_text_when_there_are_no_name_symbols(name):
    assert graph_key(name) == normalize_text(name)


def test_graph_key_is_idempotent_and_handles_punctuation_around_names():
    for name in ["C++", "C#", ".NET", "Node.js", "C/C++, C#.", "(C++)", "Python."]:
        assert graph_key(graph_key(name)) == graph_key(name)
    assert graph_key("C/C++, C#.") == "c c++ c#" and graph_key("(C++)") == "c++" and graph_key("Python.") == "python"


def test_c_family_skills_become_distinct_graph_nodes():
    b = SkillGraphBuilder()
    for name in ("C", "C++", "C#"):
        b.add_user_skill("u", f"skill:{graph_key(name)}")
    assert b.graph.number_of_nodes() == 4 and len(SkillGraphRetriever(b.graph).skills_for_user("u")) == 3


# ------------------------------------------------------------------ bug 11: provenance


def test_edges_carry_the_course_graph_builder_provenance_keys():
    b = SkillGraphBuilder(clock=lambda: "2026-01-01T00:00:00+00:00")
    b.add_user_skill("u", "s", confidence=0.9, evidence_text="Used s", source="mistral")
    b.add_job_requirement("j", "s", required=True, importance=0.7, source_sentence="Must know s.")
    b.add_job_requirement("j", "t", required=False)
    b.add_course_skill("c", "s", confidence=0.8, learning_outcome_evidence="Apply s", depth="advanced")
    has = b.graph["u"]["s"]["HAS_SKILL"]
    assert {"source", "confidence", "evidence_text", "last_seen"} <= set(has)
    assert (has["source"], has["confidence"], has["evidence_text"], has["last_seen"]) == ("mistral", 0.9, "Used s", "2026-01-01T00:00:00+00:00")
    req = b.graph["j"]["s"]["REQUIRES"]
    assert {"importance", "requirement_strength", "source_sentence"} <= set(req) and req["requirement_strength"] == "required" and req["source_sentence"] == "Must know s."
    assert b.graph["j"]["t"]["PREFERS"]["requirement_strength"] == "preferred"
    teach = b.graph["c"]["s"]["TEACHES"]
    assert {"confidence", "depth", "learning_outcome_evidence", "source"} <= set(teach) and teach["depth"] == "advanced"


def test_unknown_provenance_is_empty_not_invented():
    b = SkillGraphBuilder()
    b.add_user_skill("u", "s")
    has = b.graph["u"]["s"]["HAS_SKILL"]
    assert has["evidence_text"] == "" and has["source"] == "unknown"
    b.add_course_skill("c", "s")
    assert b.graph["c"]["s"]["TEACHES"]["depth"] is None
