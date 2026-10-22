"""Tests for the context / graph / explanation evals: hand-verifiable toy cases, controls and fixture integrity.

Where a test touches a behaviour of the module under test that is a known defect (non-deterministic content_hash,
path truncation, unknown-node leakage, ...), it asserts that the EVAL measures what is independently observable, not
that the defect must persist, so fixing the module does not break the suite.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from skillllm_v2.evals import context_evals as C
from skillllm_v2.evals import explanation_evals as X
from skillllm_v2.evals import graph_evals as G
from skillllm_v2.evals.contract import EvalContext, EvalResult

ROOT = Path(__file__).resolve().parents[2]
SMOKE = EvalContext(smoke=True, n_boot=100)


def by_name(results, name):
    return next(r for r in results if r.eval_name == name)


# ----------------------------------------------------------------------------------------------------- context


def _toy_profile():
    text = "I use Python daily. I use Python and SQL often. I tried Excel once."
    def ex(surface, conf, evidence):
        st = text.find(surface)
        return {"surface_form": surface, "label": "HARD_SKILL", "start": st, "end": st + len(surface), "source": "mistral",
                "confidence": conf, "evidence": evidence, "canonical_skill_id": surface.lower(), "canonical_skill_name": surface}
    extractions = [ex("Python", 0.95, "I use Python daily"), ex("Python", 0.9, "I use Python and SQL often"), ex("SQL", 0.8, "I use Python and SQL often"),
                   ex("Excel", 0.4, "I tried Excel once")]
    return {"profile_id": "toy", "text": text, "extractions": extractions, "job_target": "analyst", "gold_important": ["python"], "gold_skills": ["python", "sql", "excel"]}


def test_token_proxy_hand_value():
    assert C.count_tokens("Python, SQL!") == 4  # Python , SQL !


def test_profile_generator_is_deterministic_synthetic_and_consistent():
    a, b = C.make_profile(42, 10, idx=0), C.make_profile(42, 10, idx=0)
    assert a == b and a["dataset"] == "synthetic"
    assert C.make_profile(43, 10, idx=0) != a
    assert set(a["gold_important"]) <= set(a["gold_skills"])
    for e in a["extractions"]:
        assert a["text"][e["start"]:e["end"]] == e["surface_form"]  # offsets are true offsets
        if e["evidence"]:
            assert e["evidence"] in a["text"]


def test_toy_importance_ranking_keeps_the_important_skill_and_random_control_can_miss_it():
    p = _toy_profile()
    with C.fixed_clock():
        packet = C.build_production_packet(p)
    full = C.reference_compact(packet, 10_000)
    assert C.retained_ids(full) == {"python", "sql", "excel"}
    # budget that admits exactly the header plus the first (most important) entry
    hdr = C.count_tokens(C._json({"packet_type": "reference_compact", "user_id": "toy", "skills": []}))
    entry = C._entry(sorted(C.packet_skills(packet), key=lambda s: -C.importance_score(s))[0])
    budget = hdr + C.count_tokens(C._json(entry)) + 1
    ranked = C.reference_compact(packet, budget)
    assert C.retained_ids(ranked) == {"python"}  # Python: conf 0.95 x 2 evidence beats SQL and Excel
    assert C.compact_token_count(ranked) <= budget
    misses = 0
    for d in range(30):
        rc = C.reference_compact(packet, budget, order="random", rng=random.Random(d))
        misses += int("python" not in C.retained_ids(rc))
        assert C.compact_token_count(rc) <= budget
    assert misses > 0  # a random-drop compactor loses the important skill in some draws: the control has bite


def test_reference_compactor_evidence_is_substring_and_audit_passes():
    p = _toy_profile()
    with C.fixed_clock():
        packet = C.build_production_packet(p)
    comp = C.reference_compact(packet, 10_000)
    a = C.audit_compact(comp, p)
    assert a["evidence_ok"] == a["n"] == 3 and a["ungrounded_skills"] == []


def test_hallucination_audit_flags_fabricated_skill_and_corrupted_evidence_but_not_clean_packets():
    p = _toy_profile()
    with C.fixed_clock():
        packet = C.build_production_packet(p)
    clean = C.audit_production_packet(packet, p)
    assert clean["fabricated_by_compaction"] == [] and clean["evidence_substring_ok"] == clean["evidence_total"]
    bad = json.loads(C._json(packet))
    C.packet_skills(bad).append({"surface_form": "Quantum Basketweaving", "evidence": [{"text": "Quantum Basketweaving at CERN"}]})
    assert C.audit_production_packet(bad, p)["fabricated_by_compaction"]
    bad2 = json.loads(C._json(packet))
    C.packet_skills(bad2)[0]["evidence"][0]["text"] = "X" + C.packet_skills(bad2)[0]["evidence"][0]["text"][1:]
    assert C.audit_production_packet(bad2, p)["evidence_substring_ok"] < clean["evidence_substring_ok"]


def test_production_determinism_measured_exactly_as_observed():
    p = _toy_profile()
    with C.fixed_clock():
        assert C._json(C.build_production_packet(p)) == C._json(C.build_production_packet(p))
    with C.ticking_clock():
        a, b = C.build_production_packet(p), C.build_production_packet(p)
    res = C.eval_determinism(SMOKE, [p])
    r = by_name(res, "context.determinism_across_clock_ticks")
    assert r.value == float(C._json(a) == C._json(b))  # eval agrees with the independent computation
    assert by_name(res, "context.determinism_modulo_timestamps").value == float(C._json(C.strip_timestamps(a)) == C._json(C.strip_timestamps(b)))


def test_rank_metrics_hand_values():
    assert C.rr_and_recall(["x", "g", "y"], ["g"], 2) == (0.5, 1.0, 0.0)
    assert C.rr_and_recall(["g", "x"], ["g", "h"], 2) == (1.0, 0.5, 1.0)
    assert C.rr_and_recall([], ["g"], 5) == (0.0, 0.0, 0.0)


def test_context_retriever_alias_query_beats_surface_baseline_on_hand_case():
    p = _toy_profile()
    p["extractions"][2]["canonical_skill_name"] = "Structured Query Language"
    with C.fixed_clock():
        packet = C.build_production_packet(p)
    assert C._rank_ids_context_retriever(packet, "structured query language", 3)[0] == "sql"
    assert C._rank_ids_surface_baseline(packet, "structured query language", 3) == []


def test_context_run_smoke_has_controls_and_is_serialisable():
    res = C.run(SMOKE)
    assert all(isinstance(r, EvalResult) for r in res)
    assert not [r for r in res if r.status == "failed"], [(r.eval_name, r.reason) for r in res if r.status == "failed"]
    json.dumps([r.to_dict() for r in res])
    assert all("synthetic" in r.dataset for r in res if r.status != "skipped")
    assert {r.status for r in res if r.eval_name in ("context.importance_vs_human_labels", "context.downstream_answer_fidelity")} == {"skipped"}
    nc = by_name(res, "context.negative_control_random_drop")
    assert nc.value > 0 and nc.details["control_passed"]
    assert by_name(res, "context.reference_budget_adherence").value == 1.0
    assert by_name(res, "context.retriever_mrr").value > by_name(res, "context.random_baseline_mrr").value
    assert by_name(res, "context.audit_sensitivity_fabricated_skill").value == 1.0
    assert by_name(res, "context.audit_false_positive_rate").value == 0.0


# ------------------------------------------------------------------------------------------------------ graph


@pytest.mark.parametrize("case", G.TOY_CASES, ids=[c[0] for c in G.TOY_CASES])
def test_toy_paths_against_hand_computed_expectations(case):
    out = G.run_toy_case(case)
    name = case[0]
    if name == "six_hops_cap4":
        # hand-computed shortest path has 6 hops; the explainer caps at 4. Whatever it returns must either reach the target or be empty.
        assert out["endpoint_ok"] == (not out["got"] or out["got"][-1] == "g")
        assert out["expected"] == ["a", "b", "c", "d", "e", "f", "g"]
    else:
        assert out["exact"], out
        assert out["realisable"]


def test_independent_bfs_oracle_hand_values():
    adj = {"a": ["b"], "b": ["c"], "c": []}
    assert G.bfs_shortest(adj, "a", "c") == 2
    assert G.bfs_shortest(adj, "c", "a") is None
    assert G.bfs_shortest(adj, "a", "a") == 0


def test_realisable_detects_fabricated_hop():
    g = G._toy_graph([("a", "b"), ("b", "c")])
    assert G.realisable(g, ["a", "b", "c"])
    assert not G.realisable(g, ["a", "c"])


def test_structural_report_hand_graphs():
    from skillllm_v2.graph.graph_builder import SkillGraphBuilder
    from skillllm_v2.graph.schema import EdgeType

    b = SkillGraphBuilder()
    b.add_user_skill("u", "python")
    clean = G.structural_report(b.graph)
    assert clean["dangling_edge_rate"] == 0.0 and clean["edge_signature_valid"] == 1.0
    assert 0.0 <= clean["provenance_complete"] < 1.0  # SkillGraphBuilder now carries source/confidence; last_seen/evidence stay empty without a clock/evidence input (BUGFIX_LOG bug 11)
    b.add_edge("ghost-course", "ghost-skill", EdgeType.TEACHES)
    dirty = G.structural_report(b.graph)
    assert dirty["untyped_nodes"] == 2 and dirty["dangling_edge_rate"] > 0.0


def test_course_graph_matches_spec_and_perturbation_is_detected():
    w = G.make_world(42, 0, n_skills=12, n_courses=6, n_jobs=3, n_users=3)
    with G.fixed_graph_clock():
        g = G.build_course_graph(w)
        g2 = G.build_course_graph(w)
        twice = G.build_course_graph(w, repeat=2)
    exp = G.expected_edges(w)
    assert G.edge_multiset(g.graph) == exp
    assert G.canonical_form(g.graph) == G.canonical_form(g2.graph)
    assert twice.graph.number_of_edges() == 2 * g.graph.number_of_edges()  # builder is not idempotent (parallel edges)
    rep = G.structural_report(g.graph)
    assert rep["provenance_complete"] == 1.0 and rep["edge_signature_valid"] == 1.0 and rep["dangling_edge_rate"] == 0.0
    # control: removing one edge breaks exact-match fidelity
    u, v, k = next(iter(g.graph.edges(keys=True)))
    g.graph.remove_edge(u, v, k)
    assert G.edge_multiset(g.graph) != exp


def test_graph_determinism_across_clock_ticks_matches_independent_computation():
    w = G.make_world(42, 0, n_skills=12, n_courses=6, n_jobs=3, n_users=3)
    with G.ticking_graph_clock():
        a = G.build_course_graph(w)
        b = G.build_course_graph(w)
    assert G.canonical_form(a.graph, drop=("last_seen",)) == G.canonical_form(b.graph, drop=("last_seen",))
    res = G.eval_fidelity_idempotence_determinism(SMOKE, [w])
    assert by_name(res, "graph.determinism_across_clock_ticks").value == float(G.canonical_form(a.graph) == G.canonical_form(b.graph))
    assert by_name(res, "graph.order_invariance").value == 1.0
    assert by_name(res, "graph.ingest_idempotence").value == pytest.approx(0.5)


def test_unknown_node_probe_matches_direct_call_and_course_builder_is_safe():
    from skillllm_v2.graph.graph_builder import SkillGraphBuilder
    from skillllm_v2.graph.graph_retriever import SkillGraphRetriever

    b = SkillGraphBuilder()
    for d in "0123456789":
        b.add_user_skill(d, f"skill-{d}")
    direct = float(bool(SkillGraphRetriever(b.graph).skills_for_user("14")))
    res = G.eval_graph_retriever(SMOKE, G.make_worlds(SMOKE))
    leak = by_name(res, "graph.retriever_unknown_node_leak_rate")
    assert leak.details["CourseGraphBuilder.get_user_skills_leak_rate"] == 0.0
    assert leak.value >= 0.0 and (direct == 0.0 or leak.value > 0.0)


def test_skill_id_normalisation_collision_probe():
    assert G.norm_id("C++") == G.norm_id("C#") == G.norm_id("C")
    assert G.norm_id("Python") != G.norm_id("PyTorch")
    r = by_name(G.eval_id_collisions(SMOKE), "graph.skill_id_collision_rate")
    assert r.value == 1.0 and r.n == 3


def test_graph_run_smoke():
    res = G.run(SMOKE)
    assert not [r for r in res if r.status == "failed"], [(r.eval_name, r.reason) for r in res if r.status == "failed"]
    json.dumps([r.to_dict() for r in res])
    assert by_name(res, "graph.course_builder.exact_edge_match").value == 1.0
    assert 0.0 < by_name(res, "graph.skill_builder.provenance_complete").value < 1.0  # was 0.0 before the bug-11 fix; <1 because demo inputs carry no timestamps
    assert by_name(res, "graph.path_shortest_correct_within_hops").value == 1.0
    assert by_name(res, "graph.random_baseline_recall_at_5").value < by_name(res, "graph.contextual_retriever_recall_at_5").value
    assert by_name(res, "graph.retriever_exact_user_skills").value == 1.0


# ------------------------------------------------------------------------------------------------ explanation


def _hand_graph_and_explanation():
    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder

    cg = CourseGraphBuilder()
    with G.fixed_graph_clock():
        cg.ingest_user_profile({"user_id": "u", "skills": [{"surface_form": "SQL", "canonical_skill_name": "SQL", "confidence": 0.9, "evidence": "Used SQL daily"}]})
    cg.ingest_job({"job_id": "j", "title": "Data Engineer", "required_skills": [{"skill_name": "Python", "importance": 1.0, "source_sentence": "You must know Python."}], "preferred_skills": []})
    cg.ingest_course({"course_id": "c", "title": "Python Basics", "description": "Learn Python", "teaches_skills": ["Python"], "learning_outcomes": ["Apply Python in production"], "prerequisites": []})
    cg.ingest_course({"course_id": "c2", "title": "Go Basics", "teaches_skills": ["Go"], "learning_outcomes": ["Apply Go"], "prerequisites": []})
    cg.ingest_course({"course_id": "c3", "title": "Rust Basics", "teaches_skills": ["Rust"], "learning_outcomes": ["Apply Rust"], "prerequisites": []})
    expl = {
        "course_id": "c", "title": "Python Basics", "rank": 1, "score": 0.9, "role": "primary", "unlocks": [], "covered_missing_skills": ["Python"],
        "prerequisite_status": {"is_satisfied": True, "status_code": "satisfied_none_required", "satisfied": [], "missing": [], "explanation": "Course has no prerequisite requirements."},
        "evidence": [{"skill": "Python", "canonical_skill_id": "python", "job_requirement_evidence": "You must know Python.", "gap_classification": "missing_critical",
                      "user_evidence": "User missing verified skill", "course_learning_outcome_evidence": "Apply Python in production"}],
        "explanation_path": ["Target role 'Data Engineer' requires 'Python' -> User is missing critical -> Course 'Python Basics' teaches 'Python' via outcome: \"Apply Python in production\"",
                             "Course has no prerequisite requirements."],
        "limitations": ["No blocking constraints detected."],
        "next_profile_update_action": {"skills_to_add": [{"canonical_skill_id": "python"}]},
    }
    return cg, expl


def test_audit_hand_case_all_supported_and_parse_coverage_full():
    cg, expl = _hand_graph_and_explanation()
    a = X.audit_course_explanation(expl, X.GraphIndex(cg.graph), "u", "j")
    assert X.unsupported_keys(a) == set(), a
    assert a["n_steps"] == a["n_parsed"] == 2
    assert a["completeness"]["complete"]
    kinds = {c["kind"] for c in a["claims"]}
    assert {"ev.course_teaches", "ev.job_requires", "ev.user_lacks", "path.job_requires", "path.course_teaches", "path.user_state"} <= kinds


@pytest.mark.parametrize("mutation", ["remove_teaches", "remove_job_requires", "user_gains_skill", "swap_teaches"])
def test_audit_flags_graph_corruption_on_hand_case(mutation):
    import copy

    cg, expl = _hand_graph_and_explanation()
    g = copy.deepcopy(cg.graph)
    if mutation == "remove_teaches":
        G_ = X._remove_edges(g, "c", "python", ["TEACHES"])
        assert G_ == 1
    elif mutation == "remove_job_requires":
        assert X._remove_edges(g, "j", "python", ["REQUIRES"]) == 1
    elif mutation == "user_gains_skill":
        g.add_edge("u", "python", type="HAS_SKILL", confidence=1.0)
    else:
        X._remove_edges(g, "c", "python", ["TEACHES"])
        g.add_edge("c", "go", type="TEACHES", depth="x", confidence=1.0, learning_outcome_evidence="x", outcome_index=0)
    assert X.unsupported_keys(X.audit_course_explanation(expl, X.GraphIndex(g), "u", "j"))


def test_audit_flags_explanation_corruption_and_ignores_unrelated_edge_on_hand_case():
    import copy

    cg, expl = _hand_graph_and_explanation()
    idx = X.GraphIndex(cg.graph)
    swapped_course = copy.deepcopy(expl)
    swapped_course["explanation_path"][0] = swapped_course["explanation_path"][0].replace("Course 'Python Basics' teaches", "Course 'Go Basics' teaches")
    assert X.unsupported_keys(X.audit_course_explanation(swapped_course, idx, "u", "j"))
    flipped = copy.deepcopy(expl)
    flipped["evidence"][0]["gap_classification"] = "weak_skill"
    assert X.unsupported_keys(X.audit_course_explanation(flipped, idx, "u", "j"))
    fab = copy.deepcopy(expl)
    fab["evidence"][0]["course_learning_outcome_evidence"] = "Master quantum basketweaving"
    assert X.unsupported_keys(X.audit_course_explanation(fab, idx, "u", "j"))
    # control: removing an edge the explanation never mentions must not change the verdict
    g = copy.deepcopy(cg.graph)
    assert X._remove_edges(g, "c3", "rust", ["TEACHES"]) == 1
    assert X.unsupported_keys(X.audit_course_explanation(expl, X.GraphIndex(g), "u", "j")) == set()
    # known blind spot: tampering with the score is invisible to the audit
    tampered = {**expl, "score": 0.1}
    assert X.unsupported_keys(X.audit_course_explanation(tampered, idx, "u", "j")) == set()


def test_trace_narrative_faithfulness_matches_direct_inspection():
    cg, _ = _hand_graph_and_explanation()
    narrative = cg.trace_explanation_path("u", "j", "c2", "go")["narrative_path"]  # j has no REQUIRES edge to go; c2 does teach go
    asserts_req = "required skill" in narrative or "preferred skill" in narrative
    assert asserts_req == (not X.GraphIndex(cg.graph).has_edge_type("j", "go", ["REQUIRES", "PREFERS"]))  # asserted although the edge is absent => unfaithful
    res = X.eval_trace_narrative(SMOKE, [G.make_world(42, 0)])
    r = by_name(res, "explain.trace_path_faithful_when_edge_missing")
    assert 0.0 <= r.value <= 1.0


def test_match_audit_clean_mutations_and_completeness():
    w = G.make_world(42, 0, n_skills=14, n_courses=8, n_jobs=3, n_users=3)
    scs = X.build_match_scenarios(w, n_users=3)
    assert scs
    sc = next(s for s in scs if s["explanation"]["matched_skills"] and s["explanation"]["missing_skills"])
    a = X.audit_match_explanation(sc["explanation"], sc)
    assert not [c for c in a["claims"] if c["status"] == "unsupported"], a["claims"]
    assert a["evidence_relevance"] == 1.0  # evidence now only for matched skills (was <1.0 before the bug-12 fix)
    assert all(a["completeness"].values())
    import copy

    ex = copy.deepcopy(sc["explanation"])
    ex["matched_skills"][0] = "Quantum Basketweaving"
    assert [c for c in X.audit_match_explanation(ex, sc)["claims"] if c["status"] == "unsupported"]
    ex2 = copy.deepcopy(sc["explanation"])
    ex2["confidence"] = 0.123456
    assert [c for c in X.audit_match_explanation(ex2, sc)["claims"] if c["kind"] == "match.confidence" and c["status"] == "unsupported"]
    # limitations are static boilerplate and are never counted as supported
    assert {c["status"] for c in a["claims"] if c["kind"] == "match.limitation"} == {"static"}


def test_explanation_run_smoke_controls():
    res = X.run(SMOKE)
    assert not [r for r in res if r.status == "failed"], [(r.eval_name, r.reason) for r in res if r.status == "failed"]
    json.dumps([r.to_dict() for r in res])
    assert by_name(res, "explain.audit_false_alarm_control").value == 0.0
    assert by_name(res, "explain.audit_sensitivity_graph_mutations").value > 0.9
    assert by_name(res, "explain.audit_sensitivity_explanation_mutations").value > 0.9
    assert by_name(res, "explain.audit_blind_spots").value == 0.0
    assert by_name(res, "explain.match.audit_false_alarm_control").value == 0.0
    assert by_name(res, "explain.course.parse_coverage").value == 1.0


# ---------------------------------------------------------------------------------------------------- fixtures


def test_checked_in_fixtures_match_generators_and_are_marked_synthetic():
    ctx_fx = json.loads((ROOT / "inputs" / "evals" / "context_profiles_synthetic.json").read_text())
    gr_fx = json.loads((ROOT / "inputs" / "evals" / "graph_cases_synthetic.json").read_text())
    assert ctx_fx["dataset"] == gr_fx["dataset"] == "synthetic"
    assert ctx_fx == json.loads(json.dumps(C.fixture_payload(ctx_fx["seed"]), sort_keys=True))
    assert gr_fx == json.loads(json.dumps(G.fixture_payload(gr_fx["seed"]), sort_keys=True))
    assert all(p["dataset"] == "synthetic" for p in ctx_fx["profiles"])
