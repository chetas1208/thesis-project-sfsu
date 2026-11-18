"""Unit tests for the L03 perturbation generator: exact offset remapping, label preservation, determinism."""
from __future__ import annotations

import random

import pytest

from skillllm_v2.evals.research import perturbations as P


def sk(tokens, spans, did="d1", source="tech"):
    """Build a SkillSpan-like doc from tokens and token-range spans."""
    text = " ".join(tokens)
    offs, pos = [], 0
    for t in tokens:
        offs.append((pos, pos + len(t)))
        pos += len(t) + 1
    gold = [{"surface_form": text[offs[i][0]:offs[j - 1][1]], "label": l, "start": offs[i][0], "end": offs[j - 1][1]}
            for i, j, l in spans]
    return {"id": did, "tokens": tokens, "text": text, "gold_spans": gold, "source": source}


DOC = sk(["Experience", "with", "machine", "learning", ",", "Python", "and", "strong", "communication", "skills", "."],
         [(2, 4, "HARD_SKILL"), (5, 6, "HARD_SKILL"), (8, 9, "SOFT_SKILL")])
BULLET = sk(["-", "Knowledge", "of", "Amazon", "Web", "Services", ";", "team", "work"], [(3, 6, "HARD_SKILL"), (7, 9, "SOFT_SKILL")], "d2")
NESTED = sk(["Java", "/", "JavaScript", "development"], [(0, 1, "HARD_SKILL"), (2, 4, "HARD_SKILL"), (0, 4, "SOFT_SKILL")], "d3")


def check_spans(row, base_doc, surface_may_change):
    text = row["text"]
    for sp in row["gold_spans"]:
        assert text[sp["start"]:sp["end"]] == sp["surface_form"]
    assert sorted(s["label"] for s in row["gold_spans"]) == sorted(s["label"] for s in base_doc["gold_spans"])
    if not surface_may_change:
        assert sorted(s["surface_form"] for s in row["gold_spans"]) == sorted(s["surface_form"] for s in base_doc["gold_spans"])
    # token offsets agree with the rendered text
    for tok, (s, e) in zip(row["tokens"], row["token_offsets"]):
        assert text[s:e] == tok


def test_from_skillspan_roundtrip():
    t = P.from_skillspan(DOC)
    text, _ = t.render()
    assert text == DOC["text"]
    assert t.char_spans() == sorted(DOC["gold_spans"], key=lambda x: (x["start"], x["end"], x["label"]))


@pytest.mark.parametrize("family", P.FAMILIES)
@pytest.mark.parametrize("doc", [DOC, BULLET, NESTED])
def test_every_family_offsets_and_labels(family, doc):
    aliases = P.AliasTable({"c:aws": ["Amazon Web Services", "AWS"], "c:ml": ["machine learning", "ML"]}, [])
    rows, na = P.build([doc], [family], seed=42, aliases=aliases)
    if rows:
        assert len(rows) == 1 and not na
        check_spans(rows[0], doc, family in P.SURFACE_CHANGING)
        assert rows[0]["meta"]["family"] == family and rows[0]["meta"]["base_id"] == doc["id"]
    else:
        assert na and na[0]["reason"]


def test_determinism_and_seed_dependence():
    a, _ = P.build([DOC], P.FAMILIES, seed=42)
    b, _ = P.build([DOC], P.FAMILIES, seed=42)
    c, _ = P.build([DOC], ["typo_outside", "whitespace"], seed=7)
    assert a == b
    texts42 = {r["meta"]["family"]: r["text"] for r in a}
    assert any(texts42[r["meta"]["family"]] != r["text"] for r in c)


def test_identity_is_exact_copy():
    rows, _ = P.build([DOC], ["identity"])
    assert rows[0]["text"] == DOC["text"]
    assert rows[0]["gold_spans"] == sorted(DOC["gold_spans"], key=lambda x: (x["start"], x["end"], x["label"]))


def test_case_families():
    rows, _ = P.build([DOC], ["case_upper", "case_lower", "case_title"])
    by = {r["meta"]["family"]: r for r in rows}
    assert by["case_upper"]["text"] == DOC["text"].upper()
    assert "MACHINE LEARNING" in [s["surface_form"] for s in by["case_upper"]["gold_spans"]]
    assert by["case_lower"]["text"] == DOC["text"].lower()
    assert by["case_title"]["text"].startswith("Experience With Machine Learning")


def test_placeholders_untouched_by_case():
    d = sk(["<ORGANIZATION>", "needs", "Python"], [(2, 3, "HARD_SKILL")], "d9")
    rows, _ = P.build([d], ["case_lower"])
    assert rows[0]["text"].startswith("<ORGANIZATION>")


def test_alias_substitution_updates_spans():
    aliases = P.AliasTable({"c:aws": ["Amazon Web Services", "AWS"]}, [])
    rows, na = P.build([BULLET], ["alias_sub"], aliases=aliases)
    r = rows[0]
    assert "AWS" in r["text"] and "Amazon" not in r["text"]
    assert {"AWS", "team work"} == {s["surface_form"] for s in r["gold_spans"]}
    check_spans(r, BULLET, True)
    rows, na = P.build([DOC], ["alias_sub"], aliases=P.AliasTable({}, []))
    assert not rows and na[0]["reason"].startswith("no gold span")


def test_typo_outside_never_touches_spans_and_inside_does():
    rows, _ = P.build([DOC], ["typo_outside", "typo_inside"])
    by = {r["meta"]["family"]: r for r in rows}
    base_surfaces = sorted(s["surface_form"] for s in DOC["gold_spans"])
    assert sorted(s["surface_form"] for s in by["typo_outside"]["gold_spans"]) == base_surfaces
    assert by["typo_outside"]["text"] != DOC["text"]
    assert sorted(s["surface_form"] for s in by["typo_inside"]["gold_spans"]) != base_surfaces


def test_typo_function():
    rng = random.Random(0)
    for w in ["communication", "Python", "teamwork", "data"]:
        for _ in range(20):
            t = P.typo(w, rng)
            assert t != w and t[0] == w[0] and abs(len(t) - len(w)) <= 1


def test_punct_detok_and_strip():
    rows, _ = P.build([DOC], ["punct_detok", "punct_strip"])
    by = {r["meta"]["family"]: r for r in rows}
    assert "learning, Python" in by["punct_detok"]["text"] and by["punct_detok"]["text"].endswith("skills.")
    assert "," not in by["punct_strip"]["text"] and "." not in by["punct_strip"]["text"]


def test_bullets_both_directions():
    rows, _ = P.build([DOC, BULLET], ["bullets_prose"])
    by = {r["meta"]["base_id"]: r for r in rows}
    assert by["d1"]["text"].lstrip("\n")[0] in "•-*" and not by["d1"]["text"].endswith(".")
    assert not by["d2"]["text"].startswith("-") and by["d2"]["text"].endswith(".")


def test_insertions_shift_spans():
    for fam in ("boilerplate", "distractor"):
        rows, _ = P.build([DOC], [fam])
        r = rows[0]
        assert len(r["text"]) > len(DOC["text"])
        check_spans(r, DOC, False)


def test_reorder_sections_keeps_spans_and_changes_order():
    d = sk(["Python", "and", "SQL", ";", "Docker", ".", "good", "teamwork", "."],
           [(0, 1, "HARD_SKILL"), (2, 3, "HARD_SKILL"), (4, 5, "HARD_SKILL"), (7, 8, "SOFT_SKILL")], "d5")
    rows, na = P.build([d], ["reorder_clauses"])
    r = rows[0]
    assert r["text"] != d["text"] and r["text"].endswith(" .") and r["meta"]["note"].startswith("sections")
    assert r["tokens"][3] == ";" or r["tokens"].count(";") == 1
    assert sorted(r["tokens"]) == sorted(d["tokens"])
    check_spans(r, d, False)
    single = sk(["Python", "experience"], [(0, 1, "HARD_SKILL")], "d6")
    rows, na = P.build([single], ["reorder_clauses"])
    assert not rows and "coordination" in na[0]["reason"]


def test_whitespace_keeps_internal_separators():
    rows, _ = P.build([DOC], ["whitespace"])
    r = rows[0]
    assert "machine learning" in r["text"]  # internal span separator untouched
    check_spans(r, DOC, False)


def test_paraphrase_only_context():
    rows, _ = P.build([DOC], ["paraphrase_context"])
    r = rows[0]
    assert r["text"].startswith("Background with") and "solid communication" in r["text"]
    check_spans(r, DOC, False)


def test_nested_overlapping_spans_survive():
    rows, _ = P.build([NESTED], ["boilerplate", "case_upper", "whitespace"])
    for r in rows:
        assert len(r["gold_spans"]) == 3
        check_spans(r, NESTED, r["meta"]["family"] in P.SURFACE_CHANGING)


def test_stratified_sample_deterministic_and_proportional():
    docs = [sk(["a", "b"], [(0, 1, "HARD_SKILL")] * (1 + i % 5), f"x{i}", "tech" if i % 3 else "house") for i in range(300)]
    s1 = P.stratified_sample(docs, 60, 42)
    s2 = P.stratified_sample(docs, 60, 42)
    assert [d["id"] for d in s1] == [d["id"] for d in s2] and len(s1) == 60
    assert len({d["id"] for d in s1}) == 60
    house = sum(d["source"] == "house" for d in s1)
    assert 15 <= house <= 25  # 1/3 of 60


def test_validate_catches_broken_offsets():
    t = P.from_skillspan(DOC)
    t.spans = [(0, 99, "HARD_SKILL")]
    with pytest.raises(AssertionError):
        P.validate(t)


def test_reorder_coordination_swap_and_bullet_prefix():
    d = sk(["*", "•", "Python", "and", "SQL", "for", "strong", "teamwork"],
           [(2, 3, "HARD_SKILL"), (4, 5, "HARD_SKILL"), (7, 8, "SOFT_SKILL")], "d7")
    rows, na = P.build([d], ["reorder_clauses"])
    r = rows[0]
    assert r["text"] == "* • SQL and Python for strong teamwork"
    assert r["meta"]["note"].startswith("coordination_swap")
    check_spans(r, d, False)
    inspan = sk(["Python", "and", "Django"], [(0, 3, "HARD_SKILL")], "d8")
    rows, na = P.build([inspan], ["reorder_clauses"])
    assert not rows
    sections = sk(["*", "•", "Node", "JS", "*", "•", "React", "skills"], [(2, 4, "HARD_SKILL"), (6, 7, "HARD_SKILL")], "d10")
    rows, na = P.build([sections], ["reorder_clauses"])
    assert rows[0]["text"] == "* • React skills * • Node JS"
    check_spans(rows[0], sections, False)


def test_esco_alias_filter():
    ok = P.esco_alias_ok
    assert ok("TypeScript", "TypeScript programming language") and ok("Information Technology", "IT")
    assert ok("reactjs", "React.js") and ok("communication skills", "communication") and ok("IT", "information technology")
    assert not ok("PowerShell", "Tcl") and not ok("machine learning", "semi-supervised learning")
    assert not ok("python", "Python 3K") and not ok("German", "competent in German") and not ok("SQL", "Adaptive Server Anywhere")
    t = P.AliasTable({"esco:x": ["PowerShell", "Tcl", "PowerShell programming language"]}, [])
    assert t.alternatives("PowerShell") == ["PowerShell programming language"]
    t = P.AliasTable({"repo:x": ["machine learning", "ML"]}, [])
    assert t.alternatives("ML") == ["machine learning"]


def test_bullet_prefix_fully_removed_and_insertions_on_own_line():
    d = sk(["*", "•", "Advanced", "Python"], [(3, 4, "HARD_SKILL")], "d11")
    rows, _ = P.build([d], ["bullets_prose", "boilerplate", "distractor", "whitespace"])
    by = {r["meta"]["family"]: r for r in rows}
    assert by["bullets_prose"]["text"] == "Advanced Python ."
    for fam in ("boilerplate", "distractor"):
        assert "\n" in by[fam]["text"]
        check_spans(by[fam], d, False)
    assert "0 inter-token" not in by["whitespace"]["meta"]["note"]
