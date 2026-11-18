"""L11 packet scoring, span-cell parsing and blinding checks (synthetic fixtures; built packets checked if present)."""

import csv

import pytest

from skillllm_v2.evals.research import human_eval_samples as he


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_parse_span_cell_occurrences_and_problems():
    text = "Python and SQL; more Python later"
    spans, probs = he.parse_span_cell(text, "Python|knowledge; Python#2|KNOWLEDGE; SQL|KNOWLEDGE; Rust|KNOWLEDGE; bad")
    assert (0, 6, "KNOWLEDGE") in spans and (21, 27, "KNOWLEDGE") in spans and (11, 14, "KNOWLEDGE") in spans
    assert len(probs) == 2


def test_score_packet_pending_then_measured(tmp_path):
    recs = he.score_packet("P5", tmp_path, 4, 50, 42)
    assert recs and all(r["status"] == "pending_human" for r in recs)
    rows_a = [{"item_id": f"P5-{i}", "q1_relevance": str(g), "q2_gap_closure": "partial", "q3_prereq_ready": "yes"}
              for i, g in enumerate([0, 1, 2, 3])]
    rows_b = [{"item_id": f"P5-{i}", "q1_relevance": str(g), "q2_gap_closure": "partial", "q3_prereq_ready": "yes"}
              for i, g in enumerate([0, 1, 3, 3])]
    _write(tmp_path / "P5__A.csv", rows_a)
    _write(tmp_path / "P5__B.csv", rows_b)
    recs = he.score_packet("P5", tmp_path, 4, 200, 42)
    rel = [r for r in recs if r["metric"] == "q1_relevance_cohen_kappa_quadratic"][0]
    assert rel["status"] == "measured" and 0.5 < rel["value"] < 1.0
    todo = he.adjudication_todo("P5", tmp_path)
    with open(todo, newline="") as fh:
        assert [r["item_id"] for r in csv.DictReader(fh)] == ["P5-2"]


def test_score_span_packet(tmp_path):
    sents = {"P2-1": "Use Python and SQL daily", "P2-2": "No skills here at all"}
    _write(tmp_path / "P2__A.csv", [{"item_id": "P2-1", "spans": "Python|KNOWLEDGE; SQL|KNOWLEDGE", "completed": "y"},
                                    {"item_id": "P2-2", "spans": "", "completed": "y"}])
    _write(tmp_path / "P2__B.csv", [{"item_id": "P2-1", "spans": "Python|KNOWLEDGE", "completed": "y"},
                                    {"item_id": "P2-2", "spans": "", "completed": "y"}])
    recs = he.score_packet("P2", tmp_path, 2, 100, 42, sents)
    ex = [r for r in recs if r["metric"] == "span_iaa_f1_exact"][0]
    assert ex["status"] == "measured" and ex["value"] == pytest.approx(2 / 3)


def test_explanation_text_top_level_fields():
    rec = {"explanation_path": ["Role requires X -> course teaches X"], "limitations": ["60 hours"]}
    txt = he._explanation_text(rec)
    assert "course teaches X" in txt and "Limitations: 60 hours" in txt
    assert he._explanation_text({}) == ""


HIDDEN = ("system", "gold", "stratum", "auto_", "rule_label", "rank", "in_gold", "method", "doc_id", "scenario_id", "course_id")


@pytest.mark.parametrize("packet", ["P1", "P2", "P3", "P4a", "P4b", "P5", "P6"])
def test_built_packets_are_blind(packet):
    path = he.OUT / "packets" / f"{packet}__A.csv"
    if not path.exists():
        pytest.skip("packets not built")
    with open(path, newline="", encoding="utf-8") as fh:
        cols = next(csv.reader(fh))
    assert not [c for c in cols if any(h in c.lower() for h in HIDDEN)], cols
    with open(he.OUT / "packets" / f"{packet}__B.csv", newline="", encoding="utf-8") as fh:
        ids_b = [r["item_id"] for r in csv.DictReader(fh)]
    with open(path, newline="", encoding="utf-8") as fh:
        ids_a = [r["item_id"] for r in csv.DictReader(fh)]
    assert sorted(ids_a) == sorted(ids_b) and (len(ids_a) < 5 or ids_a != ids_b)
