"""Leaderboard aggregation on synthetic layer outputs (tolerates missing layers; never invents values)."""

import json

from skillllm_v2.evals.research import leaderboard as lb
from skillllm_v2.evals.research.common import record


def _write(root, layer, records):
    d = root / layer
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"meta": {"utc": "2026-11-17T00:00:00+00:00", "git": {"commit": "abc"}},
                                                "records": records}))


def _rec(layer, metric, value, **kw):
    base = dict(n_units=10, unit="document", split="test", dataset="toy", direction="higher")
    base.update(kw)
    return record(layer, "sys", metric, value, **base)


def test_missing_layers_and_headline_and_verdicts(tmp_path, monkeypatch):
    monkeypatch.setattr(lb, "PROGRESS", tmp_path / "Progress.md")
    (tmp_path / "Progress.md").write_text("t | A3 | integrity-flag | something odd\nt | A1 | progress | fine\n")
    _write(tmp_path, "L01_extraction", [
        _rec("L01_extraction", "exact_span_f1", 0.5, ci=(0.4, 0.6), details={"headline": True}),
        _rec("L01_extraction", "H1_verdict", 0.3, ci=(0.2, 0.4), caveats=["straw-man baseline"],
             details={"verdict": "SUPPORTED", "rule": "r", "p_raw": 0.001, "p_holm": 0.002, "family": "H1"}),
    ])
    _write(tmp_path, "L04_calibration", [
        _rec("L04_calibration", "H4_verdict", None, status="pending_gpu", reason="no logprobs yet"),
        _rec("L04_calibration", "ece", None, status="pending_gpu", reason="no logprobs yet", details={"headline": True}),
    ])
    out = lb.build(root=tmp_path)
    status = {s["layer"]: s["status"] for s in out["layers"]}
    assert status["L01_extraction"] == "measured"
    assert status["L04_calibration"] == "pending_gpu"
    assert status["L12_systems"] == "missing"
    heads = {(h["layer"], h["metric"]): h for h in out["headline"]}
    assert heads[("L01_extraction", "exact_span_f1")]["value"] == 0.5
    assert heads[("L04_calibration", "ece")]["value"] is None
    v = {r["hypothesis"]: r for r in out["hypotheses"]}
    assert v["H1"]["verdict"] == "SUPPORTED" and v["H1"]["delta"] == 0.3 and "straw-man" in v["H1"]["caveat"]
    assert v["H4"]["verdict"] == "NOT TESTABLE" and v["H4"]["status"] == "pending_gpu"
    assert v["H7"]["verdict"] == "NOT YET REPORTED"
    assert len(out["integrity_flags"]) == 1
    stm = lb._verdict_statements(out)
    assert "**H1: SUPPORTED**" in stm and "**H4: NOT TESTABLE**" in stm and "**H7: NOT YET REPORTED**" in stm
    assert "Pre-registered rule: r." in stm and ".." not in stm.split("H4")[0]
    md = lb.render_md(out)
    assert "NOT YET REPORTED" in md and "0.5000" in md and "[0.4000, 0.6000]" in md
    tex = lb.render_tex(out)
    assert r"\toprule" in tex["evaluation_hypotheses.tex"] and "SUPPORTED" in tex["evaluation_hypotheses.tex"]
    assert r"exact\_span\_f1" in tex["evaluation_headline.tex"]


def test_chapter_blocks_only_replace_marked_regions(tmp_path):
    out = lb.build(root=tmp_path)  # nothing exists -> everything missing, nothing invented
    text = ("Intro prose stays.\n<!-- BEGIN AUTO:hypotheses -->\nold\n<!-- END AUTO:hypotheses -->\n"
            "Middle prose.\n<!-- BEGIN AUTO:layer:L09_course_rec -->x<!-- END AUTO:layer:L09_course_rec -->\n"
            "<!-- BEGIN AUTO:unknown_block -->keep me<!-- END AUTO:unknown_block -->\n")
    new = lb.fill_chapter(text, lb.chapter_blocks(out))
    assert new.startswith("Intro prose stays.") and "Middle prose." in new
    assert "old" not in new and "NOT YET REPORTED" in new
    assert "does not exist yet" in new
    assert "keep me" in new
    assert lb.fill_chapter(new, lb.chapter_blocks(out)).count("BEGIN AUTO:hypotheses") == 1


def test_tex_escaping():
    assert lb.tex("a_b & 50% #1") == r"a\_b \& 50\% \#1"
