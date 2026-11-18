"""Schema checks for the L08/L09 per_unit_scores.jsonl exports consumed by L10 (skipped until the layers have been run)."""

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EVALS = ROOT / "reports" / "skillllm_v2" / "evals"
FILES = [EVALS / "L08_graph_retrieval" / "per_unit_scores.jsonl", EVALS / "L09_course_rec" / "per_unit_scores.jsonl"]


def _rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.parent.name)
def test_schema_and_declared_references_resolve(path):
    if not path.exists():
        pytest.skip("layer not run yet")
    rows = _rows(path)
    assert rows
    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        assert {"layer", "family", "system", "metric", "unit_id", "value"} <= set(r)
        assert r["value"] is not None and math.isfinite(float(r["value"]))
        g = (r["layer"], r["family"], r["metric"], r.get("bucket"))
        assert r["unit_id"] not in groups[g][r["system"]], f"duplicate unit {r['unit_id']} for {r['system']} in {g}"
        groups[g][r["system"]][r["unit_id"]] = r
    for g, by_sys in groups.items():
        for s, units in by_sys.items():
            refs = {u.get("reference") for u in units.values()}
            assert len(refs) == 1, f"{g} {s}: inconsistent reference declarations {refs}"
            ref = refs.pop()
            if ref:
                assert ref in by_sys, f"{g}: reference {ref!r} of {s!r} has no rows"
                assert len(set(units) & set(by_sys[ref])) >= 2, f"{g}: {s} and {ref} share <2 units"


def test_l09_headline_flags_only_on_full():
    p = EVALS / "L09_course_rec" / "results.json"
    if not p.exists():
        pytest.skip("L09 not run yet")
    recs = json.loads(p.read_text())["records"]
    head = {r["metric"] for r in recs if (r.get("details") or {}).get("headline")}
    assert {"precision@5", "wsgc@5", "prereq_violation_rate", "ndcg@10"} <= head
    assert all(r["system"] == "SkillLLM_v2 Full" for r in recs if (r.get("details") or {}).get("headline"))
    h6 = [r for r in recs if r["metric"] == "H6_verdict"]
    assert len(h6) == 1 and h6[0]["details"]["verdict"] in ("SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE", "NOT TESTABLE")
