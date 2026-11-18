"""Tests for skillllm_v2.evals.research.extraction_full (tiny synthetic fixtures, hand-checkable values)."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from skillllm_v2.evals import extraction_evals as EE
from skillllm_v2.evals.research import extraction_full as X


def sp(s, e, lab="HARD_SKILL"):
    return {"start": s, "end": e, "label": lab, "surface_form": "x"}


# ----------------------------------------------------------------------------- SemEval / nervaluate
def test_semeval_strict_exact_partial_type_hand_example():
    gold = [sp(0, 10), sp(20, 30, "SOFT_SKILL"), sp(40, 50)]
    pred = [sp(0, 10), sp(20, 30, "HARD_SKILL"), sp(42, 50), sp(60, 70)]
    # strict: COR (0,10); (20,30) wrong label -> INC; (42,50) overlaps -> INC; (60,70) -> SPU; no MIS
    assert X.semeval_doc(gold, pred, "strict") == (1, 2, 0, 0, 1)
    # exact (boundaries only): (0,10) COR, (20,30) COR, (42,50) INC, SPU 1
    assert X.semeval_doc(gold, pred, "exact") == (2, 1, 0, 0, 1)
    # partial: (0,10) COR, (20,30) COR (boundaries), (42,50) PAR, SPU 1
    assert X.semeval_doc(gold, pred, "partial") == (2, 0, 1, 0, 1)
    # ent_type: (0,10) COR, (20,30) INC (label), (42,50) COR (overlap + same label), SPU 1
    assert X.semeval_doc(gold, pred, "ent_type") == (2, 1, 0, 0, 1)
    p, r, f = X.sem_prf(np.array([2.0, 0, 1, 0, 1]), "partial")
    assert p == pytest.approx(2.5 / 4) and r == pytest.approx(2.5 / 3)


def test_semeval_greedy_quirk_and_missed():
    # an overlapping prediction listed first consumes the gold as INC, so the later exact prediction is SPU
    gold = [sp(0, 10)]
    pred = [sp(0, 5), sp(0, 10)]
    assert X.semeval_doc(gold, pred, "strict") == (0, 1, 0, 0, 1)
    # canonical order puts (0,5) first too, so the quirk is order-independent but present; one-to-one would give COR 1
    assert EE.nervaluate_doc(gold, pred, "strict")[0] == 1
    assert X.semeval_doc([sp(0, 3), sp(5, 9)], [], "strict") == (0, 0, 0, 2, 0)


def test_canonical_order_dedups_and_sorts():
    pred = [sp(5, 9), sp(0, 3), sp(5, 9), {"start": None, "end": None, "label": "HARD_SKILL"}]
    out = X.canonical_order(pred)
    assert [(p["start"], p["end"]) for p in out] == [(0, 3), (5, 9), (None, None)]


def _random_doc(rng, n_chars=60):
    def spans(k):
        out = []
        for _ in range(k):
            s = rng.randrange(0, n_chars - 2)
            out.append(sp(s, min(n_chars, s + rng.randrange(1, 12)), rng.choice(X.LABELS)))
        return out
    gold = []
    pos = 0
    while pos < n_chars - 4 and len(gold) < 4:  # non-overlapping gold per label, like SkillSpan
        s = pos + rng.randrange(0, 6)
        e = min(n_chars, s + rng.randrange(1, 8))
        if e > s:
            gold.append(sp(s, e, rng.choice(X.LABELS)))
        pos = e + 1
    return gold, spans(rng.randrange(0, 5))


def test_nervaluate_oracle_matches_package(tmp_path):
    if not Path(X.DEFAULT_NERVALUATE_PY).exists():
        pytest.skip("nervaluate throwaway venv not present")
    rng = random.Random(7)
    docs = [_random_doc(rng) for _ in range(40)]
    gold_docs, pred_docs = [d[0] for d in docs], [d[1] for d in docs]
    texts = ["a " * 30 for _ in docs]
    evals = {"toy": {"gold_docs": gold_docs, "pred_docs": pred_docs,
                     "rows": X.score_documents(gold_docs, pred_docs, texts)}}
    res = X.nervaluate_oracle(evals, tmp_path)
    assert res["available"], res
    assert res["all_match"], res["per_system"]


# ----------------------------------------------------------------------------- IoU
def test_iou_doc_exact_partial_and_threshold():
    text = "alpha beta gamma delta epsilon"
    gold = [sp(0, 10)]  # "alpha beta"
    assert X.iou_doc(gold, [sp(0, 10)], text)[:3] == [1.0, 1.0, 1.0]
    row = X.iou_doc(gold, [sp(0, 5)], text)  # "alpha": char IoU 5/10, word IoU 1/2
    assert row[0] == pytest.approx(0.5) and row[1] == 1 and row[2] == 1  # 0.5 counts as >= 0.5
    assert row[3] == pytest.approx(0.5) and row[5] == 1
    row = X.iou_doc(gold, [sp(0, 3)], text)  # char IoU 0.3 -> not >= 0.5
    assert row[2] == 0
    assert X.iou_doc(gold, [sp(0, 10, "SOFT_SKILL")], text)[1] == 0  # labels must match
    assert X.iou_doc(gold, [sp(0, 10), sp(0, 10)], text)[6] == 1  # duplicates removed


# ----------------------------------------------------------------------------- boundary taxonomy
@pytest.mark.parametrize("pred,kind,sub", [
    (sp(8, 20), "left_extension", ""), (sp(10, 25), "right_extension", ""), (sp(12, 20), "left_truncation", ""),
    (sp(10, 15), "right_truncation", ""), (sp(8, 25), "both_side_shift", "superset"),
    (sp(12, 18), "both_side_shift", "subset"), (sp(12, 25), "both_side_shift", "crossing"),
    (sp(10, 20, "SOFT_SKILL"), "label_confusion_exact_boundary", "HARD_SKILL->SOFT_SKILL"),
    (sp(12, 25, "SOFT_SKILL"), "overlap_wrong_label", "HARD_SKILL->SOFT_SKILL"),
])
def test_boundary_one_to_one(pred, kind, sub):
    ev = X.boundary_events([sp(10, 20)], [pred])
    assert len(ev) == 1 and ev[0]["kind"] == kind and ev[0]["subtype"] == sub and ev[0]["fp"] == ev[0]["fn"] == 1


def test_boundary_split_merge_spurious_missed():
    assert [e["kind"] for e in X.boundary_events([sp(10, 30)], [sp(10, 18), sp(20, 30)])] == ["split"]
    ev = X.boundary_events([sp(10, 18), sp(20, 30)], [sp(10, 30)])
    assert ev[0]["kind"] == "merge" and ev[0]["fn"] == 2 and ev[0]["fp"] == 1
    ev = X.boundary_events([sp(0, 5)], [sp(0, 5), sp(2, 4), sp(40, 45)])
    assert sorted(e["subtype"] for e in ev) == ["no_overlap", "overlaps_correct_gold"]
    assert [e["kind"] for e in X.boundary_events([sp(0, 5)], [])] == ["missed"]
    ev = X.boundary_events([], [{"start": None, "end": None, "label": "HARD_SKILL", "surface_form": "q"}])
    assert ev[0]["kind"] == "spurious_unlocated"


def test_boundary_mislocated_occurrence():
    text = "javascript reactjs java"
    ev = X.boundary_events([sp(19, 23)], [sp(0, 4)], text)  # "java" located inside "javascript"
    assert ev[0]["kind"] == "missed" or ev[0].get("mislocated_occurrence")
    assert any(e.get("mislocated_occurrence") for e in ev)


def test_taxonomy_reconciles_with_evaluator_on_random_docs():
    rng = random.Random(3)
    for _ in range(200):
        gold, pred = _random_doc(rng)
        row = EE.doc_counts(gold, pred)
        ev = X.boundary_events(gold, pred)
        assert sum(e["fp"] for e in ev) == row[EE.C_FP]
        assert sum(e["fn"] for e in ev) == row[EE.C_FN]


# ----------------------------------------------------------------------------- bootstrap + rows
def test_boot_multi_shared_draws_and_point_inside():
    rng = np.random.default_rng(0)
    rows = np.stack([rng.integers(0, 5, 80), rng.integers(0, 3, 80), rng.integers(0, 3, 80)], axis=1).astype(float)
    f1 = lambda S: X._div(2 * S[..., 0], 2 * S[..., 0] + S[..., 1] + S[..., 2])  # noqa: E731
    a = X.boot_multi(rows, {"f1": f1}, 500, 42)
    b = X.boot_multi(rows, {"f1": f1, "p": lambda S: X._div(S[..., 0], S[..., 0] + S[..., 1])}, 500, 42)
    assert a["f1"] == b["f1"]  # adding a statistic does not change the draws
    point = float(f1(rows.sum(axis=0)))
    assert a["f1"][0] <= point <= a["f1"][1]
    ci, samples = X.boot_multi(rows, {"f1": f1}, 100, 1, return_samples=True)
    assert samples["f1"].shape == (100,)


def test_cluster_rows_sums_by_cluster():
    rows = np.array([[1.0, 0], [2, 1], [3, 4]])
    out = X.cluster_rows(rows, ["b", "a", "b"])
    assert out.tolist() == [[2, 1], [4, 4]]


def test_score_documents_metrics_match_repo_evaluator():
    rng = random.Random(11)
    docs = [_random_doc(rng) for _ in range(30)]
    gold_docs, pred_docs = [d[0] for d in docs], [d[1] for d in docs]
    rows = X.score_documents(gold_docs, pred_docs, ["w " * 30] * 30)
    S = rows.sum(axis=0)
    ref = EE.evaluate_extraction(gold_docs, pred_docs)
    for k in ("exact_span_f1", "token_f1", "partial_span_f1", "macro_f1"):
        assert X.METRIC_TABLE[k][0](S) == pytest.approx(ref[k], abs=1e-12)


# ----------------------------------------------------------------------------- claims reconciliation
def test_rounds_to_and_attribution():
    assert X._rounds_to("0.6701", 0.670069) and not X._rounds_to("0.6702", 0.670069)
    assert X._rounds_to("0.670069", 0.6700687818948303)
    assert "mistral_lora" in X._attribute_systems("Raw Mistral LoRA (anchor) Exact Span F1")
    assert X._attribute_systems("Mistral + Boundary Refiner (safe wrapper)") == ["safe_wrapper_boundary_refiner"]
    assert X._attribute_metric("Token F1") == "token" and X._attribute_metric("Exact Span F1") == "exact_span"


def test_reconcile_claims_on_toy_repo(tmp_path):
    (tmp_path / "README.md").write_text(
        "| System | Exact Span F1 | Sign-test p |\n|---|---|---|\n| Raw Mistral LoRA (anchor) | 0.670069 | 0.0078 |\n"
        "| Old Full CAG/Graph | 0.675034 | 0.5 |\n\nThe anchor exact F1 is 0.6701 on the dev set 0.7029.\n")
    comps = [("mistral_lora", "exact_span_f1", 0.6700687818948303), ("full_cag_graph", "exact_span_f1", 0.6711022399645153)]
    out = X.reconcile_claims(comps, root=tmp_path)
    st = {(r["claimed"]): r["status"] for r in out["rows"]}
    assert st["0.670069"] == "match"
    assert st["0.675034"] == "mismatch"
    assert st["0.6701"] == "match"
    assert "0.0078" not in st  # p-value column skipped
    assert out["known_values"]["0.675034"]["occurrences"] == 1
