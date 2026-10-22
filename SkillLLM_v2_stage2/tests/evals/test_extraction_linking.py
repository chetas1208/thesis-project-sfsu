"""Tests for the extraction + linking evals: hand-verifiable toy cases, controls, an independent span scorer
(nervaluate-style strict/exact/partial/ent_type, brute force or scipy assignment, no shared code with the
module under test) and the reproduction check against stored metrics."""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np
import pytest

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.evals import extraction_evals as E
from skillllm_v2.evals import linking_evals as L
from skillllm_v2.evals.contract import EvalContext

ROOT = Path(__file__).resolve().parents[2]
CTX = EvalContext(root=ROOT, n_boot=200)
H, S = "HARD_SKILL", "SOFT_SKILL"


def sp(start, end, label=H, surface="x"):
    return {"surface_form": surface, "label": label, "start": start, "end": end}


# ============================================================================ independent re-implementation
def indep_counts(gold, pred, mode):
    """(COR, PAR, ACT, POS) by brute force over all one-to-one assignments (small docs only).

    strict: exact boundaries + same type.  exact: exact boundaries, type ignored.  ent_type: any overlap + same type.
    partial: exact boundaries (type ignored) is COR, any other overlap (type ignored) is PAR; lexicographic max."""
    def overlap(a, b):
        return a["start"] is not None and min(a["end"], b["end"]) > max(a["start"], b["start"])

    def same_b(a, b):
        return a["start"] == b["start"] and a["end"] == b["end"]

    def kind(p, g):  # 2 = correct, 1 = partial, 0 = no match
        if mode == "strict":
            return 2 if same_b(p, g) and p["label"] == g["label"] else 0
        if mode == "exact":
            return 2 if same_b(p, g) else 0
        if mode == "ent_type":
            return 2 if overlap(p, g) and p["label"] == g["label"] else 0
        return 2 if same_b(p, g) else (1 if overlap(p, g) else 0)

    best = (0, 0)
    for assign in itertools.product(range(len(gold) + 1), repeat=len(pred)):
        used = [a for a in assign if a < len(gold)]
        if len(used) != len(set(used)):
            continue
        cor = par = 0
        for pi, a in enumerate(assign):
            if a < len(gold):
                k = kind(pred[pi], gold[a])
                cor += k == 2
                par += k == 1
        best = max(best, (cor, par))
    return best[0], best[1], len(pred), len(gold)


def indep_counts_scipy(gold, pred, mode):
    """Same quantity via a scipy assignment (weights make exact-boundary matches dominate partial ones)."""
    from scipy.optimize import linear_sum_assignment

    if not gold or not pred:
        return 0, 0, len(pred), len(gold)
    w = np.zeros((len(pred), len(gold)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if p["start"] is None:
                continue
            ov = min(p["end"], g["end"]) > max(p["start"], g["start"])
            sb = p["start"] == g["start"] and p["end"] == g["end"]
            if mode == "strict":
                w[i, j] = 1000 if sb and p["label"] == g["label"] else 0
            elif mode == "exact":
                w[i, j] = 1000 if sb else 0
            elif mode == "ent_type":
                w[i, j] = 1000 if ov and p["label"] == g["label"] else 0
            else:
                w[i, j] = 1000 if sb else (1 if ov else 0)
    rows, cols = linear_sum_assignment(-w)
    vals = [w[r, c] for r, c in zip(rows, cols)]
    return sum(v == 1000 for v in vals), sum(v == 1 for v in vals), len(pred), len(gold)


def indep_prf(counts):
    cor = sum(c[0] for c in counts)
    par = sum(c[1] for c in counts)
    act = sum(c[2] for c in counts)
    pos = sum(c[3] for c in counts)
    p = (cor + 0.5 * par) / act if act else 0.0
    r = (cor + 0.5 * par) / pos if pos else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ============================================================================ toy extraction metrics
TOY_TEXT = "aaaaa bbbbb ccccc"
TOY_GOLD = [sp(0, 5, H), sp(6, 11, S)]
TOY_PRED = [sp(0, 5, H), sp(6, 10, S), sp(12, 17, H)]


def test_toy_counts_are_hand_verifiable():
    row = E.doc_counts(TOY_GOLD, TOY_PRED, TOY_TEXT)
    # exact: (0,5,H) matches; (6,10,S) and (12,17,H) are FP; (6,11,S) is FN
    assert (row[E.C_TP], row[E.C_FP], row[E.C_FN]) == (1, 2, 1)
    assert row[E.C_PTP] == 2  # (6,10,S) overlaps (6,11,S) with the same label
    # character tokens: gold 5+5=10, pred 5+4+5=14, overlap 5+4=9
    assert (row[E.C_TTP], row[E.C_TFP], row[E.C_TFN]) == (9, 5, 1)
    # word tokens: gold {w0 H, w1 S}, pred {w0 H, w1 S, w2 H}
    assert (row[E.C_WTP], row[E.C_WFP], row[E.C_WFN]) == (2, 1, 0)
    assert (row[E.C_HTP], row[E.C_HFP], row[E.C_HFN]) == (1, 1, 0)
    assert (row[E.C_STP], row[E.C_SFP], row[E.C_SFN]) == (0, 1, 1)
    m = E.aggregate(np.array([row], dtype=float))
    assert m["exact_span_precision"] == pytest.approx(1 / 3)
    assert m["exact_span_recall"] == pytest.approx(1 / 2)
    assert m["exact_span_f1"] == pytest.approx(0.4)
    assert m["partial_span_precision"] == pytest.approx(2 / 3)
    assert m["partial_span_recall"] == pytest.approx(1.0)
    assert m["partial_span_f1"] == pytest.approx(0.8)
    assert m["token_precision"] == pytest.approx(9 / 14)
    assert m["token_recall"] == pytest.approx(0.9)
    assert m["word_token_f1"] == pytest.approx(0.8)
    assert m["hard_skill_f1"] == pytest.approx(2 / 3)
    assert m["soft_skill_f1"] == 0.0
    assert m["macro_f1"] == pytest.approx(1 / 3)


def test_aggregate_equals_repository_evaluator_on_toy_corpus():
    rng = random.Random(0)
    gold_docs, pred_docs, rows = [], [], []
    for _ in range(40):
        gold = [sp(a, a + rng.randint(1, 6), rng.choice([H, S])) for a in rng.sample(range(0, 60, 7), rng.randint(1, 4))]
        pred = [sp(max(0, g["start"] + rng.choice([-1, 0, 0, 1])), g["end"] + rng.choice([0, 0, 1]), g["label"] if rng.random() > 0.2 else S)
                for g in gold if rng.random() > 0.2] + [sp(rng.randint(0, 80), rng.randint(81, 90))]
        gold_docs.append(gold)
        pred_docs.append(pred)
        rows.append(E.doc_counts(gold, pred))
    ref = evaluate_extraction(gold_docs, pred_docs)
    mine = E.aggregate(np.array(rows, dtype=float))
    for k, v in ref.items():
        assert mine[k] == pytest.approx(v, abs=1e-12), k


# ============================================================================ independent scorer cross-check
@pytest.mark.parametrize("mode", E.NER_MODES)
def test_toy_nervaluate_modes_hand_values(mode):
    gold = [sp(0, 5, H), sp(6, 11, S), sp(20, 25, H)]
    pred = [sp(0, 5, H), sp(6, 10, S), sp(20, 25, S), sp(40, 45, H)]
    expected = {"strict": (1, 0, 4, 3), "exact": (2, 0, 4, 3), "partial": (2, 1, 4, 3), "ent_type": (2, 0, 4, 3)}[mode]
    # strict: only (0,5,H).  exact: + (20,25) ignoring type.  partial: exact boundaries count 1, (6,10)~(6,11) overlap 0.5.
    # ent_type: (0,5,H) and (6,10,S)~(6,11,S) overlap with same type.
    assert E.nervaluate_doc(gold, pred, mode) == expected
    assert indep_counts(gold, pred, mode) == expected


@pytest.mark.parametrize("mode", E.NER_MODES)
def test_module_nervaluate_matches_independent_bruteforce_on_random_docs(mode):
    rng = random.Random(7 + len(mode))
    for _ in range(250):
        def rand_spans(k):
            out, used = [], set()
            for _ in range(k):
                a = rng.randint(0, 30)
                b = a + rng.randint(1, 5)
                lab = rng.choice([H, S])
                if (a, b, lab) not in used:
                    used.add((a, b, lab))
                    out.append(sp(a, b, lab))
            return out

        gold, pred = rand_spans(rng.randint(0, 4)), rand_spans(rng.randint(0, 4))
        assert E.nervaluate_doc(gold, pred, mode) == indep_counts(gold, pred, mode), (gold, pred)


def test_strict_mode_equals_repository_exact_span_scorer_on_random_docs():
    rng = random.Random(3)
    gold_docs, pred_docs = [], []
    for _ in range(60):
        gold = [sp(a, a + 3, rng.choice([H, S])) for a in rng.sample(range(0, 50, 4), 3)]
        pred = [dict(g) if rng.random() < 0.6 else sp(g["start"] + 1, g["end"], g["label"]) for g in gold] + [sp(70, 75)]
        gold_docs.append(gold)
        pred_docs.append(pred)
    ref = evaluate_extraction(gold_docs, pred_docs)
    p, r, f = indep_prf([indep_counts(g, q, "strict") for g, q in zip(gold_docs, pred_docs)])
    assert f == pytest.approx(ref["exact_span_f1"], abs=1e-12)
    assert p == pytest.approx(ref["exact_span_precision"], abs=1e-12)
    assert r == pytest.approx(ref["exact_span_recall"], abs=1e-12)


def _real(system):
    path = ROOT / E.SYSTEMS[system]["pred"]
    if not path.exists() or not (ROOT / E.DATASET_ROOT / "test.json").exists():
        pytest.skip("saved predictions / SkillSpan data not available")
    gold = E.load_gold(CTX)
    preds = E.load_predictions(path)
    ids = [i for i in preds if i in gold]
    return ids, [gold[i]["gold_spans"] for i in ids], [preds[i] for i in ids]


@pytest.mark.parametrize("system", ["mistral_lora", "safe_wrapper"])
def test_real_data_independent_scorer_agrees_with_repository_scorer(system):
    pytest.importorskip("scipy")
    ids, gold_docs, pred_docs = _real(system)
    ref = evaluate_extraction(gold_docs, pred_docs)
    strict = indep_prf([indep_counts_scipy(g, p, "strict") for g, p in zip(gold_docs, pred_docs)])
    # unlocated predictions (start=None) are false positives in both implementations
    assert strict[2] == pytest.approx(ref["exact_span_f1"], abs=1e-9)
    # one-to-one overlap+type matching: the repository's greedy matcher can under-count versus an optimal assignment
    ent = indep_prf([indep_counts_scipy(g, p, "ent_type") for g, p in zip(gold_docs, pred_docs)])
    assert ent[2] >= ref["partial_span_f1"] - 1e-9
    assert ent[2] - ref["partial_span_f1"] < 0.01
    # the module's nervaluate implementation agrees exactly with the independent assignment solver
    for mode in E.NER_MODES:
        mine = E.nervaluate_counts(gold_docs, pred_docs)[mode].sum(axis=0)
        theirs = np.array([indep_counts_scipy(g, p, mode) for g, p in zip(gold_docs, pred_docs)]).sum(axis=0)
        assert tuple(mine) == tuple(theirs), mode


# ============================================================================ reproduction against stored metrics
@pytest.mark.parametrize("system", list(E.SYSTEMS))
def test_recomputed_metrics_reproduce_stored_metrics_within_1e6(system):
    ids, gold_docs, pred_docs = _real(system)
    counts = np.array([E.doc_counts(g, p) for g, p in zip(gold_docs, pred_docs)], dtype=float)
    metrics = E.aggregate(counts)
    rep = E.reproduction_check(system, metrics, CTX)
    if not rep["files"]:
        pytest.skip("stored metric files missing")
    assert rep["reproduced"], rep
    assert rep["max_abs_diff"] <= 1e-6
    ref = evaluate_extraction(gold_docs, pred_docs)
    for k in ref:
        assert abs(ref[k] - metrics[k]) <= 1e-6


# ============================================================================ error taxonomy
def test_error_taxonomy_toy_each_kind_once():
    gold = [sp(0, 5, H), sp(6, 11, S), sp(20, 25, H), sp(30, 35, H)]
    pred = [sp(0, 5, H), sp(0, 5, H), sp(6, 10, S), sp(20, 25, S), sp(40, 45, H)]
    ev = E.classify_errors(gold, pred)
    kinds = [e["kind"] for e in ev]
    assert sorted(kinds) == sorted(["duplicate", "boundary_shift", "label_confusion", "spurious", "missed"])
    by = {e["kind"]: e for e in ev}
    assert by["boundary_shift"]["gold"]["end"] == 11 and by["boundary_shift"]["pred"]["end"] == 10
    assert by["label_confusion"]["subtype"] == "same_boundaries"
    assert by["missed"]["gold"]["start"] == 30
    assert by["spurious"]["pred"]["start"] == 40
    # FP = boundary + label + spurious; FN = missed + boundary + label  (duplicates are invisible to the set scorer)
    row = E.doc_counts(gold, pred)
    assert row[E.C_FP] == 3 and row[E.C_FN] == 3
    # result does not depend on prediction order
    assert sorted(e["kind"] for e in E.classify_errors(gold, list(reversed(pred)))) == sorted(kinds)


def test_unlocated_prediction_counts_as_spurious_like_the_evaluator():
    pred = [{"surface_form": "foo", "label": H, "start": None, "end": None}]
    ev = E.classify_errors([sp(0, 3)], pred)
    assert sorted(e["kind"] for e in ev) == ["missed", "spurious"]
    assert E.doc_counts([sp(0, 3)], pred)[E.C_FP] == 1


def test_error_taxonomy_reconciles_with_evaluator_on_real_predictions():
    for system in ("mistral_lora", "safe_wrapper"):
        ids, gold_docs, pred_docs = _real(system)
        texts = ["" for _ in ids]
        tax = E.error_taxonomy(ids, gold_docs, pred_docs, [" " * 400 for _ in ids])
        c = tax["counts"]
        counts = np.array([E.doc_counts(g, p) for g, p in zip(gold_docs, pred_docs)])
        assert c["spurious"] + c["boundary_shift"] + c["label_confusion"] == counts[:, E.C_FP].sum()
        assert c["missed"] + c["boundary_shift"] + c["label_confusion"] == counts[:, E.C_FN].sum()
        for kind, n in c.items():
            if n:
                assert tax["examples"][kind] and tax["examples"][kind][0]["doc_id"] in ids


# ============================================================================ controls, CIs, slices
def test_shuffle_is_a_derangement_and_collapses_toy_f1():
    rng = random.Random(1)
    gold_docs = [[sp(i, i + 3, H)] for i in range(10)]
    pred_docs = [list(g) for g in gold_docs]  # perfect predictions
    shuffled = E.shuffled_predictions(pred_docs, rng)
    assert all(s != p for s, p in zip(shuffled, pred_docs))
    assert evaluate_extraction(gold_docs, pred_docs)["exact_span_f1"] == 1.0
    assert evaluate_extraction(gold_docs, shuffled)["exact_span_f1"] == 0.0
    ctrl = E.negative_control(gold_docs, pred_docs, n_shuffles=5)
    assert ctrl["mean"]["exact_span_f1"] == 0.0
    assert ctrl["positive_control_gold_as_pred"]["exact_span_f1"] == 1.0
    assert ctrl["empty_predictions"]["exact_span_f1"] == 0.0


def test_negative_control_collapses_on_real_predictions():
    ids, gold_docs, pred_docs = _real("mistral_lora")
    ctrl = E.negative_control(gold_docs, pred_docs, n_shuffles=5)
    real = evaluate_extraction(gold_docs, pred_docs)["exact_span_f1"]
    assert real > 0.6
    assert ctrl["max"]["exact_span_f1"] < 0.05
    # overlap-based metrics do not collapse to ~0 (short job-ad docs share positions by chance); they must still sit
    # far below the real score, which is why exact/strict F1 is the control's pass criterion
    assert ctrl["max"]["partial_span_f1"] < 0.30 and ctrl["max"]["partial_span_f1"] < 0.5 * evaluate_extraction(gold_docs, pred_docs)["partial_span_f1"]
    assert ctrl["max"]["token_f1"] < 0.5 * evaluate_extraction(gold_docs, pred_docs)["token_f1"]


def test_bootstrap_ci_is_deterministic_and_brackets_point_estimate():
    rng = np.random.default_rng(0)
    counts = np.zeros((120, E.N_COLS))
    counts[:, E.C_TP] = rng.integers(0, 4, 120)
    counts[:, E.C_FP] = rng.integers(0, 3, 120)
    counts[:, E.C_FN] = rng.integers(0, 3, 120)
    stat = E.METRICS["exact_span_f1"]
    point = float(stat(counts.sum(axis=0)))
    ci1 = E.boot_ci(counts, stat, 300, 5)
    ci2 = E.boot_ci(counts, stat, 300, 5)
    assert ci1 == ci2 and ci1[0] <= point <= ci1[1]
    wide = E.boot_ci(counts[:20], stat, 300, 5)
    assert wide[1] - wide[0] > ci1[1] - ci1[0]  # fewer documents -> wider interval
    d = E.paired_boot(counts, counts, stat, 200, 1)
    assert d["delta"] == 0.0 and d["ci_low"] == 0.0 and d["ci_high"] == 0.0


def test_slices_partition_documents_and_spans():
    ids, gold_docs, pred_docs = _real("mistral_lora")
    gold = E.load_gold(CTX)
    meta = [{"source": gold[i]["source"], "n_tokens": len(gold[i]["tokens"])} for i in ids]
    for family, buckets in E.doc_slices(ids, gold_docs, meta).items():
        flat = [i for idxs in buckets.values() for i in idxs]
        assert sorted(flat) == list(range(len(ids))), family
    texts = [gold[i]["text"] for i in ids]
    pairs = E.span_length_pairs(gold_docs, pred_docs, texts, "gold")
    n_gold = sum(len({E._key(s) for s in g}) for g in gold_docs)
    assert sum(arr[:, 2].sum() for arr in pairs.values()) == n_gold


def test_extraction_run_smoke_results_are_well_formed():
    if not (ROOT / E.SYSTEMS["mistral_lora"]["pred"]).exists():
        pytest.skip("artifacts missing")
    res = E.run(EvalContext(root=ROOT, smoke=True, n_boot=50, extra={"smoke_docs": 40}))
    names = {r.eval_name for r in res}
    assert "extract.mistral_lora.exact_span_f1" in names
    assert "extract.safe_wrapper.negative_control" in names
    assert any(n.startswith("extract.mistral_lora.slice.source_domain") for n in names)
    for r in res:
        assert r.module == "extract" and r.dataset and r.status in {"ok", "skipped", "failed"}
        assert r.status != "failed", (r.eval_name, r.reason)
        if r.status == "ok":
            assert r.n > 0 and r.caveats
    repro = [r for r in res if r.eval_name.endswith(".reproduction")]
    assert repro and all(r.status == "skipped" and r.reason for r in repro)  # subset: stored metrics are full-set


# ============================================================================ linking: toy
def toy_rows():
    def r(cid, name, alias, src="manual_curated"):
        return {"canonical_skill_id": cid, "canonical_skill_name": name, "alias": alias,
                "normalized_alias": alias.lower(), "taxonomy": "CUSTOM", "label_hint": None, "source": src}

    return [r("amazon-web-services", "Amazon Web Services", "AWS"), r("amazon-web-services", "Amazon Web Services", "Amazon Web Services"),
            r("python", "Python", "Python"), r("kubernetes", "Kubernetes", "Kubernetes"),
            r("machine-learning", "machine learning", "ML"), r("machine-learning", "machine learning", "machine learning"),
            r("communication", "communication", "communication"), r("communication", "communication", "communication skills"),
            r("data-science", "data science", "data science")]


@pytest.fixture()
def toy_linker(tmp_path):
    path = tmp_path / "toy_aliases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in toy_rows()) + "\n", encoding="utf-8")
    return L.make_linker(CTX, table=path)


def test_item_scores_hand_values():
    ranked = [{"canonical_skill_id": c, "confidence": 1.0, "method": "alias"} for c in ("a", "b", "c")]
    top1 = {"canonical_skill_id": "a"}
    s = L.item_scores("b", ranked, top1)
    assert (s["recall@1"], s["recall@5"], s["recall@10"], s["rr"]) == (0.0, 1.0, 1.0, 0.5)
    assert s["correct_top1"] == 0.0 and s["linked"] == 1.0
    s = L.item_scores("a", ranked, top1)
    assert (s["recall@1"], s["rr"], s["correct_top1"]) == (1.0, 1.0, 1.0)
    s = L.item_scores("zzz", ranked, top1)
    assert (s["recall@10"], s["rr"]) == (0.0, 0.0)
    s = L.item_scores("c", ranked, None)
    assert s["linked"] == 0.0 and s["recall@5"] == 1.0 and s["rr"] == pytest.approx(1 / 3)


def test_rank_one_equals_link_and_stage_order(toy_linker):
    for surf in ["AWS", "aws", "Python", "Kubernets", "machine", "communicaton skills", "data", "pottery", "ML", "science"]:
        link, ranked = toy_linker.link(surf), L.rank_candidates(toy_linker, surf)
        if link is None:
            assert ranked == [], surf
        else:
            assert ranked[0]["canonical_skill_id"] == link["canonical_skill_id"], surf
            assert ranked[0]["method"] == link["link_method"], surf
    r = L.rank_candidates(toy_linker, "AWS")
    assert r[0]["method"] == "alias" and r[0]["confidence"] == 1.0
    assert len({c["canonical_skill_id"] for c in r}) == len(r)  # de-duplicated
    assert L.rank_candidates(toy_linker, "Kubernets")[0]["method"] == "fuzzy"
    assert L.rank_candidates(toy_linker, "machine")[0]["method"] == "bm25"


def test_ambiguity_margin():
    mk = lambda *confs: [{"canonical_skill_id": str(i), "confidence": c, "method": "bm25"} for i, c in enumerate(confs)]
    assert L.margin_of(mk(0.8)) is None and not L.is_ambiguous(mk(0.8))
    assert L.is_ambiguous(mk(0.80, 0.78)) and not L.is_ambiguous(mk(0.84, 0.60))


def test_ratio_ci_hand_values():
    ci = L._ratio_ci([1, 1, 1, 1], [2, 2, 2, 2], EvalContext(root=ROOT, n_boot=100))
    assert ci == (0.5, 0.5)
    ci = L._ratio_ci([0, 0, 1, 1], [1, 1, 1, 1], EvalContext(root=ROOT, n_boot=300))
    assert ci[0] < 0.5 < ci[1]


def test_abstention_never_removes_skills_and_the_check_has_teeth(toy_linker, tmp_path):
    cands = [[{"surface_form": s, "label": H, "start": i, "end": i + 3, "source": "mistral", "confidence": 0.9,
               "evidence": s, "canonical_skill_id": None, "canonical_skill_name": None, "metadata": {}}
              for i, s in enumerate(["AWS", "machine", "pottery", "Kubernets", "forklift operation"])]]
    rep = L.check_never_removes(toy_linker, cands)
    assert rep["skills_removed"] == 0 and rep["skills_edited"] == 0 and rep["skills_in"] == 5
    assert rep["kept_without_link"] >= 1  # unlinked skills are kept
    for thr in (0.0, 0.88, 1.01):
        path = tmp_path / "t.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in toy_rows()) + "\n", encoding="utf-8")
        r2 = L.check_never_removes(L.make_linker(CTX, fuzzy_threshold=thr, table=path), cands)
        assert r2["skills_removed"] == 0 and r2["skills_edited"] == 0

    class Dropping(type(toy_linker)):  # a (wrong) linker that abstains by dropping unlinked skills
        def link_candidates(self, candidates):
            return [c for c in super().link_candidates(candidates) if c.get("canonical_skill_id")]

    bad = Dropping(alias_table_paths=[str(tmp_path / "t.jsonl")])
    assert L.check_never_removes(bad, cands)["skills_removed"] >= 1


def test_negative_control_random_ids_near_zero_but_oracle_is_perfect():
    table = L.load_table(CTX)
    ids = sorted({r["canonical_skill_id"] for r in table})
    rng = random.Random(0)
    items = [{"gold_id": g, "top1": {"canonical_skill_id": g}} for g in rng.sample(ids, 60)]  # oracle linker, distinct golds
    res = L.negative_control_result(items, CTX, table, n_draws=100)
    assert res.status == "ok"
    assert res.value < 0.02 and res.details["random_ids_top1_max"] < 0.1
    assert res.details["shuffled_links_across_items_top1_mean"] == 0.0  # distinct golds, deranged links
    assert res.details["expected_chance_top1"] == pytest.approx(1 / len(ids))
    assert sum(i["gold_id"] == i["top1"]["canonical_skill_id"] for i in items) / len(items) == 1.0


# ============================================================================ linking: real table / probe / run
def test_probe_file_is_valid_hand_written_supplementary_data():
    rows, problems = L.load_probe(CTX)
    assert not problems, problems
    assert len(rows) >= 100
    assert all(r["source"] == "hand-written" for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    cats = {r["category"] for r in rows}
    assert {"exact", "alias", "fuzzy_typo", "phrase", "ambiguous", "no_match"} <= cats
    assert sum(r["gold_id"] is None for r in rows) >= 10


def test_probe_metrics_are_consistent():
    cache = L._Cache(L.make_linker(CTX))
    res = {r.eval_name: r for r in L.probe_results(CTX, cache)}
    r1, r5, r10, mrr = (res[f"link.probe.{k}"].value for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"))
    assert r1 <= mrr <= r10 and r1 <= r5 <= r10
    assert res["link.probe.category.exact.recall_at_1"].value == 1.0  # table entries link to themselves
    assert res["link.probe.rank_consistency"].value == 1.0 and res["link.probe.rank_consistency"].status == "ok"
    assert res["link.negative_control"].status == "ok" and res["link.negative_control"].value < 0.02
    assert res["link.probe.method.dense.precision"].status == "skipped"
    for r in res.values():
        assert r.dataset.startswith("linking_probe") and r.caveats or r.status == "skipped"
    assert res["link.probe.coverage"].n == len(L.load_probe(CTX)[0])
    assert res["link.probe.no_match_false_link_rate"].details["false_links"] is not None


def test_leave_one_alias_out_items_come_from_curated_rows_only():
    items = L.loo_items(CTX)
    table = L.load_table(CTX)
    assert items and all(table[i["row_index"]]["source"] == "manual_curated" for i in items)
    scored = L.score_loo(CTX, items[:3])
    assert all(s["source"] == "alias_table_loo" for s in scored)


def test_linking_run_smoke_skips_what_it_cannot_measure_and_never_fails():
    if not (ROOT / L.PRED_PATH).exists():
        pytest.skip("artifacts missing")
    res = L.run(EvalContext(root=ROOT, smoke=True, n_boot=50, extra={"smoke_docs": 25}))
    by = {r.eval_name: r for r in res}
    assert by["link.skillspan_gold_links"].status == "skipped" and "no taxonomy ids" in by["link.skillspan_gold_links"].reason
    assert by["link.method.dense.share"].status == "skipped"
    assert by["link.abstention.never_removes_skill"].status == "ok" and by["link.abstention.never_removes_skill"].value == 1.0
    assert by["link.relink_agreement"].value == 1.0
    assert 0.0 < by["link.coverage"].value <= 1.0 and by["link.coverage"].ci is not None
    for r in res:
        assert r.module == "link" and r.dataset and r.status in {"ok", "skipped", "failed"}
        assert r.status != "failed", (r.eval_name, r.reason)
        if r.status == "skipped":
            assert r.reason
        else:
            assert r.n > 0 and r.caveats, r.eval_name
    # shares of link methods (incl. none) partition the predicted spans
    total = sum(by[f"link.method.{m}.share"].value for m in ("alias", "normalized", "fuzzy", "bm25", "none"))
    assert total == pytest.approx(1.0)


# ============================================================================ optional nervaluate oracle
def _nervaluate_results(gold, pred):
    """Run the real nervaluate package if importable (0.2.0 API; guarded because 1.x changed it)."""
    nv = pytest.importorskip("nervaluate")
    version = getattr(nv, "__version__", "0")
    if version.split(".")[0] not in ("0",):
        pytest.skip(f"nervaluate {version}: API differs from the pinned 0.2.0")
    to = lambda docs: [[{"label": s["label"], "start": s["start"], "end": s["end"] - 1} for s in d] for d in docs]  # inclusive end
    ev = nv.Evaluator(to(gold), to(pred), tags=[H, S], loader="list")
    out = ev.evaluate()
    return out[0] if isinstance(out, tuple) else out


def test_optional_nervaluate_oracle_strict_and_exact_agree():
    rng = random.Random(11)
    gold_docs, pred_docs = [], []
    for _ in range(40):
        gold = [sp(a, a + 3, rng.choice([H, S])) for a in rng.sample(range(0, 60, 5), 3)]
        pred = [dict(g) if rng.random() < 0.6 else sp(g["start"], g["end"] + 1, g["label"]) for g in gold] + [sp(80, 85, H)]
        gold_docs.append(gold)
        pred_docs.append(pred)
    res = _nervaluate_results(gold_docs, pred_docs)
    nc = E.nervaluate_counts(gold_docs, pred_docs)
    for mode in ("strict", "exact"):
        _, _, f = E.ner_prf(nc[mode].sum(axis=0))
        assert float(f) == pytest.approx(res[mode]["f1"], abs=1e-9), mode
