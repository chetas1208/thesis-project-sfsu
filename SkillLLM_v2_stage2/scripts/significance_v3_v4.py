"""Paired significance for the short-prompt (v3) and distillation (v4) adapters on the 974 held-out SkillSpan test docs.

Uses the same machinery as scripts/significance_qlora_v2.py (skillllm_v2.eval.significance unchanged): paired document bootstrap, sign-flip
permutation, 5 metrics, Holm (within a comparison over 5 metrics, and across the pre-declared family of exact-span tests of that experiment).

    PYTHONPATH=src python scripts/significance_v3_v4.py --exp v3   # v3 vs v2 (non-inferiority), v3 vs anchor
    PYTHONPATH=src python scripts/significance_v3_v4.py --exp v4   # v4 vs anchor (A), v4 vs ensemble (B), v4 vs v2

Pre-declared criteria are evaluated in the output (see docs/GPU_EXPERIMENTS_LOG.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_eval as E  # noqa: E402
import significance_qlora_v2 as Q  # noqa: E402

from skillllm_v2.eval import significance as S  # noqa: E402
from skillllm_v2.io_utils import write_json  # noqa: E402

PRED = {
    "v3": "artifacts/skillllm_v2/qlora_loop_v3_eval/predictions.jsonl",
    "v4": "artifacts/skillllm_v2/qlora_loop_v4_eval/predictions.jsonl",
}
LABEL = {"anchor": "Raw Mistral LoRA anchor", "v1": "QLoRA-loop v1", "v2": "QLoRA-loop v2", "ensemble": "3-adapter majority ensemble (>=2/3)",
         "v3": "v3 short prompt", "v4": "v4 distilled (single adapter)"}
FAMILY = {"v3": [("v3", "v2"), ("v3", "anchor")], "v4": [("v4", "anchor"), ("v4", "ensemble"), ("v4", "v2")]}


def exact_pr(m) -> tuple[float, float]:
    tp, fp, fn = (m[:, c].sum() for c in S.COUNT_COLUMNS["exact_span_f1"])
    return float(tp / max(tp + fp, 1)), float(tp / max(tp + fn, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["v3", "v4"], required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--pred", default=None, help="override predictions path of the new system")
    args = ap.parse_args()
    exp = args.exp
    out_prefix = args.out_prefix or f"reports/skillllm_v2/qlora_loop_{exp}_significance"
    t0 = time.time()
    docs = S.load_gold_docs()
    ids = [d["id"] for d in docs]
    gold = [d["gold_spans"] for d in docs]
    paths = {"anchor": S.ANCHOR_PREDICTIONS, "v1": E.SYSTEMS["v1"], "v2": E.SYSTEMS["v2"], exp: args.pred or PRED[exp]}
    preds = {k: S.load_system_predictions(p, ids) for k, p in paths.items()}
    preds["ensemble"] = E.vote([preds["anchor"], preds["v1"], preds["v2"]], 2)
    mats = {k: S.count_matrix(gold, p) for k, p in preds.items()}
    metrics_json = Path(paths[exp]).parent / "metrics.json"
    repro = S.assert_reproduction(exp, gold, preds[exp], mats[exp], Q.read_json(str(metrics_json), default=None))
    family = FAMILY[exp]
    results = {f"{n}_vs_{b}": Q.compare(mats, n, b, args.n_resamples, args.seed, 0.05) for n, b in family}
    # Holm across the exact-span tests of the family (primary metric only)
    for kind in ("bootstrap", "permutation"):
        keys = list(results)
        adj = S.holm([results[k]["metrics"]["exact_span_f1"][kind]["p_two_sided"] for k in keys])
        for k, p in zip(keys, adj):
            results[k]["metrics"]["exact_span_f1"][kind]["p_holm_family_exact_span"] = p
    systems = {}
    for k, m in mats.items():
        p, r = exact_pr(m)
        systems[k] = {**{mm: S.pooled_f1_from_matrix(m, mm) for mm in S.METRIC_ORDER}, "exact_precision": p, "exact_recall": r}
    cfg = {"n_docs": len(docs), "n_resamples": args.n_resamples, "seed": args.seed, "runtime_seconds": round(time.time() - t0, 1)}
    verdict = {}
    if exp == "v3":
        r = results["v3_vs_v2"]["metrics"]["exact_span_f1"]
        b = r["bootstrap"]
        verdict = {
            "criterion_1_point_within_0.01_of_v2": bool(r["new_f1"] >= r["base_f1"] - 0.01),
            "criterion_1_formal_noninferiority_ci_low_gt_minus_0.01": bool(b["ci95_low"] > -0.01),
            "delta_v3_minus_v2": b["observed_delta"], "ci95": [b["ci95_low"], b["ci95_high"]],
        }
    else:
        a = results["v4_vs_anchor"]["metrics"]["exact_span_f1"]["bootstrap"]
        e = results["v4_vs_ensemble"]["metrics"]["exact_span_f1"]
        verdict = {
            "criterion_A_beats_anchor_ci_excludes_0": bool(a["ci95_low"] > 0 or a["ci95_high"] < 0) and a["observed_delta"] > 0,
            "A_delta": a["observed_delta"], "A_ci95": [a["ci95_low"], a["ci95_high"]],
            "criterion_B_within_0.01_of_ensemble_point": bool(e["new_f1"] >= e["base_f1"] - 0.01),
            "B_delta_v4_minus_ensemble": e["bootstrap"]["observed_delta"], "B_ci95": [e["bootstrap"]["ci95_low"], e["bootstrap"]["ci95_high"]],
        }
    write_json(f"{out_prefix}.json", {"config": cfg, "inputs": {k: {"path": p, "sha256_16": S._sha(p)} for k, p in paths.items()},
                                      "reproduction": repro, "systems": systems, "comparisons": results, "verdict": verdict})
    L = [f"# {exp}: paired significance on the held-out test (974 gold-bearing SkillSpan docs)", "",
         f"Paired document bootstrap ({args.n_resamples:,} resamples, seed {args.seed}) + sign-flip permutation; Holm over 5 metrics within each "
         f"comparison and over the {len(family)} exact-span tests of this experiment. Delta = first minus second. One training seed per system.", "",
         "| System | exact P | exact R | exact F1 | token F1 | partial F1 | HARD F1 | SOFT F1 |", "|---|---|---|---|---|---|---|---|"]
    for k, r in systems.items():
        L.append(f"| {LABEL[k]} | {r['exact_precision']:.4f} | {r['exact_recall']:.4f} | " + " | ".join(f"{r[m]:.4f}" for m in S.METRIC_ORDER) + " |")
    for (n, b_) in family:
        res = results[f"{n}_vs_{b_}"]
        L += ["", f"## {LABEL[n]} minus {LABEL[b_]}", "", "| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | MDE (80%) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in res["metrics"].items():
            bt, pm = r["bootstrap"], r["permutation"]
            L.append(f"| {m} | {r['base_f1']:.4f} | {r['new_f1']:.4f} | {bt['observed_delta']:+.4f} | [{bt['ci95_low']:+.4f}, {bt['ci95_high']:+.4f}] | "
                     f"{bt['p_two_sided']:.4f} | {pm['p_two_sided']:.4f} | {bt['p_holm_5_metrics']:.4f} | {pm['p_holm_5_metrics']:.4f} | {r['mde_alpha05_power80']:.4f} |")
        ex = res["metrics"]["exact_span_f1"]
        L.append(f"\nExact-span F1 Holm over the family: boot {ex['bootstrap']['p_holm_family_exact_span']:.4f} / perm {ex['permutation']['p_holm_family_exact_span']:.4f}.")
        d = res["doc_level"]
        L.append(f"Doc level: {n} wins {d['doc_f1_wins']}, losses {d['doc_f1_losses']}, ties {d['doc_f1_ties']} (sign p = {d['doc_f1_sign_test_p']:.4f}); "
                 f"perfect docs {n}-only {d['perfect_doc_variant_only']} vs {b_}-only {d['perfect_doc_anchor_only']} (McNemar p = {d['perfect_doc_mcnemar_exact_p']:.4f}).")
    L += ["", "## Pre-declared criteria", "", "```", __import__("json").dumps(verdict, indent=1), "```", ""]
    Path(f"{out_prefix}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:14])); print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
