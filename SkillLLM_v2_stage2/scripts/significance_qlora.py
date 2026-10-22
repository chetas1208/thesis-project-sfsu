"""New-vs-anchor paired significance: QLoRA-loop best adapter vs Raw Mistral LoRA (974 test docs).

Reuses skillllm_v2.eval.significance unchanged (count matrices, paired document bootstrap,
sign-flip permutation, win/loss/tie + McNemar, reproduction asserts). One comparison x 5 metrics,
so Holm is applied across those 5 tests. Nothing in significance.py or significance_f1.* is modified.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from skillllm_v2.eval import significance as S
from skillllm_v2.io_utils import read_json, write_json

NEW_PRED = "artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl"
NEW_METRICS = "artifacts/skillllm_v2/qlora_loop_eval/metrics.json"


def run(n_resamples: int, seed: int, alpha: float) -> dict:
    t0 = time.time()
    docs = S.load_gold_docs()
    ids = [d["id"] for d in docs]
    gold = [d["gold_spans"] for d in docs]
    mats, repro, prov = {}, {}, {}
    for name, pred_path, met_path in (("anchor", S.ANCHOR_PREDICTIONS, S.ANCHOR_METRICS_JSON), ("qlora_loop", NEW_PRED, NEW_METRICS)):
        pred = S.load_system_predictions(pred_path, ids)
        cm = S.count_matrix(gold, pred)
        repro[name] = S.assert_reproduction(name, gold, pred, cm, read_json(met_path, default=None))
        mats[name] = cm
        prov[name] = {"path": pred_path, "sha256_16": S._sha(pred_path)}
    a, b = mats["anchor"], mats["qlora_loop"]
    boot = S.paired_bootstrap(a, b, S.METRIC_ORDER, n_resamples, seed)
    metrics = {}
    for mi, m in enumerate(S.METRIC_ORDER):
        perm = S.sign_flip_permutation(a, b, m, seed + 1000 + mi)
        metrics[m] = {"anchor_f1": S.pooled_f1_from_matrix(a, m), "new_f1": S.pooled_f1_from_matrix(b, m),
                      "bootstrap": boot[m], "permutation": perm}
    keys = list(S.METRIC_ORDER)
    for kind in ("bootstrap", "permutation"):
        adj = S.holm([metrics[m][kind]["p_two_sided"] for m in keys])
        for m, p in zip(keys, adj):
            metrics[m][kind]["p_holm_5_metrics"] = p
    for m in keys:
        metrics[m]["mde_alpha05_power80"] = S.minimal_detectable_effect(metrics[m]["bootstrap"]["boot_delta_sd"], alpha, 0.8)
    return {
        "config": {"n_docs": len(docs), "n_resamples": n_resamples, "seed": seed, "alpha": alpha,
                   "comparison": "qlora_loop_best (round 1, r=8) minus Raw Mistral LoRA anchor (80 epochs, r=16)",
                   "resampling_unit": "document", "split": "SkillSpan test, docs with >=1 gold span",
                   "reproduction_tolerance": S.REPRO_TOL, "runtime_seconds": round(time.time() - t0, 1), "inputs": prov},
        "reproduction": repro, "metrics": metrics, "doc_level": S.doc_level_counts(a, b),
    }


def render(res: dict) -> str:
    c = res["config"]
    L = ["# QLoRA-loop best adapter vs Raw Mistral LoRA anchor: paired significance (held-out test)", "",
         f"{c['comparison']}. Paired document bootstrap ({c['n_resamples']:,} resamples, seed {c['seed']}) and paired "
         f"sign-flip permutation on {c['n_docs']} SkillSpan test docs (same docs/gold filter as the anchor). Delta = new minus anchor. "
         f"Pooled-count F1s reproduce evaluate_extraction and each system's stored metrics.json to {c['reproduction_tolerance']:g}. "
         "Holm correction is across the 5 metrics only. One training seed, one split.", "",
         "| Metric | Anchor | New | Delta | 95% CI | p boot | p perm | Holm boot | Holm perm | MDE (80% power) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in res["metrics"].items():
        b, p = r["bootstrap"], r["permutation"]
        L.append(f"| {m} | {r['anchor_f1']:.6f} | {r['new_f1']:.6f} | {b['observed_delta']:+.6f} | [{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] | "
                 f"{b['p_two_sided']:.4f} | {p['p_two_sided']:.4f} | {b['p_holm_5_metrics']:.4f} | {p['p_holm_5_metrics']:.4f} | {r['mde_alpha05_power80']:.4f} |")
    d = res["doc_level"]
    L += ["", "## Permutation details", "", "| Metric | Discordant docs | Method |", "|---|---|---|"]
    for m, r in res["metrics"].items():
        L.append(f"| {m} | {r['permutation']['n_discordant_docs']} | {r['permutation']['method']} |")
    L += ["", "## Document-level (doc exact-span F1)", "",
          f"New wins {d['doc_f1_wins']}, losses {d['doc_f1_losses']}, ties {d['doc_f1_ties']} of {d['n_docs']} docs "
          f"(sign-test p = {d['doc_f1_sign_test_p']:.4f}). Perfectly extracted docs: new-only {d['perfect_doc_variant_only']}, "
          f"anchor-only {d['perfect_doc_anchor_only']} (McNemar exact p = {d['perfect_doc_mcnemar_exact_p']:.4f}).", "",
          "Inputs (sha256 prefix): " + "; ".join(f"{k}: `{v['path']}` ({v['sha256_16']})" for k, v in c["inputs"].items()), ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output-json", default="reports/skillllm_v2/significance_qlora_vs_anchor.json")
    ap.add_argument("--output-md", default="reports/skillllm_v2/significance_qlora_vs_anchor.md")
    args = ap.parse_args()
    res = run(args.n_resamples, args.seed, args.alpha)
    write_json(args.output_json, res)
    Path(args.output_md).write_text(render(res), encoding="utf-8")
    for m, r in res["metrics"].items():
        b = r["bootstrap"]
        print(f"{m}: delta {b['observed_delta']:+.4f} CI [{b['ci95_low']:+.4f},{b['ci95_high']:+.4f}] p_boot {b['p_two_sided']:.4f} p_perm {r['permutation']['p_two_sided']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
