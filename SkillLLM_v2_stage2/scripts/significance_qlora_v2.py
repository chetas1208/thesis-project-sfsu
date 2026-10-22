"""QLoRA-loop v2 paired significance: v2 vs Raw Mistral LoRA anchor, and v2 vs QLoRA-loop v1.

Same machinery as scripts/significance_qlora.py (skillllm_v2.eval.significance unchanged): paired
document bootstrap, sign-flip permutation, 5 metrics, Holm. Holm is reported both within each
comparison (5 tests) and across all 10 tests (2 comparisons x 5 metrics). Delta = first minus second.
Writes reports/skillllm_v2/qlora_loop_v2_significance_{vs_anchor,vs_v1}.{json,md}; v1 outputs untouched.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from skillllm_v2.eval import significance as S
from skillllm_v2.io_utils import read_json, write_json

SYSTEMS = {
    "anchor": (S.ANCHOR_PREDICTIONS, S.ANCHOR_METRICS_JSON, "Raw Mistral LoRA anchor (80 epochs, r=16)"),
    "v1": ("artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl", "artifacts/skillllm_v2/qlora_loop_eval/metrics.json", "QLoRA-loop v1 (r=8, round 1)"),
    "v2": ("artifacts/skillllm_v2/qlora_loop_v2_eval/predictions.jsonl", "artifacts/skillllm_v2/qlora_loop_v2_eval/metrics.json", "QLoRA-loop v2 (r=16)"),
}
COMPARISONS = [("v2", "anchor"), ("v2", "v1")]


def compare(mats, new, base, n_resamples, seed, alpha):
    a, b = mats[base], mats[new]
    boot = S.paired_bootstrap(a, b, S.METRIC_ORDER, n_resamples, seed)
    metrics = {}
    for mi, m in enumerate(S.METRIC_ORDER):
        perm = S.sign_flip_permutation(a, b, m, seed + 1000 + mi)
        metrics[m] = {"base_f1": S.pooled_f1_from_matrix(a, m), "new_f1": S.pooled_f1_from_matrix(b, m),
                      "bootstrap": boot[m], "permutation": perm}
        metrics[m]["mde_alpha05_power80"] = S.minimal_detectable_effect(boot[m]["boot_delta_sd"], alpha, 0.8)
    keys = list(S.METRIC_ORDER)
    for kind in ("bootstrap", "permutation"):
        adj = S.holm([metrics[m][kind]["p_two_sided"] for m in keys])
        for m, p in zip(keys, adj):
            metrics[m][kind]["p_holm_5_metrics"] = p
    return {"metrics": metrics, "doc_level": S.doc_level_counts(a, b)}


def render(name, res, cfg, prov):
    new, base = name
    L = [f"# {SYSTEMS[new][2]} vs {SYSTEMS[base][2]}: paired significance (held-out test)", "",
         f"Delta = {new} minus {base}. Paired document bootstrap ({cfg['n_resamples']:,} resamples, seed {cfg['seed']}) and paired "
         f"sign-flip permutation on {cfg['n_docs']} SkillSpan test docs (docs with >=1 gold span). Pooled-count F1s reproduce "
         f"evaluate_extraction and stored metrics.json to {cfg['reproduction_tolerance']:g}. Holm: within this comparison (5 metrics) "
         "and across all 10 tests (2 comparisons x 5 metrics). One training seed per system, one split.", "",
         "| Metric | Base | New | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm | Holm10 boot | Holm10 perm | MDE (80% power) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in res["metrics"].items():
        b, p = r["bootstrap"], r["permutation"]
        L.append(f"| {m} | {r['base_f1']:.6f} | {r['new_f1']:.6f} | {b['observed_delta']:+.6f} | [{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] | "
                 f"{b['p_two_sided']:.4f} | {p['p_two_sided']:.4f} | {b['p_holm_5_metrics']:.4f} | {p['p_holm_5_metrics']:.4f} | "
                 f"{b['p_holm_10_tests']:.4f} | {p['p_holm_10_tests']:.4f} | {r['mde_alpha05_power80']:.4f} |")
    d = res["doc_level"]
    L += ["", "## Document-level (doc exact-span F1)", "",
          f"{new} wins {d['doc_f1_wins']}, losses {d['doc_f1_losses']}, ties {d['doc_f1_ties']} of {d['n_docs']} docs "
          f"(sign-test p = {d['doc_f1_sign_test_p']:.4f}). Perfectly extracted docs: {new}-only {d['perfect_doc_variant_only']}, "
          f"{base}-only {d['perfect_doc_anchor_only']} (McNemar exact p = {d['perfect_doc_mcnemar_exact_p']:.4f}).", "",
          "Inputs (sha256 prefix): " + "; ".join(f"{k}: `{v['path']}` ({v['sha256_16']})" for k, v in prov.items()), ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-prefix", default="reports/skillllm_v2/qlora_loop_v2_significance")
    args = ap.parse_args()
    t0 = time.time()
    docs = S.load_gold_docs()
    ids = [d["id"] for d in docs]
    gold = [d["gold_spans"] for d in docs]
    mats, prov, repro = {}, {}, {}
    for name, (pp, mp, _) in SYSTEMS.items():
        pred = S.load_system_predictions(pp, ids)
        cm = S.count_matrix(gold, pred)
        repro[name] = S.assert_reproduction(name, gold, pred, cm, read_json(mp, default=None))
        mats[name] = cm
        prov[name] = {"path": pp, "sha256_16": S._sha(pp)}
    results = {f"{n}_vs_{b}": compare(mats, n, b, args.n_resamples, args.seed, args.alpha) for n, b in COMPARISONS}
    keys = list(S.METRIC_ORDER)
    for kind in ("bootstrap", "permutation"):
        flat = [(c, m) for c in results for m in keys]
        adj = S.holm([results[c]["metrics"][m][kind]["p_two_sided"] for c, m in flat])
        for (c, m), p in zip(flat, adj):
            results[c]["metrics"][m][kind]["p_holm_10_tests"] = p
    cfg = {"n_docs": len(docs), "n_resamples": args.n_resamples, "seed": args.seed, "alpha": args.alpha,
           "reproduction_tolerance": S.REPRO_TOL, "runtime_seconds": round(time.time() - t0, 1)}
    for (n, b) in COMPARISONS:
        key = f"{n}_vs_{b}"
        suffix = "vs_anchor" if b == "anchor" else f"vs_{b}"
        write_json(f"{args.out_prefix}_{suffix}.json", {"config": cfg, "inputs": prov, "reproduction": repro, **results[key]})
        Path(f"{args.out_prefix}_{suffix}.md").write_text(render((n, b), results[key], cfg, prov), encoding="utf-8")
        for m, r in results[key]["metrics"].items():
            bt = r["bootstrap"]
            print(f"{key} {m}: delta {bt['observed_delta']:+.4f} CI [{bt['ci95_low']:+.4f},{bt['ci95_high']:+.4f}] "
                  f"p_boot {bt['p_two_sided']:.4f} p_perm {r['permutation']['p_two_sided']:.4f} holm10_perm {r['permutation']['p_holm_10_tests']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
