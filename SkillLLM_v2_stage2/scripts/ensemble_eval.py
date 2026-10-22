"""Span-voting ensemble of the three Mistral LoRA adapters (anchor, QLoRA-loop v1, v2) on the held-out test split.

The rule is fixed BEFORE looking at test results and is not tuned: a predicted span (start, end, label) is kept
if at least 2 of the 3 systems predict it (strict majority). Unanimity (3/3) and union (>=1/3) are reported
for context only and are NOT candidates for selection, so the single pre-declared comparison is
`majority vs anchor` (Holm over the 5 metrics). Uses the same pooled-count scorer and paired bootstrap /
sign-flip permutation as the other significance reports (skillllm_v2.eval.significance, unchanged).

    PYTHONPATH=src python scripts/ensemble_eval.py [--out-prefix reports/skillllm_v2/ensemble_eval]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from skillllm_v2.eval import significance as S

SYSTEMS = {
    "anchor": S.ANCHOR_PREDICTIONS,
    "v1": "artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl",
    "v2": "artifacts/skillllm_v2/qlora_loop_v2_eval/predictions.jsonl",
}
RULES = {"majority(>=2/3)": 2, "unanimous(3/3)": 3, "union(>=1/3)": 1}
PRIMARY = "majority(>=2/3)"


def vote(per_system_docs: list[list[list[dict[str, Any]]]], min_votes: int) -> list[list[dict[str, Any]]]:
    """Combine per-system per-doc span lists by exact (start, end, label) voting."""
    n_docs = len(per_system_docs[0])
    out: list[list[dict[str, Any]]] = []
    for d in range(n_docs):
        votes: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for sys_docs in per_system_docs:
            seen = set()
            for sp in sys_docs[d]:
                key = (sp["start"], sp["end"], sp.get("label"))
                if key in seen:  # one vote per system per span
                    continue
                seen.add(key)
                votes[key].append(sp)
        kept = []
        for key, sps in sorted(votes.items(), key=lambda kv: str(kv[0])):
            if len(sps) >= min_votes:
                sp = dict(sps[0])
                confs = [s.get("confidence") for s in sps if s.get("confidence") is not None]
                if confs:
                    sp["confidence"] = round(sum(confs) / len(confs), 4)
                sp["source"] = f"ensemble:{len(sps)}of{len(per_system_docs)}"
                kept.append(sp)
        out.append(kept)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="reports/skillllm_v2/ensemble_eval")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    docs = S.load_gold_docs()
    ids = [d["id"] for d in docs]
    gold = [d["gold_spans"] if "gold_spans" in d else d.get("spans", []) for d in docs]
    preds = {name: S.load_system_predictions(path, ids) for name, path in SYSTEMS.items()}
    mats = {name: S.count_matrix(gold, p) for name, p in preds.items()}
    for name in SYSTEMS:
        mats[name]  # noqa: B018 - fail early if a system is unreadable
    per_system = [preds["anchor"], preds["v1"], preds["v2"]]
    for rule, k in RULES.items():
        mats[rule] = S.count_matrix(gold, vote(per_system, k))

    results: dict[str, Any] = {"n_docs": len(ids), "seed": args.seed, "n_resamples": args.n_resamples,
                               "primary": f"{PRIMARY} vs anchor", "systems": {}, "comparisons": {}}
    for name, m in mats.items():
        results["systems"][name] = {metric: S.pooled_f1_from_matrix(m, metric) for metric in S.METRIC_ORDER}
        tp, fp, fn = (m[:, c].sum() for c in S.COUNT_COLUMNS["exact_span_f1"])
        results["systems"][name]["exact_precision"] = float(tp / max(tp + fp, 1))
        results["systems"][name]["exact_recall"] = float(tp / max(tp + fn, 1))

    for rule in RULES:
        a, b = mats["anchor"], mats[rule]
        boot = S.paired_bootstrap(a, b, S.METRIC_ORDER, args.n_resamples, args.seed)
        cmp: dict[str, Any] = {}
        keys = list(S.METRIC_ORDER)
        for i, m in enumerate(keys):
            perm = S.sign_flip_permutation(a, b, m, args.seed + 1000 + i)
            cmp[m] = {"anchor_f1": S.pooled_f1_from_matrix(a, m), "ensemble_f1": S.pooled_f1_from_matrix(b, m),
                      "bootstrap": boot[m], "permutation": perm}
        for kind in ("bootstrap", "permutation"):
            adj = S.holm([cmp[m][kind]["p_two_sided"] for m in keys])
            for m, p in zip(keys, adj):
                cmp[m][kind]["p_holm_5_metrics"] = p
        results["comparisons"][rule] = {"metrics": cmp, "doc_level": S.doc_level_counts(a, b)}

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1, default=float))

    L = ["# Span-voting ensemble of the three adapters (held-out test)", "",
         "Pre-declared rule: keep a span (start, end, label) if >= 2 of 3 systems (anchor, QLoRA-loop v1, v2) predict it. "
         f"The primary and only selection candidate is **{PRIMARY}**; unanimity and union are descriptive. "
         f"{len(ids)} SkillSpan test docs, paired bootstrap ({args.n_resamples:,} resamples) + sign-flip permutation, Holm over 5 metrics.", "",
         "## Pooled F1 per system", "", "| System | exact P | exact R | exact F1 | token F1 | partial F1 | HARD F1 | SOFT F1 |", "|---|---|---|---|---|---|---|---|"]
    for name, r in results["systems"].items():
        L.append(f"| {name} | {r['exact_precision']:.4f} | {r['exact_recall']:.4f} | " + " | ".join(f"{r[m]:.4f}" for m in S.METRIC_ORDER) + " |")
    for rule in RULES:
        L += ["", f"## {rule} minus anchor" + ("  (PRIMARY)" if rule == PRIMARY else "  (descriptive)"), "",
              "| Metric | Anchor | Ensemble | Delta | 95% CI | p boot | p perm | Holm5 boot | Holm5 perm |", "|---|---|---|---|---|---|---|---|---|"]
        for m, c in results["comparisons"][rule]["metrics"].items():
            b, p = c["bootstrap"], c["permutation"]
            L.append(f"| {m} | {c['anchor_f1']:.4f} | {c['ensemble_f1']:.4f} | {b['observed_delta']:+.4f} | "
                     f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}] | {b['p_two_sided']:.4f} | {p['p_two_sided']:.4f} | "
                     f"{b['p_holm_5_metrics']:.4f} | {p['p_holm_5_metrics']:.4f} |")
        d = results["comparisons"][rule]["doc_level"]
        L.append(f"\nDoc-level exact-span F1: ensemble wins {d['doc_f1_wins']}, losses {d['doc_f1_losses']}, ties {d['doc_f1_ties']} (sign test p = {d['doc_f1_sign_test_p']:.4f}).")
    L += ["", "## Caveats", "",
          "- The three adapters share the same base model and training data, so their errors are correlated; voting gains are bounded by that.",
          "- One test split, one seed per adapter. The rule was fixed in advance, but only one rule is a valid test; the others are context.",
          "- Ensembling costs 3x inference (three adapters over one 4-bit base, adapters can be hot-swapped but each needs its own generation pass)."]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
