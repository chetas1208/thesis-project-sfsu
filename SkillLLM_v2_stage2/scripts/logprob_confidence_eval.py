"""Experiment 3: calibration of token-log-probability confidence vs the constant 0.86 vs vote-count confidence (held-out test, CPU only).

Unit = a predicted span of the new system X (v3 or v4, predictions generated with record_logprobs=True). Correct = its (start, end, label)
exactly matches a gold span of the document (project exact-span definition, same as evals/calibration_evals.py). Unaligned spans
(start None) are kept and count as incorrect. Spans whose log-probability could not be aligned get score -20 (lowest rank); their count is
reported.

Confidence candidates
  const0.86     the shipped constant
  p_mean        exp(mean token log-prob of the span string)   [pre-declared primary]
  p_seq         exp(sum token log-prob)  (probability of the span string as generated)
  p_min         exp(min token log-prob)
  votes/3       number of {anchor, v1, v2} that predict the same (start, end, label), divided by 3
Raw values are scored for ECE/Brier. Re-calibrated variants (logistic regression, 5-fold document-grouped cross-fitting so no fold is scored
by a model fitted on itself): logit(mean lp), logit(sum lp), logit(min lp), votes one-hot, mean lp + votes one-hot.
Metrics via skillllm_v2.evals.calibration_evals primitives (read-only import): ECE (10 equal-width bins), Brier, AURC (lower=better), AUROC,
selective accuracy at 50/70/90% coverage, with document-cluster bootstrap CIs; paired cluster bootstrap for mean-lp vs votes.

    PYTHONPATH=src python scripts/logprob_confidence_eval.py --pred artifacts/skillllm_v2/qlora_loop_v3_eval/predictions.jsonl --tag v3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_eval as E  # noqa: E402

from skillllm_v2.eval import significance as S  # noqa: E402
from skillllm_v2.evals import calibration_evals as C  # noqa: E402

MISSING = -20.0


def build_table(gold: list[list[dict]], pred: list[list[dict]], refs: list[list[list[dict]]]) -> dict[str, np.ndarray]:
    rows = []
    for d, (g, ps) in enumerate(zip(gold, pred)):
        gk = {S._key(x) for x in g}
        ref_keys = [{S._key(x) for x in r[d] if x.get("start") is not None} for r in refs]
        for sp in ps:
            lp = (sp.get("metadata") or {}).get("logprob")
            aligned = sp.get("start") is not None
            key = S._key(sp)
            rows.append((
                d, int(aligned and key in gk),
                lp["mean"] if lp else MISSING, lp["sum"] if lp else MISSING, lp["min"] if lp else MISSING,
                sum(key in rk for rk in ref_keys) if aligned else 0, int(lp is None), int(not aligned),
            ))
    a = np.array(rows, dtype=float)
    return {"doc": a[:, 0].astype(int), "y": a[:, 1].astype(int), "mean": a[:, 2], "sum": a[:, 3], "min": a[:, 4],
            "votes": a[:, 5].astype(int), "lp_missing": a[:, 6].astype(int), "unaligned": a[:, 7].astype(int)}


def crossfit(features: np.ndarray, y: np.ndarray, docs: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    uniq = np.unique(docs)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    folds = np.array([fold_of[d] for d in docs])
    out = np.zeros(len(y))
    for f in range(k):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2:
            out[te] = y[tr].mean()
            continue
        model = LogisticRegression(C=1e6, max_iter=1000).fit(features[tr], y[tr])
        out[te] = model.predict_proba(features[te])[:, 1]
    return out


def onehot_votes(votes: np.ndarray) -> np.ndarray:
    return np.stack([(votes == v).astype(float) for v in (1, 2, 3)], axis=1)


def summarise(conf: np.ndarray, t: dict[str, np.ndarray], n_docs: int, n_boot: int, seed: int) -> dict:
    s = C.calibration_summary(conf, t["y"])
    keep = ("n", "base_rate", "mean_confidence", "ece_width", "brier", "aurc", "aurc_oracle", "auroc", "sel_acc@50", "sel_acc@70", "sel_acc@90", "confidence_is_constant", "n_distinct_confidences")
    out = {k: s[k] for k in keep}
    ci = C.cluster_bootstrap(t["doc"], n_docs, conf, t["y"], n_boot=n_boot, seed=seed)
    for k in ("ece_width", "brier", "aurc", "auroc"):
        out[f"{k}_ci95"] = ci.get(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    docs = S.load_gold_docs()
    ids = [d["id"] for d in docs]
    gold = [d["gold_spans"] for d in docs]
    pred = S.load_system_predictions(args.pred, ids)
    refs = [S.load_system_predictions(S.ANCHOR_PREDICTIONS, ids), S.load_system_predictions(E.SYSTEMS["v1"], ids), S.load_system_predictions(E.SYSTEMS["v2"], ids)]
    t = build_table(gold, pred, refs)
    n_docs, n = len(ids), len(t["y"])
    lp, y, dd = t["mean"], t["y"], t["doc"]
    feats = {
        "logit(mean lp)": lp[:, None], "logit(sum lp)": t["sum"][:, None], "logit(min lp)": t["min"][:, None],
        "logit(votes one-hot)": onehot_votes(t["votes"]),
        "logit(mean lp + votes)": np.concatenate([lp[:, None], onehot_votes(t["votes"])], axis=1),
    }
    conf = {
        "const0.86": np.full(n, 0.86),
        "p_mean = exp(mean lp)": np.exp(np.clip(lp, MISSING, 0)),
        "p_seq = exp(sum lp)": np.exp(np.clip(t["sum"], MISSING, 0)),
        "p_min = exp(min lp)": np.exp(np.clip(t["min"], MISSING, 0)),
        "votes/3": t["votes"] / 3.0,
    }
    for name, x in feats.items():
        conf[name + " [cross-fitted]"] = crossfit(x, y, dd)
    results = {name: summarise(c, t, n_docs, args.n_boot, args.seed) for name, c in conf.items()}
    paired = {}
    arm = lambda name: (dd, conf[name], y)  # noqa: E731
    for a, b in [("p_mean = exp(mean lp)", "votes/3"), ("p_mean = exp(mean lp)", "const0.86"), ("logit(mean lp) [cross-fitted]", "logit(votes one-hot) [cross-fitted]"),
                 ("logit(mean lp + votes) [cross-fitted]", "logit(votes one-hot) [cross-fitted]")]:
        paired[f"{a} minus {b}"] = C.paired_cluster_bootstrap(arm(a), arm(b), n_docs, ("auroc", "aurc", "ece_width", "brier"), args.n_boot, args.seed)
    vote_table = {int(v): {"spans": int((t["votes"] == v).sum()), "correct": int(y[t["votes"] == v].sum())} for v in range(4)}
    deciles = []
    order = np.argsort(-lp, kind="stable")
    for q in range(5):
        idx = order[q * n // 5:(q + 1) * n // 5]
        deciles.append({"quintile_by_mean_lp": q + 1, "spans": int(len(idx)), "precision": float(y[idx].mean()), "mean_lp_range": [float(lp[idx].min()), float(lp[idx].max())]})
    out = {"tag": args.tag, "predictions": args.pred, "n_docs": n_docs, "n_spans": n, "spans_correct": int(y.sum()), "spans_logprob_missing": int(t["lp_missing"].sum()),
           "spans_unaligned": int(t["unaligned"].sum()), "confidences": results, "paired_vs": paired, "vote_count_table": vote_table, "quintiles": deciles,
           "config": {"n_boot": args.n_boot, "seed": args.seed, "crossfit_folds": 5, "missing_logprob_score": MISSING}}
    Path(f"reports/skillllm_v2/logprob_confidence_{args.tag}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    def f(v): return "n/a" if v is None else f"{v:.4f}"
    def ci(v): return "" if not v else f" [{v[0]:.3f}, {v[1]:.3f}]"
    L = [f"# Log-probability confidence of {args.tag} predictions (held-out test, {n_docs} docs)", "",
         f"{n} predicted spans, {int(y.sum())} exactly correct (precision {y.mean():.4f}); {int(t['lp_missing'].sum())} spans without an alignable log-prob, "
         f"{int(t['unaligned'].sum())} unaligned spans. Document-cluster bootstrap ({args.n_boot}) CIs. Cross-fitted = 5-fold, grouped by document.", "",
         "| Confidence | ECE | Brier | AURC (lower better) | AUROC | sel.acc@50 | @70 | @90 |", "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        L.append(f"| {name} | {f(r['ece_width'])}{ci(r['ece_width_ci95'])} | {f(r['brier'])}{ci(r['brier_ci95'])} | {f(r['aurc'])}{ci(r['aurc_ci95'])} | "
                 f"{f(r['auroc'])}{ci(r['auroc_ci95'])} | {f(r['sel_acc@50'])} | {f(r['sel_acc@70'])} | {f(r['sel_acc@90'])} |")
    L += ["", f"AURC of a perfect ranker = {results['const0.86']['aurc_oracle']:.4f}; of a constant = error rate = {1 - y.mean():.4f}.", "", "## Paired differences (first minus second; AUROC higher better, others lower better)", "",
          "| Pair | metric | delta | 95% CI | p |", "|---|---|---|---|---|"]
    for pair, d in paired.items():
        for m, v in d.items():
            L.append(f"| {pair} | {m} | {v['delta']:+.4f} | [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {v['p']:.3f} |")
    L += ["", "## Precision by agreement with anchor/v1/v2 (vote count)", "", "| Votes | Spans | Correct | Precision |", "|---|---|---|---|"]
    for v, r in vote_table.items():
        L.append(f"| {v} of 3 | {r['spans']} | {r['correct']} | {r['correct'] / max(1, r['spans']):.4f} |")
    L += ["", "## Precision by mean-log-prob quintile (1 = most confident)", "", "| Quintile | Spans | Precision | mean lp range |", "|---|---|---|---|"]
    for q in deciles:
        L.append(f"| {q['quintile_by_mean_lp']} | {q['spans']} | {q['precision']:.4f} | [{q['mean_lp_range'][0]:.3f}, {q['mean_lp_range'][1]:.3f}] |")
    Path(f"reports/skillllm_v2/logprob_confidence_{args.tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
