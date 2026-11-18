"""L04 supplement (post-hoc, labelled secondary; D-B2-01): fairer constant baselines for the anchor log-prob calibration.

The pre-registered H4 baseline is the stored constant 0.86 (D-A2-03). A constant equal to the DEV base rate has ECE close to
0 by construction, so the headline ECE gain over 0.86 mostly reflects 0.86 being mis-set. This module reports, on the same
test spans and the same document-cluster bootstrap replicates, the dev-base-rate constant next to the dev-fit Platt
calibration of the dev-selected score (sum log-prob), and the paired deltas. Nothing here changes the H4 verdict.

    python -m skillllm_v2.evals.research.calibration_supplement [--n-boot 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from skillllm_v2.evals.research import calibration_selective as K
from skillllm_v2.evals.research import stats as S
from skillllm_v2.evals.research.common import EVALS_ROOT, REPO_ROOT, sha256_file

MET = ("ece_w10", "ece_m10", "brier", "nll", "aurc", "auroc", "sel_prec@80", "sel_risk@80")


def run(n_boot: int, seed: int) -> dict:
    gt, gd = K.load_gold("test"), K.load_gold("dev")
    jt, jd = REPO_ROOT / K.P["anchor_test_lp"], REPO_ROOT / K.P["anchor_dev_lp"]
    at, ad = K.read_rows(jt / "predictions.jsonl"), K.read_rows(jd / "predictions.jsonl")
    tt = K.Table("t", "test", gt, at, ids=[i for i in gt if i in at])
    td = K.Table("d", "dev", gd, ad, ids=[i for i in gd if i in ad])
    g_dev, g_test = td.logit("lp_sum"), tt.logit("lp_sum")  # dev-selected score (report: sum_lp)
    platt = K.fit_calibrator("platt", g_dev, td.y)(g_test)
    base = float(td.y.mean())
    series = {"platt_sum_lp": platt, "const_dev_base_rate": np.full(tt.n, base), "const_0.86": np.full(tt.n, 0.86)}
    res = K.evaluate(tt, {k: v for k, v in series.items()}, platt, n_boot, seed + 31)
    out = {"dev_base_rate": base, "n_spans_test": tt.n, "n_docs_test": tt.n_docs, "n_boot": n_boot,
           "inputs": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in (jt / "predictions.jsonl", jd / "predictions.jsonl")},
           "point": {}, "ci": {}, "paired_platt_minus_base_rate": {}}
    for k in series:
        out["point"][k] = {m: res["point"][k][m] for m in MET}
        out["ci"][k] = {m: list(res["ci"][k][m]) for m in MET}
    for m in MET:
        obs = res["point"]["platt_sum_lp"][m] - res["point"]["const_dev_base_rate"][m]
        s = S.summarize_samples(res["boot"]["platt_sum_lp"][m] - res["boot"]["const_dev_base_rate"][m], obs, paired=True)
        out["paired_platt_minus_base_rate"][m] = {k: s[k] for k in ("value", "ci_low", "ci_high", "p")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.chdir(REPO_ROOT)
    out = run(a.n_boot, a.seed)
    path = EVALS_ROOT / "L04_calibration" / "supplement_base_rate.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
