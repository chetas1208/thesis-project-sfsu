# Experiment 3: token log-probability as span confidence (held-out test, CPU analysis of the v3 and v4 generation runs)

Option: `MistralExtractor(record_logprobs=True)` stores `metadata.logprob = {sum, mean, min, n_tokens}` per span (log-probs of the generated tokens overlapping the span's JSON string; misaligned rows get `None`; verified on 8 docs that predictions are identical to the stored v2 predictions and all spans are aligned; in the full runs 0 of 2,065 (v3) / 2,132 (v4) spans lacked a log-prob). No extra generation was needed.
Unit = predicted span; correct = exact (start, end, label) match. Details and CIs: `logprob_confidence_v3.{md,json}`, `logprob_confidence_v4.{md,json}` (document-cluster bootstrap 1,000; cross-fitted = 5-fold by document). Vote count = how many of anchor/v1/v2 predict the same span (0-3).

| System | Confidence | ECE | Brier | AURC (lower better) | AUROC [95% CI] |
|---|---|---|---|---|---|
| v4 (2,132 spans, precision 0.706) | constant 0.86 | 0.154 | 0.231 | 0.294 | 0.500 |
| | exp(mean logprob) (primary) | 0.265 | 0.269 | 0.168 | 0.701 [0.676, 0.725] |
| | exp(sum logprob) | 0.203 | 0.228 | 0.163 | 0.719 [0.694, 0.745] |
| | vote count / 3 | 0.156 | 0.209 | 0.198 | 0.695 [0.670, 0.717] |
| | logit(mean lp) cross-fitted | 0.058 | 0.199 | 0.200 | 0.677 |
| | logit(sum lp) cross-fitted | 0.026 | 0.189 | 0.180 | 0.705 |
| | logit(votes) cross-fitted | 0.019 | 0.181 | 0.202 | 0.687 |
| | logit(mean lp + votes) cross-fitted | 0.017 | 0.180 | 0.162 | 0.734 [0.708, 0.758] |
| v3 (2,065 spans, precision 0.675) | constant 0.86 | 0.185 | 0.254 | 0.325 | 0.500 |
| | exp(mean logprob) | 0.258 | 0.270 | 0.183 | 0.714 [0.690, 0.737] |
| | exp(sum logprob) | 0.125 | 0.197 | 0.170 | 0.754 [0.728, 0.778] |
| | vote count / 3 | 0.163 | 0.210 | 0.207 | 0.724 [0.699, 0.746] |
| | logit(mean lp + votes) cross-fitted | 0.026 | 0.177 | 0.171 | 0.767 [0.742, 0.790] |

Findings
- The constant 0.86 has no ranking information (AUROC 0.5, AURC = error rate). Log-prob confidence is a real, free ranking signal: AUROC 0.70-0.75 and AURC roughly halved (v4 mean-lp AURC 0.168 vs 0.294; paired delta -0.126, CI [-0.140, -0.112]). Selective accuracy at 50% coverage 0.83 (v4) vs 0.71.
- It is comparable to the 3x-cost vote count at 1x cost: AUROC difference mean-lp minus votes +0.006 (v4, CI [-0.024, +0.034]) / -0.010 (v3, CI [-0.036, +0.019]) i.e. not distinguishable; AURC slightly better for log-probs (-0.030 v4, CI [-0.046, -0.014]; -0.024 v3). The sequence probability exp(sum) is the best single log-prob signal.
- Combining log-prob and votes (cross-fitted logistic) is best: AUROC 0.734 (v4) / 0.767 (v3), +0.048 / +0.059 over votes alone (CIs exclude 0).
- **Raw log-prob "probabilities" are over-confident**: exp(mean lp) ECE 0.26 and Brier 0.27 are WORSE than the constant 0.86 (ECE 0.15-0.18) because most spans have log-prob near 0 (the model is very peaked) while precision is 0.68-0.71. Use them as a ranking score, or recalibrate (5-fold cross-fitted logistic brings ECE to 0.02-0.06 and Brier below the constant).
- Precision by mean-log-prob quintile (v3): 0.88 / 0.81 / 0.68 / 0.54 / 0.46 (monotone).

Caveats: recalibration was cross-fitted on the same test docs (no separate calibration split); two systems only (v3 short-prompt, v4), not the anchor/v2 (no log-probs stored for them; vote counts for the new systems are agreement with the three older adapters, not the 3-way vote of the ensemble); log-prob capture adds a small generation overhead; descriptive analysis, no multiplicity correction beyond the CIs shown.
