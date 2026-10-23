# Log-probability confidence of v4 predictions (held-out test, 974 docs)

2132 predicted spans, 1505 exactly correct (precision 0.7059); 0 spans without an alignable log-prob, 2 unaligned spans. Document-cluster bootstrap (1000) CIs. Cross-fitted = 5-fold, grouped by document.

| Confidence | ECE | Brier | AURC (lower better) | AUROC | sel.acc@50 | @70 | @90 |
|---|---|---|---|---|---|---|---|
| const0.86 | 0.1541 [0.127, 0.182] | 0.2313 [0.212, 0.251] | 0.2941 [0.267, 0.322] | 0.5000 [0.500, 0.500] | 0.7059 | 0.7059 | 0.7059 |
| p_mean = exp(mean lp) | 0.2647 [0.240, 0.292] | 0.2690 [0.245, 0.294] | 0.1682 [0.146, 0.193] | 0.7008 [0.676, 0.725] | 0.8255 | 0.7843 | 0.7358 |
| p_seq = exp(sum lp) | 0.2026 [0.179, 0.227] | 0.2281 [0.208, 0.249] | 0.1629 [0.140, 0.188] | 0.7186 [0.694, 0.745] | 0.8358 | 0.7950 | 0.7410 |
| p_min = exp(min lp) | 0.2123 [0.188, 0.237] | 0.2345 [0.214, 0.256] | 0.1647 [0.142, 0.190] | 0.7122 [0.688, 0.738] | 0.8321 | 0.7924 | 0.7379 |
| votes/3 | 0.1562 [0.134, 0.178] | 0.2087 [0.191, 0.227] | 0.1981 [0.174, 0.224] | 0.6950 [0.670, 0.717] | 0.8218 | 0.8005 | 0.7433 |
| logit(mean lp) [cross-fitted] | 0.0582 [0.040, 0.080] | 0.1990 [0.188, 0.210] | 0.1996 [0.171, 0.230] | 0.6766 [0.647, 0.703] | 0.8236 | 0.7843 | 0.7353 |
| logit(sum lp) [cross-fitted] | 0.0257 [0.018, 0.050] | 0.1893 [0.178, 0.201] | 0.1800 [0.153, 0.207] | 0.7046 [0.677, 0.731] | 0.8349 | 0.7924 | 0.7400 |
| logit(min lp) [cross-fitted] | 0.0280 [0.019, 0.051] | 0.1919 [0.181, 0.204] | 0.1826 [0.156, 0.210] | 0.6973 [0.670, 0.725] | 0.8246 | 0.7917 | 0.7374 |
| logit(votes one-hot) [cross-fitted] | 0.0185 [0.015, 0.046] | 0.1813 [0.170, 0.193] | 0.2024 [0.177, 0.230] | 0.6866 [0.658, 0.712] | 0.8188 | 0.7927 | 0.7411 |
| logit(mean lp + votes) [cross-fitted] | 0.0174 [0.015, 0.043] | 0.1795 [0.168, 0.191] | 0.1623 [0.140, 0.188] | 0.7344 [0.708, 0.758] | 0.8546 | 0.8011 | 0.7426 |

AURC of a perfect ranker = 0.0483; of a constant = error rate = 0.2941.

## Paired differences (first minus second; AUROC higher better, others lower better)

| Pair | metric | delta | 95% CI | p |
|---|---|---|---|---|
| p_mean = exp(mean lp) minus votes/3 | auroc | +0.0058 | [-0.0236, +0.0343] | 0.680 |
| p_mean = exp(mean lp) minus votes/3 | aurc | -0.0298 | [-0.0459, -0.0136] | 0.001 |
| p_mean = exp(mean lp) minus votes/3 | ece_width | +0.1085 | [+0.0875, +0.1314] | 0.001 |
| p_mean = exp(mean lp) minus votes/3 | brier | +0.0603 | [+0.0432, +0.0781] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | auroc | +0.2008 | [+0.1763, +0.2250] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | aurc | -0.1258 | [-0.1400, -0.1116] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | ece_width | +0.1106 | [+0.1081, +0.1146] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | brier | +0.0376 | [+0.0322, +0.0433] | 0.001 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | auroc | -0.0100 | [-0.0443, +0.0232] | 0.582 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | aurc | -0.0028 | [-0.0262, +0.0204] | 0.786 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | ece_width | +0.0397 | [+0.0021, +0.0523] | 0.028 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | brier | +0.0177 | [+0.0100, +0.0248] | 0.001 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | auroc | +0.0478 | [+0.0344, +0.0626] | 0.001 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | aurc | -0.0400 | [-0.0529, -0.0277] | 0.001 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | ece_width | -0.0011 | [-0.0194, +0.0140] | 0.826 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | brier | -0.0019 | [-0.0041, +0.0003] | 0.100 |

## Precision by agreement with anchor/v1/v2 (vote count)

| Votes | Spans | Correct | Precision |
|---|---|---|---|
| 0 of 3 | 131 | 43 | 0.3282 |
| 1 of 3 | 235 | 102 | 0.4340 |
| 2 of 3 | 419 | 253 | 0.6038 |
| 3 of 3 | 1347 | 1107 | 0.8218 |

## Precision by mean-log-prob quintile (1 = most confident)

| Quintile | Spans | Precision | mean lp range |
|---|---|---|---|
| 1 | 426 | 0.8944 | [-0.000, -0.000] |
| 2 | 426 | 0.7911 | [-0.002, -0.000] |
| 3 | 427 | 0.7354 | [-0.008, -0.002] |
| 4 | 426 | 0.6385 | [-0.044, -0.008] |
| 5 | 427 | 0.4707 | [-0.698, -0.044] |
