# Log-probability confidence of v3 predictions (held-out test, 974 docs)

2065 predicted spans, 1394 exactly correct (precision 0.6751); 0 spans without an alignable log-prob, 0 unaligned spans. Document-cluster bootstrap (1000) CIs. Cross-fitted = 5-fold, grouped by document.

| Confidence | ECE | Brier | AURC (lower better) | AUROC | sel.acc@50 | @70 | @90 |
|---|---|---|---|---|---|---|---|
| const0.86 | 0.1849 [0.158, 0.215] | 0.2536 [0.234, 0.275] | 0.3249 [0.298, 0.355] | 0.5000 [0.500, 0.500] | 0.6751 | 0.6751 | 0.6751 |
| p_mean = exp(mean lp) | 0.2578 [0.232, 0.285] | 0.2698 [0.248, 0.294] | 0.1832 [0.159, 0.211] | 0.7140 [0.690, 0.737] | 0.8238 | 0.7607 | 0.7041 |
| p_seq = exp(sum lp) | 0.1249 [0.105, 0.148] | 0.1965 [0.181, 0.213] | 0.1704 [0.146, 0.199] | 0.7536 [0.728, 0.778] | 0.8393 | 0.7870 | 0.7192 |
| p_min = exp(min lp) | 0.1547 [0.133, 0.179] | 0.2112 [0.194, 0.228] | 0.1760 [0.150, 0.205] | 0.7375 [0.713, 0.762] | 0.8306 | 0.7766 | 0.7111 |
| votes/3 | 0.1634 [0.144, 0.189] | 0.2098 [0.193, 0.229] | 0.2074 [0.184, 0.234] | 0.7243 [0.699, 0.746] | 0.8178 | 0.7930 | 0.7161 |
| logit(mean lp) [cross-fitted] | 0.0656 [0.047, 0.088] | 0.2041 [0.194, 0.215] | 0.2122 [0.182, 0.244] | 0.6934 [0.669, 0.719] | 0.8132 | 0.7580 | 0.7025 |
| logit(sum lp) [cross-fitted] | 0.0433 [0.033, 0.065] | 0.1842 [0.173, 0.196] | 0.1917 [0.163, 0.224] | 0.7406 [0.713, 0.767] | 0.8345 | 0.7877 | 0.7176 |
| logit(min lp) [cross-fitted] | 0.0367 [0.027, 0.060] | 0.1919 [0.181, 0.204] | 0.1991 [0.170, 0.232] | 0.7232 [0.696, 0.750] | 0.8267 | 0.7766 | 0.7117 |
| logit(votes one-hot) [cross-fitted] | 0.0144 [0.012, 0.043] | 0.1804 [0.169, 0.192] | 0.2189 [0.190, 0.249] | 0.7078 [0.679, 0.734] | 0.8078 | 0.7882 | 0.7093 |
| logit(mean lp + votes) [cross-fitted] | 0.0256 [0.021, 0.049] | 0.1772 [0.166, 0.189] | 0.1711 [0.146, 0.198] | 0.7672 [0.742, 0.790] | 0.8558 | 0.7994 | 0.7149 |

AURC of a perfect ranker = 0.0598; of a constant = error rate = 0.3249.

## Paired differences (first minus second; AUROC higher better, others lower better)

| Pair | metric | delta | 95% CI | p |
|---|---|---|---|---|
| p_mean = exp(mean lp) minus votes/3 | auroc | -0.0103 | [-0.0361, +0.0192] | 0.496 |
| p_mean = exp(mean lp) minus votes/3 | aurc | -0.0242 | [-0.0396, -0.0098] | 0.002 |
| p_mean = exp(mean lp) minus votes/3 | ece_width | +0.0944 | [+0.0723, +0.1090] | 0.001 |
| p_mean = exp(mean lp) minus votes/3 | brier | +0.0599 | [+0.0435, +0.0762] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | auroc | +0.2140 | [+0.1901, +0.2367] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | aurc | -0.1417 | [-0.1561, -0.1280] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | ece_width | +0.0729 | [+0.0688, +0.0775] | 0.001 |
| p_mean = exp(mean lp) minus const0.86 | brier | +0.0162 | [+0.0113, +0.0211] | 0.001 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | auroc | -0.0144 | [-0.0462, +0.0187] | 0.394 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | aurc | -0.0067 | [-0.0277, +0.0157] | 0.504 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | ece_width | +0.0512 | [+0.0140, +0.0662] | 0.004 |
| logit(mean lp) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | brier | +0.0237 | [+0.0143, +0.0328] | 0.001 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | auroc | +0.0594 | [+0.0450, +0.0740] | 0.001 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | aurc | -0.0478 | [-0.0590, -0.0356] | 0.001 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | ece_width | +0.0112 | [-0.0092, +0.0269] | 0.340 |
| logit(mean lp + votes) [cross-fitted] minus logit(votes one-hot) [cross-fitted] | brier | -0.0031 | [-0.0060, -0.0002] | 0.040 |

## Precision by agreement with anchor/v1/v2 (vote count)

| Votes | Spans | Correct | Precision |
|---|---|---|---|
| 0 of 3 | 128 | 37 | 0.2891 |
| 1 of 3 | 230 | 76 | 0.3304 |
| 2 of 3 | 379 | 195 | 0.5145 |
| 3 of 3 | 1328 | 1086 | 0.8178 |

## Precision by mean-log-prob quintile (1 = most confident)

| Quintile | Spans | Precision | mean lp range |
|---|---|---|---|
| 1 | 413 | 0.8814 | [-0.004, -0.000] |
| 2 | 413 | 0.8136 | [-0.020, -0.004] |
| 3 | 413 | 0.6804 | [-0.054, -0.020] |
| 4 | 413 | 0.5351 | [-0.127, -0.054] |
| 5 | 413 | 0.4649 | [-1.098, -0.128] |
