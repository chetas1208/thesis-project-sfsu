# L10_significance — central significance re-test

Paired document (or unit) bootstrap, 10000 resamples, seed 42; approximate randomization 10000 shuffles (exact enumeration when ≤16 units differ). p (bootstrap) = percentile inversion, +1 smoothed; Holm within declared family; BH across all non-cross-check tests (secondary). Statistics module: `research/stats.py` (Decisions D-A2-01).

Central extraction systems (22): anchor, ensemble_majority_2of3, ensemble_unanimous_3of3, ensemble_union_1of3, fair_runs_bert-base-cased_s42, fair_runs_bert-base-cased_s43, fair_runs_bert-base-cased_s44, fair_runs_jjzha__jobbert-base-cased_s42, fair_runs_jjzha__jobbert-base-cased_s43, fair_runs_jjzha__jobbert-base-cased_s44, gliner_zero_shot, legacy_full_cag_graph, legacy_mistral_gliner_taxonomy, qlora_v1, qlora_v2, qlora_v3, qlora_v4, safe_wrapper_boundary_refiner, safe_wrapper_final, safe_wrapper_full_cag_graph, safe_wrapper_mistral_gliner_taxonomy, token_lookup_bert. Fair BERT-family baselines found: {'fair_runs_bert-base-cased_s42': 'artifacts/skillllm_v2/evals_gpu/bert_fair/runs/bert-base-cased_s42/predictions.jsonl', 'fair_runs_bert-base-cased_s43': 'artifacts/skillllm_v2/evals_gpu/bert_fair/runs/bert-base-cased_s43/predictions.jsonl', 'fair_runs_bert-base-cased_s44': 'artifacts/skillllm_v2/evals_gpu/bert_fair/runs/bert-base-cased_s44/predictions.jsonl', 'fair_runs_jjzha__jobbert-base-cased_s42': 'artifacts/skillllm_v2/evals_gpu/bert_fair/runs/jjzha__jobbert-base-cased_s42/predictions.jsonl', 'fair_runs_jjzha__jobbert-base-cased_s43': 'artifacts/skillllm_v2/evals_gpu/bert_fair/runs/jjzha__jobbert-base-cased_s43/predictions.jsonl', 'fair_runs_jjzha__jobbert-base-cased_s44': 'artifacts/skillllm_v2/evals_gpu/bert_fair/runs/jjzha__jobbert-base-cased_s44/predictions.jsonl'}.

## Central extraction families (exact-span F1 primary)

| family | system − reference | Δ | 95% CI | p boot | p AR | p Holm | q BH | docs differ | MDE80 |
|---|---|---|---|---|---|---|---|---|---|
| H1_anchor_vs_baselines | anchor − gliner_zero_shot | 0.5887 | [+0.5599, +0.6164] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 886 | 0.0402 |
| H1_anchor_vs_baselines:token_f1 | anchor − gliner_zero_shot | 0.6675 | [+0.6411, +0.6925] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 899 | 0.0367 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − gliner_zero_shot | 0.6674 | [+0.6311, +0.7024] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 545 | 0.0511 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − gliner_zero_shot | 0.5013 | [+0.4610, +0.5395] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 542 | 0.0559 |
| H1_anchor_vs_baselines | anchor − token_lookup_bert | 0.5325 | [+0.5110, +0.5541] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 922 | 0.0308 |
| H1_anchor_vs_baselines:token_f1 | anchor − token_lookup_bert | 0.2689 | [+0.2493, +0.2891] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 928 | 0.0285 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − token_lookup_bert | 0.4524 | [+0.4214, +0.4821] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 807 | 0.0436 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − token_lookup_bert | 0.5576 | [+0.5265, +0.5888] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 853 | 0.0444 |
| H1_anchor_vs_baselines | anchor − fair_runs_bert-base-cased_s42 | 0.0504 | [+0.0290, +0.0718] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 479 | 0.0306 |
| H1_anchor_vs_baselines:token_f1 | anchor − fair_runs_bert-base-cased_s42 | 0.0003 | [-0.0153, +0.0159] | 0.9715 | 0.9685 | 1.0000 | 1.0000 | 515 | 0.0225 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − fair_runs_bert-base-cased_s42 | 0.0407 | [+0.0105, +0.0723] | 0.0110 | 0.0105 | 0.0440 | 0.0209 | 286 | 0.0442 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − fair_runs_bert-base-cased_s42 | 0.0636 | [+0.0329, +0.0945] | 0.0002 | 0.0002 | 0.0016 | 0.0010 | 291 | 0.0439 |
| H1_anchor_vs_baselines | anchor − fair_runs_bert-base-cased_s43 | 0.0553 | [+0.0331, +0.0773] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 485 | 0.0321 |
| H1_anchor_vs_baselines:token_f1 | anchor − fair_runs_bert-base-cased_s43 | 0.0027 | [-0.0132, +0.0196] | 0.7419 | 0.7383 | 1.0000 | 0.9064 | 519 | 0.0236 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − fair_runs_bert-base-cased_s43 | 0.0516 | [+0.0211, +0.0823] | 0.0014 | 0.0010 | 0.0084 | 0.0029 | 285 | 0.0438 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − fair_runs_bert-base-cased_s43 | 0.0618 | [+0.0300, +0.0941] | 0.0006 | 0.0001 | 0.0018 | 0.0021 | 308 | 0.0456 |
| H1_anchor_vs_baselines | anchor − fair_runs_bert-base-cased_s44 | 0.0586 | [+0.0357, +0.0809] | 0.0002 | 0.0002 | 0.0016 | 0.0010 | 495 | 0.0319 |
| H1_anchor_vs_baselines:token_f1 | anchor − fair_runs_bert-base-cased_s44 | 0.0041 | [-0.0124, +0.0204] | 0.6309 | 0.6171 | 1.0000 | 0.8011 | 535 | 0.0236 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − fair_runs_bert-base-cased_s44 | 0.0388 | [+0.0076, +0.0696] | 0.0122 | 0.0147 | 0.0440 | 0.0230 | 296 | 0.0444 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − fair_runs_bert-base-cased_s44 | 0.0833 | [+0.0521, +0.1149] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 313 | 0.0452 |
| H1_anchor_vs_baselines | anchor − fair_runs_jjzha__jobbert-base-cased_s42 | 0.0370 | [+0.0167, +0.0574] | 0.0010 | 0.0004 | 0.0016 | 0.0021 | 451 | 0.0292 |
| H1_anchor_vs_baselines:token_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s42 | -0.0148 | [-0.0287, -0.0012] | 0.0300 | 0.0339 | 0.1800 | 0.0528 | 501 | 0.0194 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s42 | 0.0404 | [+0.0120, +0.0699] | 0.0066 | 0.0082 | 0.0330 | 0.0130 | 297 | 0.0416 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s42 | 0.0332 | [+0.0051, +0.0613] | 0.0190 | 0.0213 | 0.0190 | 0.0344 | 269 | 0.0402 |
| H1_anchor_vs_baselines | anchor − fair_runs_jjzha__jobbert-base-cased_s43 | 0.0388 | [+0.0170, +0.0605] | 0.0004 | 0.0003 | 0.0016 | 0.0016 | 469 | 0.0313 |
| H1_anchor_vs_baselines:token_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s43 | -0.0157 | [-0.0303, -0.0010] | 0.0374 | 0.0385 | 0.1870 | 0.0645 | 505 | 0.0207 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s43 | 0.0263 | [-0.0039, +0.0570] | 0.0876 | 0.0912 | 0.1076 | 0.1412 | 289 | 0.0433 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s43 | 0.0524 | [+0.0219, +0.0830] | 0.0012 | 0.0004 | 0.0024 | 0.0025 | 295 | 0.0436 |
| H1_anchor_vs_baselines | anchor − fair_runs_jjzha__jobbert-base-cased_s44 | 0.0502 | [+0.0285, +0.0716] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 487 | 0.0307 |
| H1_anchor_vs_baselines:token_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s44 | -0.0122 | [-0.0267, +0.0020] | 0.0922 | 0.0959 | 0.3688 | 0.1467 | 524 | 0.0204 |
| H1_anchor_vs_baselines:hard_skill_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s44 | 0.0280 | [-0.0004, +0.0563] | 0.0538 | 0.0546 | 0.1076 | 0.0909 | 305 | 0.0402 |
| H1_anchor_vs_baselines:soft_skill_f1 | anchor − fair_runs_jjzha__jobbert-base-cased_s44 | 0.0772 | [+0.0463, +0.1082] | 0.0002 | 0.0001 | 0.0016 | 0.0010 | 302 | 0.0445 |
| H2_wrapper_vs_anchor | safe_wrapper_final − anchor | 0.0018 | [-0.0006, +0.0042] | 0.1330 | 0.1280 | 0.5319 | 0.2001 | 19 | 0.0034 |
| H2_wrapper_vs_anchor:token_f1 | safe_wrapper_final − anchor | 0.0001 | [-0.0005, +0.0007] | 0.8711 | 0.8516 | 1.0000 | 1.0000 | 9 | 0.0009 |
| H2_wrapper_vs_anchor:hard_skill_f1 | safe_wrapper_final − anchor | 0.0035 | [-0.0004, +0.0076] | 0.0772 | 0.0781 | 0.3088 | 0.1266 | 9 | 0.0057 |
| H2_wrapper_vs_anchor:soft_skill_f1 | safe_wrapper_final − anchor | 0.0000 | [-0.0018, +0.0018] | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 10 | 0.0026 |
| H2_wrapper_vs_anchor | safe_wrapper_boundary_refiner − anchor | 0.0024 | [+0.0009, +0.0042] | 0.0008 | 0.0020 | 0.0040 | 0.0021 | 10 | 0.0024 |
| H2_wrapper_vs_anchor:token_f1 | safe_wrapper_boundary_refiner − anchor | 0.0001 | [+0.0000, +0.0003] | 0.0916 | 0.2500 | 0.4580 | 0.1460 | 3 | 0.0002 |
| H2_wrapper_vs_anchor:hard_skill_f1 | safe_wrapper_boundary_refiner − anchor | 0.0031 | [+0.0006, +0.0063] | 0.0126 | 0.0625 | 0.0630 | 0.0237 | 5 | 0.0041 |
| H2_wrapper_vs_anchor:soft_skill_f1 | safe_wrapper_boundary_refiner − anchor | 0.0015 | [+0.0003, +0.0029] | 0.0136 | 0.0625 | 0.0488 | 0.0253 | 5 | 0.0018 |
| H2_wrapper_vs_anchor | safe_wrapper_mistral_gliner_taxonomy − anchor | -0.0006 | [-0.0023, +0.0010] | 0.5163 | 0.6133 | 1.0000 | 0.6656 | 9 | 0.0023 |
| H2_wrapper_vs_anchor:token_f1 | safe_wrapper_mistral_gliner_taxonomy − anchor | -0.0001 | [-0.0006, +0.0006] | 0.8211 | 0.8125 | 1.0000 | 0.9778 | 6 | 0.0008 |
| H2_wrapper_vs_anchor:hard_skill_f1 | safe_wrapper_mistral_gliner_taxonomy − anchor | 0.0004 | [-0.0025, +0.0032] | 0.7419 | 0.7500 | 0.9677 | 0.9064 | 4 | 0.0041 |
| H2_wrapper_vs_anchor:soft_skill_f1 | safe_wrapper_mistral_gliner_taxonomy − anchor | -0.0015 | [-0.0028, -0.0003] | 0.0122 | 0.0625 | 0.0488 | 0.0230 | 5 | 0.0018 |
| H2_wrapper_vs_anchor | legacy_full_cag_graph − anchor | 0.0010 | [-0.0015, +0.0037] | 0.4354 | 0.4796 | 1.0000 | 0.5771 | 25 | 0.0037 |
| H2_wrapper_vs_anchor:token_f1 | legacy_full_cag_graph − anchor | -0.0003 | [-0.0011, +0.0005] | 0.4554 | 0.4678 | 1.0000 | 0.5953 | 16 | 0.0012 |
| H2_wrapper_vs_anchor:hard_skill_f1 | legacy_full_cag_graph − anchor | 0.0022 | [-0.0020, +0.0066] | 0.3226 | 0.3712 | 0.9677 | 0.4397 | 14 | 0.0062 |
| H2_wrapper_vs_anchor:soft_skill_f1 | legacy_full_cag_graph − anchor | -0.0003 | [-0.0021, +0.0016] | 0.8771 | 1.0000 | 1.0000 | 1.0000 | 11 | 0.0027 |
| H2_wrapper_vs_anchor | legacy_mistral_gliner_taxonomy − anchor | -0.0000 | [-0.0023, +0.0024] | 0.9819 | 0.9336 | 1.0000 | 1.0000 | 19 | 0.0033 |
| H2_wrapper_vs_anchor:token_f1 | legacy_mistral_gliner_taxonomy − anchor | -0.0003 | [-0.0011, +0.0005] | 0.4560 | 0.4678 | 1.0000 | 0.5953 | 16 | 0.0012 |
| H2_wrapper_vs_anchor:hard_skill_f1 | legacy_mistral_gliner_taxonomy − anchor | 0.0016 | [-0.0025, +0.0058] | 0.4556 | 0.5071 | 0.9677 | 0.5953 | 13 | 0.0059 |
| H2_wrapper_vs_anchor:soft_skill_f1 | legacy_mistral_gliner_taxonomy − anchor | -0.0017 | [-0.0032, -0.0006] | 0.0030 | 0.0312 | 0.0150 | 0.0061 | 6 | 0.0020 |
| QLoRA_vs_anchor | qlora_v1 − anchor | -0.0135 | [-0.0340, +0.0070] | 0.2012 | 0.1972 | 0.4024 | 0.2886 | 412 | 0.0296 |
| QLoRA_vs_anchor:token_f1 | qlora_v1 − anchor | -0.0033 | [-0.0198, +0.0132] | 0.6731 | 0.6913 | 0.6751 | 0.8493 | 447 | 0.0235 |
| QLoRA_vs_anchor:hard_skill_f1 | qlora_v1 − anchor | -0.0218 | [-0.0514, +0.0074] | 0.1436 | 0.1504 | 0.5591 | 0.2134 | 260 | 0.0421 |
| QLoRA_vs_anchor:soft_skill_f1 | qlora_v1 − anchor | -0.0044 | [-0.0315, +0.0235] | 0.7413 | 0.7575 | 1.0000 | 0.9064 | 262 | 0.0394 |
| QLoRA_vs_anchor | qlora_v2 − anchor | -0.0086 | [-0.0288, +0.0112] | 0.4062 | 0.3960 | 0.4062 | 0.5435 | 359 | 0.0288 |
| QLoRA_vs_anchor:token_f1 | qlora_v2 − anchor | 0.0078 | [-0.0083, +0.0235] | 0.3376 | 0.3413 | 0.6751 | 0.4577 | 402 | 0.0228 |
| QLoRA_vs_anchor:hard_skill_f1 | qlora_v2 − anchor | -0.0105 | [-0.0396, +0.0185] | 0.4950 | 0.4921 | 0.8747 | 0.6404 | 245 | 0.0418 |
| QLoRA_vs_anchor:soft_skill_f1 | qlora_v2 − anchor | -0.0051 | [-0.0307, +0.0200] | 0.6885 | 0.6965 | 1.0000 | 0.8656 | 223 | 0.0362 |
| QLoRA_vs_anchor | qlora_v3 − anchor | -0.0262 | [-0.0465, -0.0054] | 0.0118 | 0.0121 | 0.0472 | 0.0223 | 406 | 0.0293 |
| QLoRA_vs_anchor:token_f1 | qlora_v3 − anchor | -0.0178 | [-0.0338, -0.0014] | 0.0332 | 0.0354 | 0.0996 | 0.0581 | 436 | 0.0231 |
| QLoRA_vs_anchor:hard_skill_f1 | qlora_v3 − anchor | -0.0218 | [-0.0515, +0.0073] | 0.1398 | 0.1479 | 0.5591 | 0.2091 | 253 | 0.0418 |
| QLoRA_vs_anchor:soft_skill_f1 | qlora_v3 − anchor | -0.0347 | [-0.0622, -0.0065] | 0.0188 | 0.0175 | 0.0752 | 0.0341 | 255 | 0.0400 |
| QLoRA_vs_anchor | qlora_v4 − anchor | 0.0145 | [-0.0042, +0.0333] | 0.1312 | 0.1391 | 0.3936 | 0.1977 | 356 | 0.0267 |
| QLoRA_vs_anchor:token_f1 | qlora_v4 − anchor | 0.0207 | [+0.0049, +0.0367] | 0.0132 | 0.0110 | 0.0528 | 0.0246 | 388 | 0.0227 |
| QLoRA_vs_anchor:hard_skill_f1 | qlora_v4 − anchor | 0.0100 | [-0.0158, +0.0353] | 0.4374 | 0.4354 | 0.8747 | 0.5777 | 216 | 0.0361 |
| QLoRA_vs_anchor:soft_skill_f1 | qlora_v4 − anchor | 0.0235 | [-0.0038, +0.0510] | 0.0926 | 0.0924 | 0.2778 | 0.1471 | 227 | 0.0394 |
| ensemble_vs_anchor | ensemble_majority_2of3 − anchor | 0.0198 | [+0.0040, +0.0361] | 0.0154 | 0.0158 | 0.0154 | 0.0282 | 281 | 0.0228 |
| ensemble_vs_anchor:token_f1 | ensemble_majority_2of3 − anchor | 0.0083 | [-0.0037, +0.0204] | 0.1820 | 0.1667 | 0.1820 | 0.2633 | 297 | 0.0173 |
| ensemble_vs_anchor:hard_skill_f1 | ensemble_majority_2of3 − anchor | 0.0209 | [-0.0030, +0.0455] | 0.0942 | 0.0957 | 0.0942 | 0.1487 | 176 | 0.0349 |
| ensemble_vs_anchor:soft_skill_f1 | ensemble_majority_2of3 − anchor | 0.0191 | [+0.0002, +0.0381] | 0.0478 | 0.0404 | 0.0478 | 0.0814 | 150 | 0.0267 |

## Hypothesis re-tests (owners decide the verdict)

- H1: {"value": 0.03695331348524411, "n_units": 974, "central_verdict_by_rule": "NOT SUPPORTED", "baselines": ["gliner_zero_shot", "token_lookup_bert", "fair_runs_bert-base-cased_s42", "fair_runs_bert-base-cased_s43", "fair_runs_bert-base-cased_s44", "fair_runs_jjzha__jobbert-base-cased_s42", "fair_runs_jjzha__jobbert-base-cased_s43", "fair_runs_jjzha__jobbert-base-cased_s44"], "owner_verdict_L01": "NOT SUPPORTED", "agrees_with_owner": true, "note": "L10 re-applies the H1 rule to the baselines it can see from stored predictions; A1 (L01) owns the verdict."}
- H2: {"value": 0.0017853755084601408, "n_units": 974, "f1_noninferior_margin_-0.005": true, "owner_verdict_L05": "NOT SUPPORTED", "note": "only the F1 non-inferiority half of H2 is re-tested here; APR half lives in L05"}

## Legacy significance files cross-check

| file | comparison | metric | legacy Δ | new Δ | legacy p boot | new p boot | legacy p perm | new p AR | flags |
|---|---|---|---|---|---|---|---|---|---|
| significance_f1.json | safe_wrapper_boundary_refiner − anchor | exact_span_f1 | 0.0024 | 0.0024 | 0.0002 | 0.0008 | 0.0020 | 0.0020 | agree |
| significance_f1.json | safe_wrapper_boundary_refiner − anchor | token_f1 | 0.0001 | 0.0001 | 0.1032 | 0.0916 | 0.2500 | 0.2500 | agree |
| significance_f1.json | safe_wrapper_boundary_refiner − anchor | hard_skill_f1 | 0.0031 | 0.0031 | 0.0142 | 0.0126 | 0.0625 | 0.0625 | agree |
| significance_f1.json | safe_wrapper_boundary_refiner − anchor | soft_skill_f1 | 0.0015 | 0.0015 | 0.0138 | 0.0136 | 0.0625 | 0.0625 | HOLM_DECISION_DIFFERS (legacy family 0.1794 vs 0.0488; families differ) |
| significance_f1.json | safe_wrapper_mistral_gliner_taxonomy − anchor | exact_span_f1 | -0.0006 | -0.0006 | 0.5099 | 0.5163 | 0.6133 | 0.6133 | agree |
| significance_f1.json | safe_wrapper_mistral_gliner_taxonomy − anchor | token_f1 | -0.0001 | -0.0001 | 0.8135 | 0.8211 | 0.8125 | 0.8125 | agree |
| significance_f1.json | safe_wrapper_mistral_gliner_taxonomy − anchor | hard_skill_f1 | 0.0004 | 0.0004 | 0.7269 | 0.7419 | 0.7500 | 0.7500 | agree |
| significance_f1.json | safe_wrapper_mistral_gliner_taxonomy − anchor | soft_skill_f1 | -0.0015 | -0.0015 | 0.0144 | 0.0122 | 0.0625 | 0.0625 | HOLM_DECISION_DIFFERS (legacy family 0.1794 vs 0.0488; families differ) |
| significance_f1.json | safe_wrapper_full_cag_graph − anchor | exact_span_f1 | 0.0018 | 0.0018 | 0.1342 | 0.1282 | 0.1237 | 0.1251 | agree |
| significance_f1.json | safe_wrapper_full_cag_graph − anchor | token_f1 | 0.0001 | 0.0001 | 0.8823 | 0.8823 | 0.8516 | 0.8516 | agree |
| significance_f1.json | safe_wrapper_full_cag_graph − anchor | hard_skill_f1 | 0.0035 | 0.0035 | 0.0834 | 0.0760 | 0.0781 | 0.0781 | agree |
| significance_f1.json | safe_wrapper_full_cag_graph − anchor | soft_skill_f1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | agree |
| significance_qlora_vs_anchor.json | qlora_v1 − anchor | exact_span_f1 | -0.0135 | -0.0135 | 0.2006 | 0.2012 | 0.1992 | 0.1972 | agree |
| significance_qlora_vs_anchor.json | qlora_v1 − anchor | token_f1 | -0.0033 | -0.0033 | 0.6921 | 0.6731 | 0.6955 | 0.6913 | agree |
| significance_qlora_vs_anchor.json | qlora_v1 − anchor | hard_skill_f1 | -0.0218 | -0.0218 | 0.1544 | 0.1436 | 0.1524 | 0.1504 | agree |
| significance_qlora_vs_anchor.json | qlora_v1 − anchor | soft_skill_f1 | -0.0044 | -0.0044 | 0.7617 | 0.7413 | 0.7580 | 0.7575 | agree |
| qlora_loop_v2_significance_vs_anchor.json | qlora_v2 − anchor | exact_span_f1 | -0.0086 | -0.0086 | 0.4034 | 0.4062 | 0.3971 | 0.3960 | agree |
| qlora_loop_v2_significance_vs_anchor.json | qlora_v2 − anchor | token_f1 | 0.0078 | 0.0078 | 0.3350 | 0.3376 | 0.3425 | 0.3413 | agree |
| qlora_loop_v2_significance_vs_anchor.json | qlora_v2 − anchor | hard_skill_f1 | -0.0105 | -0.0105 | 0.4774 | 0.4950 | 0.4851 | 0.4921 | agree |
| qlora_loop_v2_significance_vs_anchor.json | qlora_v2 − anchor | soft_skill_f1 | -0.0051 | -0.0051 | 0.6955 | 0.6885 | 0.6913 | 0.6965 | agree |
| qlora_loop_v2_significance_vs_v1.json | qlora_v2 − qlora_v1 | exact_span_f1 | 0.0049 | 0.0049 | 0.6017 | 0.5961 | 0.6106 | 0.6127 | agree |
| qlora_loop_v2_significance_vs_v1.json | qlora_v2 − qlora_v1 | token_f1 | 0.0111 | 0.0111 | 0.1590 | 0.1570 | 0.1591 | 0.1629 | agree |
| qlora_loop_v2_significance_vs_v1.json | qlora_v2 − qlora_v1 | hard_skill_f1 | 0.0113 | 0.0113 | 0.3524 | 0.3620 | 0.3590 | 0.3691 | agree |
| qlora_loop_v2_significance_vs_v1.json | qlora_v2 − qlora_v1 | soft_skill_f1 | -0.0007 | -0.0007 | 0.9675 | 0.9635 | 0.9663 | 0.9653 | agree |
| qlora_loop_v3_significance.json | qlora_v3 − qlora_v2 | exact_span_f1 | -0.0176 | -0.0176 | 0.0614 | 0.0610 | 0.0567 | 0.0574 | agree |
| qlora_loop_v3_significance.json | qlora_v3 − qlora_v2 | token_f1 | -0.0256 | -0.0256 | 0.0046 | 0.0020 | 0.0032 | 0.0037 | agree |
| qlora_loop_v3_significance.json | qlora_v3 − qlora_v2 | hard_skill_f1 | -0.0112 | -0.0112 | 0.3218 | 0.3064 | 0.3175 | 0.3151 | agree |
| qlora_loop_v3_significance.json | qlora_v3 − qlora_v2 | soft_skill_f1 | -0.0296 | -0.0296 | 0.0398 | 0.0356 | 0.0361 | 0.0341 | agree |
| qlora_loop_v3_significance.json | qlora_v3 − anchor | exact_span_f1 | -0.0262 | -0.0262 | 0.0126 | 0.0118 | 0.0119 | 0.0121 | HOLM_DECISION_DIFFERS (legacy family 0.0630 vs 0.0472; families differ) |
| qlora_loop_v3_significance.json | qlora_v3 − anchor | token_f1 | -0.0178 | -0.0178 | 0.0342 | 0.0332 | 0.0333 | 0.0354 | agree |
| qlora_loop_v3_significance.json | qlora_v3 − anchor | hard_skill_f1 | -0.0218 | -0.0218 | 0.1532 | 0.1398 | 0.1492 | 0.1479 | agree |
| qlora_loop_v3_significance.json | qlora_v3 − anchor | soft_skill_f1 | -0.0347 | -0.0347 | 0.0156 | 0.0188 | 0.0163 | 0.0175 | agree |
| qlora_loop_v4_significance.json | qlora_v4 − anchor | exact_span_f1 | 0.0145 | 0.0145 | 0.1380 | 0.1312 | 0.1324 | 0.1391 | agree |
| qlora_loop_v4_significance.json | qlora_v4 − anchor | token_f1 | 0.0207 | 0.0207 | 0.0096 | 0.0132 | 0.0103 | 0.0110 | HOLM_DECISION_DIFFERS (legacy family 0.0384 vs 0.0528; families differ) |
| qlora_loop_v4_significance.json | qlora_v4 − anchor | hard_skill_f1 | 0.0100 | 0.0100 | 0.4420 | 0.4374 | 0.4301 | 0.4354 | agree |
| qlora_loop_v4_significance.json | qlora_v4 − anchor | soft_skill_f1 | 0.0235 | 0.0235 | 0.0906 | 0.0926 | 0.0934 | 0.0924 | agree |
| qlora_loop_v4_significance.json | qlora_v4 − qlora_v2 | exact_span_f1 | 0.0231 | 0.0231 | 0.0088 | 0.0058 | 0.0086 | 0.0098 | agree |
| qlora_loop_v4_significance.json | qlora_v4 − qlora_v2 | token_f1 | 0.0129 | 0.0129 | 0.0702 | 0.0670 | 0.0678 | 0.0643 | agree |
| qlora_loop_v4_significance.json | qlora_v4 − qlora_v2 | hard_skill_f1 | 0.0205 | 0.0205 | 0.0898 | 0.0818 | 0.0854 | 0.0833 | agree |
| qlora_loop_v4_significance.json | qlora_v4 − qlora_v2 | soft_skill_f1 | 0.0286 | 0.0286 | 0.0170 | 0.0104 | 0.0148 | 0.0158 | agree |
| ensemble_eval.json | ensemble_majority_2of3 − anchor | exact_span_f1 | 0.0198 | 0.0198 | 0.0162 | 0.0154 | 0.0146 | 0.0158 | HOLM_DECISION_DIFFERS (legacy family 0.0810 vs 0.0154; families differ) |
| ensemble_eval.json | ensemble_majority_2of3 − anchor | token_f1 | 0.0083 | 0.0083 | 0.1748 | 0.1820 | 0.1715 | 0.1667 | agree |
| ensemble_eval.json | ensemble_majority_2of3 − anchor | hard_skill_f1 | 0.0209 | 0.0209 | 0.0966 | 0.0942 | 0.0936 | 0.0957 | agree |
| ensemble_eval.json | ensemble_majority_2of3 − anchor | soft_skill_f1 | 0.0191 | 0.0191 | 0.0398 | 0.0478 | 0.0428 | 0.0404 | HOLM_DECISION_DIFFERS (legacy family 0.1592 vs 0.0478; families differ) |
| ensemble_eval.json | ensemble_unanimous_3of3 − anchor | exact_span_f1 | -0.0500 | -0.0500 | 0.0002 | 0.0002 | 0.0000 | 0.0001 | agree |
| ensemble_eval.json | ensemble_unanimous_3of3 − anchor | token_f1 | -0.1296 | -0.1296 | 0.0002 | 0.0002 | 0.0000 | 0.0001 | agree |
| ensemble_eval.json | ensemble_unanimous_3of3 − anchor | hard_skill_f1 | -0.0375 | -0.0375 | 0.0038 | 0.0050 | 0.0049 | 0.0049 | agree |
| ensemble_eval.json | ensemble_unanimous_3of3 − anchor | soft_skill_f1 | -0.0654 | -0.0654 | 0.0002 | 0.0002 | 0.0000 | 0.0001 | agree |
| ensemble_eval.json | ensemble_union_1of3 − anchor | exact_span_f1 | 0.0003 | 0.0003 | 0.9611 | 0.9753 | 0.9717 | 0.9728 | agree |
| ensemble_eval.json | ensemble_union_1of3 − anchor | token_f1 | 0.0114 | 0.0114 | 0.0642 | 0.0584 | 0.0630 | 0.0616 | agree |
| ensemble_eval.json | ensemble_union_1of3 − anchor | hard_skill_f1 | -0.0177 | -0.0177 | 0.0690 | 0.0698 | 0.0624 | 0.0660 | agree |
| ensemble_eval.json | ensemble_union_1of3 − anchor | soft_skill_f1 | 0.0217 | 0.0217 | 0.0116 | 0.0122 | 0.0105 | 0.0095 | agree |

## Ingested layer files

- `reports/skillllm_v2/evals/L01_extraction/per_unit_scores.jsonl`: 90234 rows, 4 groups, 93 tests, 0 skipped groups, 0 unparsed rows.
- `reports/skillllm_v2/evals/L02_generalization/per_unit_scores.jsonl`: 31188 rows, 4 groups, 84 tests, 0 skipped groups, 0 unparsed rows.
- `reports/skillllm_v2/evals/L03_robustness/per_unit_scores.jsonl`: 15530 rows, 16 groups, 64 tests, 0 skipped groups, 0 unparsed rows.
- `reports/skillllm_v2/evals/L05_wrapper/per_unit_scores.jsonl`: 120776 rows, 20 groups, 120 tests, 0 skipped groups, 0 unparsed rows.
- `reports/skillllm_v2/evals/L06_linking/per_unit_scores.jsonl`: 48860 rows, 4 groups, 36 tests, 0 skipped groups, 0 unparsed rows.
- `reports/skillllm_v2/evals/L07_compaction/per_unit_scores.jsonl`: 27030 rows, 22 groups, 244 tests, 0 skipped groups, 0 unparsed rows.
- `reports/skillllm_v2/evals/L08_graph_retrieval/per_unit_scores.jsonl`: 4908 rows, 11 groups, 86 tests, 1 skipped groups, 0 unparsed rows.
- `reports/skillllm_v2/evals/L09_course_rec/per_unit_scores.jsonl`: 17747 rows, 54 groups, 218 tests, 1 skipped groups, 0 unparsed rows.

Per ingested family: tests / significant after Holm (p<0.05).

| family | tests | Holm-significant | BH-significant |
|---|---|---|---|
| L01_extraction:exact_span | 93 | 19 | 35 |
| L02_generalization:bucket_counts | 84 | 14 | 18 |
| L03_robustness:perturbation:alias_sub | 4 | 2 | 2 |
| L03_robustness:perturbation:boilerplate | 4 | 2 | 2 |
| L03_robustness:perturbation:bullets_prose | 4 | 2 | 2 |
| L03_robustness:perturbation:case_lower | 4 | 2 | 2 |
| L03_robustness:perturbation:case_title | 4 | 2 | 2 |
| L03_robustness:perturbation:case_upper | 4 | 2 | 2 |
| L03_robustness:perturbation:distractor | 4 | 2 | 2 |
| L03_robustness:perturbation:paraphrase_context | 4 | 2 | 2 |
| L03_robustness:perturbation:punct_detok | 4 | 2 | 2 |
| L03_robustness:perturbation:punct_strip | 4 | 2 | 2 |
| L03_robustness:perturbation:reorder_clauses | 4 | 2 | 2 |
| L03_robustness:perturbation:typo_inside | 4 | 2 | 2 |
| L03_robustness:perturbation:typo_outside | 4 | 2 | 2 |
| L03_robustness:perturbation:whitespace | 4 | 2 | 2 |
| L03_robustness:xdata:green | 4 | 3 | 3 |
| L03_robustness:xdata:sayfullina | 4 | 2 | 2 |
| L05_wrapper:H2_wrapper_vs_anchor | 48 | 1 | 5 |
| L05_wrapper:L05_leakage_ablation | 4 | 0 | 0 |
| L05_wrapper:L05_leave_one_out | 20 | 0 | 4 |
| L05_wrapper:L05_policy_off_counterfactual | 24 | 18 | 18 |
| L05_wrapper:L05_single | 24 | 0 | 2 |
| L06_linking:L06_linking | 36 | 32 | 32 |
| L07_compaction:H7 | 244 | 193 | 198 |
| L08_graph_retrieval:H5 | 3 | 1 | 1 |
| L08_graph_retrieval:retrieval | 72 | 17 | 39 |
| L08_graph_retrieval:retrieval_SG-v1 | 11 | 11 | 11 |
| L09_course_rec:H6 | 5 | 5 | 5 |
| L09_course_rec:H6-ndcg10-noninferiority | 5 | 4 | 4 |
| L09_course_rec:L09-vs-baselines[ndcg10] | 7 | 5 | 5 |
| L09_course_rec:L09-vs-baselines[p5] | 7 | 6 | 6 |
| L09_course_rec:L09-vs-baselines[viol5] | 7 | 6 | 6 |
| L09_course_rec:L09-vs-baselines[wsgc5] | 7 | 3 | 3 |
| L09_course_rec:rec | 132 | 48 | 74 |
| L09_course_rec:rec_SG-v1 | 48 | 43 | 44 |

## Caveats

- Document resampling captures test-sample noise only; no training-seed variance (single run per system).
- Wrapper deltas rest on ≤25 differing documents: AR (exact when ≤16 differ) is the more trustworthy p-value there.
- Ingested families are re-tested as declared by their owners; descriptive per-unit metrics inside a large declared family make Holm conservative. Owners' own verdicts take precedence; disagreements are listed above.
- Runtime 180s. Files: `results.json`, `all_tests.csv`, `forest_plot.png`, `legacy_crosscheck.csv`.
