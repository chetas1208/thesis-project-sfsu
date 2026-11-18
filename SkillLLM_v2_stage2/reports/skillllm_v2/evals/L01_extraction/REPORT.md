# L01 - Extraction benchmark (SkillSpan test)

Owner A1. Module `src/skillllm_v2/evals/research/extraction_full.py`; decisions D-A1-01..05 and D-A1-08 in `Decisions.md`. Population: 974 test sentences with gold spans (from 65 job postings); skill-free sentences: 2595. CIs: 95% percentile sentence-level bootstrap (seed 42; 10,000 resamples for the anchor and safe wrapper, 2000 otherwise); `ci_posting` in results.json = posting-level cluster bootstrap (sensitivity).

## 1. Reproduction of stored metric files

| system | reproduced (<=1e-6) | max abs diff | files compared |
|---|---|---|---|
| mistral_lora | True | 0.00e+00 | artifacts/skillllm_v2/mistral_lora/metrics.json, reports/skillllm_v2/raw_mistral_lora_metrics.json, artifacts/skillllm_v2/safe_wrapper/metrics.json:anchor_metrics |
| bert_baseline | True | 0.00e+00 | artifacts/skillllm_v2/bert_baseline/metrics.json, reports/skillllm_v2/bert_baseline_metrics.json |
| gliner_baseline | True | 0.00e+00 | artifacts/skillllm_v2/gliner_baseline/metrics.json, reports/skillllm_v2/gliner_baseline_metrics.json |
| mistral_boundary | True | 0.00e+00 | artifacts/skillllm_v2/mistral_boundary/metrics.json, reports/skillllm_v2/mistral_boundary_metrics.json |
| mistral_gliner_taxonomy | True | 0.00e+00 | artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json, reports/skillllm_v2/mistral_gliner_taxonomy_metrics.json |
| full_cag_graph | True | 0.00e+00 | artifacts/skillllm_v2/full_cag_graph/metrics.json, reports/skillllm_v2/full_cag_graph_metrics.json |
| safe_wrapper | True | 0.00e+00 | artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json:selected_metrics |
| safe_wrapper_boundary_refiner | True | 0.00e+00 | artifacts/skillllm_v2/safe_wrapper/boundary_refiner/metrics.json:candidate_metrics, artifacts/skillllm_v2/safe_wrapper/boundary_refiner/metrics.json:selected_metrics |
| safe_wrapper_mistral_gliner_taxonomy | True | 0.00e+00 | artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/metrics.json:candidate_metrics, artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/metrics.json:selected_metrics |
| safe_wrapper_full_cag_graph | True | 0.00e+00 | artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json:candidate_metrics, artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json:selected_metrics |
| qlora_loop_v1 | True | 0.00e+00 | artifacts/skillllm_v2/qlora_loop_eval/metrics.json, reports/skillllm_v2/qlora_loop_test_metrics.json |
| qlora_loop_v2 | True | 0.00e+00 | artifacts/skillllm_v2/qlora_loop_v2_eval/metrics.json, reports/skillllm_v2/qlora_loop_v2_test_metrics.json |
| qlora_loop_v3 | True | 0.00e+00 | artifacts/skillllm_v2/qlora_loop_v3_eval/metrics.json, reports/skillllm_v2/qlora_loop_v3_test_metrics.json |
| qlora_loop_v4 | True | 0.00e+00 | artifacts/skillllm_v2/qlora_loop_v4_eval/metrics.json, reports/skillllm_v2/qlora_loop_v4_test_metrics.json |

Stored BERT/GLiNER files are over all 3,569 sentences and reproduce on that population; re-scored on the 974-sentence protocol they are different numbers (table 2).

## 2. All systems on the same 974 sentences

| system | role | exact P | exact R | exact F1 [95% CI] | posting CI | token F1 (char) | word-token F1 | partial F1 (repo) | char-IoU>=.5 F1 | mean char-IoU | HARD F1 | SOFT F1 | macro F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mistral_lora | anchor | 0.6735 | 0.6667 | 0.6701 [0.6465, 0.6938] | [0.6378, 0.7014] | 0.7879 | 0.7809 | 0.8138 | 0.7548 | 0.9156 | 0.7265 | 0.6040 | 0.6652 |
| gpu_anchor_test_rerun | anchor_rerun | 0.6748 | 0.6680 | 0.6714 [0.6473, 0.6959] | [0.6362, 0.7033] | 0.7880 | 0.7812 | 0.8152 | 0.7562 | 0.9158 | 0.7289 | 0.6040 | 0.6665 |
| safe_wrapper_boundary_refiner | safe_wrapper | 0.6770 | 0.6680 | 0.6724 [0.6485, 0.6967] | [0.6373, 0.7044] | 0.7880 | 0.7812 | 0.8164 | 0.7573 | 0.9158 | 0.7295 | 0.6055 | 0.6675 |
| safe_wrapper | safe_wrapper | 0.6767 | 0.6671 | 0.6719 [0.6481, 0.6956] | [0.6390, 0.7031] | 0.7879 | 0.7812 | 0.8168 | 0.7572 | 0.9154 | 0.7300 | 0.6040 | 0.6670 |
| safe_wrapper_full_cag_graph | safe_wrapper | 0.6767 | 0.6671 | 0.6719 [0.6477, 0.6962] | [0.6364, 0.7040] | 0.7879 | 0.7812 | 0.8168 | 0.7572 | 0.9154 | 0.7300 | 0.6040 | 0.6670 |
| safe_wrapper_mistral_gliner_taxonomy | safe_wrapper | 0.6732 | 0.6658 | 0.6695 [0.6455, 0.6936] | [0.6344, 0.7015] | 0.7878 | 0.7809 | 0.8142 | 0.7547 | 0.9153 | 0.7269 | 0.6026 | 0.6647 |
| full_cag_graph | legacy_wrapper | 0.6742 | 0.6680 | 0.6711 [0.6472, 0.6958] | [0.6362, 0.7032] | 0.7876 | 0.7808 | 0.8157 | 0.7563 | 0.9155 | 0.7286 | 0.6038 | 0.6662 |
| mistral_boundary | legacy_wrapper | 0.6735 | 0.6667 | 0.6701 [0.6464, 0.6944] | [0.6353, 0.7021] | 0.7879 | 0.7809 | 0.8138 | 0.7548 | 0.9156 | 0.7265 | 0.6040 | 0.6652 |
| mistral_gliner_taxonomy | legacy_wrapper | 0.6721 | 0.6680 | 0.6701 [0.6460, 0.6948] | [0.6353, 0.7019] | 0.7876 | 0.7808 | 0.8144 | 0.7551 | 0.9155 | 0.7280 | 0.6023 | 0.6652 |
| qlora_loop_v4 | qlora_loop | 0.7059 | 0.6645 | 0.6846 [0.6599, 0.7089] | [0.6488, 0.7176] | 0.8086 | 0.8028 | 0.8365 | 0.7814 | 0.9173 | 0.7364 | 0.6275 | 0.6820 |
| qlora_loop_v2 | qlora_loop | 0.6720 | 0.6512 | 0.6614 [0.6370, 0.6858] | [0.6192, 0.6975] | 0.7957 | 0.7916 | 0.8170 | 0.7610 | 0.9133 | 0.7159 | 0.5989 | 0.6574 |
| qlora_loop_v1 | qlora_loop | 0.6864 | 0.6291 | 0.6565 [0.6314, 0.6802] | [0.6239, 0.6863] | 0.7846 | 0.7790 | 0.8040 | 0.7431 | 0.9141 | 0.7046 | 0.5996 | 0.6521 |
| qlora_loop_v3 | qlora_loop | 0.6751 | 0.6155 | 0.6439 [0.6199, 0.6694] | [0.6080, 0.6793] | 0.7701 | 0.7647 | 0.8042 | 0.7386 | 0.9056 | 0.7047 | 0.5694 | 0.6370 |
| gpu_bert_fair | baseline | 0.6249 | 0.6415 | 0.6331 [0.6099, 0.6558] | [0.6038, 0.6618] | 0.8026 | 0.7948 | 0.8214 | 0.7556 | 0.8962 | 0.6860 | 0.5709 | 0.6285 |
| bert_baseline (token lookup, not BERT) | baseline | 0.0918 | 0.2751 | 0.1376 [0.1246, 0.1511] | [0.1134, 0.1636] | 0.5190 | 0.4696 | 0.4104 | 0.2648 | 0.6563 | 0.2741 | 0.0464 | 0.1602 |
| gliner_baseline (zero-shot) | baseline | 0.2940 | 0.0472 | 0.0814 [0.0656, 0.0978] | [0.0598, 0.1061] | 0.1204 | 0.1070 | 0.1453 | 0.1255 | 0.8023 | 0.0591 | 0.1028 | 0.0809 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s42 | baseline_seed_run | 0.6249 | 0.6415 | 0.6331 [0.6099, 0.6558] | [0.6038, 0.6618] | 0.8026 | 0.7948 | 0.8214 | 0.7556 | 0.8962 | 0.6860 | 0.5709 | 0.6285 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s43 | baseline_seed_run | 0.6255 | 0.6371 | 0.6312 [0.6088, 0.6540] | [0.6049, 0.6583] | 0.8035 | 0.7954 | 0.8167 | 0.7498 | 0.8943 | 0.7001 | 0.5516 | 0.6259 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s44 | baseline_seed_run | 0.6051 | 0.6353 | 0.6199 [0.5955, 0.6441] | [0.5879, 0.6516] | 0.8001 | 0.7923 | 0.8163 | 0.7461 | 0.8906 | 0.6985 | 0.5268 | 0.6126 |
| gpu_bert_fair_run_bert-base-cased_s42 | baseline_seed_run | 0.6266 | 0.6128 | 0.6196 [0.5945, 0.6439] | [0.5864, 0.6510] | 0.7875 | 0.7809 | 0.8116 | 0.7429 | 0.8900 | 0.6858 | 0.5405 | 0.6131 |
| gpu_bert_fair_run_bert-base-cased_s43 | baseline_seed_run | 0.6269 | 0.6031 | 0.6148 [0.5895, 0.6389] | [0.5829, 0.6447] | 0.7851 | 0.7781 | 0.8038 | 0.7313 | 0.8905 | 0.6749 | 0.5422 | 0.6086 |
| gpu_bert_fair_run_bert-base-cased_s44 | baseline_seed_run | 0.6249 | 0.5987 | 0.6115 [0.5880, 0.6347] | [0.5782, 0.6436] | 0.7838 | 0.7763 | 0.8018 | 0.7382 | 0.8925 | 0.6877 | 0.5208 | 0.6042 |

Identical prediction sets (not independent evidence): gpu_bert_fair = gpu_bert_fair_run_jjzha__jobbert-base-cased_s42; mistral_boundary = mistral_lora; safe_wrapper = safe_wrapper_full_cag_graph.


### SemEval-2013 / nervaluate schemes (F1)

| system | strict | exact (boundary only) | partial | type |
|---|---|---|---|---|
| mistral_lora | 0.6665 | 0.6945 | 0.7795 | 0.8107 |
| gpu_anchor_test_rerun | 0.6687 | 0.6967 | 0.7812 | 0.8121 |
| safe_wrapper_boundary_refiner | 0.6698 | 0.6978 | 0.7824 | 0.8133 |
| safe_wrapper | 0.6692 | 0.6972 | 0.7821 | 0.8133 |
| safe_wrapper_full_cag_graph | 0.6692 | 0.6972 | 0.7821 | 0.8133 |
| safe_wrapper_mistral_gliner_taxonomy | 0.6659 | 0.6939 | 0.7791 | 0.8107 |
| full_cag_graph | 0.6676 | 0.6955 | 0.7809 | 0.8122 |
| mistral_boundary | 0.6665 | 0.6945 | 0.7795 | 0.8107 |
| mistral_gliner_taxonomy | 0.6665 | 0.6944 | 0.7797 | 0.8109 |
| qlora_loop_v4 | 0.6836 | 0.7073 | 0.7933 | 0.8365 |
| qlora_loop_v2 | 0.6605 | 0.6888 | 0.7789 | 0.8166 |
| qlora_loop_v1 | 0.6565 | 0.6796 | 0.7650 | 0.8030 |
| qlora_loop_v3 | 0.6434 | 0.6693 | 0.7626 | 0.8037 |
| gpu_bert_fair | 0.6301 | 0.6510 | 0.7560 | 0.8148 |
| bert_baseline | 0.1376 | 0.1462 | 0.2923 | 0.3856 |
| gliner_baseline | 0.0814 | 0.1179 | 0.1677 | 0.1445 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s42 | 0.6301 | 0.6510 | 0.7560 | 0.8148 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s43 | 0.6299 | 0.6483 | 0.7520 | 0.8132 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s44 | 0.6156 | 0.6336 | 0.7452 | 0.8103 |
| gpu_bert_fair_run_bert-base-cased_s42 | 0.6174 | 0.6371 | 0.7438 | 0.8067 |
| gpu_bert_fair_run_bert-base-cased_s43 | 0.6125 | 0.6314 | 0.7372 | 0.7984 |
| gpu_bert_fair_run_bert-base-cased_s44 | 0.6101 | 0.6277 | 0.7330 | 0.7968 |

Oracle: the `nervaluate` 1.2.1 package, run in a throwaway venv on the same spans, gives identical COR/INC/PAR/MIS/SPU counts for every system and mode: **True**. Greedy nervaluate strict F1 can be below the repo exact F1 when an earlier boundary-overlapping prediction consumes a gold span; the one-to-one variant is in results.json.

## 3. Boundary-error taxonomy (events; component-based, D-A1-04)

| system | left_extension | right_extension | left_truncation | right_truncation | both_side_shift | split | merge | many_to_many | label_confusion_exact_boundary | overlap_wrong_label | spurious | spurious_unlocated | missed | reconciles FP/FN | mislocated-occurrence FP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mistral_lora | 30 | 90 | 27 | 38 | 11 | 58 | 80 | 1 | 62 | 40 | 208 | 7 | 214 | True | 6 |
| gpu_anchor_test_rerun | 30 | 90 | 27 | 38 | 11 | 58 | 80 | 1 | 62 | 40 | 206 | 7 | 211 | True | 1 |
| safe_wrapper_boundary_refiner | 30 | 90 | 27 | 38 | 11 | 58 | 80 | 1 | 62 | 40 | 206 | 0 | 211 | True | 1 |
| safe_wrapper | 30 | 90 | 28 | 38 | 11 | 57 | 81 | 1 | 62 | 40 | 206 | 0 | 210 | True | 0 |
| safe_wrapper_full_cag_graph | 30 | 90 | 28 | 38 | 11 | 57 | 81 | 1 | 62 | 40 | 206 | 0 | 210 | True | 0 |
| safe_wrapper_mistral_gliner_taxonomy | 30 | 90 | 28 | 38 | 11 | 57 | 81 | 1 | 62 | 40 | 208 | 7 | 213 | True | 5 |
| full_cag_graph | 30 | 90 | 28 | 37 | 11 | 58 | 81 | 1 | 62 | 41 | 212 | 0 | 207 | True | 5 |
| mistral_boundary | 30 | 90 | 27 | 38 | 11 | 58 | 80 | 1 | 62 | 40 | 208 | 7 | 214 | True | 6 |
| mistral_gliner_taxonomy | 30 | 90 | 28 | 37 | 11 | 58 | 81 | 1 | 62 | 41 | 212 | 7 | 207 | True | 5 |
| qlora_loop_v4 | 37 | 85 | 28 | 51 | 15 | 48 | 78 | 3 | 52 | 28 | 137 | 2 | 237 | True | 2 |
| qlora_loop_v2 | 45 | 96 | 26 | 30 | 12 | 49 | 102 | 3 | 63 | 33 | 194 | 1 | 197 | True | 2 |
| qlora_loop_v1 | 29 | 55 | 30 | 54 | 14 | 45 | 107 | 3 | 50 | 31 | 171 | 8 | 280 | True | 1 |
| qlora_loop_v3 | 32 | 98 | 28 | 37 | 14 | 45 | 104 | 3 | 55 | 40 | 155 | 0 | 269 | True | 2 |
| gpu_bert_fair | 21 | 92 | 49 | 57 | 16 | 108 | 76 | 12 | 42 | 37 | 223 | 0 | 205 | True | 1 |
| bert_baseline | 20 | 252 | 41 | 66 | 90 | 636 | 54 | 53 | 39 | 53 | 3667 | 0 | 220 | True | 3 |
| gliner_baseline | 29 | 6 | 10 | 14 | 10 | 8 | 12 | 0 | 48 | 42 | 69 | 0 | 1965 | True | 0 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s42 | 21 | 92 | 49 | 57 | 16 | 108 | 76 | 12 | 42 | 37 | 223 | 0 | 205 | True | 1 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s43 | 18 | 83 | 54 | 55 | 14 | 115 | 80 | 11 | 40 | 31 | 211 | 0 | 224 | True | 1 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s44 | 10 | 101 | 63 | 56 | 16 | 130 | 79 | 9 | 40 | 36 | 236 | 0 | 193 | True | 1 |
| gpu_bert_fair_run_bert-base-cased_s42 | 23 | 78 | 46 | 50 | 19 | 109 | 87 | 17 | 42 | 29 | 179 | 0 | 258 | True | 2 |
| gpu_bert_fair_run_bert-base-cased_s43 | 20 | 78 | 44 | 53 | 17 | 98 | 87 | 21 | 39 | 30 | 185 | 0 | 288 | True | 2 |
| gpu_bert_fair_run_bert-base-cased_s44 | 25 | 77 | 44 | 57 | 17 | 97 | 90 | 15 | 37 | 27 | 187 | 0 | 298 | True | 1 |

Example sentence ids per category are in results.json (`boundary_error_*` details); no text is reproduced here.

## 4. Skill-free sentences and the full 3,569-sentence test

| system | FP spans / skill-free sentence | share of skill-free sentences with >=1 prediction | full-test exact P | full-test exact R | full-test exact F1 [95% CI] | 974-subset exact F1 |
|---|---|---|---|---|---|---|
| mistral_lora | 0.062 | 0.040 | 0.6281 | 0.6667 | 0.6468 [0.6234, 0.6704] | 0.6701 |
| bert_baseline | 4.628 | 0.890 | 0.0331 | 0.2751 | 0.0592 [0.0529, 0.0657] | 0.1376 |
| gliner_baseline | 0.074 | 0.062 | 0.1921 | 0.0472 | 0.0758 [0.0610, 0.0908] | 0.0814 |
| gpu_bert_fair | 0.084 | 0.049 | 0.5716 | 0.6415 | 0.6045 [0.5809, 0.6281] | 0.6331 |
| gpu_bert_fair_run_bert-base-cased_s42 | 0.081 | 0.054 | 0.5721 | 0.6128 | 0.5918 [0.5675, 0.6157] | 0.6196 |
| gpu_bert_fair_run_bert-base-cased_s43 | 0.087 | 0.052 | 0.5680 | 0.6031 | 0.5850 [0.5601, 0.6095] | 0.6148 |
| gpu_bert_fair_run_bert-base-cased_s44 | 0.083 | 0.050 | 0.5686 | 0.5987 | 0.5832 [0.5583, 0.6070] | 0.6115 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s42 | 0.084 | 0.049 | 0.5716 | 0.6415 | 0.6045 [0.5809, 0.6281] | 0.6331 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s43 | 0.078 | 0.049 | 0.5749 | 0.6371 | 0.6044 [0.5802, 0.6278] | 0.6312 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s44 | 0.094 | 0.054 | 0.5486 | 0.6353 | 0.5888 [0.5644, 0.6133] | 0.6199 |
## 5. H1 - anchor vs baselines (pre-registered)

Rule: H1 SUPPORTED iff anchor exact-span F1 minus baseline >= 0.10 AND Holm-adjusted p < 0.05 against EVERY baseline in family H1_anchor_vs_baselines (paired, 974 test sentences; p = approximate randomization, 10,000 shuffles; CI = paired bootstrap, 10,000 resamples).

| baseline | anchor F1 | baseline F1 | Δ [95% CI] | p (AR) | p (boot) | p Holm | passes |
|---|---|---|---|---|---|---|---|
| bert_baseline | 0.6701 | 0.1376 | 0.5325 [0.5112, 0.5542] | 1.00e-04 | 2.00e-04 | 3.00e-04 | True |
| gliner_baseline | 0.6701 | 0.0814 | 0.5887 [0.5599, 0.6164] | 1.00e-04 | 2.00e-04 | 3.00e-04 | True |
| gpu_bert_fair | 0.6701 | 0.6331 | 0.0370 [0.0165, 0.0573] | 4.00e-04 | 1.80e-03 | 4.00e-04 | False |

**H1 verdict: NOT SUPPORTED**.

Fair-baseline seed runs (descriptive; selection used dev only): n=6, exact F1 on the 974 sentences mean 0.6217, sd 0.0087, range [0.6115, 0.6331].

Secondary (full 3,569-sentence protocol): vs bert_baseline: Δ 0.5877 [0.5657, 0.6089], Holm p 3.00e-04; vs gliner_baseline: Δ 0.5710 [0.5434, 0.5984], Holm p 3.00e-04; vs gpu_bert_fair: Δ 0.0423 [0.0220, 0.0628], Holm p 3.00e-04

### Baseline fairness (straw-man risk)

- The repo's 'BERT baseline' is a train-set token->tag lookup table (`baselines/bert_baseline.py`, `training_mode: token_classification_memorized_baseline`), not a BERT model. The 'GLiNER baseline' is zero-shot GLiNER medium-v2.1 with no SkillSpan training.
- The stored baseline metrics (0.059150 / 0.075833) were computed over all 3,569 sentences, the anchor's over 974; on the same 974 sentences they are higher (table 2).
- Published reference (Zhang, Jensen, Sonniks, Plank (2022). SkillSpan: Hard and Soft Skill Extraction from English Job Postings. NAACL-HLT 2022, pp. 4962-4984. doi:10.18653/v1/2022.naacl-main.366; Table 5 (Appendix E), TEST rows; columns SKILL / KNOWLEDGE / COMBINED x STL / MTL; span-F1 x100; PDF sha256 `d8eddf4c63b96c76...`), TEST span-F1:

| model | SKILL STL | KNOWLEDGE STL | COMBINED STL | COMBINED MTL |
|---|---|---|---|---|
| BERTbase | 54.34±0.74 | 62.43±0.41 | 58.16±0.47 | 57.73±0.66 |
| JobBERT | 56.11±0.49 | 63.88±0.28 | 59.73±0.38 | 59.18±0.37 |
| JobSpanBERT | 56.64±0.83 | 61.06±0.99 | 58.72±0.69 | 58.90±0.48 |

- The paper's TEST split has 4,680 sentences from BIG+HOUSE+TECH (Table 2); the HF release used here has only HOUSE+TECH (3,569 sentences; the paper lists 1,216+2,352=3,568), because BIG was not released.
- The paper scores ALL test sentences, including those without skills; the repo's anchor 0.670 is on the 974-sentence subset with gold spans, which cannot contain false positives on skill-free sentences.
- Label mapping: SkillSpan SKILL = repo SOFT_SKILL, KNOWLEDGE = repo HARD_SKILL; COMBINED (STL) = the two single-task models' spans merged, closest to the repo's labelled micro exact-span F1.
- Consequence: a supervised BERT-family tagger is expected around 0.58-0.60 span-F1 on the full test, i.e. far above 0.06-0.14. Whether the anchor beats such a model by >=0.10 is untested until A3's fair baseline exists and the anchor's skill-free predictions allow a same-protocol comparison.

## 6. Claims vs artifacts

Status counts over 651 extracted extraction-metric claims: match 300, mismatch 158, not_recomputable_ensemble_or_old_run 74, not_recomputable_dev_or_other_split 42, threshold_or_target_value 41, match_other_system_or_metric 36. Full table: `claims_reconciliation.csv`.

Known values:

| value | occurrences | files | equals recomputed |
|---|---|---|---|
| 0.670069 | 33 | 14 | mistral_boundary:exact_span_f1, mistral_boundary:micro_f1, mistral_lora:exact_span_f1, mistral_lora:micro_f1 |
| 0.6701 | 30 | 14 | mistral_boundary:exact_span_f1, mistral_boundary:micro_f1, mistral_gliner_taxonomy:exact_span_f1, mistral_gliner_taxonomy:micro_f1 |
| 0.675034 | 20 | 6 | nothing (not reproducible) |
| 0.675 | 10 | 7 | gpu_anchor_test_rerun:exact_span_precision, gpu_bert_fair_run_bert-base-cased_s43:hard_skill_f1, qlora_loop_v3:exact_span_precision |
| 0.669478 | 11 | 5 | safe_wrapper_mistral_gliner_taxonomy:exact_span_f1, safe_wrapper_mistral_gliner_taxonomy:micro_f1 |
| 0.670222 | 4 | 2 | nothing (not reproducible) |
| 0.672602 | 4 | 2 | nothing (not reproducible) |
| 0.672444 | 12 | 5 | safe_wrapper_boundary_refiner:exact_span_f1, safe_wrapper_boundary_refiner:micro_f1 |
| 0.671854 | 13 | 5 | safe_wrapper:exact_span_f1, safe_wrapper:micro_f1, safe_wrapper_full_cag_graph:exact_span_f1, safe_wrapper_full_cag_graph:micro_f1 |
| 0.671102 | 13 | 7 | full_cag_graph:exact_span_f1, full_cag_graph:micro_f1 |
| 0.670062 | 5 | 4 | mistral_gliner_taxonomy:exact_span_f1, mistral_gliner_taxonomy:micro_f1 |

Anchor/wrapper-attributed claims that equal no recomputed value (27), by file:

- `reports/skillllm_v2/extraction_ablation.md`: 15 values - stub-extractor ablation on 20 DEV documents (already flagged as unusable in docs/INFERENCE_AND_EVALS.md, items E13/I7); not a test result
- `reports/skillllm_v2/wrapper_error_analysis.md`: 12 values - June-2026 report computed from the earlier run-1 anchor (exact F1 0.675034) and pre-gate wrapper stages; no stored predictions reproduce it
- 0.675034 (20 occurrences) is the anchor's exact F1 from an earlier run ('run 1'); it is used as the wrapper gate's improvement target and appears in `wrapper_error_analysis.md`; no stored prediction file reproduces it. 0.670222 / 0.672602 (significance_f1.md) likewise match no artifact.

| file:line | claimed | attributed | nearest recomputed (system:metric) | |diff| |
|---|---|---|---|---|
| reports/skillllm_v2/extraction_ablation.md:3 | 0.0571 | mistral_lora | mistral_lora:exact_span_recall = 0.666667 | 0.609567 |
| reports/skillllm_v2/extraction_ablation.md:3 | 0.1143 | mistral_lora | mistral_lora:partial_span_recall = 0.809713 | 0.695413 |
| reports/skillllm_v2/extraction_ablation.md:3 | 0.0755 | mistral_lora | mistral_lora:hard_skill_precision = 0.702466 | 0.626966 |
| reports/skillllm_v2/extraction_ablation.md:7 | 0.2025 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:exact_span_recall = 0.665784 | 0.463284 |
| reports/skillllm_v2/extraction_ablation.md:7 | 0.2785 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:partial_span_recall = 0.809713 | 0.531213 |
| reports/skillllm_v2/extraction_ablation.md:7 | 0.1595 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:token_precision = 0.774419 | 0.614919 |
| reports/skillllm_v2/extraction_ablation.md:7 | 0.2581 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:hard_skill_precision = 0.704800 | 0.446700 |
| reports/skillllm_v2/extraction_ablation.md:8 | 0.2025 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:exact_span_recall = 0.665784 | 0.463284 |
| reports/skillllm_v2/extraction_ablation.md:8 | 0.2785 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:partial_span_recall = 0.809713 | 0.531213 |
| reports/skillllm_v2/extraction_ablation.md:8 | 0.1595 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:token_precision = 0.774419 | 0.614919 |
| reports/skillllm_v2/extraction_ablation.md:8 | 0.2581 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:hard_skill_precision = 0.704800 | 0.446700 |
| reports/skillllm_v2/extraction_ablation.md:9 | 0.2025 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:exact_span_recall = 0.665784 | 0.463284 |
| reports/skillllm_v2/extraction_ablation.md:9 | 0.2785 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:partial_span_recall = 0.809713 | 0.531213 |
| reports/skillllm_v2/extraction_ablation.md:9 | 0.1595 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:token_precision = 0.774419 | 0.614919 |
| reports/skillllm_v2/extraction_ablation.md:9 | 0.2581 | safe_wrapper,safe_wrapper_full_cag_graph,safe_wrapper_boundary_refiner,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:hard_skill_precision = 0.704800 | 0.446700 |
| reports/skillllm_v2/wrapper_error_analysis.md:16 | 0.5704 | mistral_boundary,safe_wrapper_boundary_refiner | mistral_boundary:exact_span_recall = 0.666667 | 0.096267 |
| reports/skillllm_v2/wrapper_error_analysis.md:16 | 0.7847 | mistral_boundary,safe_wrapper_boundary_refiner | mistral_boundary:token_f1 = 0.787886 | 0.003186 |
| reports/skillllm_v2/wrapper_error_analysis.md:16 | 0.7086 | mistral_boundary,safe_wrapper_boundary_refiner | safe_wrapper_boundary_refiner:hard_skill_precision = 0.705976 | 0.002624 |
| reports/skillllm_v2/wrapper_error_analysis.md:16 | 0.4258 | mistral_boundary,safe_wrapper_boundary_refiner | mistral_boundary:soft_skill_recall = 0.574702 | 0.148902 |
| reports/skillllm_v2/wrapper_error_analysis.md:17 | 0.6515 | mistral_gliner_taxonomy,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:exact_span_recall = 0.665784 | 0.014284 |
| reports/skillllm_v2/wrapper_error_analysis.md:17 | 0.7827 | mistral_gliner_taxonomy,safe_wrapper_mistral_gliner_taxonomy | mistral_gliner_taxonomy:token_f1 = 0.787575 | 0.004875 |
| reports/skillllm_v2/wrapper_error_analysis.md:17 | 0.7181 | mistral_gliner_taxonomy,safe_wrapper_mistral_gliner_taxonomy | safe_wrapper_mistral_gliner_taxonomy:hard_skill_f1 = 0.726898 | 0.008798 |
| reports/skillllm_v2/wrapper_error_analysis.md:17 | 0.5777 | mistral_gliner_taxonomy,safe_wrapper_mistral_gliner_taxonomy | mistral_gliner_taxonomy:soft_skill_recall = 0.574702 | 0.002998 |
| reports/skillllm_v2/wrapper_error_analysis.md:18 | 0.5616 | full_cag_graph,safe_wrapper_full_cag_graph,safe_wrapper | safe_wrapper:exact_span_recall = 0.667108 | 0.105508 |
| reports/skillllm_v2/wrapper_error_analysis.md:18 | 0.7482 | full_cag_graph,safe_wrapper_full_cag_graph,safe_wrapper | full_cag_graph:token_precision = 0.773495 | 0.025295 |
| reports/skillllm_v2/wrapper_error_analysis.md:18 | 0.7069 | full_cag_graph,safe_wrapper_full_cag_graph,safe_wrapper | safe_wrapper:hard_skill_precision = 0.708333 | 0.001433 |
| reports/skillllm_v2/wrapper_error_analysis.md:18 | 0.4080 | full_cag_graph,safe_wrapper_full_cag_graph,safe_wrapper | full_cag_graph:soft_skill_recall = 0.574702 | 0.166702 |

## 7. Descriptive paired Δ exact F1 vs anchor (not hypothesis tests)

| system | Δ | 95% CI | p (boot) | sentences that differ |
|---|---|---|---|---|
| qlora_loop_v4 | +0.0145 | [-0.0045, +0.0330] | 0.129 | 356 |
| safe_wrapper_boundary_refiner | +0.0024 | [+0.0009, +0.0042] | 0.001 | 10 |
| safe_wrapper | +0.0018 | [-0.0005, +0.0042] | 0.13 | 19 |
| safe_wrapper_full_cag_graph | +0.0018 | [-0.0005, +0.0042] | 0.13 | 19 |
| gpu_anchor_test_rerun | +0.0013 | [+0.0000, +0.0030] | 0.091 | 3 |
| full_cag_graph | +0.0010 | [-0.0015, +0.0036] | 0.425 | 25 |
| mistral_boundary | +0.0000 | [+0.0000, +0.0000] | 1 | 0 |
| mistral_gliner_taxonomy | -0.0000 | [-0.0023, +0.0023] | 0.998 | 19 |
| safe_wrapper_mistral_gliner_taxonomy | -0.0006 | [-0.0024, +0.0010] | 0.551 | 9 |
| qlora_loop_v2 | -0.0086 | [-0.0295, +0.0129] | 0.445 | 359 |
| qlora_loop_v1 | -0.0135 | [-0.0336, +0.0070] | 0.204 | 412 |
| qlora_loop_v3 | -0.0262 | [-0.0471, -0.0063] | 0.013 | 406 |
| gpu_bert_fair | -0.0370 | [-0.0574, -0.0164] | 0.002 | 451 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s42 | -0.0370 | [-0.0574, -0.0164] | 0.002 | 451 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s43 | -0.0388 | [-0.0610, -0.0170] | 0.001 | 469 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s44 | -0.0502 | [-0.0714, -0.0305] | 0.001 | 487 |
| gpu_bert_fair_run_bert-base-cased_s42 | -0.0504 | [-0.0723, -0.0293] | 0.001 | 479 |
| gpu_bert_fair_run_bert-base-cased_s43 | -0.0553 | [-0.0771, -0.0338] | 0.001 | 485 |
| gpu_bert_fair_run_bert-base-cased_s44 | -0.0586 | [-0.0808, -0.0365] | 0.001 | 495 |
| bert_baseline | -0.5325 | [-0.5546, -0.5115] | 0.001 | 922 |
| gliner_baseline | -0.5887 | [-0.6171, -0.5597] | 0.001 | 886 |

## Files

- `results.json` (all records), `per_unit_scores.jsonl` (rows `layer, family=exact_span, metric=exact_span_counts, unit_id, posting_id, system, population, label in ALL/HARD_SKILL/SOFT_SKILL, tp, fp, fn` [+ char-token counts on ALL rows]), `claims_reconciliation.csv`, `fig_exact_f1_ci.png`, `fig_exact_f1_ci.csv`, `fig_boundary_errors.png`, `fig_boundary_errors.csv`

## Caveats

- Single SkillSpan test split and one training run per system: CIs reflect sentence sampling only, not seed/training variance.
- Primary population = the 974 test sentences that contain gold spans; false positives on the 2,595 skill-free test sentences are only counted in the explicitly labelled full-test records.
- Resampling unit = sentence (repo 'document'); sentences come from 65 job postings, so posting-level CIs (details.ci_posting) are wider.
- Partial-span F1 and char-level token F1 are repository definitions, not SkillSpan-official metrics.
