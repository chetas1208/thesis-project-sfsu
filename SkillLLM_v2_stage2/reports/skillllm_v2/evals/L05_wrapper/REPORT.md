# L05_wrapper — contribution of each safe-wrapper component vs the Raw Mistral LoRA anchor

Data: SkillSpan test (HF jjzha/skillspan), 974 docs with >=1 gold span (2,265 gold spans); n = 974 documents. Seed 42; bootstrap 2000 (headline F1/APR/H2: 10000); approximate randomization 10,000 shuffles (exact enumeration when ≤16 documents differ). Definitions: Decisions.md D-A2-02.

## H2 verdict

**NOT SUPPORTED.** Rule: APR(safe) − APR(legacy) CI > 0 vs BOTH legacy artefacts AND F1(safe) − F1(anchor) lower CI > −0.005.

| comparator | APR safe | APR comparator | ΔAPR | 95% CI | p boot | p AR | p Holm |
|---|---|---|---|---|---|---|---|
| legacy_full_cag_graph | 0.9987 | 0.9987 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 | 1.0000 |
| legacy_mistral_gliner_taxonomy | 0.9987 | 0.9987 | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 | 1.0000 |

- Condition (i) ΔAPR > 0 vs both legacy: **False**.
- Condition (ii) non-inferiority: Δ exact F1 (safe − anchor) = +0.00179, 95% CI [-0.00058, +0.00416] (p boot 0.1314, p AR 0.1227); margin −0.005 → **True**.
- Secondary (post-hoc, same anchor): APR safe replay 0.9987 vs pre-fix-like policy-off counterfactual 0.8318, Δ +0.1669 [0.1474, 0.1875]; ΔF1 +0.1314 [0.1143, 0.1491]; upper bound (all edits forced) APR 0.4139.
- Historical pre-fix full wrapper (`reports/skillllm_v2/wrapper_error_analysis.json`, vs the run-1 anchor F1 0.6750; aggregates only, no predictions on disk): exact F1 0.5616 (Δ -0.1135); anchor TPs kept 1281, destroyed 225 → implied APR 0.8506; boundary-modified 194, verifier rejections 29, GLiNER additions 146, dictionary additions 51. Use it to judge how close the policy-off constructs are to the real pre-fix behaviour.
- Caveat: the legacy artefacts regenerated Mistral live and already contained the source-aware components (their metadata shows `accepted_anchor`), so their APR loss mixes generation non-determinism with ungated post-processing. The truly destructive pre-fix pipeline (exact F1 0.56–0.57 in `wrapper_error_analysis.md`) has no predictions on disk; the policy-off counterfactual is A2's approximation of it.

## Ladder (cumulative; replayed on the stored anchor)

| system | exact F1 | Δ vs anchor [95% CI] | p AR | APR | NWG/1000 | TP+ | TP− | FP+ | FP− | docs Δ | soft-recall loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_raw_mistral | 0.6701 | - | - | 1.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | - |
| R1_boundary | 0.6714 | +0.0013 [0.0000, 0.0031] | 0.2500 | 1.0000 | 2.6490 | 3 | 0 | 0 | 3 | 3 | 0.0000 |
| R2_gliner_addonly | 0.6716 | +0.0015 [-0.0006, +0.0037] | 0.1389 | 0.9987 | 3.9735 | 3 | 2 | 0 | 8 | 7 | 0.0000 |
| R3_dictionary | 0.6711 | +0.0010 [-0.0012, +0.0032] | 0.3603 | 0.9987 | 2.6490 | 3 | 2 | 3 | 8 | 10 | 0.0000 |
| R4_taxonomy | 0.6711 | +0.0010 [-0.0011, +0.0033] | 0.3640 | 0.9987 | 2.6490 | 3 | 2 | 3 | 8 | 10 | 0.0000 |
| R5_verifier | 0.6722 | +0.0021 [-0.0002, +0.0045] | 0.0656 | 0.9987 | 5.7395 | 3 | 2 | 3 | 15 | 17 | 0.0000 |
| R6_calibration | 0.6722 | +0.0021 [-0.0002, +0.0044] | 0.0617 | 0.9987 | 5.7395 | 3 | 2 | 3 | 15 | 17 | 0.0000 |
| R7_safe_wrapper_stored | 0.6719 | +0.0018 [-0.0006, +0.0041] | 0.1185 | 0.9987 | 4.8565 | 3 | 2 | 5 | 15 | 19 | 0.0000 |

Stage diagnostics (pooled counts): verifier rejections / of which gold-correct; calibrator rejections / gold-correct; merger: gold-correct GLiNER/dictionary candidates dropped by the add-only policy, gold-correct anchor spans dropped, anchor spans removed by grouping/subsumption.

| system | verifier rej | verifier false rej | calib rej | calib false rej | merger gold drop (new) | merger gold drop (anchor) | merger anchor dedupe |
|---|---|---|---|---|---|---|---|
| R1_boundary | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R2_gliner_addonly | 0 | 0 | 0 | 0 | 29 | 2 | 7 |
| R3_dictionary | 0 | 0 | 0 | 0 | 42 | 2 | 7 |
| R4_taxonomy | 0 | 0 | 0 | 0 | 42 | 2 | 7 |
| R5_verifier | 7 | 0 | 0 | 0 | 42 | 2 | 7 |
| R6_calibration | 7 | 0 | 0 | 0 | 42 | 2 | 7 |

## Order-dependence: single components and leave-one-out

| system | exact F1 | Δ vs anchor [95% CI] | p AR | APR | NWG/1000 | TP+ | TP− | FP+ | FP− | docs Δ | soft-recall loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| alone_refiner | 0.6714 | +0.0013 [0.0000, 0.0031] | 0.2500 | 1.0000 | 2.6490 | 3 | 0 | 0 | 3 | 3 | 0.0000 |
| alone_merger_dedupe | 0.6702 | +0.0002 [-0.0015, +0.0015] | 0.7500 | 0.9987 | 1.3245 | 0 | 2 | 0 | 5 | 4 | 0.0000 |
| alone_gliner | 0.6702 | +0.0002 [-0.0013, +0.0016] | 0.7493 | 0.9987 | 1.3245 | 0 | 2 | 0 | 5 | 4 | 0.0000 |
| alone_dictionary | 0.6702 | +0.0002 [-0.0015, +0.0016] | 0.7601 | 0.9987 | 1.3245 | 0 | 2 | 0 | 5 | 4 | 0.0000 |
| alone_verifier | 0.6711 | +0.0010 [0.0004, 0.0019] | 0.0156 | 1.0000 | 3.0905 | 0 | 0 | 0 | 7 | 7 | 0.0000 |
| alone_calibrator | 0.6701 | +0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| loo_minus_refine | 0.6708 | +0.0007 [-0.0012, +0.0024] | 0.3824 | 0.9987 | 3.0905 | 0 | 2 | 3 | 12 | 14 | 0.0000 |
| loo_minus_gliner | 0.6726 | +0.0025 [0.0003, 0.0049] | 0.0188 | 0.9987 | 7.0640 | 3 | 2 | 0 | 15 | 14 | 0.0000 |
| loo_minus_dictionary | 0.6726 | +0.0025 [0.0003, 0.0049] | 0.0154 | 0.9987 | 7.0640 | 3 | 2 | 0 | 15 | 14 | 0.0000 |
| loo_minus_verify | 0.6711 | +0.0010 [-0.0012, +0.0033] | 0.3690 | 0.9987 | 2.6490 | 3 | 2 | 3 | 8 | 10 | 0.0000 |
| loo_minus_calibrate | 0.6722 | +0.0021 [-0.0002, +0.0044] | 0.0607 | 0.9987 | 5.7395 | 3 | 2 | 3 | 15 | 17 | 0.0000 |

## Stored artefacts, legacy comparators, policy-off counterfactual, leakage ablation

| system | exact F1 | Δ vs anchor [95% CI] | p AR | APR | NWG/1000 | TP+ | TP− | FP+ | FP− | docs Δ | soft-recall loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stored_boundary_refiner | 0.6724 | +0.0024 [0.0009, 0.0043] | 0.0020 | 1.0000 | 5.7395 | 3 | 0 | 0 | 10 | 10 | 0.0000 |
| stored_mistral_gliner_taxonomy | 0.6695 | -0.0006 [-0.0024, +0.0009] | 0.6133 | 0.9987 | -0.8830 | 0 | 2 | 5 | 5 | 9 | 0.0000 |
| legacy_full_cag_graph | 0.6711 | +0.0010 [-0.0015, +0.0036] | 0.4818 | 0.9987 | 1.7660 | 5 | 2 | 11 | 12 | 25 | 0.0000 |
| legacy_mistral_gliner_taxonomy | 0.6701 | -0.0000 [-0.0023, +0.0024] | 0.9350 | 0.9987 | -1.3245 | 5 | 2 | 11 | 5 | 19 | 0.0000 |
| legacy_mistral_boundary | 0.6701 | +0.0000 [0.0000, 0.0000] | 1.0000 | 1.0000 | 0.0000 | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| policyoff_full | 0.2901 | -0.3800 [-0.4065, -0.3544] | 0.0001 | 0.4139 | -877.7042 | 85 | 885 | 1754 | 566 | 781 | 0.4418 |
| policyoff_prefix_like | 0.5408 | -0.1293 [-0.1470, -0.1122] | 0.0001 | 0.8318 | -345.2539 | 80 | 254 | 841 | 233 | 478 | 0.1292 |
| policyoff_refiner_only | 0.2926 | -0.3774 [-0.4040, -0.3521] | 0.0001 | 0.4126 | -839.7351 | 78 | 887 | 1658 | 565 | 760 | 0.4455 |
| policyoff_merger_only | 0.6587 | -0.0114 [-0.0168, -0.0056] | 0.0003 | 0.9987 | -46.7991 | 29 | 2 | 148 | 15 | 151 | -0.0128 |
| policyoff_verifier_only | 0.6596 | -0.0104 [-0.0181, -0.0032] | 0.0057 | 0.9298 | 15.4525 | 3 | 106 | 3 | 141 | 223 | 0.0889 |
| policyoff_calibrator_only | 0.6722 | +0.0021 [-0.0001, +0.0044] | 0.0600 | 0.9987 | 5.7395 | 3 | 2 | 3 | 15 | 17 | 0.0000 |
| leakage_no_test_mined_aliases | 0.6722 | +0.0021 [-0.0003, +0.0045] | 0.0641 | 0.9987 | 5.7395 | 3 | 2 | 3 | 15 | 17 | 0.0000 |

Leakage ablation: removing the 76 alias rows mined from test-split anchor predictions changes the replayed safe wrapper's exact F1 from 0.67215 to 0.67215 (each vs anchor: +0.00208 and +0.00208).

## Where the changes come from

Boundary edits applied (by type): n / original span was gold / edited span is gold.

- `R1_boundary`: realign_word_boundary: 3 / 0 / 3
- `R6_calibration`: realign_word_boundary: 3 / 0 / 3
- `R7_safe_wrapper_stored`: realign_word_boundary: 3 / 0 / 3
- `policyoff_full`: head_skill: 1175 / 695 / 39; realign_word_boundary: 3 / 0 / 3; split_conjunction: 547 / 328 / 32; taxonomy_normalized: 5 / 0 / 1; trim_leading_filler: 11 / 5 / 2; trim_trailing_filler: 67 / 36 / 5
- `policyoff_prefix_like`: split_conjunction: 701 / 391 / 53; trim_leading_filler: 16 / 5 / 4; trim_trailing_filler: 75 / 40 / 5
- `legacy_full_cag_graph`: none

Spans not in the anchor, by origin: n / gold.

- `R3_dictionary`: boundary_edit_of_anchor: 3 / 3; dictionary+gliner: 3 / 0
- `R6_calibration`: boundary_edit_of_anchor: 3 / 3; dictionary+gliner: 3 / 0
- `R7_safe_wrapper_stored`: boundary_edit_of_anchor: 3 / 3; dictionary+gliner: 5 / 0
- `legacy_full_cag_graph`: dictionary: 13 / 5; dictionary+gliner: 3 / 0
- `legacy_mistral_gliner_taxonomy`: dictionary: 13 / 5; dictionary+gliner: 3 / 0
- `policyoff_full`: boundary_edit_of_anchor: 1712 / 77; dictionary: 41 / 9; dictionary+gliner: 3 / 0; gliner: 111 / 4
- `policyoff_prefix_like`: boundary_edit_of_anchor: 766 / 57; dictionary: 41 / 10; dictionary+gliner: 3 / 0; gliner: 126 / 16

## Replay reproduction

- R6_calibration_vs_R7_safe_wrapper_stored: 968/974 documents identical (differing: test-507, test-522, test-523, test-2435, test-3098, test-3188).
- replay_refine_verify_calibrate_vs_stored_boundary_refiner: 974/974 documents identical (differing: none).
- replay_merge_link_vs_stored_mistral_gliner_taxonomy: 968/974 documents identical (differing: test-507, test-522, test-523, test-2435, test-3098, test-3188).

## Tuning / test-contact audit (contract rule 2)

| component | configured on | test contact | flag |
|---|---|---|---|
| Raw Mistral LoRA anchor | SkillSpan train (SFT, 80 epochs); final checkpoint | none found |  |
| CandidateMerger thresholds | hand-set YAML artifacts/skillllm_v2/candidate_merger/tuned_config.yaml (gliner 0.95, default 0.85; library default gliner 0.9); values inside tuner grid: False; dev predictions file present: False | not tuner output; no dev tuning evidence on disk | FLAG |
| BoundarySelector config | YAML identical to DEFAULT_CONFIG: True; allowed_edit_types in tuner grid: False | realign_word_boundary / anchor protection were introduced after the TEST-split error analysis (reports/skillllm_v2/wrapper_error_analysis.md, split=test) | FLAG |
| SpanVerifier | rule list GENERIC_REJECTS + min_confidence 0.35 (no fitting); the trained TF-IDF/LogReg verifier is never loaded | anchor protection designed after test error analysis | FLAG |
| ConfidenceCalibrator | library defaults (tuned YAML present: False) | none (defaults) |  |
| Alias table (merger evidence gate, boundary alias bonus, taxonomy_normalized, linker) | {"manual_curated": 34, "skillspan_gold_train_dev": 601, "mistral_high_confidence": 76} | 76 rows mined from TEST-split anchor predictions | FLAG |
| DictionaryExtractor default list | 22 hard-coded strings | strings that occur as gold spans only in TEST (never train/dev): [] |  |
| Non-regression gate | evaluated on the 974-doc TEST subset; decides which output ships | selection on test (the final system choice is test-informed) | FLAG |
| GLiNER (zero-shot urchade/gliner_medium-v2.1, thr 0.35) | library default | none |  |

## Reading guide / caveats

- All wrapper effects are a handful of documents; read Δ with `docs Δ` and the AR p-value (exact when ≤16 documents differ).
- TP gained ≡ recovered anchor FN (set semantics, fixed gold).
- Taxonomy linking never changes spans (asserted); its contribution is linking coverage, evaluated in L06.
- The replayed GLiNER source is the stored zero-shot GLiNER output, not the wrapper's own (unpersisted) live calls.
- Single anchor training run, single split; no training-seed variance in any CI.

Files: `ladder/ladder_summary.{csv,json}`, `ladder/per_doc_counts.csv`, `per_unit_scores.jsonl`, `wrapper_ladder.{png,csv}`, `wrapper_policy_off.png`, replayed predictions `inputs/evals/research/L05_wrapper/replay_predictions/`.
