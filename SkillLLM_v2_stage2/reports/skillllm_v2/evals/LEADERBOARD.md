# SkillLLM_v2 research evaluation — leaderboard

Generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/*/results.json` (git 54aab906d548 dirty). Values are copied from layer records, never recomputed here. `-` = no value. A missing layer means its results.json does not exist yet; nothing is estimated in its place.

## Layer status

| layer | title | owner | status | records | measured | pending_gpu | pending_human | not_available | generated_utc |
|---|---|---|---|---|---|---|---|---|---|
| L01_extraction | Span extraction (SkillSpan test) | A1 | measured | 1370 | 1370 | 0 | 0 | 0 | 2026-11-18T06:54:54+00:00 |
| L02_generalization | Unseen-skill generalization | A1 | measured | 747 | 747 | 0 | 0 | 0 | 2026-11-18T06:58:38+00:00 |
| L03_robustness | Robustness to perturbations | A3 | measured + not_available | 329 | 324 | 0 | 0 | 5 | 2026-11-18T07:35:14+00:00 |
| L04_calibration | Calibration and selective prediction | A2 | measured | 1409 | 1409 | 0 | 0 | 0 | 2026-11-18T07:08:45+00:00 |
| L05_wrapper | Safe-wrapper contribution | A2 | measured | 826 | 826 | 0 | 0 | 0 | 2026-11-17T21:26:54+00:00 |
| L06_linking | Taxonomy linking / normalization | A4 | measured + pending_human | 828 | 827 | 0 | 1 | 0 | 2026-11-18T08:23:32+00:00 |
| L07_compaction | Profile compaction | A4 | measured | 560 | 560 | 0 | 0 | 0 | 2026-11-17T21:16:49+00:00 |
| L08_graph_retrieval | Graph-augmented retrieval | A5 | measured + not_available | 174 | 169 | 0 | 0 | 5 | 2026-11-18T07:08:07+00:00 |
| L09_course_rec | Course recommendation | A5 | measured | 385 | 385 | 0 | 0 | 0 | 2026-11-18T07:14:29+00:00 |
| L10_significance | Central significance testing | A2 | measured | 1056 | 1056 | 0 | 0 | 0 | 2026-11-18T08:28:22+00:00 |
| L11_human_eval | Human evaluation | A6 | measured + pending_human | 57 | 8 | 0 | 49 | 0 | 2026-11-18T08:25:23+00:00 |
| L12_systems | Systems: latency, memory | A3 | measured + not_available | 48 | 45 | 0 | 0 | 3 | 2026-11-18T07:42:00+00:00 |

## Headline metrics

Records flagged `details.headline` by their owning layer (95% CI; n = independent units).

| layer | system | metric | value | ci | n | unit | split | provenance |
|---|---|---|---|---|---|---|---|---|
| L01_extraction | mistral_lora | exact_span_f1 | 0.6701 | [0.6465, 0.6938] | 974 | sentence | test[974 sentences with >=1 gold span] | artifacts/skillllm_v2/mistral_lora/predictions.jsonl@ac5917d026 (+1) |
| L01_extraction | mistral_lora | token_f1 | 0.7879 | [0.7701, 0.8053] | 974 | sentence | test[974 sentences with >=1 gold span] | artifacts/skillllm_v2/mistral_lora/predictions.jsonl@ac5917d026 (+1) |
| L02_generalization | mistral_lora | unseen_exact_f1 | 0.5874 | [0.5589, 0.6157] | 974 | sentence | test[974 sentences with >=1 gold span] | artifacts/skillllm_v2/mistral_lora/predictions.jsonl@ac5917d026 (+5) |
| L03_robustness | anchor | mean_degradation_exact_f1 | 0.0223 | [0.0035, 0.0417] | 120 | document | test | inputs/evals/research/L03_robustness/perturbed_docs.jsonl@d5935c230d (+2) |
| L04_calibration | anchor_logprob:primary[sum_lp]_platt | sel_risk@80 | 0.2664 | [0.2394, 0.2950] | 974 | document (cluster of spans) | test | artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/predictions.jsonl@b4c75aa87b (+3) |
| L04_calibration | anchor_logprob:primary[sum_lp]_platt | ece_w10 | 0.0187 | [0.0133, 0.0422] | 974 | document (cluster of spans) | test | artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/predictions.jsonl@b4c75aa87b (+3) |
| L05_wrapper | R7_safe_wrapper_stored | exact_span_f1 | 0.6719 | [0.6477, 0.6954] | 974 | document | test | data/source/skillspan_hf/test.json@8438eda349 (+2) |
| L05_wrapper | R7_safe_wrapper_stored | apr | 0.9987 | [0.9958, 1.0000] | 974 | document | test | data/source/skillspan_hf/test.json@8438eda349 (+2) |
| L06_linking | shipped_bridge | top1_acc_linkable | 0.1573 | [0.1258, 0.1903] | 55 | posting (cluster); mentions in details | test | data/external/decorte_skill_extraction/tech/validation.csv@7efcc66c27 (+4) |
| L06_linking | shipped_bridge | recall@5 | 0.1620 | [0.1295, 0.1968] | 55 | posting (cluster); mentions in details | test | data/external/decorte_skill_extraction/tech/validation.csv@7efcc66c27 (+4) |
| L06_linking | shipped_bridge | mrr@10 | 0.1594 | [0.1280, 0.1930] | 55 | posting (cluster); mentions in details | test | data/external/decorte_skill_extraction/tech/validation.csv@7efcc66c27 (+4) |
| L06_linking | cascade_esco@tau_val | top1_acc_linkable | 0.3239 | [0.2856, 0.3599] | 55 | posting (cluster); mentions in details | test | data/external/decorte_skill_extraction/tech/validation.csv@7efcc66c27 (+4) |
| L06_linking | cascade_esco@tau_val | recall@5 | 0.5056 | [0.4604, 0.5505] | 55 | posting (cluster); mentions in details | test | data/external/decorte_skill_extraction/tech/validation.csv@7efcc66c27 (+4) |
| L06_linking | cascade_esco@tau_val | mrr@10 | 0.3999 | [0.3619, 0.4380] | 55 | posting (cluster); mentions in details | test | data/external/decorte_skill_extraction/tech/validation.csv@7efcc66c27 (+4) |
| L07_compaction | compact@0.5[informative] | token_reduction_vs_full | 0.0928 | [0.0110, 0.1674] | 120 | profile | synthetic | artifacts/skillllm_v2/mistral_lora/model/tokenizer.json@260c7a2c57 (+6) |
| L07_compaction | compact@0.5[informative] | job_ndcg10 | 0.2487 | [0.2007, 0.2993] | 119 | profile | synthetic | artifacts/skillllm_v2/mistral_lora/model/tokenizer.json@260c7a2c57 (+6) |
| L07_compaction | compact@0.5[informative] | course_ndcg10 | 0.7284 | [0.6939, 0.7604] | 117 | profile | synthetic | artifacts/skillllm_v2/mistral_lora/model/tokenizer.json@260c7a2c57 (+6) |
| L07_compaction | compact@0.5[informative] | gap_jaccard_vs_full | 0.3483 | [0.2961, 0.4014] | 120 | profile | synthetic | artifacts/skillllm_v2/mistral_lora/model/tokenizer.json@260c7a2c57 (+6) |
| L08_graph_retrieval | hybrid+graph | recall@10 | 0.8424 | [0.7578, 0.9199] | 24 | scenario | test | inputs/courses/course_catalog_large.jsonl@1b42b903e7 (+3) |
| L08_graph_retrieval | hybrid+graph | ndcg@10 | 0.8244 | [0.7777, 0.8691] | 24 | scenario | test | inputs/courses/course_catalog_large.jsonl@1b42b903e7 (+3) |
| L08_graph_retrieval | SkillLLM_v2 Full recommender (explanation paths) | path_validity | 1.0000 | [1.0000, 1.0000] | 24 | scenario | test | inputs/courses/course_catalog_large.jsonl@1b42b903e7 (+3) |
| L09_course_rec | SkillLLM_v2 Full | precision@5 | 0.6083 | [0.5333, 0.6833] | 24 | scenario | test | inputs/courses/course_catalog_large.jsonl@1b42b903e7 (+3) |
| L09_course_rec | SkillLLM_v2 Full | ndcg@10 | 0.8854 | [0.8514, 0.9160] | 24 | scenario | test | inputs/courses/course_catalog_large.jsonl@1b42b903e7 (+3) |
| L09_course_rec | SkillLLM_v2 Full | wsgc@5 | 0.8382 | [0.7735, 0.8994] | 24 | scenario | test | inputs/courses/course_catalog_large.jsonl@1b42b903e7 (+3) |
| L09_course_rec | SkillLLM_v2 Full | prereq_violation_rate | 0.0000 | [0.0000, 0.0000] | 24 | scenario | test | inputs/courses/course_catalog_large.jsonl@1b42b903e7 (+3) |
| L12_systems | anchor | latency_p95 | 16.6614 | [15.1548, 18.3863] | 974 | document | test | artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/predictions.jsonl@b4c75aa87b (+1) |
| L12_systems | anchor | peak_vram_gb | 18.4843 | - | 974 | document | test | artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/predictions.jsonl@b4c75aa87b (+1) |

## Pre-registered hypotheses H1–H7

Rules: `docs/thesis/evaluation/hypotheses_preregistration.md`. Verdicts are emitted by the owning layer.

| hypothesis | owner_layer | verdict | delta | ci | p_raw | p_holm | status | note |
|---|---|---|---|---|---|---|---|---|
| H1 | L01_extraction | NOT SUPPORTED | 0.0370 | [0.0165, 0.0573] | 0.0004 | 0.0004 | measured | H1_anchor_vs_baselines |
| H2 | L05_wrapper | NOT SUPPORTED | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 | measured | H2 Single training run of the anchor and one test split: CIs reflect document sampling only.; SOFT_SKILL = SkillSpan 'skill' component (tags_skill), HARD_SKILL = 'knowledge' (tags_knowledge).; 974-doc subset with >=1 gold s |
| H3 | L02_generalization | SUPPORTED | 0.2063 | [0.1704, 0.2415] | 0.0002 | 0.0002 | measured | H3_seen_vs_unseen (single test; Holm = raw) Single SkillSpan test split and one training run per system: CIs reflect sentence sampling only, not seed/training variance.; Primary population = the 974 test sentences that contain gold spans; false positives on the 2, |
| H4 | L04_calibration | SUPPORTED | 0.0587 | [0.0477, 0.0682] | 0.0002 | 0.0006 | measured | H4 Primary effect = selective precision@80% minus precision@100% under the dev-selected score. |
| H5 | L08_graph_retrieval | SUPPORTED | 0.0855 | [0.0478, 0.1204] | 0.0002 | 0.0006 | measured | H5 All 24 test scenarios have ≥4 system-visible gaps: the 1-gap stratum is empty on the original benchmark.; Graph hop 1 equals flat skill overlap here because the KG's TEACHES edges are a 1:1 projection of the catalogue fi |
| H6 | L09_course_rec | SUPPORTED | -0.1833 | [-0.2583, -0.1167] | 0.0004 | 0.0015 | measured | H6 The violation criterion is satisfied by construction when the strict mode enforces the same rule the metric checks; nDCG non-inferiority is the informative half. |
| H7 | L07_compaction | NOT SUPPORTED | 0.1615 | [0.1004, 0.2241] | 0.0005 | 0.0010 | measured | H7[informative][compact@0.5] |
| H7 | L07_compaction | NOT SUPPORTED | 0.0000 | [0.0000, 0.0000] | 0.0005 | 0.0010 | measured | H7[informative][compact@shipped] |
| H7 | L07_compaction | NOT SUPPORTED | 0.1615 | [0.1004, 0.2241] | 0.0005 | 0.0010 | measured | H7[uninformative][compact@0.5] |
| H7 | L07_compaction | NOT SUPPORTED | 0.0000 | [0.0000, 0.0000] | 0.0005 | 0.0010 | measured | H7[uninformative][compact@shipped] |

## Integrity flags raised in Progress.md

- 2026-11-17T20:56Z | A2 | integrity-flag | minor: src/skillllm_v2/evals/extraction_evals.py::paired_boot p-value = 2*min(P(d<=0), 1-P(d<=0)) returns p=0 when all bootstrap deltas are exactly 0 (identical systems) and under-counts ties at 0; not edited (read-only). stats.paired_bootstrap counts ties on both sides with +1 smoothing. Affects only near-degenerate comparisons (e.g. wrapper variants differing on <20 docs) — L10 recomputes them.
- 2026-11-17T21:03Z | A3 | integrity-flag | The stored "BERT baseline" (artifacts/skillllm_v2/bert_baseline, exact F1 0.059) is NOT a neural BERT: src/skillllm_v2/baselines/bert_baseline.py memorises the most frequent BIO tag per lower-cased training token and tags by lookup (model.json training_mode "token_classification_memorized_baseline"; configs name bert-base-uncased but it is never loaded). It must be labelled "token-lookup baseline" in any table; H1 against it is a straw man. GLiNER baseline is zero-shot (urchade/gliner_medium-v2.1, not fine-tuned on SkillSpan; metrics.json training_mode "zero_shot").
- 2026-11-17T21:03Z | A3 | integrity-flag | Offset code drift: anchor predictions (artifacts/skillllm_v2/mistral_lora/predictions.jsonl, Jul 13) were aligned with the OLD schema.find_offsets (first plain substring match), the current code (commit 1f8bd68) uses word-boundary matching first. E.g. test-12 "javascript reactjs java": stored span "java" at 0-4 (inside "javascript"), current code gives 19-23. Re-running the anchor today therefore cannot reproduce stored offsets byte-for-byte even with identical model output; J1 reproduction check reports model-output identity and offset-code effects separately.
- 2026-11-17T21:11Z | A5 | integrity-flag | src/skillllm_v2/evals/contract.py::paired_bootstrap has the same p=0-for-identical-systems defect A2 flagged in extraction_evals.paired_boot (p = 2*min(below, 1-below) with ties counted only in 'below'); A5 uses its own +1-smoothed bootstrap p plus a sign-flip permutation p. Not edited.
- 2026-11-17T21:27Z | A4 | integrity-flag | The money-table "Linking Accuracy" (0.890374 Full CAG/Graph, 0.890271 Mistral+GLiNER+Taxonomy, 0.889834/0.889732 safe wrapper) is NOT an accuracy: pipeline/smoke_runner.py sets linking_accuracy = linked_count/total over PREDICTED spans (share with any non-null canonical id); no gold link is compared (none existed). It is also partly circular: the 711-row alias table it links into = 601 SkillSpan train/dev gold surfaces + 76 aliases mined from anchor predictions on the SAME test split + 34 manual, and the BM25 fallback links almost any string sharing a token. linker_{train,dev}.jsonl are 2 toy rows; the taxonomy reranker is a placeholder; linking_eval.csv's 0.0 = "reranker labels unavailable". Must be relabelled "linking coverage (diagnostic)" everywhere; real accuracy against public ESCO gold is in L06_linking.
- 2026-11-17T21:16Z | A1 | integrity-flag | money_table compares different populations: stored BERT/GLiNER metrics (0.059150 / 0.075833) are over all 3,569 test sentences, the anchor 0.670069 over the 974 with gold spans. Same 974 sentences: lookup baseline 0.1376, GLiNER 0.0814. Anchor's own full-test F1 is unmeasured until A3's anchor_test_nospan (pending_gpu)
- 2026-11-17T21:16Z | A1 | integrity-flag | claims vs artifacts (L01_extraction/claims_reconciliation.csv): 0.675034 (20 occurrences; run-1 anchor exact F1, used as wrapper-gate target) and 0.670222 / 0.672602 (significance_f1.md) match no stored predictions; reports/skillllm_v2/wrapper_error_analysis.md (12 anchor/wrapper values, old run) and extraction_ablation.md (15 values, stub ablation on 20 dev docs) are not reproducible from artifacts. 0.670069 / 0.669478 / 0.671854 / 0.672444 / 0.671102 reproduce exactly
- 2026-11-17T21:16Z | A1 | integrity-flag | data/skillllm_v2/taxonomy/skill_aliases.jsonl contains 76 rows (source mistral_high_confidence) mined from the anchor's TEST predictions (normalize/skill_alias_builder.py) = transductive test use; 0 of the stored wrapper outputs' dictionary-sourced spans trace to those rows (121 in safe_wrapper: 81 manual, 40 train/dev gold), so no measured effect; excluded from L02 S3. scripts/build_offline_aliases.py itself uses no test data
- 2026-11-17T21:18Z | A2 | integrity-flag | wrapper tuning provenance (confirms docs/INFERENCE_AND_EVALS.md I10/I12): candidate_merger/tuned_config.yaml values are outside the tuner grid, boundary YAML == DEFAULT_CONFIG, no dev predictions/tuning CSVs on disk, calibrator untuned -> NO wrapper component is verifiably dev-tuned; the non-regression gate that chose the shipped variant ran on the TEST subset (test-set selection); anchor-protection/realign designs followed a TEST-split error analysis; 76 alias rows come from TEST predictions (measured effect on spans: 0). Wrapper gains are therefore not clean held-out improvements (they are also not significant).
- 2026-11-18T07:15Z | B3 | integrity-flag | L06 circularity audit (label-free diagnostics): legacy 0.890374 reproduced exactly from full_cag_graph/linked_skills.jsonl = coverage. Coverage on predicted spans that match a gold span 0.895 vs 0.879 on non-gold spans (no discrimination); 52% of links come from the BM25 fallback; 76% of linked spans carry taxonomy tag SKILLSPAN_GOLD; shipped linker links 66% of random non-gold n-grams; 67/76 'mistral_high_confidence' alias rows (mined from anchor test predictions) are test gold surfaces (152 test gold mentions match them). Verdict: not an accuracy, circular (dictionary-derived score), test-contaminated.
- 2026-11-18T07:16Z | B4 | integrity-flag | L09: existing Explanation Completeness@5 = 1.000 for Full, for random courses with placeholder fields and for Full packets attached to the WRONG courses -> tautological (field presence only). Claim-level faithfulness of Full 0.925 lenient (145 unsupported of 1922 claims, ALL are 'quoted learning outcome does not mention the skill': 119/264 supported), strict-field covers_gap_skill 230/300; wrong-course probe 0.650.
- 2026-11-18T07:16Z | B4 | integrity-flag | L09 label independence: label builder imports no system code, but labels' need/coverage and prerequisite-feasibility rules are the same catalogue-field logic skill-aware systems and the strict prereq mode implement: an oracle applying the v1 rule to system-VISIBLE inputs scores nDCG@10 0.958 [0.929,0.979] vs gold (kappa v1 vs v_visible 0.898); text-only labels kappa 0.586. Violation-rate zeros for Full/filter systems are by construction. System ordering Kendall tau v1 vs content_only 0.462, vs v_text 0.487.
