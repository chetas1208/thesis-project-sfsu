# L02 - Generalisation to unseen skill mentions (SkillSpan test vs train)

Owner A1. Module `src/skillllm_v2/evals/research/generalization.py`; rules D-A1-06/07 in `Decisions.md`. 974 test sentences, 2265 gold mentions; CIs = 95% sentence-level cluster bootstrap (seed 42; 10,000 for the anchor, 2000 otherwise); `ci_posting` in results.json = 65-posting cluster bootstrap.

## Buckets (test gold mentions vs train gold spans)

| bucket | mentions | share | HARD | SOFT | mean words |
|---|---|---|---|---|---|
| S1 exact_seen | 780 | 0.344 | 602 | 178 | 1.57 |
| S2 normalized_seen | 114 | 0.050 | 78 | 36 | 1.66 |
| S3 canonical_seen_surface_unseen | 17 | 0.008 | 13 | 4 | 1.47 |
| S4 fully_unseen | 1354 | 0.598 | 481 | 873 | 3.60 |

S3 mapping sources: {'esco': 7, 'alias': 10}. ESCO: used (13939 concepts, data/external/esco/v1.2.1/skills_en.jsonl).
Train-frequency strata of test mentions (normalised surface): 0: 1371, 1: 205, 2-5: 297, >5: 392.

### Train/test and train/dev overlap

| pair | mentions | mention share seen (exact) | mention share seen (normalised) | type share seen (norm.) | type Jaccard (norm.) | type Jaccard (exact) |
|---|---|---|---|---|---|---|
| train-test (974 subset) | 2265 | 0.344 | 0.395 | 0.210 | 0.068 | 0.061 |
| train-dev | 2163 | 0.288 | 0.329 | 0.191 | 0.065 | 0.059 |

Label-aware exact-seen share (test): 0.336; normalised + same label: 0.384.

### Leakage audit of the taxonomy

- `data/skillllm_v2/taxonomy/skill_aliases.jsonl` rows by source: {'manual_curated': 34, 'skillspan_gold_train_dev': 601, 'mistral_high_confidence': 76}. The 76 `mistral_high_confidence` rows are mined from the anchor's TEST predictions (`src/skillllm_v2/normalize/skill_alias_builder.py`), i.e. transductive use of test inputs; they are excluded from S3. `scripts/build_offline_aliases.py` builds a different file (`src/skillllm_v2/resources/offline_aliases.jsonl`) from hand-curated lists only (no SkillSpan or prediction data).
- 601 rows come from SkillSpan train+dev gold spans (count >= 2): using them for S3 is not test leakage.

## Exact F1 by bucket (gold recall x bucketed precision)

| system | S1 F1 [CI] | S2 F1 [CI] | S3 F1 [CI] | S4 unseen F1 [CI] | seen (S1+S2) F1 | Δ seen-unseen [CI] |
|---|---|---|---|---|---|---|
| mistral_lora | 0.8034 [0.7735, 0.8313] | 0.7248 [0.6481, 0.7923] | 0.8333 [0.6667, 0.9474] | 0.5874 [0.5589, 0.6157] | 0.7937 | 0.2063 [0.1704, 0.2415] |
| bert_baseline | 0.4088 [0.3733, 0.4420] | 0.1538 [0.1142, 0.1954] | 0.0278 [0.0000, 0.0889] | 0.0394 [0.0322, 0.0468] | 0.3528 | 0.3134 [0.2810, 0.3455] |
| gliner_baseline | 0.0878 [0.0627, 0.1148] | 0.0328 [0.0000, 0.0813] | 0.0952 [0.0000, 0.3000] | 0.0815 [0.0613, 0.1013] | 0.0810 | -0.0005 [-0.0299, 0.0304] |
| mistral_boundary | 0.8034 [0.7744, 0.8313] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5874 [0.5594, 0.6148] | 0.7937 | 0.2063 [0.1708, 0.2416] |
| mistral_gliner_taxonomy | 0.8003 [0.7717, 0.8283] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5885 [0.5602, 0.6163] | 0.7910 | 0.2024 [0.1679, 0.2378] |
| full_cag_graph | 0.8008 [0.7725, 0.8286] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5898 [0.5618, 0.6176] | 0.7914 | 0.2016 [0.1666, 0.2372] |
| safe_wrapper | 0.8047 [0.7756, 0.8330] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5893 [0.5615, 0.6170] | 0.7948 | 0.2055 [0.1709, 0.2405] |
| safe_wrapper_boundary_refiner | 0.8078 [0.7789, 0.8361] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5887 [0.5611, 0.6164] | 0.7975 | 0.2088 [0.1729, 0.2438] |
| safe_wrapper_mistral_gliner_taxonomy | 0.8003 [0.7717, 0.8281] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5880 [0.5601, 0.6158] | 0.7909 | 0.2029 [0.1675, 0.2382] |
| safe_wrapper_full_cag_graph | 0.8047 [0.7756, 0.8330] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5893 [0.5615, 0.6170] | 0.7948 | 0.2055 [0.1709, 0.2405] |
| qlora_loop_v1 | 0.7745 [0.7431, 0.8052] | 0.6762 [0.5961, 0.7500] | 0.6486 [0.4444, 0.8085] | 0.5898 [0.5614, 0.6160] | 0.7621 | 0.1724 [0.1375, 0.2099] |
| qlora_loop_v2 | 0.7732 [0.7423, 0.8052] | 0.7018 [0.6278, 0.7711] | 0.7222 [0.5405, 0.8718] | 0.5957 [0.5671, 0.6237] | 0.7637 | 0.1680 [0.1317, 0.2049] |
| qlora_loop_v3 | 0.7705 [0.7381, 0.8020] | 0.6825 [0.6031, 0.7586] | 0.7097 [0.5000, 0.8750] | 0.5703 [0.5407, 0.5989] | 0.7593 | 0.1890 [0.1540, 0.2267] |
| qlora_loop_v4 | 0.8003 [0.7699, 0.8295] | 0.6796 [0.6028, 0.7489] | 0.7647 [0.5882, 0.9000] | 0.6178 [0.5888, 0.6465] | 0.7858 | 0.1680 [0.1349, 0.2035] |
| gpu_anchor_test_rerun | 0.8072 [0.7781, 0.8353] | 0.7248 [0.6477, 0.7935] | 0.8333 [0.6856, 0.9500] | 0.5874 [0.5594, 0.6148] | 0.7971 | 0.2097 [0.1745, 0.2446] |
| gpu_bert_fair | 0.7815 [0.7543, 0.8099] | 0.6667 [0.5882, 0.7436] | 0.6061 [0.3702, 0.7919] | 0.5442 [0.5178, 0.5722] | 0.7675 | 0.2233 [0.1886, 0.2580] |
| gpu_bert_fair_run_bert-base-cased_s42 | 0.7832 [0.7557, 0.8117] | 0.6637 [0.5882, 0.7349] | 0.5806 [0.3200, 0.7778] | 0.5199 [0.4893, 0.5486] | 0.7683 | 0.2483 [0.2125, 0.2845] |
| gpu_bert_fair_run_bert-base-cased_s43 | 0.7689 [0.7389, 0.7987] | 0.6233 [0.5357, 0.7045] | 0.6452 [0.4167, 0.8276] | 0.5227 [0.4927, 0.5528] | 0.7513 | 0.2285 [0.1910, 0.2675] |
| gpu_bert_fair_run_bert-base-cased_s44 | 0.7705 [0.7420, 0.7997] | 0.6941 [0.6132, 0.7650] | 0.6250 [0.4000, 0.8148] | 0.5112 [0.4813, 0.5385] | 0.7610 | 0.2499 [0.2157, 0.2857] |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s42 | 0.7815 [0.7543, 0.8099] | 0.6667 [0.5882, 0.7436] | 0.6061 [0.3702, 0.7919] | 0.5442 [0.5178, 0.5722] | 0.7675 | 0.2233 [0.1886, 0.2580] |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s43 | 0.7855 [0.7579, 0.8134] | 0.6996 [0.6244, 0.7675] | 0.6061 [0.3846, 0.7895] | 0.5351 [0.5071, 0.5630] | 0.7750 | 0.2398 [0.2052, 0.2749] |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s44 | 0.7872 [0.7601, 0.8143] | 0.7289 [0.6533, 0.7968] | 0.7059 [0.5000, 0.8696] | 0.5118 [0.4832, 0.5403] | 0.7801 | 0.2683 [0.2317, 0.3034] |

## Gold-side exact recall by bucket (primary)

| system | S1 | S2 | S3 | S4 | train freq 1 | 2-5 | >5 |
|---|---|---|---|---|---|---|---|
| mistral_lora | 0.7962 | 0.6930 | 0.8824 | 0.5871 | 0.7024 | 0.7744 | 0.8316 |
| bert_baseline | 0.5808 | 0.4211 | 0.0588 | 0.0894 | 0.4439 | 0.5084 | 0.6607 |
| gliner_baseline | 0.0487 | 0.0175 | 0.0588 | 0.0487 | 0.0537 | 0.0572 | 0.0306 |
| mistral_boundary | 0.7962 | 0.6930 | 0.8824 | 0.5871 | 0.7024 | 0.7744 | 0.8316 |
| mistral_gliner_taxonomy | 0.7987 | 0.6930 | 0.8824 | 0.5879 | 0.6976 | 0.7744 | 0.8393 |
| full_cag_graph | 0.7987 | 0.6930 | 0.8824 | 0.5879 | 0.6976 | 0.7744 | 0.8393 |
| safe_wrapper | 0.7974 | 0.6930 | 0.8824 | 0.5871 | 0.7024 | 0.7744 | 0.8342 |
| safe_wrapper_boundary_refiner | 0.8000 | 0.6930 | 0.8824 | 0.5871 | 0.7073 | 0.7744 | 0.8367 |
| safe_wrapper_mistral_gliner_taxonomy | 0.7936 | 0.6930 | 0.8824 | 0.5871 | 0.6976 | 0.7744 | 0.8291 |
| safe_wrapper_full_cag_graph | 0.7974 | 0.6930 | 0.8824 | 0.5871 | 0.7024 | 0.7744 | 0.8342 |
| qlora_loop_v1 | 0.7244 | 0.6228 | 0.7059 | 0.5739 | 0.6488 | 0.6633 | 0.7806 |
| qlora_loop_v2 | 0.7385 | 0.7018 | 0.7647 | 0.5953 | 0.6732 | 0.7104 | 0.7832 |
| qlora_loop_v3 | 0.7167 | 0.6316 | 0.6471 | 0.5554 | 0.6341 | 0.6532 | 0.7832 |
| qlora_loop_v4 | 0.7756 | 0.6140 | 0.7647 | 0.6034 | 0.6488 | 0.7508 | 0.8138 |
| gpu_anchor_test_rerun | 0.8000 | 0.6930 | 0.8824 | 0.5871 | 0.7073 | 0.7744 | 0.8367 |
| gpu_bert_fair | 0.8000 | 0.6491 | 0.5882 | 0.5502 | 0.6780 | 0.7677 | 0.8444 |
| gpu_bert_fair_run_bert-base-cased_s42 | 0.7872 | 0.6491 | 0.5294 | 0.5103 | 0.6585 | 0.7576 | 0.8367 |
| gpu_bert_fair_run_bert-base-cased_s43 | 0.7679 | 0.5877 | 0.5882 | 0.5096 | 0.6098 | 0.7273 | 0.8291 |
| gpu_bert_fair_run_bert-base-cased_s44 | 0.7641 | 0.6667 | 0.5882 | 0.4978 | 0.6488 | 0.7542 | 0.8036 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s42 | 0.8000 | 0.6491 | 0.5882 | 0.5502 | 0.6780 | 0.7677 | 0.8444 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s43 | 0.8051 | 0.6842 | 0.5882 | 0.5369 | 0.6634 | 0.7879 | 0.8571 |
| gpu_bert_fair_run_jjzha__jobbert-base-cased_s44 | 0.8179 | 0.7193 | 0.7059 | 0.5222 | 0.6927 | 0.7879 | 0.8776 |

## H3 - seen vs unseen (anchor, pre-registered)

Rule: H3 SUPPORTED iff, for the anchor, exact F1 on seen mentions (S1 exact_seen + S2 normalized_seen) minus exact F1 on fully unseen mentions (S4) is >= 0.05 AND its 95% paired sentence-bootstrap CI (10,000) excludes 0. S3 is reported separately.

- seen (S1+S2) exact F1 0.7937; unseen (S4) exact F1 0.5874; S3 0.8333.
- Δ = 0.2063, 95% CI [0.1704, 0.2415], p = 0.0002; posting-level CI [0.1713, 0.2413].
- Within-label check: HARD_SKILL: seen 0.8347 vs unseen 0.5816; SOFT_SKILL: seen 0.6551 vs unseen 0.5910.
- Length check (gold-side recall, same word length): 1 word(s): seen 0.775 (n=581) vs unseen 0.648 (n=244), Δ 0.127 [0.047, 0.206]; 2 word(s): seen 0.796 (n=226) vs unseen 0.588 (n=325), Δ 0.209 [0.127, 0.288]; 3+ word(s): seen 0.805 (n=87) vs unseen 0.568 (n=785), Δ 0.236 [0.149, 0.327].

**H3 verdict: SUPPORTED.**

## Files

- `results.json`, `per_unit_scores.jsonl` (sentence x system x bucket counts), `fig_f1_by_bucket.png`, `fig_f1_by_bucket.csv`, `fig_recall_by_train_frequency.png`, `fig_recall_by_train_frequency.csv`

## Caveats

- Buckets are lexical: an unseen surface may still be a paraphrase of a seen skill; a seen surface may be annotated differently in context. S3 coverage depends on the taxonomy and is small without ESCO.
- Unseen mentions are longer and more often SOFT_SKILL, so part of the gap may reflect span length / label difficulty rather than memorisation (see the within-label check).
- Predictions are bucketed by their own surface; a false positive with an unseen surface lowers unseen precision.
- Single SkillSpan test split and one training run per system: CIs reflect sentence sampling only, not seed/training variance.
- Primary population = the 974 test sentences that contain gold spans; false positives on the 2,595 skill-free test sentences are only counted in the explicitly labelled full-test records.
- Resampling unit = sentence (repo 'document'); sentences come from 65 job postings, so posting-level CIs (details.ci_posting) are wider.
