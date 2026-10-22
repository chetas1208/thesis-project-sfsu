# Evaluation framework: metrics, protocol, results, defects found, and open threats

Draft for the thesis committee. Status as of 2026-10-22 (about 09:00 UTC). Every number was read from a file in this repository (cited as `reports/...`, `artifacts/...`, `docs/...`, `logs/...`) or is marked **derived** (arithmetic shown) or **pending** (experiment or re-run not finished). I ran no model, no training and no test; the evaluation suite results are those in `reports/skillllm_v2/eval_suite/REPORT.md` (a CPU run of 287.4 s at 08:35 to 08:40 UTC, seed 42, 2,000 bootstrap resamples, git commit `f196af97f395f51bf023b641c81301a0db9a9529` with a dirty tree).

Companion drafts: `training.md`, `pretraining.md`, `inference.md`; plan and claims: `docs/BEST_PLAN.md`. Master audit: `docs/INFERENCE_AND_EVALS.md`; suite description: `docs/EVAL_SUITE.md`; tool survey: `docs/EVAL_LANDSCAPE.md`; defect logs: `docs/BUGFIX_LOG.md`, `docs/BUGFIX_LOG_MATCH.md`.

## 0. What the committee should know first

1. Only **extraction** (the 24B LoRA adapters, their wrapper variants and the ensemble) is measured on real, public, human-annotated data (SkillSpan, 974 gold-bearing test documents). Everything downstream (linking, verification, calibration, context packets, graph, matching, gaps, explanations, recommendation, end-to-end) is evaluated on **synthetic, author-written data** whose ground truth was authored by the same people who wrote the system. Those evaluations find real defects and verify properties; their absolute scores are not evidence of real-world quality (section 7).
2. Human review is **pending**: `reports/skillllm_v2/human_review_sample.csv` has 60 rows and 0 of 60 `human_grade_0to3` cells filled (counted directly; the `auto_grade_0to3` column is machine-generated).
3. The suite found real bugs; most are fixed by two fix passes (`docs/BUGFIX_LOG.md`, `docs/BUGFIX_LOG_MATCH.md`), but **`reports/skillllm_v2/eval_suite/REPORT.md` has not been regenerated since** and is a mixed-state snapshot (section 5.1). Fix-time numbers below are the fix agents' own re-measurements; an independent full re-run is **pending**.
4. Twelve inconsistencies were listed by the master audit; none of the twelve is fully closed (section 4), and I found eight more (Table E6).

---

## 1. Metric definitions

Code: `src/skillllm_v2/eval/extraction_eval.py::evaluate_extraction`; statistics `eval/significance.py`; ranking `eval/ranking_metrics.py`; module evaluations `src/skillllm_v2/evals/*_evals.py`.

**Table E1. Metrics and their exact meaning.**

| Metric | Definition (as implemented) | Differs from the standard definition how |
|---|---|---|
| **Exact-span F1** (strict) | A predicted span `(start, end, label)` is correct iff it equals a gold `(start, end, label)`. Pooled (micro) counts over documents; predictions de-duplicated per document. Offsets are character offsets in the space-joined token string. | Matches nervaluate `strict` and conlleval-style chunk F1 (landscape doc: agreement checked on synthetic data in a throwaway environment, 0.6502/0.5778/0.6119 against nervaluate's 0.6503/0.58/0.61; **[tested]** per `docs/EVAL_LANDSCAPE.md` section 3). |
| **Partial-span F1** | A prediction counts if it overlaps a not-yet-matched gold span with the same label (one-to-one, greedy). | Not equal to nervaluate `partial` (which ignores type and gives half credit); landscape doc: ours 0.7433 against 0.84 on its synthetic test, "cannot be compared directly". |
| **"Token F1"** | Despite the name, **character-level**: `(character index, label)` pairs; label must match. The suite adds `word_token_f1` (anchor 0.7809 [0.7621, 0.7983]). | Not word-token F1 and not the SkillSpan official metric. |
| **HARD / SOFT F1** | Exact-span F1 restricted to one label; `macro_f1` is their mean; `micro_f1` equals exact F1. | SkillSpan "knowledge" is mapped to HARD and "skill" to SOFT by an assumption in the code. |
| nervaluate-style modes (suite, in-house) | strict, exact (boundary only, type ignored), partial, ent_type. Anchor: strict 0.6701, exact 0.6980 [0.6759, 0.7193], partial 0.7812 [0.7643, 0.7974], ent_type 0.8143 [0.7966, 0.8315]. | The suite cross-check against the **nervaluate library** was **skipped** (not importable in the default environment); only the earlier throwaway-venv check above exists. |
| nDCG@k, P@k, R@k | exponential gain `2^rel - 1`; relevant = grade at least 2; scenarios with no relevant item are **dropped**. | trec_eval-style code scores them 0 and keeps them (landscape doc: nDCG@10 mean 0.2799 ours against 0.2659 trec-style on its synthetic test). |
| **MAP@k** | AP@k divides by `min(R, k)` (repo convention). | trec_eval and ranx divide by `R`; per-scenario difference up to 0.249 in the landscape test; after rescaling the difference is 5.6e-17. The suite reports both (`match.map_at_10` 0.9848 and `match.map_at_10_trec_convention` 0.5452 [0.4521, 0.6362]). **Quote the convention with every MAP.** |
| ECE / Brier / AUROC / AURC | 10 (and mass-based) equal-width bins on binary correctness; sklearn has no ECE function (landscape doc). | Binning choices change ECE; both widths reported. |
| Paired bootstrap, sign-flip permutation, Holm | Unit of resampling = document (or scenario / pair / case), 10,000 resamples and seed 20260710 in the significance reports, 2,000 resamples seed 42 in the suite. | Verified against `scipy.stats.bootstrap(paired=True)` (CI endpoints within 7.4e-4 in the suite cross-check) and `statsmodels` Holm (landscape doc). |

Caveat against SkillSpan itself. The SkillSpan paper reports span-level precision/recall/F1 averaged over five seeds with an almost-stochastic-dominance test (landscape doc, read from the paper PDF, section 4(a)); its public repository has no scorer (`metrics/` is empty). The repository's numbers are therefore **not** directly comparable with published SkillSpan numbers: different split handling (here the 974 gold-bearing test rows), different tokenisation into character offsets, single seed, different tag mapping. Do not place them in one table with published values without saying so.

---

## 2. Protocol

- **Extraction population.** SkillSpan test rows with at least one gold span: 974 of 3,569 rows (2,265 gold spans). The BERT-lookup and GLiNER baselines were scored on **all 3,569 rows**, so their precision is deflated and they are not comparable with the Mistral rows (inconsistency I5).
- **Systems.** One trained model per row (one seed). The anchor in all current tables is run 2 (0.670069), see `training.md` Table T5 for run 1 (0.675034).
- **Statistics.** Paired document bootstrap plus sign-flip permutation with Holm correction; a delta is called significant only if both Holm-adjusted p-values are below 0.05 (`reports/skillllm_v2/significance_f1.md`). Minimum detectable effect (MDE, two-sided alpha 0.05, 80% power) is reported with each comparison.
- **Controls.** The suite includes experiments with known answers (oracle ranking gives nDCG 1; shuffled predictions give about 0 F1; random confidence gives the analytic ECE; and so on) so a broken metric or harness fails the run (section 3.2).
- **Regression gate** inside the suite: a gated metric regresses if it is worse than a stored baseline by more than 0.03 absolute (tolerance ignores confidence intervals; it guards code regressions, not significance).

---

## 3. The module evaluation suite

### 3.1 Scope and results at a glance

Command: `scripts/run_eval_suite.sh` (about 5 minutes, CPU, offline). Outputs `reports/skillllm_v2/eval_suite/{REPORT.md,results.json,baseline.json}`. Verdict in the file: **PASS**.

**Table E2. Suite summary (from `reports/skillllm_v2/eval_suite/REPORT.md`).**

| Module | Results | Controls (pass/fail) | Skipped | Data |
|---|---:|---:|---:|---|
| extraction | 107 | 2 / 0 | 0 | SkillSpan test, 974 rows, cached predictions |
| verifier | 173 | 0 / 0 | 8 | synthetic negatives plus real merged candidates |
| calibration | 113 | 12 / 0 | 4 | real predicted spans (n = 2,242 for the anchor) |
| linking | 50 | 1 / 0 | 4 | authored probe set (127 items) plus the 974-row test predictions |
| context | 38 | 2 / 0 | 3 | synthetic profiles (40) |
| graph | 42 | 1 / 0 | 2 | synthetic graph cases (12 worlds) |
| match | 42 | 7 / 0 | 0 | synthetic: 112 candidates against 42 jobs |
| gap | 55 | 6 / 0 | 0 | synthetic: 240 (user, job) pairs |
| explanation | 44 | 2 / 0 | 2 | synthetic worlds (8) |
| e2e | 47 | 4 / 0 | 0 | synthetic: 12 cases, stub extractor |
| **Total** | **711** | **37 / 0** | 23 listed | |

Coverage of pipeline stages (suite coverage table): 12 of 13 stages covered directly (extraction, candidate merging, boundary refinement, span verification, confidence calibration, taxonomy linking, context building, graph, job matching, gap detection, explanation, end to end); course retrieval/ranking is "partial" (indirect, through e2e); the course-ranking benchmark proper (`eval/recommender_benchmark.py`, section 6) lives outside the suite.

### 3.2 Integrity: controls and oracle cross-checks

All **37 controls pass** (0 failures, none without a verdict). Examples (measured value): shuffled-prediction F1 0.0010 (anchor) and 0.0009 (safe wrapper); random-confidence ECE equals its analytic expectation within 0.05 (0.2796 against 0.279); constant-confidence ECE 0.0000; random taxonomy ids score top-1 accuracy 0.0013; oracle ranking nDCG@10 = 1.0, anti-oracle 0.0, label-shuffled 0.1455, random 0.1900; a clean e2e output validates while all 7 corruptions are rejected; metric reproduction against stored files `max_abs_diff 0.0` for all four extraction systems on 974 rows.

Oracle cross-check: **5 agree, 0 disagree, 6 skipped** because libraries are not installed (`ir_measures`/`pytrec_eval`: four ranking checks; `statsmodels`: Holm; `nervaluate`: span modes). The five that ran agree: paired-bootstrap delta (difference 0.0 against `scipy.stats.bootstrap`), its CI endpoints (7.4e-4, tolerance 6.6e-3), bootstrap CI of a mean (4.3e-3, tolerance 2.7e-2), Kendall tau-b and Spearman (both 0.0 against `scipy`). The ranking-metric agreement with `trec_eval` (1.1e-16 for nDCG, 0 for P/R/AP when R is at most k) was established once in a throwaway environment (`docs/EVAL_LANDSCAPE.md`), not by the committed suite run. Controls prove the harness is not broken; they do not prove the *system* is good.

---

## 4. Master table of evaluations with confidence intervals

All extraction rows: SkillSpan test, 974 gold-bearing documents, exact-span matching unless stated; intervals from the suite are 95% percentile bootstrap over documents (2,000 resamples). "Real" = real human-annotated data; "synthetic" = authored by the project.

**Table E3. Master table (every evaluation family; the 711 individual results are in `reports/skillllm_v2/eval_suite/results.json`).**

| ID | Evaluation | Metric: value [95% CI] | n | Data | Source | Status and caveat |
|---|---|---|---:|---|---|---|
| M1 | Anchor (80-epoch LoRA, run 2) | exact F1 0.6701 [0.6464, 0.6944]; P 0.6735 [0.6477, 0.6977]; R 0.6667 [0.6397, 0.6922]; partial F1 0.8138 [0.7968, 0.8304]; char-token F1 0.7879 [0.7698, 0.8051]; HARD 0.7265 [0.6952, 0.7566]; SOFT 0.6040 [0.5707, 0.6342] | 974 | real | `reports/skillllm_v2/eval_suite/REPORT.md`, `reports/skillllm_v2/raw_mistral_lora_metrics.json` | one seed; run 1 scored 0.6750 (log only) |
| M2 | Anchor slices | exact F1: tech 0.7251 [0.6927, 0.7561], house 0.6037 [0.5683, 0.6368]; sentences of 1 to 5 words 0.8834, 13 to 24 words 0.6380, 25+ words 0.6073; gold spans of 1 word 0.7386, 4+ words 0.5811; documents with 1 gold skill 0.5562, 5+ gold skills 0.7319 | 974 | real | same | descriptive; shorter spans and fewer-skill rows are easier or harder in a way a single aggregate hides |
| M3 | Anchor error census | missed 317; spurious 294; boundary shifts 325; label confusions 113; duplicates 0 | 974 | real | same | boundary shifts are the largest class |
| M4 | Safe wrapper variants | exact F1 0.672444 (boundary_refiner), 0.669478 (gliner_taxonomy), 0.671854 (full chain) | 974 | real | `reports/skillllm_v2/safe_wrapper_summary.json`, `money_table.md` | cached anchor predictions plus CPU stages; gate "accept" decided on the same 974 documents |
| M5 | Wrapper deltas vs anchor (paired) | boundary_refiner +0.0024 [+0.0009, +0.0042], Holm p 0.003 / 0.029 (significant); gliner_taxonomy -0.0006 [-0.0023, +0.0009], p 1.0; full chain +0.0018 [-0.0006, +0.0042], p 0.94 / 0.99 | 974 (10 / 9 / 19 documents differ) | real | `reports/skillllm_v2/significance_f1.md` | the only significant row is offset cleanup on 10 documents (section 5.2) |
| M6 | Loops vs anchor (paired) | v1 -0.0135 [-0.0342, +0.0071]; v2 -0.0086 [-0.0286, +0.0114]; v2 minus v1 +0.0049 [-0.0133, +0.0232] | 974 | real | `reports/skillllm_v2/significance_qlora_vs_anchor.md`, `qlora_loop_v2_significance_vs_anchor.md`, `qlora_loop_v2_significance_vs_v1.md` | MDE about 0.03; see `training.md` |
| M7 | Majority ensemble vs anchor (paired) | +0.0198 [+0.0041, +0.0356]; raw p 0.0162; Holm-5 0.0810; precision 0.7220, recall 0.6605 | 974 | real | `reports/skillllm_v2/ensemble_eval.md` | suggestive, not confirmed; single seeds; forking path |
| M8 | Vote-count precision | 1 vote 0.2901 (273 of 941); 2 votes 0.5450 (351 of 644); 3 votes 0.8018 (1,145 of 1,428) | 3,013 spans | real | `reports/skillllm_v2/ensemble_vote_confidence.md` | estimated on the test split itself |
| M9 | Baselines | BERT-lookup exact F1 0.0592; GLiNER zero-shot 0.0758 | 3,569 | real | `reports/skillllm_v2/money_table.md` | different population from M1; "BERT" is a token-memorisation table, not BERT (I8) |
| M10 | Calibration of anchor confidence | ECE 0.1865 [0.1627, 0.2104]; Brier 0.2547 [0.2376, 0.2719]; AUROC 0.5 (constant 0.86); base rate 0.6735; wrapper-final ECE 0.1888, AUROC 0.5281 [0.5190, 0.5376]; HARD ECE 0.158, SOFT 0.223 | 2,242 spans (anchor), 2,233 (wrapper) | real | suite, calibration module | text-disjoint ECE 0.1902; confidence carries no information, so only vote counts or log-probabilities can help (log-probability test is **pending**, `training.md` Table T13) |
| M11 | Rule verifier vs synthetic negatives (dev) | precision 0.5130, recall 0.8738, F1 0.6465 [0.6398, 0.6529]; false-drop rate 0.1262, false-keep 0.6845 | 1,096 | synthetic negatives | suite, verifier module | negatives are synthetic (boundary-expanded gold plus generic phrases) |
| M12 | Learned (TF-IDF + logistic) verifier vs synthetic negatives | dev F1 0.7539 [0.7428, 0.7654], AUROC 0.8352; text-disjoint AUROC 0.8330; in-sample train AUROC 0.9085 | 1,096 / 1,082 / 1,840 | synthetic negatives | suite | the model is **not used at inference by default** (`verify.learned.enabled: false`) |
| M13 | Verifiers on real extractor candidates | learned model AUROC 0.5787 [0.5467, 0.6112] (n = 903), false-keep 0.799, F1 0.7534 against a keep-everything F1 of 0.8047; end-to-end exact-F1 change on 974 documents: shipped anchor-protected rule +0.00104 [+0.00030, +0.00191], unprotected rule -0.01156 [-0.01893, -0.00445] (recall -0.0459), learned model at threshold 0.5 on all spans -0.05465 [-0.07099, -0.03899] (recall -0.1073), learned model on non-anchor spans only +0.00059 | 974 | real | suite | on real errors the learned verifier is barely better than chance and would harm the anchor if applied to its spans; the anchor-protection design is what keeps it safe; 4.4% of candidates come from test sentences whose text also occurs in train or dev (suite leakage flag) |
| M14 | Linking: coverage | 0.8898 [0.8736, 0.9061]; HARD 0.8894, SOFT 0.8904; on correct spans 0.8981, on false positives 0.8726; method mix alias 0.4098, BM25 0.4662, fuzzy 0.0139, none 0.1102 | 974 rows (2,233 spans) | real | suite, linking module | **coverage is not accuracy**: no gold canonical ids exist; BM25 returns a link for nearly any string |
| M15 | Linking: authored probe | recall@1 0.9252 [0.8783, 0.9720], MRR 0.9299, link precision 0.9167; by method alias 1.000 (n = 67), fuzzy 0.9524 (n = 21), BM25 **0.60 [0.35, 0.80]** (n = 20); no-match items falsely linked 0.35 [0.15, 0.55] (n = 20; items sharing tokens with a real skill 1.0, n = 6); unseen abbreviations recall@1 0.0 (n = 5) | 107 / 127 | synthetic (authored) | suite, `inputs/evals/linking_probe.jsonl` | small n; authored by the project |
| M16 | Linking: alias held-out | recall@1 0.4667 [0.20, 0.73] | 15 | derived from table | suite | tiny n |
| M17 | Linking: alias-table provenance | 168 of 915 alias-resolved links (18.4%) go through aliases mined from the test-split predictions; 635 come from SkillSpan train/dev gold, 112 manual | 915 links | real | suite (`link.alias_provenance`) | circular by construction (inconsistency I12; table has 711 rows of which 76 are test-mined) |
| M18 | Context packet (post context fixes) | tokens(packet) / tokens(raw) 10.55 [10.21, 10.96]; important-skill retention at budgets 256 / 512 / 1,024 tokens (reference compactor) 0.548 / 0.774 / 0.885; retriever MRR 0.783 [0.763, 0.803] against surface baseline 0.651 and random 0.098; recall@5 0.994; evidence-preservation 1.0; fabrication rate 0.0 | 40 profiles | synthetic | suite | date conflicts not detected (0.0, documented limitation); the packet is **larger** than the raw text it summarises |
| M19 | Graph (before fixes in the snapshot) | structural validity 1.0; edge precision / recall 1.0 (12 worlds); contextual retriever recall@5 0.986 [0.974, 0.996], MRR 0.890; title baseline 0.298, random 0.164; defects: explainer connects user to job 0.0; unknown-node leak 1.0; skill-id collision 1.0 (C, C++, C# share id `c`); ingest idempotence 0.5 | 12 worlds | synthetic | suite | defects fixed for `SkillGraphBuilder` (Table E8) |
| M20 | Explanations (before fixes) | unsupported-claim rate 0.0; graph-mutation audit sensitivity 1.0 (40 mutations) and explanation-mutation sensitivity 1.0 (40); match explanations evidence relevance 0.147 [0.103, 0.194]; raw skill ids in 33.1% of course-explanation skill mentions | 8 worlds | synthetic | suite | structural audit only: semantic faithfulness and human usefulness are **not evaluated** (skipped) |
| M21 | Job matching (before fixes) | nDCG@10 0.9905 [0.9824, 0.9963]; keyword-Jaccard baseline 0.9567 (paired diff +0.0338 [+0.0199, +0.0503]); keyword-count 0.9266; random 0.1900; hit@1 1.0; MAP@10 0.9848 (trec convention 0.5452); lower-casing user skills cost 0.7119 nDCG@10; about half of job scores are exactly 0 (clamp rate 0.4766) and 0.4306 of top-10 lists contain ties | 109 candidates | synthetic | suite | **ceiling effect**: the matcher and the authored grade both reason about required-skill coverage; informative rows are the baselines, robustness and ties |
| M22 | Gap detection (before fixes) | detector micro P 1.0, R 0.8045 [0.7811, 0.8261], F1 0.8917; engine P = R = F1 = 1.0; severity Kendall tau-b 0.8106 [0.7705, 0.8460]; baselines all-job-skills F1 0.7526, missing-only 0.7466; random tau-b 0.016 | 240 pairs | synthetic | suite | engine F1 = 1.0 because the authored states use the engine's own vocabulary; the informative numbers are detector against engine, baselines and tau-b |
| M23 | End to end (stub extractor, before fixes) | profile F1 0.6357, gap F1 0.8665 (pipeline wiring), 0.6259 (demo wiring), 1.0 (oracle wiring); precision@3 0.6296 [0.5185, 0.7407] against oracle 0.8889 and random 0.1728; gap-to-course linkage@5 0.6204; schema validity 1.0; traceability 1.0; determinism 1.0 (in process and across hash seeds); graceful degradation 0.8182 (9 of 11 scenarios) | 12 cases | synthetic | suite | tiny n, stub extractor |
| M24 | End-to-end latency (CPU, stub extractors) | profile extraction 15.1 ms, job extraction 15.9 ms, match 0.017 ms, gap detection 0.008 ms, gap vector 0.042 ms, recommend 0.36 ms; total 31.4 ms per case; pipeline initialisation 1,327 ms; first call 623 ms | 12 | synthetic | suite (units are milliseconds, from `details.median_ms`) | excludes the 24B model: **not** the latency of the real system (see `inference.md`) |
| M25 | Course recommender v1 | legacy (pre-tuning) Full nDCG@10 0.868 [0.846, 0.889], P@5 0.617, R@10 0.813; strongest baseline (Gap-Overlap Greedy + prereq filter) 0.852; difference +0.016 [-0.004, +0.036], Holm p 0.117 (not significant) | 48 scenarios, 171 courses | synthetic | `reports/skillllm_v2/course_recommendation_benchmark.md` | rule-derived labels; no human validation |
| M26 | Course recommender v2 (held-out half) | tuned Full nDCG@10 0.885 [0.851, 0.916], P@5 0.608 [0.533, 0.683], R@10 0.878, prerequisite violation 0.000; strongest baseline 0.851 [0.808, 0.890]; difference +0.035 [-0.006, +0.072], Holm p 0.094 (not significant); tuned minus legacy +0.030 [+0.007, +0.055] (raw p 0.005) | 24 test scenarios | synthetic | `reports/skillllm_v2/course_recommendation_benchmark_v2.md` | split fixed before tuning; content-only-label sensitivity on this half: Full minus the user-aware graph-free Gap-Overlap Greedy -0.069 (Holm p 0.002, Full significantly worse), minus the prerequisite-filtered Greedy +0.023 (p 0.894, indistinguishable) |
| M27 | Smoke and legacy | smoke F1 0.3158 (5 dev rows, stub extractor); extraction ablation A to G 0.057 to 0.203 (20 dev rows, **stub**); original destructive wrapper stages 0.5704 / 0.6515 / 0.5616 exact F1 against the run-1 anchor | 5 / 20 / 974 | mixed | `smoke_results.json`, `extraction_ablation.md`, `wrapper_error_analysis.md` | do not cite smoke or ablation numbers; the wrapper-stage row is historical |

---

## 5. Defects the evaluations found, and what happened to them

### 5.1 State of the evidence (important)

The suite report in the repository (`REPORT.md`, 08:40) was produced between the fix passes: it already contains the **context fixes** (for example `context.label_conflict_detection` 1.0, `context.order_invariance_modulo_timestamps` 1.0, `context.duplicate_evidence_rate` 0.033, `context.packet_over_raw_tokens` 10.55; the log's "before" values are 0.474, 0.0, 0.067 and 18.9) but **not** the graph, explanation, matcher, gap-detector or extractor-degradation fixes (for example `graph.explainer_connects_user_to_job` 0.0, `gap.detector.micro_recall` 0.8045, `e2e.degradation.graceful_share` 0.8182). `reports/skillllm_v2/eval_suite/baseline.json` was written by that same run. `docs/BUGFIX_LOG_MATCH.md` states that the baseline "should be refreshed (`--update-baseline`) once the owner accepts the numbers" and reports two e2e metrics below that baseline (gap-to-course linkage@5 0.6204 to 0.5185 and precision@3 0.6296 to 0.5185) which already appear in its untouched "before" run, so they come from other uncommitted edits in the tree and are **not attributed**. A full re-run and baseline refresh are **pending**; until then the "after" values below are the fix agents' own measurements, not committed results.

### 5.2 The wrapper: from destructive to non-regressive

Historical (`reports/skillllm_v2/wrapper_error_analysis.md`, 2026-06-10, run-1 anchor 0.675034): the original boundary refiner lowered exact F1 to 0.5704 (194 regressed documents, all `bad_boundary_edit`), GLiNER plus taxonomy to 0.6515, the full chain to 0.5616 (225 regressions: 196 boundary-refiner, 29 verifier false rejections). The redesign (anchor-protected add-only merge, keep-original-unless-margin refiner, verifier that never rejects anchor spans) restored non-regression: 0.672444 / 0.669478 / 0.671854 (Table E3 M4). The audit diff shows the sole significant gain (+0.0024) comes from 7 offset-less Mistral spans dropped by the verifier and 3 spans re-aligned by `realign_word_boundary`, i.e. **cleanup of the offset-reconstruction defect, not a learned improvement** (row E7 of the master audit's evaluation table, `docs/INFERENCE_AND_EVALS.md` section 3.1). The suite adds that the learned verifier would cost 5.5 points if applied to anchor spans (M13).

### 5.3 Defects found by the module suite and fix passes

**Table E8. Bugs found, fixed and open.** "Before" and "after" are from the cited fix logs unless noted; all data synthetic; **none re-confirmed by a committed suite re-run (pending)**.

| # | Module | Defect (eval name) | Before | After | Status |
|---|---|---|---|---|---|
| 1 | context | `compact_context` had no token budget; every skill stored twice (`context.production_budget_adherence`) | 0.0 at budgets 128 to 2,048 | 0.0 / 1.0 / 1.0 / 1.0 / 1.0 (128 is infeasible: the minimum valid packet is about 129 proxy tokens) | fixed (`budget_tokens` parameter) |
| 2 | context | Important-skill recall under budget (probe) | 1.0 (nothing dropped, nothing fits) | 0.00 / 0.21 / 0.59 / 0.83 / 0.90 at 128 / 256 / 512 / 1,024 / 2,048 | fixed; the production packet pays about 130 tokens of structure |
| 3 | context | `content_hash` changed every second; order dependent | equal-when-clock-ticks 0.0; under permuted order 0.0 | 1.0; 1.0 | fixed; whole-packet bytes still differ by `generated_at` (`determinism_across_clock_ticks` stays 0.0 by design) |
| 4 | context | Label conflicts lost (`context.label_conflict_detection`) | 0.474 | 1.0 | fixed for labels |
| 5 | context | Date conflicts not detected (`context.date_conflict_detection`) | 0.0 | 0.0 | **open** (no robust date parser) |
| 6 | context | Repeated (skill, evidence) pairs (`duplicate_evidence_rate`) | 0.067 | 0.033 | fixed (residual is injected distinct-id conflicts) |
| 7 | graph | `explain_skill_path` never connected a user to a job (`graph.explainer_connects_user_to_job`; also `path_toy_exact_match`) | 0.0; 0.857 | 1.0; 1.0 | fixed (BFS over an explanatory view) |
| 8 | graph | Paths beyond `max_hops` truncated but labelled with the original target (`path_ends_at_target_beyond_hops`) | 0.0 | 1.0 | fixed |
| 9 | graph | `SkillGraphRetriever` leaked other users' skills for unknown ids (`retriever_unknown_node_leak_rate`) | 1.0 | 0.0 | fixed |
| 10 | graph | `trace_explanation_path` fabricates "required" and "teaches" relations (`explain.trace_path_faithful_when_edge_missing`) | 0.0 (fabrication rate 1.0) | unchanged in `CourseGraphBuilder`; new `narrate_skill_trace` has fabrication 0.0 | **open** (the method lives in a file the fix pass could not edit; delegate to the new function) |
| 11 | graph | Builder hygiene: re-ingest doubled edges, silent node retyping, dangling edges (`graph.ingest_idempotence`, `builder_rejects_dangling_edge`) | 0.5; 1.0 (bug); 0.0 | `SkillGraphBuilder`: 1.0; 0.0; strict mode 1.0; `CourseGraphBuilder` unchanged | partly fixed; `CourseGraphBuilder` **open** (0.5, and stamps wall-clock `last_seen`) |
| 12 | graph | `normalize_text` collapses C, C++, C#, F#, .NET to the same node id (`graph.skill_id_collision_rate` 1.0) | 1.0 | new `graph_key` keeps `+`, `#`, leading `.`; `normalize_text` unchanged | fixed for the pipeline graph; `CourseGraphBuilder` ids still collide (**open**) |
| 13 | graph | SkillGraphBuilder edges carry almost no provenance (`provenance_complete`) | 0.0 | 0.578 (1.0 when a clock or `last_seen` is supplied, probe) | fixed |
| 14 | explanation | `generate_recommendation_explanation` evidence padded with unrelated skills; boilerplate limitations (`explain.match.evidence_relevance`) | 0.147 | 1.0 | fixed; `completeness` fell 1.0 to 0.729 because 13 of 48 explanations now have an honest empty evidence list (eval artefact) |
| 15 | match | Exact-string skill comparison: case, whitespace, punctuation, no alias resolution (`match.robust.lowercase_user_skills`, `match.prop.case_invariance`) | nDCG@10 delta -0.7119; violation 0.2767 | 0.0000; 0.0000 | fixed (`match/skill_keys.py`) |
| 16 | match | Alias blindness (`match.prop.alias_invariance`; 25% / 50% alias substitution) | 0.1400; -0.0334 / -0.0962 | 0.0533; -0.0133 / -0.0342 | partly fixed; remainder is alias-table coverage (data work: `py`, `PowerBI`, `TS`, `REST`, ...) |
| 17 | gap | `GapDetector.detect`: held weak skill relabelled "covered by transferable"; duplicate gaps for required + preferred; weak preferred skills ignored (`gap.detector.micro_recall` / F1) | 0.8045 / 0.8917; relabel violation 1.0; duplicate violation 1.0 | 1.0 / 1.0; 0.0; 0.0 | fixed (note: F1 = 1.0 is on authored data in the detector's own vocabulary) |
| 18 | e2e | Extractor canonical names against surface names in jobs and catalogs (`e2e.gap_demo.f1`, false-gap rate 0.25) | 0.6147 (log) [0.6259 in the 08:39 snapshot, see note]; 0.25 | 0.7171; 0.0 | fixed in matcher, detector, vector and pipeline match stage |
| 19 | e2e | Extractor exceptions escaped `SkillLLMv2Pipeline.extract` (`e2e.degradation.graceful_share`) | 0.8182 (9 of 11) | 1.0 (11 of 11) | fixed (degraded status with error text, optional strict mode) |
| 20 | e2e | Gap-to-course linkage@5: catalog `SQL` against gap id `structured query language` | 0.6204 in baseline; 0.5185 in the fix pass's "before" run (unexplained drift) | 0.5185 | **open** (course ranking out of scope; shared `SkillKeyResolver` can be reused) |
| 21 | gap | `GapEngine`: a weakly held skill outweighs a transferable-covered one, so acquiring a skill can raise the gap (`gap.prop.engine.weakly_held_skill_not_worse_than_transferable_cover`) | violation 0.8775 | unchanged | **open** (`recommend/gap_vector.py` out of scope; fix proposed in the log) |

Notes on Table E8. Row 18: the suite snapshot reports `e2e.gap_demo.f1` 0.6259 whereas the match log's "before" is 0.6147, another sign that the tree was moving between runs. Two eval tests encode old buggy behaviour and now fail by design (`tests/evals/test_match_gap_e2e.py` two tests; `tests/evals/test_context_graph_explanation.py` two tests asserting `provenance_complete == 0.0` and `evidence_relevance < 1.0`); the stale "FINDING" caveat strings in the eval modules should be updated. The match log reports the full test suite as "438 passed, 1 skipped, 2 failed" (the two eval tests); I did not run the tests, so their pass status is **unverified by me**.

### 5.4 Wiring defects found by the audits

`docs/WIRING_AUDIT.md` (code audit): silent stub anchors (the Mistral and GLiNER fallbacks emitted alias-matched candidates labelled `source="mistral"` that policies treated as the protected anchor, unflagged); dictionary-only extraction produced zero skills under the default add-only merger (dictionary confidence 0.82 below the 0.85 bar); `TaxonomyLinker.link_candidates` overwrote canonical ids regardless of link confidence; the YAML configs were not read by the inference API; component defaults depended on the working directory; 12 modules were unreachable from any entry point. Status: stub anchors are now opt-in and flagged in a trace; the merger gates are opened in a flagged no-anchor mode ("unevaluated"); links can be suppressed below a configured confidence (default 0.0 keeps legacy behaviour); the new pipeline wires every previously unreachable module (`WIRING_AUDIT.md` "AFTER"). Reachability claims were verified by an AST import walk and grep, not by execution.

---

## 6. The twelve inconsistencies of the master audit, and eight more

**Table E5. Inconsistencies listed in `docs/INFERENCE_AND_EVALS.md` section 3.4, with status as of today.**

| # | Inconsistency | Status (checked 2026-10-22) |
|---|---|---|
| I1 | Anchor exact-F1 0.675034 (run 1) hard-coded in `configs/skillllm_v2_wrapper_guardrails.yaml`, `src/skillllm_v2/eval/non_regression_gate.py` (lines 10, 17, 57, 62) and docstrings versus 0.670069 (run 2) in the actual metrics; `improved_over_anchor` is wrong; YAML comment floor 0.665000 is stale (actual 0.660035) | **open** (grep confirms the stale values; I may not edit `src/`) |
| I2 | Full wrapper 0.671854 (current) versus 0.671102 in `README.md`, the thesis, `codebase_audit.md`, `metric_sanity_report.md`; the "safe wrapper" row there is a hybrid of anchor metrics with exact and soft overwritten | **open** |
| I3 | Money-table "Course Rec P@5 = 1.000000" matches no benchmark file (current P@5 0.617 v1, 0.608 v2 test); `README.md` says labels are not persisted although they now are | **open** (`reports/skillllm_v2/money_table.md` still shows 1.000000) |
| I4 | Linking coverage for one stage: 0.890374 (money table) versus 0.838908 (`linking_error_analysis.md`) versus 0.889834 (safe variant); `linking_eval.csv` reports accuracy 0.0 | **open**, cause unresolved |
| I5 | Baselines scored on 3,569 rows, Mistral rows on 974, one column in the money table | **open** (documented in this draft) |
| I6 | `artifacts/skillllm_v2/gliner_baseline/model/config.json` says GLiNER skipped while predictions and metrics show real zero-shot GLiNER | **open** (re-read today: still `skipped_dependency_missing`) |
| I7 | Extraction ablation A to G ran the 8-alias stub on 20 dev rows | **open**; do not cite |
| I8 | "BERT baseline" is a token-memorisation table; YAMLs list `bert-base-uncased` and 80 epochs, unused | **open** (naming in the thesis should change) |
| I9 | Two same-seed 80-epoch runs differ by 0.0050; run-1 adapter overwritten | **open**, irreducible; reported as noise floor (`training.md` section 2.4) |
| I10 | Tuned merger/boundary YAMLs lack `tuning_meta`, contain values outside the tuner's grids, and the tuning reports and dev predictions are absent; no calibrator tuned config | **open**; no dev F1 of any tuning is verifiable |
| I11 | `docs/loop_training.md` cites "0.6701/0.6726" (0.6726 matches no current file) | **open** (confirmed today, line 44) |
| I12 | 76 of 711 alias rows are mined from test-split predictions and feed the merger, boundary selector and linker evaluated on test | **open**; the suite now quantifies it: 18.4% of alias-resolved test links (M17) |

I re-checked I1, I3, I6 and I11 directly against the files today; the others are structural and no file I saw changes them.

**Table E6. Additional discrepancies found while preparing these drafts.**

| # | Discrepancy | Files |
|---|---|---|
| A1 | The `Research Papers` folder is **empty**, and `Resume+Job Data/` is empty; the committee brief expected papers there. Only bibliography entries exist (`docs/thesis/references.bib`, 69 entries). | `/home/923873155/Sem 4 Research/Research Papers`, `.../Resume+Job Data` |
| A2 | `Project Desc.md` does not mention the 1 November deployment or the 5 December final deadline (I searched all `.md` and `.tex` files under `Sem 4 Research`); the dates used in `BEST_PLAN.md` come from the task brief. `skillMatch.md` is an empty file. | `Project Desc.md`, `skillMatch.md` |
| A3 | Per-document generation time: 3.6 s (loop v1: 3,524.4 s / 974), 4.4 s (loop v2: 4,303.1 s / 974; also `docs/LIBRARY.md`), 4.37 s (anchor stage including load: 4,257.5 s / 974), 4.65 s mean and 3.79 s median over 901 documents (audit, `metadata.latency_seconds`). | `reports/...qlora_loop_summary.md`, `qlora_loop_v2_summary.md`, `docs/INFERENCE_AND_EVALS.md` |
| A4 | Training peak memory for loop v1: 16.4 GB (`docs/loop_training.md`) versus 17.9 GB (`qlora_loop_summary.md`). | same |
| A5 | Tokens per training example and fixed-prompt share: 351.9 and 86.4% (my recomputation), about 85% (`loop_training.md`), 382 and 78.5% (`docs/GPU_EXPERIMENTS_LOG.md`). | `training.md` Table T2 |
| A6 | Short-prompt text differs between the GPU log's plan and `mistral_extractor.py` (system "Extract skill spans." against "Extract skill spans from text. Return JSON only."); fixed-token counts 41 (log) against about 77 (mine). | `docs/GPU_EXPERIMENTS_LOG.md`, `src/skillllm_v2/extract/mistral_extractor.py` |
| A7 | `docs/EVAL_SUITE.md` and `docs/EVAL_LANDSCAPE.md` say the `ir_measures`/`statsmodels` adapters were exercised and agree; the committed suite report shows those cross-checks **skipped** (libraries not installed in the default environment). Both statements are true of different runs. | `reports/skillllm_v2/eval_suite/REPORT.md` |
| A8 | The thesis PDF and chapters are dated 2026-09-02 and contain none of the loop, ensemble, vote-confidence or module-suite results (a case-insensitive search of `docs/thesis/chapters/*.tex` for "qlora loop", "ensemble" and "loop adapter" returned no match); chapter 10 states that no nDCG or P@5 result exists, although the recommender benchmark now does. `docs/INFERENCE_AND_EVALS.md` section 1.6 and `docs/INFERENCE_AND_EVALS.md` section 0 item 4 describe the pre-pipeline state (stub default, unreachable modules) and are superseded by `docs/WIRING_AUDIT.md` "AFTER". | `docs/thesis/` |

---

## 7. Circularity, leakage and validity threats

**Table E7. Threats, with what is known.**

| Threat | Evidence | Effect and mitigation |
|---|---|---|
| Test set used for decisions | The non-regression gate (accept or fall back) is evaluated on the 974 documents it then reports on; the vote-precision table and operating points are estimated on test; variants compared on test | Decisions are post hoc; effects are tiny (at most +0.0024) so the harm is limited, but "accept" is not a held-out result |
| Test-derived artefacts inside the system | 76 of 711 alias rows mined from test-split anchor predictions; 18.4% of alias-resolved test links use them (M17) | Linking coverage, the merger alias gate and the boundary selector's alias bonus are partly circular; remove and re-run (not done) |
| Tuning provenance unknown | I10 | No dev F1 of the merger/boundary tuning can be verified; treat "dev-tuned" claims as unsupported |
| Train/test text repetition | 177 train and 169 dev normalised texts also occur among test rows (quoted from the loop report); the verifier eval flags 4.4% of candidates | Shared by all systems; not recomputed |
| Synthetic, self-authored ground truth | Matching, gap, context, graph, explanation and e2e data are authored with the system's vocabulary; recommender labels are rule-derived from catalogue skill and prerequisite metadata the skill-aware systems also use | Absolute scores (nDCG@10 0.99, gap F1 1.0, explanation completeness 1.0) are ceilings or by-construction values; only baselines, robustness deltas, properties and defects carry information |
| No human validation | `human_review_sample.csv` 0 of 60; no human ratings of explanations or importance | **Pending**; the recommender's label validity is unknown |
| Domain shift | SkillSpan is job-posting sentences; deployment input is resumes; no resume text exists locally | Resume performance unmeasured; false positives on skill-free text unmeasured (2,595 gold-free test rows excluded) |
| Seed variance | One training seed per recipe; two identical runs differ by 0.005 | No CI includes training variance |
| Metric convention | AP divisor, dropped scenarios, character "token F1", tag mapping | State conventions with every number (Table E1) |
| Controls test the harness, not the system | 37 controls pass | A passing control says the metric is not broken |
| Small n | e2e 12 cases; explanation 8 worlds; graph 12 worlds; linking alias held-out 15 | Wide intervals; percentile bootstrap undercovers at small n |

---

## 8. What is still not evaluated

No gold canonical links (so no linking accuracy, recall@k or MRR against a real taxonomy; no ESCO, O*NET or Lightcast data is loaded); no resume text; no false-positive rate on skill-free text; no real job-matching or gap labels; no explanation faithfulness beyond structural audit (semantic judge and human usefulness skipped); no multi-seed variance for any trained model; no real-model end-to-end run in the recommendation path (the e2e evaluation uses the stub extractor, so extraction errors are not propagated to recommendations); no real-model latency per stage; no peak memory for the 80-epoch training. The master audit's items 1 to 18 are partly closed by the suite: the verifier, calibrator, linker (probe only), context, graph, explanation, matcher, gap detector and e2e are now evaluated, **on synthetic data**.

## 9. Pending

| Item | Owner / location | Status |
|---|---|---|
| Re-run the suite on the fixed tree and refresh `baseline.json`; update stale eval tests and caveat strings | `docs/BUGFIX_LOG_MATCH.md` | pending |
| Human grades for 60 rows | `reports/skillllm_v2/human_review_sample.csv` | pending |
| Log-probability confidence study (AUROC, AURC, ECE against vote count) | `docs/GPU_EXPERIMENTS_LOG.md` Experiment 3 | pending |
| Short-prompt non-inferiority and distillation tests | Experiments 1 and 2 | pending (`training.md` Table T13) |
| Install `ir_measures`, `statsmodels`, `nervaluate` in the eval environment so the six skipped oracle checks run | `docs/EVAL_LANDSCAPE.md` section 6 | pending |
| Remaining open defects (Table E8 rows 5, 10, 11, 12, 20, 21) | `CourseGraphBuilder`, `recommend/` | open |
