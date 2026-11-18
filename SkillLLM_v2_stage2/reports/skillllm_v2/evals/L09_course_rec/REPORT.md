# L09 — Course recommendation: systems, WSGC@5, prerequisites, explanation faithfulness, label audit, H6

Owner A5. Definitions pre-registered in `Decisions.md` D-A5-01..07 (D-A5-04 = formal metric definitions incl. WSGC@5). Primary = 24 held-out TEST scenarios; unit = scenario; headline CIs B=10000. Catalogue and scenarios are SYNTHETIC.

## Headline (test half, SkillLLM_v2 Full, mean [95% CI])

- wsgc@5: 0.838 [0.773, 0.899]
- prereq_violation_rate: 0.000 [0.000, 0.000]
- ndcg@10: 0.885 [0.851, 0.916]
- precision@5: 0.608 [0.533, 0.683]
- **H6 verdict: SUPPORTED.** Full − Full-noPrereq: violation Δ -0.183 [-0.258, -0.117] (Holm p 0.001); nDCG@10 Δ +0.114 [+0.074, +0.154] (margin −0.02).
- Explanation faithfulness (Full, test): lenient 0.925 (145 unsupported of 1922 claims); strict field semantics 0.888. Existing Explanation Completeness = 1.000 for Full, 1.000 for random courses with placeholder fields, 1.000 for Full packets attached to the WRONG courses (faithfulness of those: 0.650) → **the completeness metric is tautological** (rule of D-B4-01(3) met).
- Full vs the pre-specified strongest graph-free baseline `prerequisite_aware(greedy+filter)` (test, n=24): nDCG@10 Δ +0.035 [-0.007, +0.073], Holm p 0.108 → not distinguishable (Holm p >= 0.05); WSGC@5 Δ +0.001 [-0.057, +0.059], Holm p 1.000. Strongest baseline chosen on DEV only: `prerequisite_aware(greedy+filter)` (dev nDCG@10 0.853).
- Label audit: a ranker that just applies the label rule to the systems' own visible inputs reaches nDCG@10 0.958 [0.929, 0.979] against the gold labels; κ(v1, v_visible) = 0.898, κ(v1, v_text) = 0.586.

## 1. Systems (test half, held out)

| system | precision@5 | recall@5 | ndcg@10 | mrr@10 | wsgc@5 | prereq_violation_rate | redundant_skill_coverage@5 | redundancy@5(existing) | learning_path_validity | learning_path_repairable | n_returned | ndcg@10[content_only] | ndcg@10[v_text] | ndcg@10[v_visible] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.0167 | 0.0153 | 0.0453 | 0.0432 | 0.0854 | 0.5333 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 10.0000 | 0.0645 | 0.0558 | 0.0494 |
| lexical(BM25) | 0.3917 | 0.4172 | 0.6040 | 0.7622 | 0.7188 | 0.3167 | 0.2648 | 0.0351 | 0.0833 | 0.1250 | 10.0000 | 0.6818 | 0.6146 | 0.6176 |
| semantic(MiniLM) | 0.4250 | 0.4244 | 0.6013 | 0.8160 | 0.7055 | 0.2000 | 0.2598 | 0.0347 | 0.1667 | 0.3750 | 10.0000 | 0.6488 | 0.5431 | 0.5998 |
| skill_overlap | 0.4750 | 0.4678 | 0.7594 | 0.7966 | 0.7911 | 0.2500 | 0.3373 | 0.0581 | 0.0833 | 0.2500 | 10.0000 | 0.8504 | 0.6560 | 0.8088 |
| skill_gap(greedy) | 0.5083 | 0.5163 | 0.8194 | 0.8819 | 0.8869 | 0.1667 | 0.2640 | 0.0456 | 0.2083 | 0.2500 | 10.0000 | 0.9055 | 0.6899 | 0.8702 |
| hybrid(RRF) | 0.5083 | 0.5200 | 0.7843 | 0.8740 | 0.8436 | 0.2000 | 0.3265 | 0.0511 | 0.1667 | 0.2917 | 10.0000 | 0.8674 | 0.7204 | 0.8310 |
| prerequisite_aware(greedy+filter) | 0.5500 | 0.5744 | 0.8508 | 1.0000 | 0.8372 | 0.0000 | 0.2702 | 0.0426 | 1.0000 | 1.0000 | 10.0000 | 0.8141 | 0.6194 | 0.8952 |
| SkillLLM_v2 Full | 0.6083 | 0.6646 | 0.8854 | 0.9792 | 0.8382 | 0.0000 | 0.2563 | 0.0357 | 1.0000 | 1.0000 | 9.8750 | 0.8367 | 0.6755 | 0.9116 |
| Full-noPrereq (ablation) | 0.4583 | 0.4825 | 0.7711 | 0.8472 | 0.8895 | 0.1833 | 0.2599 | 0.0412 | 0.1250 | 0.2500 | 10.0000 | 0.8593 | 0.7086 | 0.8132 |
| Full legacy (pre-tuning) | 0.6000 | 0.6270 | 0.8550 | 0.9583 | 0.8823 | 0.0667 | 0.2595 | 0.0438 | 0.5417 | 0.5417 | 8.2917 | 0.8467 | 0.6548 | 0.8955 |
| lexical+filter | 0.5417 | 0.5704 | 0.6948 | 0.9097 | 0.6842 | 0.0000 | 0.2782 | 0.0372 | 1.0000 | 1.0000 | 10.0000 | 0.6044 | 0.5537 | 0.6946 |
| semantic+filter | 0.4917 | 0.5203 | 0.6508 | 0.8958 | 0.7127 | 0.0000 | 0.2427 | 0.0338 | 1.0000 | 1.0000 | 10.0000 | 0.5987 | 0.5058 | 0.6452 |
| hybrid+filter | 0.6000 | 0.6333 | 0.8347 | 0.9792 | 0.7812 | 0.0000 | 0.3057 | 0.0443 | 1.0000 | 1.0000 | 10.0000 | 0.7689 | 0.6173 | 0.8708 |

CIs (test): SkillLLM_v2 Full ndcg@10 0.885 [0.851, 0.916]; SkillLLM_v2 Full wsgc@5 0.838 [0.773, 0.899]; SkillLLM_v2 Full prereq_violation_rate 0.000 [0.000, 0.000]; prerequisite_aware(greedy+filter) ndcg@10 0.851 [0.808, 0.890]; prerequisite_aware(greedy+filter) wsgc@5 0.837 [0.760, 0.905]; prerequisite_aware(greedy+filter) prereq_violation_rate 0.000 [0.000, 0.000]; hybrid(RRF) ndcg@10 0.784 [0.732, 0.832]; hybrid(RRF) wsgc@5 0.844 [0.785, 0.899]; hybrid(RRF) prereq_violation_rate 0.200 [0.108, 0.300]; lexical(BM25) ndcg@10 0.604 [0.533, 0.674]; lexical(BM25) wsgc@5 0.719 [0.652, 0.785]; lexical(BM25) prereq_violation_rate 0.317 [0.217, 0.417]

### Dev half (TUNED-ON for Full; context only)

| system | precision@5 | recall@5 | ndcg@10 | mrr@10 | wsgc@5 | prereq_violation_rate | redundant_skill_coverage@5 | redundancy@5(existing) | learning_path_validity | learning_path_repairable | n_returned | ndcg@10[content_only] | ndcg@10[v_text] | ndcg@10[v_visible] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.0250 | 0.0234 | 0.0262 | 0.0481 | 0.0509 | 0.5333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10.0000 | 0.0265 | 0.0406 | 0.0259 |
| lexical(BM25) | 0.4167 | 0.4263 | 0.5907 | 0.7250 | 0.7184 | 0.3250 | 0.2846 | 0.0427 | 0.0417 | 0.1250 | 10.0000 | 0.6839 | 0.6067 | 0.6072 |
| semantic(MiniLM) | 0.3833 | 0.4046 | 0.5591 | 0.7236 | 0.6815 | 0.1583 | 0.2624 | 0.0311 | 0.2083 | 0.4167 | 10.0000 | 0.6080 | 0.5449 | 0.5670 |
| skill_overlap | 0.5500 | 0.5362 | 0.7598 | 0.7882 | 0.7648 | 0.1417 | 0.3660 | 0.0634 | 0.1667 | 0.2917 | 10.0000 | 0.8246 | 0.6722 | 0.7981 |
| skill_gap(greedy) | 0.5500 | 0.5576 | 0.7959 | 0.8472 | 0.9143 | 0.1333 | 0.2423 | 0.0413 | 0.2083 | 0.3750 | 10.0000 | 0.8852 | 0.7286 | 0.8357 |
| hybrid(RRF) | 0.5333 | 0.5496 | 0.7907 | 0.8278 | 0.8566 | 0.1583 | 0.3211 | 0.0516 | 0.0833 | 0.2500 | 10.0000 | 0.8633 | 0.7529 | 0.8287 |
| prerequisite_aware(greedy+filter) | 0.6167 | 0.6184 | 0.8533 | 1.0000 | 0.8824 | 0.0000 | 0.2675 | 0.0480 | 1.0000 | 1.0000 | 10.0000 | 0.8341 | 0.7028 | 0.8943 |
| SkillLLM_v2 Full | 0.6167 | 0.6418 | 0.9022 | 1.0000 | 0.8625 | 0.0000 | 0.2675 | 0.0357 | 1.0000 | 1.0000 | 9.8750 | 0.8474 | 0.7429 | 0.9271 |
| Full-noPrereq (ablation) | 0.4250 | 0.4479 | 0.7993 | 0.9097 | 0.9099 | 0.2167 | 0.2461 | 0.0375 | 0.1250 | 0.2917 | 9.8333 | 0.8748 | 0.7595 | 0.8321 |
| Full legacy (pre-tuning) | 0.6333 | 0.6425 | 0.8814 | 1.0000 | 0.8707 | 0.0250 | 0.2864 | 0.0460 | 0.5417 | 0.5833 | 8.4167 | 0.8712 | 0.7404 | 0.9197 |
| lexical+filter | 0.5000 | 0.5062 | 0.6685 | 0.9167 | 0.6459 | 0.0000 | 0.2389 | 0.0286 | 1.0000 | 1.0000 | 10.0000 | 0.5925 | 0.5095 | 0.6749 |
| semantic+filter | 0.4333 | 0.4512 | 0.6073 | 0.8681 | 0.6465 | 0.0000 | 0.2664 | 0.0311 | 1.0000 | 1.0000 | 10.0000 | 0.5654 | 0.4741 | 0.6099 |
| hybrid+filter | 0.6000 | 0.6256 | 0.8469 | 0.9792 | 0.7983 | 0.0000 | 0.3348 | 0.0499 | 1.0000 | 1.0000 | 10.0000 | 0.7755 | 0.6905 | 0.8813 |

## 2. H6 — prerequisite awareness

Rule: H6: prerequisite-aware vs same ranker without prerequisite constraints: violation rate lower with CI excluding 0, AND nDCG@10 non-inferior (lower CI bound of Δ > −0.02).

| pair (test) | n | Δ violation [95% CI] | Holm p (violation family) | Δ nDCG@10 [95% CI] | non-inferior (−0.02)? |
|---|---|---|---|---|---|
| primary: Full vs same ranker without prerequisite constraints/score | 24 | -0.183 [-0.258, -0.117] | 0.001 | +0.114 [+0.074, +0.154] | yes |
| greedy + filter vs greedy | 24 | -0.167 [-0.258, -0.083] | 0.002 | +0.031 [-0.006, +0.075] | yes |
| BM25 + filter vs BM25 | 24 | -0.317 [-0.417, -0.217] | 0.001 | +0.091 [+0.036, +0.156] | yes |
| MiniLM + filter vs MiniLM | 24 | -0.200 [-0.300, -0.108] | 0.001 | +0.049 [+0.022, +0.081] | yes |
| hybrid + filter vs hybrid | 24 | -0.200 [-0.300, -0.108] | 0.001 | +0.050 [+0.012, +0.092] | yes |

**Verdict: SUPPORTED** (primary pair). The violation half holds by construction for strict/filtered rankers; read the nDCG column. Label sensitivity of the primary nDCG Δ (post-hoc, D-B4-02): content_only: -0.023 [-0.056, +0.006] → non-inferiority (lower CI > −0.02) FAILS; v_text: -0.033 [-0.062, -0.005] → non-inferiority (lower CI > −0.02) FAILS. The formal verdict is the pre-registered one (v1 labels); the nDCG half is therefore label-rule dependent whenever a variant FAILS.

### Full vs the other 7 systems (descriptive, D-B4-01; Holm over the 7 baselines per metric; test half)

| baseline | nDCG@10 Δ [CI] (Holm p) | P@5 Δ [CI] (Holm p) | WSGC@5 Δ [CI] (Holm p) | violation Δ [CI] (Holm p) |
|---|---|---|---|---|
| random | +0.840 [+0.801, +0.878] (0.001) | +0.592 [+0.517, +0.667] (0.001) | +0.753 [+0.664, +0.838] (0.001) | -0.533 [-0.608, -0.458] (0.001) |
| lexical(BM25) | +0.281 [+0.204, +0.357] (0.001) | +0.217 [+0.142, +0.292] (0.001) | +0.119 [+0.047, +0.195] (0.024) | -0.317 [-0.417, -0.217] (0.001) |
| semantic(MiniLM) | +0.284 [+0.225, +0.350] (0.001) | +0.183 [+0.100, +0.267] (0.004) | +0.133 [+0.060, +0.206] (0.007) | -0.200 [-0.300, -0.108] (0.001) |
| skill_overlap | +0.126 [+0.063, +0.191] (0.002) | +0.133 [+0.042, +0.233] (0.060) | +0.047 [-0.027, +0.126] (0.752) | -0.250 [-0.367, -0.142] (0.001) |
| skill_gap(greedy) | +0.066 [+0.007, +0.125] (0.094) | +0.100 [+0.025, +0.175] (0.072) | -0.049 [-0.107, +0.006] (0.463) | -0.167 [-0.258, -0.083] (0.005) |
| hybrid(RRF) | +0.101 [+0.049, +0.158] (0.002) | +0.100 [+0.025, +0.175] (0.072) | -0.005 [-0.050, +0.050] (1.000) | -0.200 [-0.300, -0.108] (0.002) |
| prerequisite_aware(greedy+filter) (pre-specified strongest) (dev-selected strongest) | +0.035 [-0.007, +0.073] (0.108) | +0.058 [-0.000, +0.117] (0.117) | +0.001 [-0.057, +0.059] (1.000) | +0.000 [+0.000, +0.000] (1.000) |

SG-v1 (240 generated, held out; secondary):

| pair | n | Δ violation [CI] | Δ nDCG@10 [CI] |
|---|---|---|---|
| SG all: primary: Full vs same ranker without prerequisite constraints/score | 240 | -0.481 [-0.518, -0.442] | +0.147 [+0.128, +0.167] |
| SG needs prerequisite hop: primary: Full vs same ranker without prerequisite constraints/score | 198 | -0.553 [-0.590, -0.516] | +0.177 [+0.157, +0.198] |
| SG all: greedy + filter vs greedy | 240 | -0.482 [-0.514, -0.449] | -0.023 [-0.050, +0.002] |
| SG needs prerequisite hop: greedy + filter vs greedy | 198 | -0.539 [-0.574, -0.509] | -0.027 [-0.058, +0.004] |
| SG all: BM25 + filter vs BM25 | 240 | -0.420 [-0.453, -0.388] | +0.009 [-0.014, +0.030] |
| SG needs prerequisite hop: BM25 + filter vs BM25 | 198 | -0.470 [-0.504, -0.438] | +0.002 [-0.025, +0.028] |
| SG all: MiniLM + filter vs MiniLM | 240 | -0.154 [-0.175, -0.134] | +0.006 [-0.004, +0.014] |
| SG needs prerequisite hop: MiniLM + filter vs MiniLM | 198 | -0.180 [-0.203, -0.158] | +0.003 [-0.008, +0.012] |
| SG all: hybrid + filter vs hybrid | 240 | -0.470 [-0.504, -0.436] | -0.019 [-0.046, +0.004] |
| SG needs prerequisite hop: hybrid + filter vs hybrid | 198 | -0.528 [-0.564, -0.492] | -0.025 [-0.055, +0.005] |

SG-v1 Full vs strongest graph-free / greedy baselines (exploratory, post-hoc D-B4-02; Holm over 2 per metric):

| baseline | metric | n | Δ [CI] | Holm p |
|---|---|---|---|---|
| prerequisite_aware(greedy+filter) | p5 | 240 | +0.003 [-0.008, +0.015] | 0.685 |
| skill_gap(greedy) | p5 | 240 | +0.042 [+0.025, +0.059] | 0.002 |
| prerequisite_aware(greedy+filter) | ndcg10 | 239 | +0.148 [+0.119, +0.180] | 0.002 |
| skill_gap(greedy) | ndcg10 | 239 | +0.126 [+0.105, +0.146] | 0.002 |
| prerequisite_aware(greedy+filter) | wsgc5 | 240 | +0.161 [+0.119, +0.204] | 0.002 |
| skill_gap(greedy) | wsgc5 | 240 | -0.167 [-0.200, -0.138] | 0.002 |
| prerequisite_aware(greedy+filter) | viol5 | 240 | +0.000 [+0.000, +0.000] | 1.000 |
| skill_gap(greedy) | viol5 | 240 | -0.482 [-0.514, -0.449] | 0.002 |

## 3. Explanation faithfulness

Every claim in each Full explanation packet is checked against the catalogue, the job/profile and the per-scenario knowledge graph (covered gap skill, job requirement, user gap class, quoted learning outcome mentions the skill, prerequisite status, unlock/bridge claims, limitations).

| claim type (test, Full) | supported / total |
|---|---|
| covers_gap_skill[lenient: enables via bridge] | 28 / 28 |
| covers_gap_skill[lenient: teaches] | 230 / 230 |
| covers_gap_skill[lenient: text-related] | 42 / 42 |
| covers_gap_skill[strict field semantics] | 230 / 300 |
| job_requires_skill | 306 / 306 |
| limitation_bridge | 19 / 19 |
| limitation_overlap | 106 / 106 |
| limitation_workload | 42 / 42 |
| outcome_quote_mentions_skill | 119 / 264 |
| prerequisite_all_satisfied_claim | 93 / 93 |
| prerequisite_satisfied_claim | 120 / 120 |
| prerequisite_status | 144 / 144 |
| unlock_placement | 74 / 74 |
| unlock_requires_prereq | 74 / 74 |
| unlock_this_teaches_prereq | 74 / 74 |
| user_gap_classification | 306 / 306 |

Faithfulness of the other Full variants (lenient): SkillLLM_v2 Full 0.925; Full-noPrereq (ablation) 0.905; Full legacy (pre-tuning) 0.887; probe: Full packets rotated to wrong courses 0.650

SG-v1 Full faithfulness: lenient 0.925, strict 0.856.

**Is 'Explanation Completeness 1.0' tautological? Yes (rule D-B4-01(3)).** The existing metric (`course_recommendation_eval.explanation_completeness_rate`) only checks that four fields are non-empty; `teaches_skills` and `prerequisites` are present on every catalogue record, so two conditions always hold, and any list-valued `explanation`/`evidence` satisfies the rest. Random courses with placeholder fields score 1.000; Full's own packets moved onto the wrong courses still score 1.000 while their faithfulness drops to 0.650.

Unsupported-claim examples (first 12): [outcome_quote_mentions_skill] quote for Docker: Able to build, tag and run container images with Dockerfiles; [outcome_quote_mentions_skill] quote for Amazon Web Services: Able to provision and operate core AWS services such as EC2,; [outcome_quote_mentions_skill] quote for CI/CD: Able to design automated build, test and deployment pipeline; [outcome_quote_mentions_skill] quote for Amazon Web Services: Able to provision and operate core AWS services such as EC2,; [outcome_quote_mentions_skill] quote for Docker: Able to build, tag and run container images with Dockerfiles; [outcome_quote_mentions_skill] quote for CI/CD: Able to build declarative Jenkins pipelines and agents; [covers_gap_skill[strict field semantics]] ml-001 covers Machine Learning (enabling_prerequisite); [outcome_quote_mentions_skill] quote for prerequisite linear algebra: Able to compute with vectors, matrices, eigendecompositions ; [covers_gap_skill[strict field semantics]] data-022 covers Machine Learning (enabling_prerequisite); [covers_gap_skill[strict field semantics]] ml-004 covers Deep Learning (enabling_prerequisite); [covers_gap_skill[strict field semantics]] ml-004 covers Natural Language Processing (enabling_prerequisite); [covers_gap_skill[strict field semantics]] ml-004 covers PyTorch (enabling_prerequisite)

## 4. Label-independence audit

- Code: `build_relevance_labels.py` imports ['__future__', 'argparse', 'collections', 'csv', 'json', 'pathlib', 'random', 'skillllm_v2.course.ingest_catalog', 'skillllm_v2.eval.benchmark_scenarios_spec'] — no recommender/graph code (0 system modules).
- Implicit needs (hidden from every system) carry 17.5% of the open-need mass on average.

- Need/coverage: labels = exact-string intersection of catalogue teaches_skills with the need set; every skill-aware system (skill overlap, gap greedy, Full's gap_coverage and graph TEACHES edges) scores the same intersection.
- Prerequisite feasibility: labels give full grade only if P(c) ⊆ known ∪ weak and cap bridge-only courses at 1; the strict prerequisite mode (compat_basis='learner': prefer courses startable now), the '+ prereq filter' baselines and the violation metric implement the same catalogue rule.
- Weights differ: labels 3/2/1 (required/implicit/preferred) vs systems 1.0/0.7 (GapEngine double-applies 0.7 to preferred → 0.49).
- Independence that does exist: implicit needs are hidden from systems; labels never read any system output; no shared code (AST imports).

| v1 vs | κ (grade ≥ 2) | quadratic κ | Spearman ρ | exact agreement | Kendall τ of system ordering (test nDCG@10) |
|---|---|---|---|---|---|
| content_only | 0.809 | 0.907 | 0.922 | 0.977 | 0.462 |
| v_visible | 0.898 | 0.958 | 0.948 | 0.988 | 1.000 |
| v_text | 0.586 | 0.696 | 0.691 | 0.934 | 0.487 |

Oracle 'v1 rule on system-visible inputs' nDCG@10 vs gold (test): 0.958 [0.929, 0.979]. Interpretation: the gold is largely reconstructible from the systems' own inputs with the label rule; differences come mainly from implicit needs and weights. A system tuned toward this rule (as the Full recommender's dev-tuned strict prerequisite mode is) gains on v1 partly by matching the rule, which the content-only and V-text columns expose.

## 5. Generated held-out set SG-v1 (240 scenarios; secondary)

| system | precision@5 | ndcg@10 | wsgc@5 | prereq_violation_rate | learning_path_validity | n_returned |
|---|---|---|---|---|---|---|
| random | 0.0167 | 0.0515 | 0.0571 | 0.5883 | 0.0000 | 10.0000 |
| lexical(BM25) | 0.1933 | 0.5194 | 0.6317 | 0.4200 | 0.0167 | 10.0000 |
| semantic(MiniLM) | 0.1483 | 0.4280 | 0.3890 | 0.1542 | 0.1458 | 10.0000 |
| skill_overlap | 0.2858 | 0.6823 | 0.8082 | 0.4708 | 0.0167 | 10.0000 |
| skill_gap(greedy) | 0.2800 | 0.6998 | 0.8856 | 0.4817 | 0.0167 | 10.0000 |
| hybrid(RRF) | 0.2808 | 0.6819 | 0.8576 | 0.4700 | 0.0208 | 10.0000 |
| prerequisite_aware(greedy+filter) | 0.3183 | 0.6772 | 0.5575 | 0.0000 | 1.0000 | 10.0000 |
| SkillLLM_v2 Full | 0.3217 | 0.8254 | 0.7184 | 0.0000 | 1.0000 | 8.3417 |
| Full-noPrereq (ablation) | 0.2625 | 0.6785 | 0.8853 | 0.4811 | 0.0625 | 8.5292 |
| Full legacy (pre-tuning) | 0.3192 | 0.7603 | 0.8669 | 0.3774 | 0.2500 | 5.0792 |
| lexical+filter | 0.2350 | 0.5287 | 0.3868 | 0.0000 | 1.0000 | 10.0000 |
| semantic+filter | 0.1617 | 0.4340 | 0.3277 | 0.0000 | 1.0000 | 10.0000 |
| hybrid+filter | 0.3267 | 0.6625 | 0.4958 | 0.0000 | 1.0000 | 10.0000 |

## Caveats

- Synthetic catalogue, scenarios and labels; human validation of labels is pending (L11).
- n = 24 test scenarios; bootstrap ignores label uncertainty; scenarios share catalogue/archetypes.
- SG-v1 is generated by the author's procedure (template job text favours lexical matching); it tests consistency of conclusions, not realism.
- Violation rate and the strict/filter mechanisms share the same catalogue prerequisite rule (by-construction zeros).

## Reproduce

```
cd SkillLLM_v2_stage2 && PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.course_rec --seed 42 --n-boot 2000
```
