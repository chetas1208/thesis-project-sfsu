# L08 — Knowledge graph construction, external validity, retrieval and H5

Owner A5. Definitions pre-registered in `Decisions.md` D-A5-01..07 before any run. Unit = scenario; 95% percentile bootstrap CIs; headline metrics B=10000. Nothing is tuned (RRF k=60 equal weights, λ=0.5 fixed). All course/scenario data are SYNTHETIC except the external CPR-Recover / AL-CPL / LectureBank annotations.

## Headline

- **H5 verdict: SUPPORTED.** hybrid+graph − hybrid nDCG@10 on the 24 multi-gap test scenarios = 0.086 [0.048, 0.120], Holm p = 0.001 (W/L/T 21/1/2).
- Full recommender explanation-path validity (test): 1.000 of 350 claimed paths.
- The knowledge graph is a 1:1 projection of the catalogue's authored `teaches_skills`/`prerequisites`; the extraction/linking/parsing enrichment never reaches it: of the 147 skill/prerequisite pairs the enrichment adds or renames, 0.000 appear in the graph; all 310 TEACHES edges carry the course title as their only 'evidence' (310).
- Taxonomy-linker over-merges on authored skills (would collapse distinct nodes if wired): 19 groups, e.g. [('linux', ['Linux', 'Linux System Administration']), ('testing', ['A/B Testing', 'Hypothesis Testing', 'Penetration Testing', 'Software Testing']), ('networking', ['Cloud Networking', 'Computer Networking']), ('performance', ['Performance Profiling', 'Web Performance'])].

## 1. Graph construction QA (171-course catalogue graph)

Node types {'Course': 171, 'Skill': 208}; edge types {'TEACHES': 310, 'REQUIRES_PREREQUISITE': 124}.

| system | metric | value |
|---|---|---|
| kg_builder(as_wired) | teaches_edge_precision_vs_authored | 1.0000 |
| kg_builder(as_wired) | teaches_edge_recall_vs_authored | 1.0000 |
| kg_builder(as_wired) | prerequisite_edge_precision_vs_authored | 1.0000 |
| kg_builder(as_wired) | prerequisite_edge_recall_vs_authored | 1.0000 |
| kg_builder(as_wired) | enrichment_only_pairs_reaching_graph | 0.0000 |
| prereq_parser+linker | precision_vs_authored | 0.9274 |
| prereq_parser+linker | recall_vs_authored | 0.9274 |
| prereq_parser+linker | f1_vs_authored | 0.9274 |
| course_skill_extractor | precision_vs_authored | 0.6218 |
| course_skill_extractor | recall_vs_authored | 0.6258 |
| course_skill_extractor | f1_vs_authored | 0.6238 |
| taxonomy_linker | over_merge_groups_on_authored_skills | 19.0000 |
| taxonomy_linker | canonical_id_differs_from_authored_name_share | 0.3894 |
| kg_builder(as_wired) | over_merged_skill_ids | 0.0000 |
| kg_builder(as_wired) | alias_duplicate_skill_node_groups | 1.0000 |
| kg_builder(as_wired) | orphan_nodes | 0.0000 |
| kg_builder(as_wired) | courses_without_teaches_edges | 0.0000 |
| kg_builder(as_wired) | dangling_prerequisite_skills | 0.0000 |
| kg_builder(as_wired) | infeasible_courses_from_empty_profile | 0.0000 |
| kg_builder(as_wired) | prerequisite_cycles_nontrivial_scc | 0.0000 |
| kg_builder(as_wired) | courses_teaching_own_prerequisite | 0.0000 |
| kg_builder(as_wired) | edge_type_consistency | 1.0000 |
| kg_builder(as_wired) | teaches_provenance_fields_present | 1.0000 |
| kg_builder(as_wired) | teaches_evidence_nonempty | 1.0000 |
| kg_builder(as_wired) | teaches_evidence_informative | 0.0000 |
| kg_builder(as_wired) | requires_prerequisite_provenance_fields_present | 1.0000 |
| kg_builder(as_wired) | requires_prerequisite_evidence_nonempty | 1.0000 |
| kg_builder(as_wired) | requires_prerequisite_evidence_informative | 0.0000 |
| kg_builder(as_wired) | has_skill_mastery_fidelity_for_weak_skills | 0.0000 |

- Over-merged ids (distinct names → one node): none.
- Alias-duplicate Skill node groups (repo's own SkillKeyResolver says same skill): [['software architecture', 'system design']].
- Dangling prerequisite skills (no course teaches them): none; infeasible courses from an empty profile: none; non-trivial prerequisite SCCs: none; courses teaching their own prerequisite: none; skill-level prerequisite pairs: 198; longest prerequisite chain (skills): 5.
- Linker id drift examples (authored name → canonical id): [('A/B Testing', 'testing'), ('API Design', 'api'), ('AWS Lambda', 'lambda'), ('Algorithms', 'algorithm'), ('Apache Airflow', 'apache'), ('Apache Kafka', 'apache'), ('Apache Spark', 'apache'), ('Application Security', 'security'), ('Backend Development', 'backend'), ('Bash Scripting', 'bash'), ('Business Intelligence', 'artificial-intelligence'), ('CI/CD', 'continuous-integration-continuous-deployment')].

## 2. External validity on public prerequisite data

**E1 — CPR-Recover (real CS courses, 11 US universities; CC BY-NC-SA 4.0).** 654 courses, 861 course-prerequisite edges, 3344 majority-labelled concept pairs → 5267 directed instances (1240 positive). Our majority vote reproduces 99.7% of the released cs_preqs.csv. 71.3% of annotated concepts were string-matched in ≥1 description.

| system | AP [95% CI] | precision [CI] | recall [CI] | F1 [CI] |
|---|---|---|---|---|
| graph_rule(repo semantics) | 0.279 [0.248, 0.311] | 0.300 [0.263, 0.338] | 0.340 [0.280, 0.397] | 0.319 [0.274, 0.361] |
| graph_rule_minus_cotaught | 0.258 [0.231, 0.287] | 0.272 [0.234, 0.308] | 0.269 [0.221, 0.310] | 0.270 [0.231, 0.304] |
| graph_rule_reversed(sanity) | 0.248 [0.224, 0.271] | 0.235 [0.205, 0.264] | 0.267 [0.218, 0.314] | 0.250 [0.214, 0.285] |
| frequency_direction | 0.444 [0.359, 0.522] | 0.333 [0.288, 0.384] | 0.652 [0.593, 0.706] | 0.441 [0.388, 0.496] |
| random(prior) | 0.254 [0.224, 0.288] | - | - | - |
| minilm_similarity×frequency_direction | 0.435 [0.389, 0.486] | - | - | - |

Paired F1 differences (repo graph rule minus …, cluster bootstrap): graph_rule_reversed(sanity): +0.068 [+0.034, +0.100], p=0.002; frequency_direction: -0.123 [-0.165, -0.082], p=0.002. Prior positive rate = 0.235; 135 conflicting duplicate directed instances (pair annotated in both orientations with different majorities) dropped.

Reading: the graph rule (p ⇒ t iff a course teaching t requires a course teaching p) is the repo's prerequisite semantics applied to real data; the reversed rule is a sanity check (a meaningful rule must beat its reverse). Concept→course edges use the annotated vocabulary (oracle) and candidate pairs were generated from course dependencies, so recall is optimistic.

**E2 — does the synthetic catalogue's prerequisite structure agree with independent annotations?**

| dataset | overlapping concepts | synthetic pairs in vocab | agree | contradict | unannotated | external positives in overlap | implied by synthetic |
|---|---|---|---|---|---|---|---|
| CPR-Recover | 36 | 16 | 4 | 4 | 8 | 22 | 4 |
| AL-CPL | 5 | 2 | 0 | 0 | 2 | 1 | 0 |
| LectureBank1 | 5 | 0 | 0 | 0 | 0 | 5 | 0 |
| LectureBank2 | 10 | 5 | 0 | 3 | 2 | 2 | 0 |

Descriptive only (tiny overlap). Retrieval on a public course catalogue: **not available** (no relevance judgments for job-gap queries).

## 3. Retrieval baselines (q_gap unless marked q_job; v1 labels, relevant = grade ≥ 2)

### test half (held-out)

| system | P@5 | Recall@10 | Recall@20 | MRR@20 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|
| random | 0.0167 | 0.0487 | 0.1064 | 0.0611 | 0.0247 | 0.0453 |
| bm25(q_job) | 0.3917 | 0.6167 | 0.7910 | 0.7622 | 0.5464 | 0.6040 |
| dense(q_job) | 0.4250 | 0.6231 | 0.7560 | 0.8160 | 0.5630 | 0.6013 |
| bm25 | 0.4083 | 0.6977 | 0.8638 | 0.7953 | 0.6289 | 0.7147 |
| dense | 0.4417 | 0.6544 | 0.8028 | 0.7990 | 0.5962 | 0.6370 |
| skill_overlap | 0.6167 | 0.8169 | 0.8257 | 0.8507 | 0.8069 | 0.8444 |
| graph_multihop | 0.6750 | 0.8989 | 0.9169 | 0.9146 | 0.8203 | 0.8792 |
| hybrid(bm25+dense) | 0.4917 | 0.7137 | 0.8939 | 0.8532 | 0.6810 | 0.7389 |
| hybrid+skill_overlap | 0.5083 | 0.8258 | 0.9130 | 0.8740 | 0.7235 | 0.8093 |
| hybrid+graph | 0.5250 | 0.8424 | 0.9416 | 0.8889 | 0.7461 | 0.8244 |
| repo_CourseRetriever | 0.5667 | 0.8438 | 0.9039 | 0.8785 | 0.7777 | 0.8526 |
| pool_RAG(B=20,rerank=dense) | 0.4500 | 0.6753 | 0.8939 | 0.8025 | 0.6116 | 0.6622 |
| pool_GraphRAG(B=20,rerank=dense) | 0.4500 | 0.6805 | 0.9421 | 0.8025 | 0.6116 | 0.6672 |

### dev half (TUNED-ON for the Full recommender; shown for context only)

| system | P@5 | Recall@10 | Recall@20 | MRR@20 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|
| random | 0.0250 | 0.0415 | 0.1013 | 0.0577 | 0.0177 | 0.0262 |
| bm25(q_job) | 0.4167 | 0.5810 | 0.6991 | 0.7250 | 0.5558 | 0.5907 |
| dense(q_job) | 0.3833 | 0.5761 | 0.7443 | 0.7236 | 0.5178 | 0.5591 |
| bm25 | 0.4750 | 0.6601 | 0.8229 | 0.8250 | 0.6715 | 0.7129 |
| dense | 0.4333 | 0.6132 | 0.7797 | 0.7535 | 0.5924 | 0.6273 |
| skill_overlap | 0.6833 | 0.8135 | 0.8278 | 0.9028 | 0.8378 | 0.8584 |
| graph_multihop | 0.6833 | 0.9113 | 0.9485 | 0.9306 | 0.8131 | 0.8883 |
| hybrid(bm25+dense) | 0.4833 | 0.6992 | 0.8764 | 0.8708 | 0.6962 | 0.7388 |
| hybrid+skill_overlap | 0.5417 | 0.8092 | 0.9064 | 0.8403 | 0.7340 | 0.8156 |
| hybrid+graph | 0.5667 | 0.8294 | 0.9503 | 0.8681 | 0.7557 | 0.8320 |
| repo_CourseRetriever | 0.6583 | 0.8292 | 0.8889 | 0.9028 | 0.8294 | 0.8570 |
| pool_RAG(B=20,rerank=dense) | 0.4333 | 0.6216 | 0.8764 | 0.7535 | 0.5924 | 0.6323 |
| pool_GraphRAG(B=20,rerank=dense) | 0.4417 | 0.6285 | 0.9722 | 0.7535 | 0.5979 | 0.6360 |

95% CIs (test): random recall@10 0.049 [0.015, 0.088]; random ndcg@10 0.045 [0.025, 0.070]; bm25(q_job) recall@10 0.617 [0.511, 0.724]; bm25(q_job) ndcg@10 0.604 [0.533, 0.674]; dense(q_job) recall@10 0.623 [0.539, 0.711]; dense(q_job) ndcg@10 0.601 [0.537, 0.662]; bm25 recall@10 0.698 [0.614, 0.783]; bm25 ndcg@10 0.715 [0.655, 0.771]; dense recall@10 0.654 [0.570, 0.741]; dense ndcg@10 0.637 [0.570, 0.699]; skill_overlap recall@10 0.817 [0.727, 0.897]; skill_overlap ndcg@10 0.844 [0.790, 0.895]; graph_multihop recall@10 0.899 [0.831, 0.954]; graph_multihop ndcg@10 0.879 [0.815, 0.932]; hybrid(bm25+dense) recall@10 0.714 [0.635, 0.794]; hybrid(bm25+dense) ndcg@10 0.739 [0.684, 0.791]; hybrid+skill_overlap recall@10 0.826 [0.734, 0.911]; hybrid+skill_overlap ndcg@10 0.809 [0.765, 0.851]; hybrid+graph recall@10 0.842 [0.758, 0.920]; hybrid+graph ndcg@10 0.824 [0.778, 0.869]; repo_CourseRetriever recall@10 0.844 [0.766, 0.914]; repo_CourseRetriever ndcg@10 0.853 [0.805, 0.897]; pool_RAG(B=20,rerank=dense) recall@10 0.675 [0.598, 0.758]; pool_RAG(B=20,rerank=dense) ndcg@10 0.662 [0.606, 0.717]; pool_GraphRAG(B=20,rerank=dense) recall@10 0.680 [0.603, 0.762]; pool_GraphRAG(B=20,rerank=dense) ndcg@10 0.667 [0.612, 0.721]

## 4. Controlled RAG vs GraphRAG — H5

Rule: H5: hybrid+graph vs hybrid (no graph) nDCG@10 on multi-gap (≥2 gaps) held-out queries: Δ > 0 with Holm p < 0.05.

| comparison (multi-gap test) | n | Δ nDCG@10 [95% CI] | p (max of boot, perm) | Holm p |
|---|---|---|---|---|
| primary: hybrid+graph − hybrid | 24 | +0.086 [+0.048, +0.120] | 0.000 | 0.001 |
| pool variant: GraphRAG − RAG (B=20, dense rerank) | 24 | +0.005 [+0.000, +0.013] | 0.249 | 0.249 |
| graph vs flat skill overlap inside the same fusion | 24 | +0.015 [-0.001, +0.035] | 0.105 | 0.211 |

Minimum detectable effect at 80% power (two-sided α = 0.05, normal approximation from the paired SD): primary: hybrid+graph − hybrid: 0.053 (SD of paired differences 0.093); pool variant: GraphRAG − RAG (B=20, dense rerank): 0.010 (SD of paired differences 0.017); graph vs flat skill overlap inside the same fusion: 0.026 (SD of paired differences 0.045).

Reading the family: (a) adds a structured skill signal AND graph structure to a text-only hybrid, so a gain in (a) alone does not show that graph structure helps; (c) holds the structured skill signal fixed and isolates the graph-specific part (the prerequisite hop).

**Verdict: SUPPORTED.**

Strata and label sensitivity (descriptive, no multiplicity correction):

| stratum / variant | n | Δ [95% CI] | p |
|---|---|---|---|
| multi-gap (≥2): primary: hybrid+graph − hybrid | 24 | 0.086 [0.049, 0.122] | 0.001 |
| multi-gap (≥2): graph vs flat skill overlap inside the same fusion | 24 | 0.015 [-0.000, 0.035] | 0.105 |
| 1-gap: primary: hybrid+graph − hybrid | 0 | - [-, -] | - |
| 1-gap: graph vs flat skill overlap inside the same fusion | 0 | - [-, -] | - |
| needs prerequisite hop: primary: hybrid+graph − hybrid | 21 | 0.078 [0.037, 0.114] | 0.001 |
| needs prerequisite hop: graph vs flat skill overlap inside the same fusion | 21 | 0.015 [-0.002, 0.039] | 0.165 |
| no prerequisite hop: primary: hybrid+graph − hybrid | 3 | 0.138 [0.000, 0.212] | 0.502 |
| no prerequisite hop: graph vs flat skill overlap inside the same fusion | 3 | 0.013 [0.000, 0.038] | 1.000 |
| labels=content_only | 24 | 0.091 [0.064, 0.120] | 0.001 |
| SG all: primary: hybrid+graph − hybrid | 239 | 0.137 [0.122, 0.153] | 0.001 |
| SG all: pool variant: GraphRAG − RAG (B=20, dense rerank) | 239 | 0.012 [0.007, 0.016] | 0.001 |
| SG all: graph vs flat skill overlap inside the same fusion | 239 | 0.059 [0.046, 0.073] | 0.001 |
| SG multi-gap (≥2): primary: hybrid+graph − hybrid | 213 | 0.133 [0.118, 0.148] | 0.001 |
| SG multi-gap (≥2): pool variant: GraphRAG − RAG (B=20, dense rerank) | 213 | 0.012 [0.008, 0.018] | 0.001 |
| SG multi-gap (≥2): graph vs flat skill overlap inside the same fusion | 213 | 0.056 [0.042, 0.069] | 0.001 |
| SG 1-gap: primary: hybrid+graph − hybrid | 26 | 0.175 [0.112, 0.242] | 0.001 |
| SG 1-gap: pool variant: GraphRAG − RAG (B=20, dense rerank) | 26 | 0.004 [0.000, 0.012] | 0.511 |
| SG 1-gap: graph vs flat skill overlap inside the same fusion | 26 | 0.083 [0.033, 0.141] | 0.003 |
| SG needs prerequisite hop: primary: hybrid+graph − hybrid | 197 | 0.142 [0.126, 0.159] | 0.001 |
| SG needs prerequisite hop: pool variant: GraphRAG − RAG (B=20, dense rerank) | 197 | 0.015 [0.010, 0.020] | 0.001 |
| SG needs prerequisite hop: graph vs flat skill overlap inside the same fusion | 197 | 0.071 [0.056, 0.087] | 0.001 |
| SG no prerequisite hop: primary: hybrid+graph − hybrid | 42 | 0.115 [0.078, 0.158] | 0.001 |
| SG no prerequisite hop: pool variant: GraphRAG − RAG (B=20, dense rerank) | 42 | -0.004 [-0.012, 0.001] | 1.000 |
| SG no prerequisite hop: graph vs flat skill overlap inside the same fusion | 42 | 0.001 [-0.003, 0.008] | 1.000 |

Test half: 24 scenarios, all multi-gap; 21 need a prerequisite hop. SG-v1 (240 generated, held out, D-A5-07) populates the 1-gap and no-hop strata; it is author-generated and never pooled with the test half.

SG-v1 nDCG@10 by system: random 0.051 [0.040, 0.064]; bm25(q_job) 0.519 [0.485, 0.552]; dense(q_job) 0.428 [0.400, 0.458]; bm25 0.658 [0.634, 0.683]; dense 0.552 [0.524, 0.579]; skill_overlap 0.693 [0.666, 0.719]; graph_multihop 0.793 [0.768, 0.816]; hybrid(bm25+dense) 0.636 [0.611, 0.661]; hybrid+skill_overlap 0.715 [0.689, 0.739]; hybrid+graph 0.774 [0.750, 0.797]; pool_RAG(B=20,rerank=dense) 0.569 [0.542, 0.597]; pool_GraphRAG(B=20,rerank=dense) 0.581 [0.555, 0.608]

## 5. Multi-hop path validity

- Full recommender explanation paths (test): 1.000 valid of 350 (strict REQUIRES-only wording: 255); invalid reasons: {}; text-only paths (non-graph hop): 42.
- A5 graph retriever paths: 1.000 (sanity check; by construction).

## Caveats

- Synthetic catalogue and scenarios; v1 labels read the same catalogue fields the skill-aware systems read (quantified in L09's label audit).
- In this repo the graph's TEACHES edges equal the catalogue field, so graph hop 1 ≡ flat skill overlap; the only graph-specific signal is the prerequisite hop.
- n = 24 test scenarios; scenarios share a catalogue and archetypes; bootstrap ignores label uncertainty.
- External E1 uses exact name matching in place of Wikipedia-miner and the annotated vocabulary as an oracle concept list.

## Reproduce

```
cd SkillLLM_v2_stage2 && PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.graph_retrieval --seed 42 --n-boot 2000
```
