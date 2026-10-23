# SkillLLM_v2 — Course Recommendation Benchmark (independent labels, bootstrap CIs)

- Scenarios: **48** (user, job) pairs; sources: {'synthetic': 41, 'synthetic_rewrite': 7}
- Catalogue: **171 SYNTHETIC courses** (`inputs/courses/course_catalog_large.jsonl`)
- Labels: graded 0-3 per (scenario, course), `inputs/courses/relevance_labels.jsonl` (v1-rule); grade totals over all pairs: {0: 7630, 1: 325, 3: 176, 2: 77}; mean #relevant (grade>=2) per scenario = 5.3
- Relevant for Precision/Recall/MAP = grade >= 2; nDCG@10 uses gain 2^grade - 1 with ideal over all judged courses.
- Values are mean over scenarios with 95% percentile-bootstrap CI (10000 resamples). Dense baseline backend: all-MiniLM-L6-v2.
- Gap coverage and redundancy are computed against the GOLD open needs for every system (not each system's own gap output).

## Main results (mean [95% CI])

| System | nDCG@10 ↑ | Precision@5 ↑ | Recall@10 ↑ | MAP@10 ↑ | Weighted Gap Coverage@5 ↑ | Prerequisite Violation Rate@5 ↓ | Redundancy@5 ↓ | Explanation Completeness@5 ↑ | #returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | 0.597 [0.544, 0.649] | 0.404 [0.346, 0.467] | 0.599 [0.526, 0.673] | 0.392 [0.330, 0.459] | 0.719 [0.663, 0.775] | 0.321 [0.250, 0.396] | 0.039 [0.029, 0.049] | 0.000 [0.000, 0.000] | 10.0 |
| BM25 + prereq filter | 0.682 [0.628, 0.732] | 0.521 [0.454, 0.588] | 0.675 [0.601, 0.748] | 0.547 [0.477, 0.618] | 0.665 [0.599, 0.729] | 0.000 [0.000, 0.000] | 0.033 [0.025, 0.041] | 0.000 [0.000, 0.000] | 10.0 |
| Dense Semantic | 0.580 [0.529, 0.631] | 0.404 [0.342, 0.467] | 0.600 [0.539, 0.664] | 0.396 [0.332, 0.463] | 0.693 [0.634, 0.751] | 0.179 [0.117, 0.246] | 0.033 [0.025, 0.041] | 0.000 [0.000, 0.000] | 10.0 |
| Dense + prereq filter | 0.629 [0.577, 0.679] | 0.463 [0.404, 0.525] | 0.627 [0.564, 0.692] | 0.478 [0.410, 0.547] | 0.680 [0.616, 0.741] | 0.000 [0.000, 0.000] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 10.0 |
| Skill Overlap (user-agnostic) | 0.760 [0.719, 0.799] | 0.513 [0.438, 0.588] | 0.770 [0.711, 0.826] | 0.518 [0.456, 0.582] | 0.778 [0.730, 0.824] | 0.196 [0.133, 0.267] | 0.061 [0.050, 0.071] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy (no graph) | 0.808 [0.772, 0.841] | 0.529 [0.463, 0.596] | 0.808 [0.747, 0.864] | 0.606 [0.545, 0.668] | 0.901 [0.875, 0.925] | 0.150 [0.096, 0.208] | 0.043 [0.034, 0.053] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy + prereq filter | 0.852 [0.826, 0.876] | 0.583 [0.517, 0.654] | 0.823 [0.765, 0.877] | 0.697 [0.646, 0.747] | 0.860 [0.810, 0.904] | 0.000 [0.000, 0.000] | 0.045 [0.035, 0.056] | 0.000 [0.000, 0.000] | 10.0 |
| **SkillLLM_v2 Full Graph Recommender** | 0.868 [0.846, 0.889] | 0.617 [0.550, 0.683] | 0.813 [0.754, 0.869] | 0.688 [0.630, 0.744] | 0.877 [0.842, 0.909] | 0.046 [0.017, 0.083] | 0.045 [0.036, 0.054] | 1.000 [1.000, 1.000] | 8.4 |

## Paired bootstrap: Full minus baseline (mean diff [95% CI], Holm-adjusted p)

Positive = Full higher. For the two lower-is-better metrics a NEGATIVE diff favours Full.

| Baseline | nDCG@10 | Precision@5 | Recall@10 | MAP@10 | Weighted Gap Coverage@5 | Prerequisite Violation Rate@5 | Redundancy@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | +0.271 [+0.220, +0.323] p=0.001 (46W/2L/0T) | +0.213 [+0.146, +0.279] p=0.001 (32W/4L/12T) | +0.214 [+0.144, +0.285] p=0.001 (30W/3L/15T) | +0.296 [+0.236, +0.354] p=0.001 (42W/4L/2T) | +0.158 [+0.104, +0.209] p=0.001 (34W/6L/8T) | -0.275 [-0.350, -0.204] p=0.001 (1W/33L/14T) | +0.006 [-0.006, +0.017] p=0.919 (25W/17L/6T) |
| BM25 + prereq filter | +0.187 [+0.136, +0.241] p=0.001 (42W/6L/0T) | +0.096 [+0.037, +0.158] p=0.005 (21W/7L/20T) | +0.138 [+0.075, +0.205] p=0.001 (23W/3L/22T) | +0.141 [+0.081, +0.203] p=0.001 (34W/11L/3T) | +0.211 [+0.154, +0.269] p=0.001 (37W/3L/8T) | +0.046 [+0.017, +0.083] p=0.002 (8W/0L/40T) | +0.012 [+0.002, +0.022] p=0.058 (25W/11L/12T) |
| Dense Semantic | +0.288 [+0.236, +0.340] p=0.001 (47W/1L/0T) | +0.212 [+0.146, +0.275] p=0.001 (31W/3L/14T) | +0.214 [+0.140, +0.283] p=0.001 (32W/5L/11T) | +0.292 [+0.221, +0.361] p=0.001 (40W/5L/3T) | +0.183 [+0.127, +0.238] p=0.001 (37W/5L/6T) | -0.133 [-0.196, -0.075] p=0.001 (3W/21L/24T) | +0.012 [+0.003, +0.022] p=0.056 (28W/12L/8T) |
| Dense + prereq filter | +0.239 [+0.188, +0.291] p=0.001 (46W/2L/0T) | +0.154 [+0.092, +0.217] p=0.001 (27W/5L/16T) | +0.186 [+0.114, +0.254] p=0.001 (31W/6L/11T) | +0.211 [+0.140, +0.279] p=0.001 (37W/7L/4T) | +0.197 [+0.143, +0.251] p=0.001 (37W/3L/8T) | +0.046 [+0.017, +0.083] p=0.002 (8W/0L/40T) | +0.012 [+0.003, +0.022] p=0.047 (28W/10L/10T) |
| Skill Overlap (user-agnostic) | +0.109 [+0.073, +0.146] p=0.001 (40W/7L/1T) | +0.104 [+0.046, +0.162] p=0.005 (25W/8L/15T) | +0.044 [+0.019, +0.073] p=0.001 (10W/0L/38T) | +0.170 [+0.117, +0.225] p=0.001 (40W/7L/1T) | +0.099 [+0.051, +0.145] p=0.001 (26W/8L/14T) | -0.150 [-0.212, -0.096] p=0.001 (0W/23L/25T) | -0.016 [-0.024, -0.008] p=0.001 (9W/26L/13T) |
| Gap-Overlap Greedy (no graph) | +0.061 [+0.030, +0.092] p=0.001 (30W/14L/4T) | +0.088 [+0.037, +0.137] p=0.005 (22W/5L/21T) | +0.005 [-0.012, +0.023] p=0.562 (5W/3L/40T) | +0.082 [+0.038, +0.126] p=0.001 (33W/10L/5T) | -0.024 [-0.057, +0.004] p=0.189 (5W/12L/31T) | -0.104 [-0.154, -0.058] p=0.001 (1W/18L/29T) | +0.001 [-0.005, +0.007] p=1.000 (16W/13L/19T) |
| Gap-Overlap Greedy + prereq filter | +0.016 [-0.004, +0.036] p=0.117 (26W/20L/2T) | +0.033 [-0.017, +0.079] p=0.205 (15W/7L/26T) | -0.009 [-0.025, +0.004] p=0.382 (2W/4L/42T) | -0.009 [-0.045, +0.025] p=0.623 (19W/19L/10T) | +0.017 [-0.022, +0.058] p=0.428 (9W/8L/31T) | +0.046 [+0.017, +0.083] p=0.002 (8W/0L/40T) | -0.000 [-0.007, +0.006] p=1.000 (14W/12L/22T) |

W/L/T = number of scenarios where Full is better / worse / tied (direction ignores whether higher is better).

## Sensitivity: labels without the prerequisite-feasibility rule

Same systems scored against content-only labels (a course is graded purely on how much open-need mass it teaches). If Full's advantage shrinks here, part of its advantage in the main table comes from the label rule rewarding prerequisite awareness.

| System | nDCG@10 (content-only) | Precision@5 (content-only) |
| --- | --- | --- |
| BM25 Keyword | 0.683 [0.630, 0.730] | 0.625 [0.558, 0.688] |
| BM25 + prereq filter | 0.598 [0.545, 0.650] | 0.571 [0.504, 0.638] |
| Dense Semantic | 0.628 [0.575, 0.680] | 0.575 [0.504, 0.646] |
| Dense + prereq filter | 0.582 [0.531, 0.633] | 0.558 [0.488, 0.629] |
| Skill Overlap (user-agnostic) | 0.837 [0.809, 0.865] | 0.717 [0.646, 0.783] |
| Gap-Overlap Greedy (no graph) | 0.895 [0.877, 0.913] | 0.763 [0.708, 0.813] |
| Gap-Overlap Greedy + prereq filter | 0.824 [0.785, 0.860] | 0.733 [0.671, 0.796] |
| SkillLLM_v2 Full Graph Recommender | 0.859 [0.838, 0.879] | 0.717 [0.662, 0.771] |

## Key Findings (generated from the numbers above)

- **nDCG@10**: Full = 0.868 [0.846, 0.889]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph). Not distinguishable (Holm p >= 0.05 or CI spans 0) from: Gap-Overlap Greedy + prereq filter.
- **Precision@5**: Full = 0.617 [0.550, 0.683]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph). Not distinguishable (Holm p >= 0.05 or CI spans 0) from: Gap-Overlap Greedy + prereq filter.
- **Recall@10**: Full = 0.813 [0.754, 0.869]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic). Not distinguishable (Holm p >= 0.05 or CI spans 0) from: Gap-Overlap Greedy (no graph), Gap-Overlap Greedy + prereq filter.
- **MAP@10**: Full = 0.688 [0.630, 0.744]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph). Not distinguishable (Holm p >= 0.05 or CI spans 0) from: Gap-Overlap Greedy + prereq filter.
- Strongest baseline by nDCG@10 is **Gap-Overlap Greedy + prereq filter** (0.852 [0.826, 0.876]); Full minus it = +0.016 (95% CI [-0.004, +0.036], Holm p = 0.117).
- **What the graph adds beyond a user-aware, prerequisite-aware, graph-free baseline** (Gap-Overlap Greedy + prereq filter): Full minus baseline: nDCG@10 +0.016 (p_holm=0.117, indistinguishable); Precision@5 +0.033 (p_holm=0.205, indistinguishable); Recall@10 -0.009 (p_holm=0.382, indistinguishable); MAP@10 -0.009 (p_holm=0.623, indistinguishable); Weighted Gap Coverage@5 +0.017 (p_holm=0.428, indistinguishable); Redundancy@5 -0.000 (p_holm=1.000, indistinguishable).
- **Prerequisite violations**: Full = 0.046 [0.017, 0.083]; unfiltered text/overlap baselines = BM25 Keyword 0.321, Dense Semantic 0.179, Skill Overlap (user-agnostic) 0.196, Gap-Overlap Greedy (no graph) 0.150. Filtered variants: BM25 + prereq filter 0.000, Dense + prereq filter 0.000, Gap-Overlap Greedy + prereq filter 0.000. Because the filtered baselines apply the same rule the metric checks, a 0 violation rate for them is by construction, not a finding; the finding is that Full does not reach 0.
- **Where Full loses (significant after Holm correction)**: Prerequisite Violation Rate@5 vs BM25 + prereq filter (diff +0.046, Holm p=0.002); Prerequisite Violation Rate@5 vs Dense + prereq filter (diff +0.046, Holm p=0.002); Prerequisite Violation Rate@5 vs Gap-Overlap Greedy + prereq filter (diff +0.046, Holm p=0.002); Redundancy@5 vs Dense + prereq filter (diff +0.012, Holm p=0.047).
- **Baselines with a nominally better mean than Full but not significantly** (do not read as ties in favour of Full): Recall@10: Gap-Overlap Greedy + prereq filter (0.823 vs 0.813); MAP@10: Gap-Overlap Greedy + prereq filter (0.697 vs 0.688); Weighted Gap Coverage@5: Gap-Overlap Greedy (no graph) (0.901 vs 0.877); Redundancy@5: BM25 Keyword (0.039 vs 0.045); Redundancy@5: BM25 + prereq filter (0.033 vs 0.045); Redundancy@5: Dense Semantic (0.033 vs 0.045); Redundancy@5: Gap-Overlap Greedy (no graph) (0.043 vs 0.045).
- **Redundancy@5** (lower is better): Full 0.045 [0.036, 0.054]; lowest baseline mean is Dense + prereq filter 0.032 [0.024, 0.041] (Full minus it +0.012, Holm p=0.047). Full is significantly WORSE (more redundant) than: Dense + prereq filter; significantly better than: Skill Overlap (user-agnostic). Absolute values are small (a few hundredths) for every system, so the practical difference is modest; but Full does not 'win' on redundancy.
- **Explanation completeness** is 1.0 for Full and 0.0 for all baselines essentially by construction (baselines emit no evidence/provenance fields); it measures output format, not recommendation quality.
- **List length**: Full returned 8.4 courses on average (baselines always return 10); it stops when no further candidate covers a gap, which caps its Recall@10 and nDCG@10 relative to systems that always fill the list.
- **Sensitivity to the prerequisite rule in the labels** (nDCG@10 against content-only labels with the feasibility rule removed): Full 0.859; BM25 Keyword 0.683 (Full minus it +0.176, Holm p=0.001, better); BM25 + prereq filter 0.598 (Full minus it +0.261, Holm p=0.001, better); Dense Semantic 0.628 (Full minus it +0.231, Holm p=0.001, better); Dense + prereq filter 0.582 (Full minus it +0.277, Holm p=0.001, better); Skill Overlap (user-agnostic) 0.837 (Full minus it +0.021, Holm p=0.116, indistinguishable); Gap-Overlap Greedy (no graph) 0.895 (Full minus it -0.036, Holm p=0.002, worse); Gap-Overlap Greedy + prereq filter 0.824 (Full minus it +0.035, Holm p=0.072, indistinguishable).
- **Effect of adding the prerequisite filter to a baseline, by label variant (nDCG@10)**: BM25 + prereq filter minus BM25 Keyword: +0.084 with the rule, -0.084 without; Dense + prereq filter minus Dense Semantic: +0.049 with the rule, -0.046 without; Gap-Overlap Greedy + prereq filter minus Gap-Overlap Greedy (no graph): +0.044 with the rule, -0.071 without. A sign change means the benefit of prerequisite filtering is (at least partly) induced by the label rule rewarding it.

## Caveats and limits

- **Synthetic catalogue.** All 171 courses are synthetic (hand-authored structure, templated outcomes, fictional providers); results say nothing about real MOOCs. Skill names come from a vocabulary checked against the project taxonomy and 1,887 real job postings, but course content, prerequisite chains and durations are author inventions.
- **Scenario provenance.** All 48 scenarios are synthetic: 41 archetypes plus 7 (sc-42..sc-48) whose job text is a synthetic rewrite written from authored skill lists (an earlier version used real-posting excerpts, removed for privacy). User profiles and skill annotations are authored. No real learner data.
- **Label procedure limits.** Labels come from a deterministic rule over hand-authored gold needs and catalogue skill/prerequisite fields (see build_relevance_labels.py). They do not use any system's output, but they share the catalogue's skill/prerequisite metadata with the skill-aware systems, so skill-aware systems (Skill Overlap, Gap-Overlap, Graph) are structurally favoured over text-only retrieval. The thresholds, the grade cap for bridge-needed courses, and the lenient treatment of weak skills are author choices. Human validation of the stratified sample (reports/skillllm_v2/human_review_sample.csv) has NOT been done yet; until then the absolute numbers are unvalidated.
- **Metrics on near-duplicate courses.** The catalogue deliberately contains overlapping courses (e.g. a crash course and a full course on the same skill). Graded nDCG does not penalise recommending several equally relevant duplicates; Redundancy@5 is reported separately for that reason.
- **Statistics.** n = 48 scenarios, not independent (shared catalogue, overlapping archetypes). Bootstrap CIs resample scenarios only; they ignore label uncertainty. Holm correction is applied across the 7 baseline comparisons within each metric, not across metrics.
- **Not an ablation of the system's components.** Full is compared as a black box; its unexplained behaviours (e.g. how many items it returns, taxonomy-linker canonicalisation quirks such as 'Scikit-Learn' -> 'learn') are not isolated here.
- **Dense baseline** uses all-MiniLM-L6-v2 off the shelf with no tuning; BM25 uses default k1=1.5, b=0.75 on job title + posting text. Text baselines are user-agnostic by design (they see only the job), which disadvantages them on personalised labels; Gap-Overlap Greedy is the user-aware baseline.

## Reproduction

```
PYTHONPATH=src python scripts/build_course_catalog_large.py            # synthetic catalogue
PYTHONPATH=src python -m skillllm_v2.eval.build_relevance_labels         # scenarios + labels + human-review sample
PYTHONPATH=src python -m skillllm_v2.eval.recommender_benchmark          # this report
```
