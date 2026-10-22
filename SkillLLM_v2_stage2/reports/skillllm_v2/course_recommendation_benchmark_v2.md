# SkillLLM_v2 - Course Recommendation Benchmark v2: tuned Full recommender on the held-out test half

**Headline (held-out test half, 24 scenarios, config frozen before this run):** tuned Full nDCG@10 = 0.885 [0.851, 0.916]; it is NOT significantly different from the strongest baseline (Gap-Overlap Greedy + prereq filter, 0.851 [0.808, 0.890]): diff +0.035 [-0.006, +0.072], Holm p = 0.094. Prerequisite violation rate 0.000 (legacy 0.067), avg list length 9.88. Everything in the 'dev' and 'all 48' sections is partly or wholly tuned-on data and is shown only for continuity with the earlier benchmark.

- Split: 24 dev / 24 test, fixed by sha256-sorted scenario ids before any tuning (`reports/skillllm_v2/recommender_split.json`). Tuning used dev only (`recommender_loop_log.md`; 5 rounds, one change each; round 4 rejected).
- Frozen Full config = `DEFAULT_CONFIG` == `recommender_loop_log.json` final_config. Differences from the legacy config: `{"w_prereq": 0.3, "w_redundancy": 0.1, "w_unlock": 0.2, "retrieval_pool": 40, "prereq_mode": "strict_bridge", "compat_basis": "learner", "fill_related": true, "related_match": "token", "redundancy_mode": "mmr"}`.
- Catalogue: 171 SYNTHETIC courses; labels `inputs/courses/relevance_labels.jsonl` (v1-rule), relevant = grade >= 2, nDCG@10 gain 2^g-1; baselines, labels and metric code unchanged. 10000 bootstrap resamples, 95% percentile CIs over scenarios; Holm correction over the 7 baseline comparisons per metric. Dense backend: all-MiniLM-L6-v2.
- No gold/relevance label is an input of the recommender (the tuned system reads user profile, job requirements, catalogue fields and the graph only); no scenario id is special-cased.

## 1. Held-out TEST half (primary result)

### Test half (n = 24 scenarios)

| System | nDCG@10 ↑ | Precision@5 ↑ | Recall@10 ↑ | MAP@10 ↑ | Weighted Gap Coverage@5 ↑ | Prerequisite Violation Rate@5 ↓ | Redundancy@5 ↓ | Explanation Completeness@5 ↑ | #returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | 0.598 [0.532, 0.661] | 0.383 [0.292, 0.475] | 0.609 [0.512, 0.708] | 0.381 [0.304, 0.464] | 0.711 [0.643, 0.779] | 0.300 [0.200, 0.408] | 0.033 [0.021, 0.045] | 0.000 [0.000, 0.000] | 10.0 |
| BM25 + prereq filter | 0.695 [0.629, 0.756] | 0.542 [0.433, 0.650] | 0.711 [0.609, 0.809] | 0.553 [0.458, 0.652] | 0.693 [0.606, 0.777] | 0.000 [0.000, 0.000] | 0.036 [0.024, 0.049] | 0.000 [0.000, 0.000] | 10.0 |
| Dense Semantic | 0.591 [0.513, 0.663] | 0.417 [0.308, 0.525] | 0.617 [0.523, 0.714] | 0.413 [0.328, 0.501] | 0.688 [0.587, 0.779] | 0.175 [0.092, 0.267] | 0.035 [0.024, 0.048] | 0.000 [0.000, 0.000] | 10.0 |
| Dense + prereq filter | 0.643 [0.568, 0.711] | 0.475 [0.375, 0.567] | 0.656 [0.571, 0.743] | 0.500 [0.407, 0.592] | 0.674 [0.572, 0.768] | 0.000 [0.000, 0.000] | 0.034 [0.022, 0.048] | 0.000 [0.000, 0.000] | 10.0 |
| Skill Overlap (user-agnostic) | 0.759 [0.693, 0.820] | 0.475 [0.350, 0.600] | 0.752 [0.666, 0.833] | 0.515 [0.416, 0.610] | 0.791 [0.720, 0.860] | 0.250 [0.142, 0.367] | 0.058 [0.043, 0.074] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy (no graph) | 0.819 [0.764, 0.870] | 0.508 [0.400, 0.617] | 0.802 [0.713, 0.882] | 0.610 [0.525, 0.692] | 0.887 [0.850, 0.924] | 0.167 [0.083, 0.258] | 0.046 [0.031, 0.061] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy + prereq filter | 0.851 [0.808, 0.890] | 0.550 [0.450, 0.650] | 0.817 [0.729, 0.898] | 0.695 [0.623, 0.765] | 0.837 [0.757, 0.905] | 0.000 [0.000, 0.000] | 0.043 [0.029, 0.057] | 0.000 [0.000, 0.000] | 10.0 |
| **SkillLLM_v2 Full Graph Recommender** | 0.885 [0.851, 0.916] | 0.608 [0.533, 0.683] | 0.878 [0.804, 0.942] | 0.732 [0.665, 0.796] | 0.838 [0.771, 0.900] | 0.000 [0.000, 0.000] | 0.036 [0.026, 0.047] | 1.000 [1.000, 1.000] | 9.9 |
| SkillLLM_v2 Full (legacy, pre-tuning) | 0.855 [0.820, 0.889] | 0.600 [0.500, 0.700] | 0.811 [0.724, 0.890] | 0.670 [0.589, 0.748] | 0.882 [0.833, 0.930] | 0.067 [0.017, 0.133] | 0.044 [0.031, 0.057] | 1.000 [1.000, 1.000] | 8.3 |

Paired bootstrap, tuned Full minus baseline (mean diff [95% CI], Holm p over the 7 baselines, W/L/T of scenarios). For the two lower-is-better metrics a negative diff favours Full.

| Baseline | nDCG@10 | Precision@5 | Recall@10 | MAP@10 | Weighted Gap Coverage@5 | Prerequisite Violation Rate@5 | Redundancy@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | +0.287 [+0.216, +0.357] p=0.001 (22W/2L/0T) | +0.225 [+0.150, +0.300] p=0.001 (18W/1L/5T) | +0.269 [+0.156, +0.373] p=0.001 (19W/1L/4T) | +0.351 [+0.261, +0.439] p=0.001 (22W/2L/0T) | +0.127 [+0.051, +0.201] p=0.007 (14W/5L/5T) | -0.300 [-0.400, -0.200] p=0.001 (0W/17L/7T) | +0.003 [-0.012, +0.019] p=1.000 (10W/7L/7T) |
| BM25 + prereq filter | +0.191 [+0.128, +0.252] p=0.001 (20W/4L/0T) | +0.067 [-0.017, +0.142] p=0.129 (10W/4L/10T) | +0.167 [+0.067, +0.263] p=0.003 (14W/2L/8T) | +0.179 [+0.094, +0.263] p=0.001 (17W/5L/2T) | +0.146 [+0.065, +0.221] p=0.007 (17W/2L/5T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/24T) | -0.000 [-0.014, +0.014] p=1.000 (10W/6L/8T) |
| Dense Semantic | +0.295 [+0.226, +0.368] p=0.001 (24W/0L/0T) | +0.192 [+0.108, +0.283] p=0.001 (13W/1L/10T) | +0.261 [+0.152, +0.362] p=0.001 (18W/1L/5T) | +0.319 [+0.233, +0.403] p=0.001 (21W/2L/1T) | +0.150 [+0.068, +0.239] p=0.002 (16W/4L/4T) | -0.175 [-0.267, -0.092] p=0.001 (0W/12L/12T) | +0.000 [-0.012, +0.013] p=1.000 (10W/7L/7T) |
| Dense + prereq filter | +0.243 [+0.176, +0.315] p=0.001 (24W/0L/0T) | +0.133 [+0.058, +0.217] p=0.002 (12W/2L/10T) | +0.223 [+0.117, +0.319] p=0.001 (17W/2L/5T) | +0.233 [+0.153, +0.315] p=0.001 (19W/3L/2T) | +0.164 [+0.082, +0.250] p=0.001 (18W/3L/3T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/24T) | +0.001 [-0.014, +0.016] p=1.000 (10W/7L/7T) |
| Skill Overlap (user-agnostic) | +0.126 [+0.064, +0.192] p=0.001 (18W/5L/1T) | +0.133 [+0.042, +0.225] p=0.006 (10W/2L/12T) | +0.126 [+0.057, +0.200] p=0.002 (11W/2L/11T) | +0.217 [+0.121, +0.318] p=0.001 (18W/4L/2T) | +0.047 [-0.030, +0.126] p=0.486 (9W/8L/7T) | -0.250 [-0.367, -0.142] p=0.001 (0W/14L/10T) | -0.022 [-0.033, -0.012] p=0.001 (2W/16L/6T) |
| Gap-Overlap Greedy (no graph) | +0.066 [+0.007, +0.128] p=0.055 (17W/6L/1T) | +0.100 [+0.025, +0.175] p=0.029 (12W/3L/9T) | +0.076 [-0.001, +0.153] p=0.104 (9W/2L/13T) | +0.122 [+0.033, +0.218] p=0.011 (19W/3L/2T) | -0.049 [-0.110, +0.007] p=0.274 (4W/10L/10T) | -0.167 [-0.267, -0.083] p=0.001 (0W/10L/14T) | -0.010 [-0.019, -0.001] p=0.216 (5W/11L/8T) |
| Gap-Overlap Greedy + prereq filter | +0.035 [-0.006, +0.072] p=0.094 (19W/4L/1T) | +0.058 [-0.000, +0.117] p=0.129 (11W/4L/9T) | +0.061 [-0.014, +0.138] p=0.118 (7W/2L/15T) | +0.037 [-0.024, +0.092] p=0.218 (16W/5L/3T) | +0.001 [-0.056, +0.060] p=0.973 (5W/6L/13T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/24T) | -0.007 [-0.016, +0.002] p=0.644 (5W/8L/11T) |

Tuned Full minus legacy (pre-tuning) Full, paired (raw p, no multiplicity correction): nDCG@10 +0.030 [+0.007, +0.055] p=0.005; Precision@5 +0.008 [-0.033, +0.050] p=0.771; Recall@10 +0.067 [+0.002, +0.139] p=0.043; MAP@10 +0.062 [+0.012, +0.117] p=0.012; Weighted Gap Coverage@5 -0.044 [-0.086, -0.008] p=0.016; Prerequisite Violation Rate@5 -0.067 [-0.133, -0.017] p=0.008; Redundancy@5 -0.008 [-0.015, -0.002] p=0.005; Avg #returned +1.583 [+1.042, +2.167] p=0.000

#### Findings (test half)

- **nDCG@10** (test half): tuned Full = 0.885 [0.851, 0.916]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic). Not distinguishable from: Gap-Overlap Greedy (no graph), Gap-Overlap Greedy + prereq filter.
- **Precision@5** (test half): tuned Full = 0.608 [0.533, 0.683]. Significantly higher than: BM25 Keyword, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph). Not distinguishable from: BM25 + prereq filter, Gap-Overlap Greedy + prereq filter.
- **Recall@10** (test half): tuned Full = 0.878 [0.804, 0.942]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic). Not distinguishable from: Gap-Overlap Greedy (no graph), Gap-Overlap Greedy + prereq filter.
- **MAP@10** (test half): tuned Full = 0.732 [0.665, 0.796]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph). Not distinguishable from: Gap-Overlap Greedy + prereq filter.
- Strongest-baseline comparison (pre-specified) **Gap-Overlap Greedy + prereq filter** (0.851 [0.808, 0.890]): Full minus it = +0.035 (95% CI [-0.006, +0.072], Holm p = 0.094) -> NOT a significant difference.
- **Where Full is significantly worse** (test half): none at the 0.05 level.
- Baselines with a nominally better mean than Full but not significantly so: Weighted Gap Coverage@5: Gap-Overlap Greedy (no graph) (0.887 vs 0.838); Redundancy@5: BM25 Keyword (0.033 vs 0.036); Redundancy@5: Dense Semantic (0.035 vs 0.036); Redundancy@5: Dense + prereq filter (0.034 vs 0.036).
- Prerequisite Violation Rate@5 = 0.000 [0.000, 0.000]; average list length = 9.88; Explanation Completeness@5 = 1.000 (1.0 for Full and 0.0 for baselines by construction: baselines emit no explanation fields; a 0 violation rate for the filtered baselines is also by construction).
- Content-only-label sensitivity (nDCG@10): vs BM25 Keyword +0.165 (p_holm=0.001, better); vs BM25 + prereq filter +0.229 (p_holm=0.001, better); vs Dense Semantic +0.200 (p_holm=0.001, better); vs Dense + prereq filter +0.245 (p_holm=0.001, better); vs Skill Overlap (user-agnostic) -0.014 (p_holm=0.894, indistinguishable); vs Gap-Overlap Greedy (no graph) -0.069 (p_holm=0.002, worse); vs Gap-Overlap Greedy + prereq filter +0.023 (p_holm=0.894, indistinguishable).

## 2. DEV half (the half that was tuned on - optimistic by construction)

### Dev half (n = 24 scenarios)

| System | nDCG@10 ↑ | Precision@5 ↑ | Recall@10 ↑ | MAP@10 ↑ | Weighted Gap Coverage@5 ↑ | Prerequisite Violation Rate@5 ↓ | Redundancy@5 ↓ | Explanation Completeness@5 ↑ | #returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | 0.565 [0.470, 0.649] | 0.383 [0.300, 0.467] | 0.550 [0.439, 0.661] | 0.368 [0.273, 0.469] | 0.690 [0.579, 0.793] | 0.350 [0.250, 0.458] | 0.039 [0.024, 0.055] | 0.000 [0.000, 0.000] | 10.0 |
| BM25 + prereq filter | 0.643 [0.547, 0.728] | 0.467 [0.383, 0.550] | 0.612 [0.499, 0.722] | 0.503 [0.399, 0.605] | 0.618 [0.508, 0.718] | 0.000 [0.000, 0.000] | 0.025 [0.015, 0.036] | 0.000 [0.000, 0.000] | 10.0 |
| Dense Semantic | 0.540 [0.449, 0.628] | 0.375 [0.300, 0.442] | 0.548 [0.456, 0.644] | 0.365 [0.266, 0.478] | 0.669 [0.570, 0.757] | 0.158 [0.075, 0.258] | 0.031 [0.021, 0.041] | 0.000 [0.000, 0.000] | 10.0 |
| Dense + prereq filter | 0.588 [0.498, 0.671] | 0.425 [0.350, 0.500] | 0.571 [0.476, 0.671] | 0.439 [0.342, 0.541] | 0.634 [0.532, 0.730] | 0.000 [0.000, 0.000] | 0.031 [0.021, 0.041] | 0.000 [0.000, 0.000] | 10.0 |
| Skill Overlap (user-agnostic) | 0.760 [0.710, 0.808] | 0.550 [0.467, 0.633] | 0.787 [0.707, 0.861] | 0.521 [0.445, 0.598] | 0.765 [0.702, 0.824] | 0.142 [0.083, 0.208] | 0.063 [0.049, 0.077] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy (no graph) | 0.796 [0.750, 0.839] | 0.550 [0.467, 0.633] | 0.813 [0.728, 0.889] | 0.603 [0.512, 0.692] | 0.914 [0.881, 0.946] | 0.133 [0.075, 0.200] | 0.041 [0.030, 0.053] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy + prereq filter | 0.853 [0.822, 0.883] | 0.617 [0.525, 0.708] | 0.828 [0.749, 0.898] | 0.699 [0.627, 0.770] | 0.882 [0.820, 0.935] | 0.000 [0.000, 0.000] | 0.048 [0.034, 0.063] | 0.000 [0.000, 0.000] | 10.0 |
| **SkillLLM_v2 Full Graph Recommender** | 0.902 [0.884, 0.921] | 0.617 [0.550, 0.692] | 0.879 [0.830, 0.925] | 0.746 [0.682, 0.811] | 0.863 [0.808, 0.911] | 0.000 [0.000, 0.000] | 0.036 [0.028, 0.044] | 1.000 [1.000, 1.000] | 9.9 |
| SkillLLM_v2 Full (legacy, pre-tuning) | 0.881 [0.856, 0.905] | 0.633 [0.550, 0.725] | 0.815 [0.732, 0.888] | 0.706 [0.626, 0.784] | 0.871 [0.823, 0.914] | 0.025 [0.000, 0.058] | 0.046 [0.035, 0.058] | 1.000 [1.000, 1.000] | 8.4 |

Paired bootstrap, tuned Full minus baseline (mean diff [95% CI], Holm p over the 7 baselines, W/L/T of scenarios). For the two lower-is-better metrics a negative diff favours Full.

| Baseline | nDCG@10 | Precision@5 | Recall@10 | MAP@10 | Weighted Gap Coverage@5 | Prerequisite Violation Rate@5 | Redundancy@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | +0.338 [+0.254, +0.430] p=0.001 (24W/0L/0T) | +0.233 [+0.125, +0.342] p=0.001 (16W/2L/6T) | +0.329 [+0.234, +0.430] p=0.001 (19W/0L/5T) | +0.378 [+0.290, +0.467] p=0.001 (22W/1L/1T) | +0.172 [+0.064, +0.286] p=0.009 (15W/5L/4T) | -0.350 [-0.458, -0.242] p=0.001 (0W/19L/5T) | -0.003 [-0.020, +0.013] p=1.000 (11W/9L/4T) |
| BM25 + prereq filter | +0.259 [+0.174, +0.355] p=0.001 (22W/2L/0T) | +0.150 [+0.042, +0.258] p=0.015 (13W/3L/8T) | +0.266 [+0.173, +0.371] p=0.001 (18W/0L/6T) | +0.243 [+0.155, +0.340] p=0.001 (20W/3L/1T) | +0.245 [+0.149, +0.349] p=0.001 (20W/1L/3T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/24T) | +0.011 [-0.000, +0.022] p=0.314 (13W/6L/5T) |
| Dense Semantic | +0.362 [+0.277, +0.451] p=0.001 (24W/0L/0T) | +0.242 [+0.142, +0.342] p=0.001 (19W/2L/3T) | +0.330 [+0.245, +0.415] p=0.001 (20W/0L/4T) | +0.380 [+0.283, +0.479] p=0.001 (21W/1L/2T) | +0.194 [+0.099, +0.293] p=0.001 (17W/2L/5T) | -0.158 [-0.258, -0.075] p=0.001 (0W/11L/13T) | +0.005 [-0.006, +0.016] p=1.000 (12W/7L/5T) |
| Dense + prereq filter | +0.314 [+0.231, +0.405] p=0.001 (24W/0L/0T) | +0.192 [+0.092, +0.292] p=0.002 (16W/2L/6T) | +0.307 [+0.224, +0.393] p=0.001 (19W/0L/5T) | +0.307 [+0.217, +0.403] p=0.001 (21W/1L/2T) | +0.229 [+0.151, +0.317] p=0.001 (19W/1L/4T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/24T) | +0.005 [-0.005, +0.015] p=1.000 (11W/8L/5T) |
| Skill Overlap (user-agnostic) | +0.142 [+0.095, +0.189] p=0.001 (22W/2L/0T) | +0.067 [-0.017, +0.142] p=0.282 (13W/4L/7T) | +0.092 [+0.048, +0.141] p=0.001 (11W/0L/13T) | +0.225 [+0.152, +0.298] p=0.001 (21W/3L/0T) | +0.098 [+0.024, +0.169] p=0.026 (14W/5L/5T) | -0.142 [-0.208, -0.083] p=0.001 (0W/13L/11T) | -0.028 [-0.043, -0.012] p=0.001 (4W/15L/5T) |
| Gap-Overlap Greedy (no graph) | +0.106 [+0.066, +0.147] p=0.001 (19W/3L/2T) | +0.067 [-0.008, +0.133] p=0.270 (12W/4L/8T) | +0.065 [+0.013, +0.122] p=0.024 (7W/1L/16T) | +0.143 [+0.076, +0.207] p=0.001 (19W/3L/2T) | -0.052 [-0.103, -0.007] p=0.036 (3W/9L/12T) | -0.133 [-0.200, -0.075] p=0.001 (0W/12L/12T) | -0.006 [-0.016, +0.005] p=1.000 (7W/12L/5T) |
| Gap-Overlap Greedy + prereq filter | +0.049 [+0.018, +0.081] p=0.003 (16W/7L/1T) | +0.000 [-0.083, +0.075] p=1.000 (7W/6L/11T) | +0.051 [+0.007, +0.097] p=0.024 (7W/1L/16T) | +0.046 [-0.016, +0.098] p=0.126 (16W/6L/2T) | -0.020 [-0.065, +0.020] p=0.368 (3W/7L/14T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/24T) | -0.012 [-0.027, +0.001] p=0.337 (5W/12L/7T) |

Tuned Full minus legacy (pre-tuning) Full, paired (raw p, no multiplicity correction): nDCG@10 +0.021 [+0.002, +0.040] p=0.028; Precision@5 -0.017 [-0.058, +0.025] p=0.440; Recall@10 +0.063 [+0.019, +0.115] p=0.003; MAP@10 +0.040 [+0.002, +0.078] p=0.039; Weighted Gap Coverage@5 -0.008 [-0.042, +0.018] p=0.646; Prerequisite Violation Rate@5 -0.025 [-0.050, +0.000] p=0.084; Redundancy@5 -0.010 [-0.021, -0.003] p=0.004; Avg #returned +1.458 [+0.875, +2.083] p=0.000

## 3. All 48 scenarios (continuity with course_recommendation_benchmark.md; PARTLY DEV-TUNED, do not read as a held-out result)

### All 48 (n = 48 scenarios)

| System | nDCG@10 ↑ | Precision@5 ↑ | Recall@10 ↑ | MAP@10 ↑ | Weighted Gap Coverage@5 ↑ | Prerequisite Violation Rate@5 ↓ | Redundancy@5 ↓ | Explanation Completeness@5 ↑ | #returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | 0.581 [0.525, 0.634] | 0.383 [0.321, 0.446] | 0.579 [0.505, 0.655] | 0.374 [0.313, 0.438] | 0.701 [0.634, 0.765] | 0.325 [0.250, 0.400] | 0.036 [0.026, 0.046] | 0.000 [0.000, 0.000] | 10.0 |
| BM25 + prereq filter | 0.669 [0.612, 0.722] | 0.504 [0.438, 0.575] | 0.662 [0.585, 0.737] | 0.528 [0.458, 0.601] | 0.655 [0.584, 0.723] | 0.000 [0.000, 0.000] | 0.030 [0.022, 0.039] | 0.000 [0.000, 0.000] | 10.0 |
| Dense Semantic | 0.565 [0.505, 0.624] | 0.396 [0.333, 0.462] | 0.583 [0.514, 0.652] | 0.389 [0.323, 0.459] | 0.678 [0.607, 0.743] | 0.167 [0.108, 0.233] | 0.033 [0.025, 0.041] | 0.000 [0.000, 0.000] | 10.0 |
| Dense + prereq filter | 0.615 [0.556, 0.671] | 0.450 [0.388, 0.513] | 0.613 [0.546, 0.681] | 0.469 [0.401, 0.539] | 0.654 [0.583, 0.722] | 0.000 [0.000, 0.000] | 0.033 [0.024, 0.042] | 0.000 [0.000, 0.000] | 10.0 |
| Skill Overlap (user-agnostic) | 0.760 [0.719, 0.799] | 0.513 [0.438, 0.588] | 0.770 [0.711, 0.826] | 0.518 [0.456, 0.582] | 0.778 [0.730, 0.824] | 0.196 [0.133, 0.267] | 0.061 [0.050, 0.071] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy (no graph) | 0.808 [0.772, 0.841] | 0.529 [0.463, 0.596] | 0.808 [0.747, 0.864] | 0.606 [0.545, 0.668] | 0.901 [0.875, 0.925] | 0.150 [0.096, 0.208] | 0.043 [0.034, 0.053] | 0.000 [0.000, 0.000] | 10.0 |
| Gap-Overlap Greedy + prereq filter | 0.852 [0.826, 0.876] | 0.583 [0.517, 0.654] | 0.823 [0.765, 0.877] | 0.697 [0.646, 0.747] | 0.860 [0.810, 0.904] | 0.000 [0.000, 0.000] | 0.045 [0.035, 0.056] | 0.000 [0.000, 0.000] | 10.0 |
| **SkillLLM_v2 Full Graph Recommender** | 0.894 [0.874, 0.912] | 0.612 [0.558, 0.667] | 0.878 [0.836, 0.920] | 0.739 [0.693, 0.786] | 0.850 [0.807, 0.890] | 0.000 [0.000, 0.000] | 0.036 [0.029, 0.043] | 1.000 [1.000, 1.000] | 9.9 |
| SkillLLM_v2 Full (legacy, pre-tuning) | 0.868 [0.846, 0.889] | 0.617 [0.550, 0.683] | 0.813 [0.754, 0.869] | 0.688 [0.630, 0.744] | 0.877 [0.842, 0.909] | 0.046 [0.017, 0.083] | 0.045 [0.036, 0.054] | 1.000 [1.000, 1.000] | 8.4 |

Paired bootstrap, tuned Full minus baseline (mean diff [95% CI], Holm p over the 7 baselines, W/L/T of scenarios). For the two lower-is-better metrics a negative diff favours Full.

| Baseline | nDCG@10 | Precision@5 | Recall@10 | MAP@10 | Weighted Gap Coverage@5 | Prerequisite Violation Rate@5 | Redundancy@5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 Keyword | +0.312 [+0.257, +0.371] p=0.001 (46W/2L/0T) | +0.229 [+0.163, +0.296] p=0.001 (34W/3L/11T) | +0.299 [+0.225, +0.374] p=0.001 (38W/1L/9T) | +0.364 [+0.301, +0.427] p=0.001 (44W/3L/1T) | +0.150 [+0.082, +0.219] p=0.001 (29W/10L/9T) | -0.325 [-0.400, -0.250] p=0.001 (0W/36L/12T) | -0.000 [-0.011, +0.011] p=1.000 (21W/16L/11T) |
| BM25 + prereq filter | +0.225 [+0.171, +0.284] p=0.001 (42W/6L/0T) | +0.108 [+0.042, +0.175] p=0.006 (23W/7L/18T) | +0.217 [+0.146, +0.291] p=0.001 (32W/2L/14T) | +0.211 [+0.147, +0.276] p=0.001 (37W/8L/3T) | +0.195 [+0.132, +0.262] p=0.001 (37W/3L/8T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/48T) | +0.005 [-0.003, +0.014] p=0.927 (23W/12L/13T) |
| Dense Semantic | +0.328 [+0.272, +0.387] p=0.001 (48W/0L/0T) | +0.217 [+0.150, +0.283] p=0.001 (32W/3L/13T) | +0.296 [+0.226, +0.362] p=0.001 (38W/1L/9T) | +0.350 [+0.282, +0.415] p=0.001 (42W/3L/3T) | +0.172 [+0.107, +0.239] p=0.001 (33W/6L/9T) | -0.167 [-0.233, -0.108] p=0.001 (0W/23L/25T) | +0.003 [-0.006, +0.011] p=1.000 (22W/14L/12T) |
| Dense + prereq filter | +0.279 [+0.224, +0.336] p=0.001 (48W/0L/0T) | +0.163 [+0.100, +0.225] p=0.001 (28W/4L/16T) | +0.265 [+0.197, +0.331] p=0.001 (36W/2L/10T) | +0.270 [+0.207, +0.332] p=0.001 (40W/4L/4T) | +0.196 [+0.137, +0.257] p=0.001 (37W/4L/7T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/48T) | +0.003 [-0.006, +0.012] p=1.000 (21W/15L/12T) |
| Skill Overlap (user-agnostic) | +0.134 [+0.095, +0.174] p=0.001 (40W/7L/1T) | +0.100 [+0.037, +0.158] p=0.006 (23W/6L/19T) | +0.109 [+0.067, +0.153] p=0.001 (22W/2L/24T) | +0.221 [+0.161, +0.281] p=0.001 (39W/7L/2T) | +0.072 [+0.019, +0.127] p=0.018 (23W/13L/12T) | -0.196 [-0.267, -0.133] p=0.001 (0W/27L/21T) | -0.025 [-0.035, -0.016] p=0.001 (6W/31L/11T) |
| Gap-Overlap Greedy (no graph) | +0.086 [+0.050, +0.123] p=0.001 (36W/9L/3T) | +0.083 [+0.029, +0.133] p=0.006 (24W/7L/17T) | +0.070 [+0.025, +0.117] p=0.008 (16W/3L/29T) | +0.132 [+0.077, +0.189] p=0.001 (38W/6L/4T) | -0.050 [-0.089, -0.014] p=0.013 (7W/19L/22T) | -0.150 [-0.208, -0.096] p=0.001 (0W/22L/26T) | -0.008 [-0.015, -0.001] p=0.158 (12W/23L/13T) |
| Gap-Overlap Greedy + prereq filter | +0.042 [+0.016, +0.066] p=0.002 (35W/11L/2T) | +0.029 [-0.025, +0.075] p=0.287 (18W/10L/20T) | +0.056 [+0.013, +0.100] p=0.012 (14W/3L/31T) | +0.042 [-0.002, +0.080] p=0.058 (32W/11L/5T) | -0.009 [-0.045, +0.027] p=0.589 (8W/13L/27T) | +0.000 [+0.000, +0.000] p=1.000 (0W/0L/48T) | -0.010 [-0.018, -0.002] p=0.110 (10W/20L/18T) |

Tuned Full minus legacy (pre-tuning) Full, paired (raw p, no multiplicity correction): nDCG@10 +0.026 [+0.010, +0.041] p=0.001; Precision@5 -0.004 [-0.033, +0.025] p=0.785; Recall@10 +0.065 [+0.025, +0.110] p=0.001; MAP@10 +0.051 [+0.018, +0.084] p=0.003; Weighted Gap Coverage@5 -0.026 [-0.053, -0.003] p=0.028; Prerequisite Violation Rate@5 -0.046 [-0.083, -0.017] p=0.000; Redundancy@5 -0.009 [-0.015, -0.004] p=0.000; Avg #returned +1.521 [+1.104, +1.938] p=0.000

#### Findings (all 48, partly dev-tuned)

- **nDCG@10** (all 48, partly dev-tuned): tuned Full = 0.894 [0.874, 0.912]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph), Gap-Overlap Greedy + prereq filter. Not distinguishable from: none.
- **Precision@5** (all 48, partly dev-tuned): tuned Full = 0.612 [0.558, 0.667]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph). Not distinguishable from: Gap-Overlap Greedy + prereq filter.
- **Recall@10** (all 48, partly dev-tuned): tuned Full = 0.878 [0.836, 0.920]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph), Gap-Overlap Greedy + prereq filter. Not distinguishable from: none.
- **MAP@10** (all 48, partly dev-tuned): tuned Full = 0.739 [0.693, 0.786]. Significantly higher than: BM25 Keyword, BM25 + prereq filter, Dense Semantic, Dense + prereq filter, Skill Overlap (user-agnostic), Gap-Overlap Greedy (no graph). Not distinguishable from: Gap-Overlap Greedy + prereq filter.
- Strongest-baseline comparison (pre-specified) **Gap-Overlap Greedy + prereq filter** (0.852 [0.826, 0.876]): Full minus it = +0.042 (95% CI [+0.016, +0.066], Holm p = 0.002) -> Full significantly better.
- **Where Full is significantly worse** (all 48, partly dev-tuned): Weighted Gap Coverage@5 vs Gap-Overlap Greedy (no graph) (-0.050, Holm p=0.013).
- Baselines with a nominally better mean than Full but not significantly so: Weighted Gap Coverage@5: Gap-Overlap Greedy + prereq filter (0.860 vs 0.850); Redundancy@5: BM25 + prereq filter (0.030 vs 0.036); Redundancy@5: Dense Semantic (0.033 vs 0.036); Redundancy@5: Dense + prereq filter (0.033 vs 0.036).
- Prerequisite Violation Rate@5 = 0.000 [0.000, 0.000]; average list length = 9.88; Explanation Completeness@5 = 1.000 (1.0 for Full and 0.0 for baselines by construction: baselines emit no explanation fields; a 0 violation rate for the filtered baselines is also by construction).
- Content-only-label sensitivity (nDCG@10): vs BM25 Keyword +0.177 (p_holm=0.001, better); vs BM25 + prereq filter +0.255 (p_holm=0.001, better); vs Dense Semantic +0.229 (p_holm=0.001, better); vs Dense + prereq filter +0.273 (p_holm=0.001, better); vs Skill Overlap (user-agnostic) +0.005 (p_holm=0.792, indistinguishable); vs Gap-Overlap Greedy (no graph) -0.053 (p_holm=0.003, worse); vs Gap-Overlap Greedy + prereq filter +0.018 (p_holm=0.684, indistinguishable).

## Caveats

- n = 24 test scenarios is small; CIs are wide and scenarios share a catalogue/archetypes, so they are not independent. Bootstrap ignores label uncertainty.
- Synthetic catalogue and scenarios; labels come from a deterministic rule over the same catalogue fields the skill-aware systems read, and the rule caps courses that need a bridge at grade 1 (this is what makes strict, startable-now prerequisite handling pay off; see the content-only sensitivity rows). Human validation of the labels (human_review_sample.csv) has NOT been done.
- The graph-free baseline 'Gap-Overlap Greedy + prereq filter' implements the same chained prerequisite rule as the violation metric, so its 0.000 violation rate is by construction; Full now reaches 0.000 through the same catalogue-field rule (not through the graph).
- Explanation Completeness is a format check (baselines emit no explanation fields), not a quality measure. Round 2 adds 'related_text' fill courses whose explanation states they are text-related only.
- Learner-level inference (round 4) did not help on dev and is disabled; the benchmark profiles carry no experience level, so level suitability is effectively constant here and untested.
- Tuning touched 24 scenarios with ~100 configurations in total; dev gains of a few thousandths are inside noise.

## Reproduction

```
PYTHONPATH=src python -m skillllm_v2.recommend.loop_eval        # (library) split + harness
PYTHONPATH=src python -m skillllm_v2.recommend.loop_rounds      # dev-only loop -> recommender_loop_log.{md,json}
PYTHONPATH=src python -m skillllm_v2.recommend.recommender_final  # this report (test half, run once)
```
