# Recommender improvement loop - dev-half log

Protocol: 48 benchmark scenarios were split ONCE (sha256-sorted ids; `recommender_split.json`) into 24 dev / 24 test. Every number in this file is from the 24 **dev** scenarios; the test half is not touched until `recommender_final.py` runs the frozen config once (see `course_recommendation_benchmark_v2.md`). Primary objective: dev nDCG@10. Hard constraints for an accepted config: Prerequisite Violation Rate@5 not above the previous round, Explanation Completeness@5 = 1.0, dev Redundancy@5 <= previous round + 0.001. Round 1 is a constraint round (violation -> 0 within 0.005 nDCG); rounds 2-5 need >= +0.002 dev nDCG@10 to be accepted, else they are REJECTED and the config is unchanged. (Arms within 0.001 nDCG@10 of the best count as tied; the tie goes to the arm that returns the longest list.) Labels, baselines and metric code are untouched; no gold/relevance label is an input of the recommender.

Caveat that applies to every round: with 24 dev scenarios a +0.002..0.01 nDCG change is well inside sampling noise; the rules above only stop us from accepting changes that hurt. Dev-set tuning of any kind can overfit - only the test half can say whether a gain is real.

## Round 0 - starting point (legacy recommender, soft prerequisite score)

| config | ndcg10 | p5 | recall10 | map10 | gap_cov5 | prereq_viol5 | redundancy5 | expl5 | n_returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 legacy | 0.881 | 0.633 | 0.815 | 0.706 | 0.871 | 0.025 | 0.046 | 1.000 | 8.417 |

## Round 1 - Strict prerequisite handling with bridge sequencing [ACCEPTED]

**Change:** A course is eligible only if every prerequisite is known by the learner or taught by a course placed earlier in the list; a blocked course can be unlocked by a startable bridge course (placed before it) chosen by unlock gain. Prerequisite compatibility in the score counts only the learner's own skills (prefers courses startable now).

| config | ndcg10 | p5 | recall10 | map10 | gap_cov5 | prereq_viol5 | redundancy5 | expl5 | n_returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (previous round) | 0.881 | 0.633 | 0.815 | 0.706 | 0.871 | 0.025 | 0.046 | 1.000 | 8.417 |
| strict_drop, compat=chain | 0.853 | 0.550 | 0.795 | 0.656 | 0.873 | 0.000 | 0.041 | 1.000 | 7.917 |
| strict_drop, compat=learner | 0.880 | 0.642 | 0.824 | 0.714 | 0.848 | 0.000 | 0.047 | 1.000 | 7.917 |
| strict_bridge, compat=chain, w_unlock=0.4 | 0.798 | 0.517 | 0.771 | 0.587 | 0.922 | 0.000 | 0.030 | 1.000 | 8.667 |
| strict_bridge, compat=learner, w_unlock=0.0 | 0.877 | 0.633 | 0.846 | 0.717 | 0.890 | 0.000 | 0.034 | 1.000 | 8.708 |
| strict_bridge, compat=learner, w_unlock=0.2 **<- adopted** | 0.880 | 0.650 | 0.843 | 0.725 | 0.887 | 0.000 | 0.035 | 1.000 | 8.667 |
| strict_bridge, compat=learner, w_unlock=0.4 | 0.864 | 0.642 | 0.843 | 0.703 | 0.884 | 0.000 | 0.034 | 1.000 | 8.667 |
| **after (carried forward)** | 0.880 | 0.650 | 0.843 | 0.725 | 0.887 | 0.000 | 0.035 | 1.000 | 8.667 |

Dev nDCG@10: 0.881 -> 0.880 (-0.002); per-scenario W/L/T = 8/8/8. Prereq violation 0.025 -> 0.000; avg list length 8.42 -> 8.67; Redundancy@5 0.046 -> 0.035; Explanation Completeness 1.000.

## Round 2 - Fill the list towards top-K with next-best feasible related courses [ACCEPTED]

**Change:** When no gap-covering or bridge candidate is left, append feasible courses whose text mentions an open gap skill or that teach a skill sharing a content word with it (flagged role='related_text', with an explicit text-only limitation), best score first. Candidate pool 15 -> 40.

| config | ndcg10 | p5 | recall10 | map10 | gap_cov5 | prereq_viol5 | redundancy5 | expl5 | n_returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (previous round) | 0.880 | 0.650 | 0.843 | 0.725 | 0.887 | 0.000 | 0.035 | 1.000 | 8.667 |
| fill phrase-match only, pool=40 | 0.880 | 0.650 | 0.843 | 0.725 | 0.887 | 0.000 | 0.035 | 1.000 | 9.375 |
| fill token-match, pool=15 | 0.886 | 0.650 | 0.863 | 0.735 | 0.887 | 0.000 | 0.035 | 1.000 | 9.708 |
| fill token-match, pool=40, discount=0.8 **<- adopted** | 0.886 | 0.650 | 0.863 | 0.734 | 0.887 | 0.000 | 0.035 | 1.000 | 9.875 |
| fill token-match, pool=25, discount=0.8 | 0.886 | 0.650 | 0.863 | 0.734 | 0.887 | 0.000 | 0.035 | 1.000 | 9.875 |
| **after (carried forward)** | 0.886 | 0.650 | 0.863 | 0.734 | 0.887 | 0.000 | 0.035 | 1.000 | 9.875 |

Dev nDCG@10: 0.880 -> 0.886 (+0.006); per-scenario W/L/T = 3/0/21. Prereq violation 0.000 -> 0.000; avg list length 8.67 -> 9.88; Redundancy@5 0.035 -> 0.035; Explanation Completeness 1.000.

## Round 3 - Redundancy / diversity penalty (MMR-style) tuned on dev [ACCEPTED]

**Change:** Replace the share-of-own-skills redundancy penalty by an MMR-style penalty on the already-covered gap MASS and tune its weight; constraint: dev Redundancy@5 must not exceed the previous round's.

| config | ndcg10 | p5 | recall10 | map10 | gap_cov5 | prereq_viol5 | redundancy5 | expl5 | n_returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (previous round) | 0.886 | 0.650 | 0.863 | 0.734 | 0.887 | 0.000 | 0.035 | 1.000 | 9.875 |
| redundancy=fraction, w=0.0 | 0.909 | 0.675 | 0.879 | 0.760 | 0.866 | 0.000 | 0.045 | 1.000 | 9.875 |
| redundancy=fraction, w=0.05 | 0.897 | 0.650 | 0.879 | 0.744 | 0.872 | 0.000 | 0.041 | 1.000 | 9.875 |
| redundancy=fraction, w=0.1 | 0.890 | 0.642 | 0.870 | 0.737 | 0.882 | 0.000 | 0.037 | 1.000 | 9.875 |
| redundancy=fraction, w=0.25 | 0.870 | 0.600 | 0.853 | 0.702 | 0.900 | 0.000 | 0.029 | 1.000 | 9.875 |
| redundancy=fraction, w=0.4 | 0.853 | 0.583 | 0.849 | 0.685 | 0.913 | 0.000 | 0.025 | 1.000 | 9.792 |
| redundancy=mmr, w=0.0 | 0.909 | 0.675 | 0.879 | 0.760 | 0.866 | 0.000 | 0.045 | 1.000 | 9.875 |
| redundancy=mmr, w=0.05 | 0.900 | 0.625 | 0.879 | 0.738 | 0.872 | 0.000 | 0.038 | 1.000 | 9.875 |
| redundancy=mmr, w=0.1 **<- adopted** | 0.899 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| redundancy=mmr, w=0.25 | 0.877 | 0.583 | 0.848 | 0.702 | 0.891 | 0.000 | 0.032 | 1.000 | 9.875 |
| redundancy=mmr, w=0.4 | 0.868 | 0.542 | 0.848 | 0.681 | 0.898 | 0.000 | 0.027 | 1.000 | 9.875 |
| redundancy=mmr, w=0.6 | 0.854 | 0.508 | 0.839 | 0.655 | 0.909 | 0.000 | 0.021 | 1.000 | 9.875 |
| redundancy=mmr, w=1.0 | 0.836 | 0.467 | 0.828 | 0.637 | 0.932 | 0.000 | 0.014 | 1.000 | 9.833 |
| **after (carried forward)** | 0.899 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |

Dev nDCG@10: 0.886 -> 0.899 (+0.013); per-scenario W/L/T = 11/4/9. Prereq violation 0.000 -> 0.000; avg list length 9.88 -> 9.88; Redundancy@5 0.035 -> 0.035; Explanation Completeness 1.000.

## Round 4 - Learner-level inference (user-level suitability) [REJECTED (config unchanged)]

**Change:** The benchmark's user profiles carry no experience level, so the level-suitability term used a constant 'intermediate'. Infer the level from the profile: (a) number of declared skills, (b) depth of the learner's skills (mean highest catalogue level at which each is taught).

| config | ndcg10 | p5 | recall10 | map10 | gap_cov5 | prereq_viol5 | redundancy5 | expl5 | n_returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (previous round) | 0.899 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| infer by skill count, jr<=3 sr>=6 | 0.898 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| infer by skill count, jr<=4 sr>=7 | 0.897 | 0.617 | 0.873 | 0.726 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| infer by skill count, jr<=5 sr>=8 | 0.898 | 0.600 | 0.881 | 0.726 | 0.854 | 0.000 | 0.035 | 1.000 | 9.875 |
| infer by skill depth, jr<=1.4 sr>=2.0 | 0.894 | 0.617 | 0.879 | 0.737 | 0.882 | 0.000 | 0.036 | 1.000 | 9.875 |
| infer by skill depth, jr<=1.4 sr>=2.2 | 0.894 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.036 | 1.000 | 9.875 |
| infer by skill depth, jr<=1.6 sr>=2.0 | 0.897 | 0.625 | 0.879 | 0.745 | 0.873 | 0.000 | 0.036 | 1.000 | 9.875 |
| infer by skill depth, jr<=1.6 sr>=2.2 | 0.897 | 0.625 | 0.879 | 0.744 | 0.873 | 0.000 | 0.036 | 1.000 | 9.875 |
| infer by skill depth, jr<=1.7 sr>=2.0 | 0.898 | 0.625 | 0.879 | 0.749 | 0.873 | 0.000 | 0.036 | 1.000 | 9.875 |
| infer by skill depth, jr<=1.7 sr>=2.2 | 0.899 | 0.625 | 0.879 | 0.749 | 0.873 | 0.000 | 0.036 | 1.000 | 9.875 |
| **after (carried forward)** | 0.899 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |

Dev nDCG@10: 0.899 -> 0.899 (+0.000); per-scenario W/L/T = 0/0/24. Prereq violation 0.000 -> 0.000; avg list length 9.88 -> 9.88; Redundancy@5 0.035 -> 0.035; Explanation Completeness 1.000.

Best constraint-satisfying arm: infer by skill depth, jr<=1.7 sr>=2.2; it did not reach the +0.002 dev nDCG@10 bar, so this change is **not** adopted.

## Round 5 - Weights of the interpretable score terms (coordinate search on dev) [ACCEPTED]

**Change:** One-at-a-time coordinate search (two passes max) over a coarse grid per weight; a move is taken only if it improves dev nDCG@10 by >= 0.002 and satisfies the hard constraints.

| config | ndcg10 | p5 | recall10 | map10 | gap_cov5 | prereq_viol5 | redundancy5 | expl5 | n_returned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (previous round) | 0.899 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| pass 1: w_gap=0.2 | 0.897 | 0.617 | 0.879 | 0.732 | 0.850 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_gap=0.3 | 0.898 | 0.625 | 0.879 | 0.734 | 0.860 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_gap=0.5 | 0.895 | 0.617 | 0.870 | 0.732 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| pass 1: w_gap=0.6 | 0.892 | 0.600 | 0.870 | 0.724 | 0.889 | 0.000 | 0.033 | 1.000 | 9.875 |
| pass 1: w_gap=0.8 | 0.883 | 0.592 | 0.870 | 0.715 | 0.906 | 0.000 | 0.033 | 1.000 | 9.875 |
| pass 1: w_semantic=0.0 | 0.874 | 0.608 | 0.863 | 0.706 | 0.882 | 0.000 | 0.033 | 1.000 | 9.875 |
| pass 1: w_semantic=0.05 | 0.876 | 0.592 | 0.863 | 0.711 | 0.882 | 0.000 | 0.032 | 1.000 | 9.875 |
| pass 1: w_semantic=0.25 | 0.896 | 0.625 | 0.870 | 0.729 | 0.853 | 0.000 | 0.038 | 1.000 | 9.875 |
| pass 1: w_semantic=0.35 | 0.894 | 0.617 | 0.870 | 0.727 | 0.853 | 0.000 | 0.040 | 1.000 | 9.875 |
| pass 1: w_prereq=0.0 | 0.835 | 0.508 | 0.777 | 0.616 | 0.914 | 0.000 | 0.037 | 1.000 | 9.875 |
| pass 1: w_prereq=0.1 | 0.883 | 0.567 | 0.861 | 0.702 | 0.912 | 0.000 | 0.031 | 1.000 | 9.875 |
| pass 1: w_prereq=0.3 | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_level=0.0 | 0.900 | 0.608 | 0.873 | 0.733 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_level=0.05 | 0.898 | 0.608 | 0.873 | 0.732 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_level=0.3 | 0.890 | 0.617 | 0.872 | 0.732 | 0.857 | 0.000 | 0.037 | 1.000 | 9.875 |
| pass 1: w_goal=0.0 | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_goal=0.2 | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_workload=0.2 | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 1: w_unlock=0.0 | 0.906 | 0.600 | 0.887 | 0.746 | 0.849 | 0.000 | 0.039 | 1.000 | 9.875 |
| pass 1: w_unlock=0.1 | 0.900 | 0.608 | 0.879 | 0.743 | 0.863 | 0.000 | 0.034 | 1.000 | 9.875 |
| pass 1: w_unlock=0.4 | 0.888 | 0.608 | 0.879 | 0.727 | 0.859 | 0.000 | 0.034 | 1.000 | 9.875 |
| pass 1: w_unlock=0.6 | 0.884 | 0.608 | 0.879 | 0.720 | 0.856 | 0.000 | 0.034 | 1.000 | 9.875 |
| pass 2: w_gap=0.2 | 0.902 | 0.617 | 0.879 | 0.746 | 0.850 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_gap=0.3 | 0.902 | 0.625 | 0.879 | 0.747 | 0.860 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_gap=0.5 | 0.899 | 0.617 | 0.879 | 0.737 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_gap=0.6 | 0.899 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| pass 2: w_gap=0.8 | 0.896 | 0.617 | 0.879 | 0.731 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| pass 2: w_semantic=0.0 | 0.879 | 0.608 | 0.863 | 0.723 | 0.863 | 0.000 | 0.034 | 1.000 | 9.875 |
| pass 2: w_semantic=0.05 | 0.882 | 0.592 | 0.863 | 0.728 | 0.863 | 0.000 | 0.034 | 1.000 | 9.875 |
| pass 2: w_semantic=0.25 | 0.905 | 0.625 | 0.879 | 0.745 | 0.853 | 0.000 | 0.038 | 1.000 | 9.875 |
| pass 2: w_semantic=0.35 | 0.902 | 0.633 | 0.879 | 0.744 | 0.850 | 0.000 | 0.039 | 1.000 | 9.875 |
| pass 2: w_prereq=0.0 | 0.835 | 0.508 | 0.777 | 0.616 | 0.914 | 0.000 | 0.037 | 1.000 | 9.875 |
| pass 2: w_prereq=0.1 | 0.883 | 0.567 | 0.861 | 0.702 | 0.912 | 0.000 | 0.031 | 1.000 | 9.875 |
| pass 2: w_prereq=0.2 | 0.899 | 0.617 | 0.879 | 0.736 | 0.882 | 0.000 | 0.035 | 1.000 | 9.875 |
| pass 2: w_level=0.0 | 0.900 | 0.608 | 0.873 | 0.733 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_level=0.05 | 0.898 | 0.608 | 0.873 | 0.732 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_level=0.3 | 0.890 | 0.617 | 0.872 | 0.732 | 0.857 | 0.000 | 0.037 | 1.000 | 9.875 |
| pass 2: w_goal=0.0 | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_goal=0.2 | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_workload=0.2 | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |
| pass 2: w_unlock=0.0 | 0.906 | 0.600 | 0.887 | 0.746 | 0.849 | 0.000 | 0.039 | 1.000 | 9.875 |
| pass 2: w_unlock=0.1 | 0.900 | 0.608 | 0.879 | 0.743 | 0.863 | 0.000 | 0.034 | 1.000 | 9.875 |
| pass 2: w_unlock=0.4 | 0.888 | 0.608 | 0.879 | 0.727 | 0.859 | 0.000 | 0.034 | 1.000 | 9.875 |
| pass 2: w_unlock=0.6 | 0.884 | 0.608 | 0.879 | 0.720 | 0.856 | 0.000 | 0.034 | 1.000 | 9.875 |
| **after (carried forward)** | 0.902 | 0.617 | 0.879 | 0.746 | 0.863 | 0.000 | 0.036 | 1.000 | 9.875 |

Dev nDCG@10: 0.899 -> 0.902 (+0.004); per-scenario W/L/T = 2/0/22. Prereq violation 0.000 -> 0.000; avg list length 9.88 -> 9.88; Redundancy@5 0.035 -> 0.036; Explanation Completeness 1.000.

44 configurations evaluated. pass 1: w_prereq 0.2 -> 0.3 (dev nDCG@10 0.8987 -> 0.9022)

## Final dev-tuned configuration (frozen for the test run)

```json
{
  "w_gap": 0.4,
  "w_semantic": 0.15,
  "w_prereq": 0.3,
  "w_level": 0.15,
  "w_goal": 0.1,
  "w_redundancy": 0.1,
  "w_workload": 0.05,
  "w_unlock": 0.2,
  "retrieval_pool": 40,
  "prereq_mode": "strict_bridge",
  "compat_basis": "learner",
  "bridge_discount": 1.0,
  "fill_related": true,
  "related_match": "token",
  "related_discount": 0.8,
  "redundancy_mode": "mmr",
  "min_score": 0.0,
  "infer_user_level": false,
  "level_method": "catalog_depth",
  "level_depth_junior": 1.6,
  "level_depth_senior": 2.1,
  "level_senior_skills": 7,
  "level_junior_skills": 4
}
```

Differences from the legacy config: `{"w_prereq": 0.3, "w_redundancy": 0.1, "w_unlock": 0.2, "retrieval_pool": 40, "prereq_mode": "strict_bridge", "compat_basis": "learner", "fill_related": true, "related_match": "token", "redundancy_mode": "mmr"}`

Dev summary: nDCG@10 0.881 -> 0.902, prereq violation 0.025 -> 0.000, avg list length 8.42 -> 9.88.
