# L07 — Profile compaction (H7)

**H7 (primary, `compact@0.5`, informative-confidence arm): NOT SUPPORTED** — criteria {'token': False, 'job': False, 'course': True, 'gap': False}.
Shipped default (`compact@shipped`, budget None): NOT SUPPORTED — criteria {'token': False, 'job': True, 'course': True, 'gap': True}.
Sensitivity arm (uninformative confidence): primary NOT SUPPORTED {'token': False, 'job': False, 'course': True, 'gap': False}.

Rule (pre-registered, Decisions D-A4-01): token reduction (mean, Mistral tokens) >= 0.50 AND job nDCG@10 and course nDCG@10 non-inferior to full (lower 95% CI of delta > -0.02, Holm over the two) AND mean gap Jaccard vs full >= 0.9.

Data: 120 seeded synthetic histories (short/medium/long thirds), truth authored by the generator (`research/compaction.py`, `l07-histories-v1`), inputs in `inputs/evals/research/L07_compaction/`. Tokens = Mistral tokenizer (`artifacts/skillllm_v2/mistral_lora/model/tokenizer.json`) on the exact serialisation.

## Retention vs generator truth (informative arm; means over profiles)

| representation | tokens | reduction | skill P | skill R | goal | gap R (truth) | evidence | contradictions |
|---|---|---|---|---|---|---|---|---|
| full | 323.825 | 0.000 | 0.702 | 1.000 | 1.000 | 0.566 | 1.000 | 1.000 |
| compact@shipped | 2778.017 | -8.027 | 0.702 | 1.000 | 1.000 | 0.566 | 1.000 | 0.794 |
| compact@4 | 1143.825 | -2.566 | 0.753 | 0.979 | 1.000 | 0.619 | 0.979 | 0.354 |
| compact@2 | 582.792 | -0.839 | 0.816 | 0.482 | 1.000 | 0.945 | 0.483 | 0.250 |
| compact@1 | 330.342 | -0.131 | 0.915 | 0.153 | 1.000 | 1.000 | 0.170 | 0.150 |
| compact@0.5 | 226.233 | 0.093 | 1.000 | 0.023 | 1.000 | 1.000 | 0.035 | 0.062 |
| compact@0.25 | 204.025 | 0.129 | - | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| compact@0.1 | 203.958 | 0.129 | - | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| recent@0.5 | 141.850 | 0.574 | 0.671 | 0.475 | 0.992 | 0.647 | 0.475 | 0.046 |
| recent@0.25 | 66.700 | 0.808 | 0.665 | 0.221 | 0.808 | 0.817 | 0.221 | 0.004 |
| minimal | 41.767 | 0.864 | 0.702 | 1.000 | 0.000 | 0.566 | 0.000 | 0.000 |
| truth | 28.175 | 0.906 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

## Downstream decisions (informative arm)

| representation | job nDCG@10 | course nDCG@10 | job tau | job top5 | course tau | course top5 | gap Jaccard |
|---|---|---|---|---|---|---|---|
| full | 0.914 | 0.567 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| compact@shipped | 0.914 | 0.567 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| compact@4 | 0.916 | 0.593 | 0.954 | 0.943 | 0.949 | 0.927 | 0.919 |
| compact@2 | 0.673 | 0.774 | 0.512 | 0.502 | 0.579 | 0.479 | 0.460 |
| compact@1 | 0.387 | 0.747 | 0.210 | 0.288 | 0.491 | 0.375 | 0.360 |
| compact@0.5 | 0.249 | 0.728 | 0.054 | 0.185 | 0.477 | 0.365 | 0.348 |
| compact@0.25 | 0.195 | 0.726 | 0.000 | 0.135 | 0.477 | 0.365 | 0.347 |
| compact@0.1 | 0.195 | 0.726 | 0.000 | 0.135 | 0.477 | 0.365 | 0.347 |
| recent@0.5 | 0.727 | 0.630 | 0.632 | 0.662 | 0.585 | 0.504 | 0.495 |
| recent@0.25 | 0.531 | 0.683 | 0.404 | 0.462 | 0.514 | 0.414 | 0.400 |
| minimal | 0.914 | 0.567 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| truth | 0.983 | 0.923 | 0.787 | 0.782 | 0.645 | 0.575 | 0.566 |

## Non-inferiority vs full history (paired profile bootstrap)

| arm | representation | token red. | Δ job nDCG | job CI | Δ course nDCG | course CI | gap J | verdict |
|---|---|---|---|---|---|---|---|---|
| informative | compact@shipped | -8.027 | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 1.000 | NOT SUPPORTED |
| informative | compact@4 | -2.566 | 0.003 | [-0.010, 0.014] | 0.026 | [0.013, 0.041] | 0.919 | (not an H7 config) |
| informative | compact@2 | -0.839 | -0.241 | [-0.312, -0.173] | 0.208 | [0.140, 0.278] | 0.460 | (not an H7 config) |
| informative | compact@1 | -0.131 | -0.527 | [-0.597, -0.459] | 0.180 | [0.117, 0.244] | 0.360 | (not an H7 config) |
| informative | compact@0.5 | 0.093 | -0.665 | [-0.720, -0.610] | 0.161 | [0.100, 0.224] | 0.348 | NOT SUPPORTED |
| informative | minimal | 0.864 | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 1.000 | (not an H7 config) |
| informative | recent@0.5 | 0.574 | -0.187 | [-0.238, -0.140] | 0.063 | [0.026, 0.105] | 0.495 | (not an H7 config) |
| uninformative | compact@shipped | -8.026 | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 1.000 | NOT SUPPORTED |
| uninformative | compact@4 | -2.566 | -0.022 | [-0.038, -0.009] | 0.006 | [-0.007, 0.021] | 0.922 | (not an H7 config) |
| uninformative | compact@2 | -0.841 | -0.305 | [-0.372, -0.243] | 0.139 | [0.080, 0.202] | 0.449 | (not an H7 config) |
| uninformative | compact@1 | -0.131 | -0.529 | [-0.597, -0.461] | 0.175 | [0.113, 0.239] | 0.360 | (not an H7 config) |
| uninformative | compact@0.5 | 0.093 | -0.670 | [-0.724, -0.617] | 0.161 | [0.100, 0.224] | 0.348 | NOT SUPPORTED |
| uninformative | minimal | 0.864 | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 1.000 | (not an H7 config) |
| uninformative | recent@0.5 | 0.574 | -0.187 | [-0.238, -0.140] | 0.063 | [0.026, 0.105] | 0.495 | (not an H7 config) |

Figure: `compression_vs_agreement.png` (+ `.csv`): agreement with the full-history decision vs token ratio.

## By history length (informative arm)

| representation | length | n | full tokens | token red. | skill R vs full | job tau | gap J | budget met |
|---|---|---|---|---|---|---|---|---|
| compact@shipped | short | 40 | 141.175 | -9.038 | 1.000 | 1.000 | 1.000 | - |
| compact@shipped | medium | 40 | 270.300 | -7.941 | 1.000 | 1.000 | 1.000 | - |
| compact@shipped | long | 40 | 560.000 | -7.104 | 1.000 | 1.000 | 1.000 | - |
| compact@4 | short | 40 | 141.175 | -2.687 | 0.806 | 0.895 | 0.865 | 1.000 |
| compact@4 | medium | 40 | 270.300 | -2.509 | 0.957 | 0.971 | 0.904 | 1.000 |
| compact@4 | long | 40 | 560.000 | -2.501 | 0.997 | 0.995 | 0.988 | 1.000 |
| compact@2 | short | 40 | 141.175 | -0.896 | 0.136 | 0.253 | 0.569 | 1.000 |
| compact@2 | medium | 40 | 270.300 | -0.876 | 0.374 | 0.575 | 0.432 | 1.000 |
| compact@2 | long | 40 | 560.000 | -0.745 | 0.665 | 0.707 | 0.379 | 1.000 |
| compact@1 | short | 40 | 141.175 | -0.469 | 0.000 | 0.000 | 0.554 | 0.050 |
| compact@1 | medium | 40 | 270.300 | 0.008 | 0.071 | 0.185 | 0.346 | 1.000 |
| compact@1 | long | 40 | 560.000 | 0.068 | 0.249 | 0.447 | 0.180 | 1.000 |
| compact@0.5 | short | 40 | 141.175 | -0.468 | 0.000 | 0.000 | 0.554 | 0.000 |
| compact@0.5 | medium | 40 | 270.300 | 0.231 | 0.000 | 0.000 | 0.338 | 0.000 |
| compact@0.5 | long | 40 | 560.000 | 0.515 | 0.045 | 0.161 | 0.153 | 1.000 |
| recent@0.5 | short | 40 | 141.175 | 0.605 | 0.383 | 0.624 | 0.662 | - |
| recent@0.5 | medium | 40 | 270.300 | 0.569 | 0.479 | 0.639 | 0.517 | - |
| recent@0.5 | long | 40 | 560.000 | 0.548 | 0.616 | 0.633 | 0.306 | - |
| minimal | short | 40 | 141.175 | 0.859 | 1.000 | 1.000 | 1.000 | - |
| minimal | medium | 40 | 270.300 | 0.852 | 1.000 | 1.000 | 1.000 | - |
| minimal | long | 40 | 560.000 | 0.880 | 1.000 | 1.000 | 1.000 | - |

## Contradiction preservation by type (informative arm)

| representation | retraction | label_conflict | goal_change | proficiency |
|---|---|---|---|---|
| full | 1.000 | 1.000 | 1.000 | 1.000 |
| compact@shipped | 1.000 | 1.000 | 0.000 | 1.000 |
| compact@4 | 0.034 | 1.000 | 0.000 | 0.462 |
| compact@2 | 0.000 | 0.979 | 0.000 | 0.000 |
| compact@1 | 0.000 | 0.729 | 0.000 | 0.000 |
| compact@0.5 | 0.000 | 0.375 | 0.000 | 0.000 |
| compact@0.25 | 0.000 | 0.000 | 0.000 | 0.000 |
| compact@0.1 | 0.000 | 0.000 | 0.000 | 0.000 |
| recent@0.5 | 0.000 | 0.167 | 0.048 | 0.019 |
| recent@0.25 | 0.000 | 0.021 | 0.000 | 0.000 |
| minimal | 0.000 | 0.000 | 0.000 | 0.000 |
| truth | 0.000 | 0.000 | 0.000 | 0.000 |

## Reading the result

- `full` keeps every extracted mention, including distractors/aspirations/retracted skills, so its skill precision vs truth is < 1; a compactor that drops low-confidence mentions can be *closer to truth* while *disagreeing* with full. Agreement-with-full and accuracy-vs-truth are reported separately for that reason.
- The shipped packet (`budget_tokens=None`) is lossless JSON and is LARGER than the raw history; it can only fail the token criterion.
- `goal_retained` for compact is a pass-through of the caller's `job_target` (the system does not extract goals from history text); goal *history* (goal-change contradictions) is not kept.
- `minimal` exposes the same skill set as `full`, so its downstream decisions equal full by construction; it loses goals, evidence and contradictions.
- Course nDCG@10 RISES when skills are dropped: with fewer held skills every target skill becomes a gap, so courses for the true gaps that `full` hides (aspiration mentions read as held skills) get recommended again. Course-nDCG non-inferiority is therefore satisfied trivially by an empty profile and is not sufficient on its own; the job criterion and gap agreement are what expose the information loss.
- The JSON packet costs ~35-45 proxy tokens per skill fact versus ~12-15 for the raw sentence, plus a fixed skeleton (~200 Mistral tokens incl. a 64-hex content hash); at budgets below ~4x the raw history it keeps few or no skills, and for short histories the 0.5x budget is below the skeleton itself (`budget met` < 1).

## Caveats

- Synthetic histories and authored relevance: absolute nDCG is optimistic (ranker and grades both reason about skill coverage); paired deltas and agreement are the informative quantities.
- Extraction is simulated (oracle spans, authored confidence). The informative-confidence arm favours confidence-ranked compaction; the uninformative arm is the stress test.
- Proxy-token budgets (the system's unit) differ from Mistral tokens; the token criterion is measured in Mistral tokens.
