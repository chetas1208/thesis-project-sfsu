# Human evaluation protocol (layer L11)

Status: the packets are built and the protocol is fixed. **No human judgement has been collected yet.** Every human metric
in `reports/skillllm_v2/evals/L11_human_eval/results.json` has status `pending_human`. This document was written before
any annotation. Any change after annotation starts is appended under "Amendments" with its date and reason.

Code: `src/skillllm_v2/evals/research/human_eval_samples.py` (packets, answer keys, scoring) and
`src/skillllm_v2/evals/research/agreement.py` (agreement statistics). Rebuild the packets with
`PYTHONPATH=src python -m skillllm_v2.evals.research.human_eval_samples` (seed 42; the recommender re-run is cached in
`inputs/evals/research/L11_human_eval/cache/`). Score returned annotations with the same command plus `--score`.

## 1. What human evaluation can and cannot establish here

| Question | Packet | Why automatic metrics cannot answer it |
|---|---|---|
| Are the anchor's "false positives" really wrong, or is the SkillSpan gold incomplete? | P1 (stratum `anchor_fp`) | Exact-match precision treats every non-gold span as an error. |
| Are spans the anchor missed real skills? Are boundary disagreements real errors or guideline ambiguity? | P1 (`anchor_fn`, `boundary`) | Boundary conventions (triggers, modal verbs, adverbials) are contested even among experts. |
| Do spans added by wrapper stages, or proposed only by the weak baselines, have any value? | P1 (`added_by_wrapper`, `baseline_only`) | They are scored as false positives against a gold that may miss skills. |
| How well do two humans agree on SkillSpan-style spans, i.e. what is the human ceiling? Do "skill-free" test sentences really contain no skills? | P2 | No second annotation of the test split exists. The 2,595 gold-free sentences were never scored. |
| Is the proposed canonical concept right, an acceptable alternative, wrong, or should the output be NIL? | P3 | The shipped alias/BM25 linker has no gold canonical ids. Coverage is not accuracy. |
| Are the catalogue's prerequisite edges and the recommender's learning paths pedagogically valid? | P4a, P4b | The catalogue and its prerequisites were authored by the project. Structural path validity only checks that edges exist. |
| Are recommended courses relevant? Do they close the gap? Can the learner start them? Do the rule labels agree with humans? | P5 | All recommender labels are rule-derived from the same catalogue fields the systems read. |
| Are the recommender's explanations faithful to the course record, profile and job? | P6 | The existing explanation audit is structural only. Semantic faithfulness has never been judged. |
| Are the robustness perturbations meaning-preserving and label-preserving? | P7 | The perturbation families' validity is assumed by construction (A3, D-A3-04). |

Human labels in this study are **secondary validity evidence**. They neither replace SkillSpan gold for the extraction
headline nor the pre-registered H1–H7 rules. No hypothesis verdict depends on L11.

## 2. Roles, independence and conflicts of interest

- **Two annotators (A and B)** label every item of every packet independently. They do not discuss items until both
  files are returned. Preferred profile: one annotator with recruiting/HR or labour-market experience and one with a
  computing background. Both must read SkillSpan Appendix B (Zhang et al., 2022) and this protocol, and complete the
  training round.
- **One adjudicator** resolves disagreements only (`<packet>__ADJ_todo.csv`, generated automatically). The adjudicator
  sees both labels but not system identity or gold, and may choose either label or a third one with a one-line reason.
- **Conflict of interest.** The thesis author wrote the system, the synthetic catalogue, the scenarios and the rule
  labels. The author must **not** act as annotator A or B for P4–P6. If no one else is available, the author may
  adjudicate, and this is reported as a threat to validity. If the author annotates P1–P3, that is reported too.
- **Compensation and consent.** Annotators are volunteers or paid at a flat hourly rate (never per agreement). No personal
  data are collected beyond the annotator letter (A/B/ADJ). All texts are SkillSpan sentences (CC-BY-4.0, already public)
  or synthetic.

## 3. Blinding and order

- Packets never show which system produced an item, its rank, the SkillSpan gold, automatic grades, strata or rule
  labels. These live only in `inputs/evals/research/L11_human_eval/answer_keys/`, a different directory tree that is
  never handed out. `tests/evals/research/test_human_eval_samples.py::test_built_packets_are_blind` fails if a hidden
  field appears in a packet header.
- Item ids are assigned **after** a seeded shuffle, so they carry no stratum or system information. Each annotator gets
  the same items in a **different** random order (seeds 1001 and 2002, offset by the build seed) to spread order and
  fatigue effects.
- P5 pools the top-5 of two systems per scenario (tuned Full recommender and the pre-specified strongest baseline,
  Gap-Overlap Greedy + prerequisite filter), so an annotator cannot tell which system proposed a course. The 60 rows of
  the earlier `reports/skillllm_v2/human_review_sample.csv` (which showed `auto_grade_0to3` and the rule-label input
  `needed_skills(weight)` to the annotator) are re-blinded and merged.
- **P6 cannot be system-blind.** Only the Full recommender produces explanations. This is stated wherever P6 numbers
  appear.
- P5 annotators see the job text and the job's listed required and preferred skills, i.e. what the systems see. They do
  not see the hidden `gold.needs` weights from which the rule labels were derived (D-A6-03).

## 4. Training round (T0)

`packets/training/T0_<packet>.csv`, with answer key and rationale in `answer_keys/training/`. All items are synthetic
(no posting text) and cite the SkillSpan guideline rule they illustrate.
1. Both annotators complete T0 independently.
2. The adjudicator scores T0 against the key and discusses every miss with each annotator (rule cited).
3. Pass criterion: at least 80 % key agreement on T0-P1 and T0-P2, and correct use of NONE_OF_THESE / NOT_A_SKILL and the "pick the more specific candidate" rule in T0-P3 (NIL and "acceptable
   alternative" categories in the fallback format). Otherwise repeat once with the discussion notes. T0 labels are never reported as
   results.
4. After T0, annotation guidelines are frozen. Questions that arise later are logged in `annotations/questions.md` and
   answered for both annotators identically, without reference to specific items.

## 5. Rubrics and decision rules

General rules: "unsure" or an empty cell counts as missing (pairwise deletion in agreement). Free-text columns are never
scored for agreement. Annotators must not search the web for item text.

### 5.1 P1 — extraction candidate span (one marked span per item)

The sentence shows a candidate span between ⟦ and ⟧. The type labels follow SkillSpan, which follows ESCO: **SKILL** =
"the ability to apply knowledge and use know-how to complete tasks and solve problems" (verb-led phrases; attitudes count
as SKILL, B.1.7). **KNOWLEDGE** = "the body of facts, principles, theories and practices related to a field of work or
study" (things one possesses and cannot physically execute: tools, languages, domains, degrees, licences; B.2.1, B.2.3,
B.2.8). The repository maps KNOWLEDGE to `HARD_SKILL` and SKILL to `SOFT_SKILL`. That mapping is applied only in
scoring.

| Column | Values | Decision rule |
|---|---|---|
| `q1_is_skill` | yes / no / unsure | yes if the marked text, or a span overlapping it, denotes a competence asked of the candidate under SkillSpan Appendix B. Company descriptions, benefits, job titles in the posting headline (B.3.4), ironic skills (B.1.9.1) and under-specified items such as "other test libraries" (B.2.5) are **no**. Guideline B.3.1 applies: if in doubt, annotate as a skill. |
| `q2_type` | SKILL / KNOWLEDGE | Only if q1 = yes. If in doubt, prefer SKILL (B.3.2). |
| `q3_boundary` | exact / too_long / too_short / shifted | exact = the marked span is what the guideline would tag. too_long = it includes triggers or fluff (B.3.6: "experience with", "knowledge of", "proficient in"), modal verbs (B.1.1.1), articles before attitudes (B.1.7) or trailing context (B.3.5). too_short = it omits the verb's object or a manner adverbial (B.1.1, B.1.6), or the parenthesised abbreviation (B.1.5). shifted = overlaps the right span but neither contains the other. |
| `q4_corrected_span` | free text | Required unless q3 = exact. Copy the corrected span verbatim from the sentence. If the marked text covers several coordinated components (B.1.2, B.2.8), give the first and say "+N more" in notes. |

**Strata (hidden):** `anchor_fp` (anchor span with no overlap with any gold span), `anchor_fn` (gold span the anchor did
not predict), `boundary` (anchor span overlapping gold with different offsets), `added_by_wrapper` (span added by a
wrapper stage, not by the anchor and not in gold; **census**, every such span is included), `baseline_only` (proposed
only by the token-lookup or zero-shot GLiNER baseline), `agreed_tp_control` (anchor = gold position; calibration
control). Sampling is stratified by data source (tech/house), proportional, seed 42. Pool sizes are recorded in
`packet_manifest.json`. Anchor offsets are the stored July 2026 offsets, i.e. exactly what was scored (Progress.md
integrity flag on `find_offsets` drift).

### 5.2 P2 — full-sentence span annotation (no candidate shown)

Annotate every SKILL and KNOWLEDGE component in the sentence following SkillSpan Appendix B. Nesting a KNOWLEDGE
component inside a SKILL is allowed (B.2.6). Otherwise avoid nested skills (B.1.9.2).

- `spans`: `surface|TYPE` entries separated by `;`, copied verbatim. For the k-th occurrence of a repeated string, write
  `surface#k|TYPE`. Leave the cell empty if there is no component.
- `q_headline`: yes if the sentence looks like the posting title or top headline. Components there are not tagged
  (B.3.4).
- `completed`: put `y` on every row you finished. This distinguishes "no spans" from "not done".
- Scoring converts surfaces to character offsets (`parse_span_cell`). Unparseable entries are listed, never silently
  dropped.

**Strata (hidden):** 60 gold-bearing sentences from the 974-sentence anchor subset and 40 gold-free test sentences (at
least 6 tokens, not anonymisation placeholders). The gold-free stratum directly tests the 974/3,569 caveat: if humans
find skills in "skill-free" sentences, then neither the gold nor a future spurious-prediction rate on those sentences
can be read at face value.

### 5.3 P3 — normalization / taxonomy linking

The sentence marks the mention, and `proposed_concept` is the system's canonical concept, or "(no concept proposed)".
The reference concept list is `packets/reference_taxonomy.csv` (the repository taxonomy). If A4's ESCO-based
`L06_linking/annotation_sample.csv` exists when the packets are built, it replaces the fallback sample, and its
annotator-safe columns are shown.

| `q1_link` | Rule |
|---|---|
| correct | The concept denotes exactly the competence mentioned (synonym, abbreviation, spelling variant or inflection). |
| acceptable_alternative | The concept is defensible but not the best available: a broader parent (e.g. "Microsoft Excel" for "Excel modelling") or a closely related sibling that a recruiter would accept for matching. Give the better concept in `q2_better_concept` if one exists in the reference list. |
| wrong | Different competence (e.g. Java vs JavaScript), or the mention is not a skill at all. |
| nil | No suitable concept exists in the reference list. This is the correct label when `proposed_concept` is empty and nothing fits. If a concept was proposed but nothing fits, the answer is also `nil` (the proposal is then an error). |

Reported: strict accuracy (correct), lenient accuracy (correct + acceptable_alternative), and NIL rate. The fallback
sample is stratified by link method (alias, BM25, fuzzy, none). Linking accuracy on test is partly circular: 76
alias-table rows were mined from test predictions (`docs/thesis_drafts/evals.md`, inconsistency I12). Items linked
through those rows are flagged in the answer key by method and analysed separately.

**Built version (candidate-choice mode, used because the ESCO sample now exists).** The L06 sample
(`L06_linking/annotation_sample.csv`, 300 items; a seeded random 150 are shown to annotators to bound the workload, the
rest stay in the key) shows the sentence, the `mention` and five shuffled ESCO candidates `candidate_A`..`candidate_E`;
no system, score or rank is visible. `q1_choice` takes one of **A-E** (the single best concept; when two candidates are
equally right, choose the more specific one and name the other in `q2_note`), **NONE_OF_THESE** (a real skill but no
listed concept fits, i.e. NIL is correct) or **NOT_A_SKILL**. The proposed-concept rubric above applies only to the
fallback sample. Reported in this mode: agreement (kappa, AC1, alpha) on `q1_choice`; the rate at which the adjudicated
choice equals the Decorte et al. ESCO gold on control items (human check of the gold itself); and, per linker, the rate
at which its top-ranked candidate equals the adjudicated human choice (the key maps letters back to linker ranks), which
gives an independent linking gold for 150 mentions. T0-P3 uses the same format.

### 5.4 P4a — prerequisite edge validity; P4b — learning-path validity

- P4a: "Is `claimed_prerequisite_skill` a reasonable prerequisite for this course?" **valid** = most curricula would
  require it, or a learner without it would struggle. **questionable** = helpful but not needed, or needed only for part
  of the course. **invalid** = unrelated or harmful (would block learners for no reason). Ordinal 0/1/2.
- P4b: each path is `learner knows K -> [bridge course teaches X] -> X is prerequisite of [primary course] -> primary
  teaches Y`, built from the tuned Full recommender's held-out top-10 lists (or A5's path file if present).
  `q1_all_edges_valid`: yes if every edge is valid in the P4a sense. `q2_pedagogical`: 2 = a sensible order a teacher
  would recommend for reaching the target job; 1 = workable but with an unnecessary or missing step; 0 = wrong order or
  irrelevant to the target. `q3_missing_step`: free text.

### 5.5 P5 — course recommendation

Each item is one (learner, target job, course) triple. The learner profile and job are synthetic benchmark scenarios.

| Column | Scale | Decision rule |
|---|---|---|
| `q1_relevance` | 0–3 | 3 = directly teaches one or more central missing requirements of the job and is at a suitable level. 2 = teaches a missing required or preferred skill, but only partly or less centrally. 1 = tangentially related (same area, little gap closure, or mostly re-teaches known skills). 0 = irrelevant to the gap, or teaches only what the learner already knows. Judge the gap from the job text and listed skills against the learner's known skills. |
| `q2_gap_closure` | none / partial / substantial | How much of the learner's gap to this job the course closes on its own. |
| `q3_prereq_ready` | no / needs_bridge / yes | yes = the learner can start now. needs_bridge = one short preparatory course would suffice. no = major prerequisites missing. |

Reported: weighted κ between annotators; quadratic-weighted κ between adjudicated human grades and rule labels v1 (the
benchmark's label validity); human P@5 per system with a paired difference over scenarios. The pooled design gives each
system's top-5 a human grade, but nDCG@10 over the full catalogue cannot be recomputed from human labels and is not
claimed.

### 5.6 P6 — explanation faithfulness

The item shows the learner, job, course record and the explanation the Full recommender shows the learner (its
`explanation_path` steps and limitations).

| Column | Rule |
|---|---|
| `q1_faithful` | faithful = every factual claim (what the job requires, what the learner lacks, what the course teaches, prerequisite status, "also enables") is supported by the job text/listed skills, the learner profile or the course record shown. partially = at least one claim unsupported but none contradicted. unfaithful = at least one claim is contradicted by the shown records. |
| `q2_unsupported_claims` | Quote the unsupported or contradicted fragments (free text). |
| `q3_helpful` | 1–5: would this explanation help the learner decide whether to take the course? Optional and reported separately. Helpfulness is not faithfulness. |

### 5.7 P7 — perturbation validity (built from A3's `L03_robustness/perturbation_validity_sample.csv`, 50 items; family and gold spans stay in the key)

`q1_meaning_preserved` (no / partially / yes), `q2_skills_recoverable` (yes/no: the original skill spans are still present
and recognisable in the perturbed text), `q3_fluency` (1–3). A perturbation family whose items are judged
not meaning-preserving in a large share of cases is flagged in L03 as invalid for robustness claims.

## 6. Agreement and statistics

| Scale | Agreement statistics |
|---|---|
| nominal (binary or multi-class) | Cohen's κ, percent agreement, Gwet's AC1 (robust to skewed prevalence: the "kappa paradox"), Krippendorff's α (nominal) |
| ordinal | quadratic-weighted Cohen's κ, Krippendorff's α (ordinal) |
| interval (helpfulness 1–5) | quadratic-weighted κ, Krippendorff's α (interval) |
| free spans (P2) | pairwise span F1, exact and overlap (one-to-one, label-sensitive): F1 = 2·matches/(|A|+|B|), symmetric in A and B (Hripcsak & Rothschild, 2005). Token-level κ as a chance-corrected complement. |

- CIs: 95 % percentile bootstrap over items (2,000 resamples, seed 42). For P2 the unit is the sentence, and span counts
  are pooled within each resample.
- Krippendorff's α uses the coincidence-matrix computation of Krippendorff (2011), with missing values allowed. The
  implementation reproduces that paper's worked example (4 coders × 12 units with missing data: nominal 0.743, ordinal
  0.815, interval 0.849, ratio 0.797) and matches the `krippendorff` package 0.8.1 and scikit-learn's
  `cohen_kappa_score` to ≤ 2.3·10⁻¹⁶ on 74 cases (`reports/skillllm_v2/evals/L11_human_eval/agreement_oracle_check.json`,
  `tests/evals/research/test_agreement.py`).
- Interpretation: we report the coefficients and their CIs and do not attach verbal labels such as "substantial". As a
  planning reference only, α ≥ 0.667 is Krippendorff's lowest level for tentative conclusions. A packet whose main
  question falls below it is reported with that caveat, and its substantive rates are treated as exploratory.
- Substantive metrics (e.g. the share of anchor "false positives" judged valid) use the **adjudicated** label: the agreed
  label where A = B, otherwise the adjudicator's. They get item-bootstrap CIs within each stratum. Strata are never
  pooled without stratum weights.

## 7. Sample size

Targets were fixed before annotation. The numbers come from `human_eval_samples.sample_size_table()`, reproduced in
`L11_human_eval/REPORT.md`.

- A proportion within one P1 stratum (n = 50) has a worst-case 95 % half-width of about ±0.14. A pooled proportion needs
  n = 97 for ±0.10 and n = 196 for ±0.07.
- Cohen's κ at an expected κ = 0.6 with prevalence 0.5: the large-sample SE formula of Cohen (1960) asks for about 246
  items for ±0.10. A Monte-Carlo check gives half-widths of about 0.16, 0.13 and 0.12 at n = 100, 150 and 200. The
  packet sizes (P1 201, P2 100, P3 150, P4a 80, P4b 60, P5 213, P6 120, P7 50) therefore give κ half-widths of roughly 0.11–0.17.
  This precision is adequate to tell poor from good agreement, not to rank fine differences.
- Mean relevance on the 0–3 scale with SD ≈ 1 needs about 171 items for ±0.15.
- Workload estimate, for planning only: P1 about 30 s per item, P2 about 60 s, P3 about 30 s, P4 about 30 s, P5 about
  45 s, P6 about 90 s. That is roughly 9–10 hours per annotator for all packets, plus about 1 hour of training. Packets
  can be annotated in any order, but each packet must be completed by both annotators before its agreement is computed.

## 8. LLM judges (secondary, optional)

An LLM may be used only as an additional annotator column (`C`) on the same packets, with the same rubric text and no
access to answer keys. It is reported next to the human agreement as "human–LLM agreement", never as ground truth and
never as the tie-breaker. No LLM-judge result is part of any hypothesis verdict.

## 9. Procedure checklist

1. Build the packets (done; `packet_manifest.json` lists files, counts and origins).
2. Hand annotators only `reports/skillllm_v2/evals/L11_human_eval/packets/` plus this protocol and SkillSpan Appendix B.
3. Training round T0, then the pass check (section 4).
4. Annotators return `<packet>__A.csv` and `<packet>__B.csv` into `L11_human_eval/annotations/`.
5. Run `--score`. Agreement records become `measured`, and `<packet>__ADJ_todo.csv` lists the disagreements.
6. The adjudicator returns `<packet>__ADJ.csv`. The coordinator computes the substantive metrics with the answer keys
   and re-runs the leaderboard.

## Amendments

_None._
