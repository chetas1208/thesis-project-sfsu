# Evaluation

> **How to read this chapter.** Sections 1–8 (questions, data, systems, metrics, statistics, human evaluation,
> reproducibility) were written before the layer results existed and do not depend on them. Section 9 consists only of
> blocks between `<!-- BEGIN AUTO:… -->` and `<!-- END AUTO:… -->` markers. Those blocks are regenerated from
> `reports/skillllm_v2/evals/LEADERBOARD.md` by
> `PYTHONPATH=src python -m skillllm_v2.evals.research.leaderboard --chapter`. Do not edit inside them, because the next
> refresh overwrites them. No number in this chapter is typed by hand. Each one is either a fact about the data with its
> source file, or it sits inside an auto block that copies a layer record with provenance (file + sha256). Section 10
> (interpretation) was drafted after the final refresh: its verdict list is an auto block, and its prose is qualitative
> (it holds no numbers of its own) and is for the author to review.
>
> Internal name: `SkillLLM_v2`. Proposed research name: *SkillGraphLM*. The naming audit
> (`reports/skillllm_v2/evals/naming_audit/REPORT.md`) recommends *adopt with caveat*: the exact name is unused, but the stem
> "SkillGraph" is crowded, so the full descriptive title and a disambiguation sentence are needed. Nothing has been
> renamed, and the decision is the author's.

## 1. Research questions and hypotheses

The evaluation asks whether the system's four claims hold up when measured on held-out data, with uncertainty
quantified and confounders controlled. The four claims are: accurate span extraction, non-regressive post-processing,
calibrated abstention, and graph-augmented learning-path recommendation. Negative and null results count as results.

| RQ | Question | Layer(s) | Pre-registered hypothesis |
|---|---|---|---|
| RQ1 | How accurately does the fine-tuned anchor extract SkillSpan skill and knowledge spans compared with fairly trained and zero-shot baselines? | L01 | H1 |
| RQ2 | Does accuracy hold for skills never seen in training? | L02 | H3 |
| RQ3 | How robust is extraction to realistic, label-preserving input perturbations? | L03 | (descriptive) |
| RQ4 | Can span confidence support abstention, i.e. does precision rise when the least confident spans are withheld? | L04 | H4 |
| RQ5 | Do the wrapper stages (merging, boundary refinement, verification, linking) preserve what the anchor got right? | L05 | H2 |
| RQ6 | How accurately are extracted mentions linked to a public taxonomy (ESCO), and when should the linker abstain (NIL)? | L06 | (descriptive) |
| RQ7 | Can a compact profile replace full history without hurting downstream ranking? | L07 | H7 |
| RQ8 | Does graph structure add retrieval value over a hybrid lexical+dense retriever when a learner has several gaps? | L08 | H5 |
| RQ9 | Does prerequisite awareness reduce infeasible recommendations without costing relevance? | L09 | H6 |
| RQ10 | Do humans agree with the gold standards and labels that the automatic evaluation relies on? | L11 | (validity evidence) |
| RQ11 | What does the system cost to run (latency, memory)? | L12 | (descriptive) |

The decision rules for H1–H7 were registered before any result of this programme was computed. They are reproduced
verbatim, with a timestamp and the contract's hash, in `hypotheses_preregistration.md`. Each rule yields one verdict:
SUPPORTED, NOT SUPPORTED, INCONCLUSIVE or NOT TESTABLE. L10 re-tests every comparison centrally and applies the
multiplicity corrections (Section 6).

## 2. Evaluation design: twelve layers

| Layer | Question | Primary data | Independent unit | Primary metric(s) | Code (`src/skillllm_v2/evals/research/`) |
|---|---|---|---|---|---|
| L01 extraction | span accuracy vs baselines | SkillSpan test, 974 gold-bearing sentences (+ 2,595 gold-free when available) | sentence (posting-level sensitivity) | exact-span micro-F1 | `extraction_full.py` |
| L02 generalization | seen vs unseen skills | same, bucketed against train surfaces | sentence | bucket F1, Δ seen−unseen | `generalization.py` |
| L03 robustness | perturbation sensitivity | 120 stratified test sentences × 14 families | sentence (paired) | Δ exact F1 vs identity | `perturbations.py`, `robustness.py` |
| L04 calibration | confidence quality, abstention | anchor log-probabilities on dev (fit) and test (report) | sentence (cluster) | ECE, selective risk@80 %, AURC | `calibration_selective.py` |
| L05 wrapper | non-regression of post-processing | stored anchor + replayed stage ladder | sentence | APR, NWG, Δ exact F1 | `wrapper_contribution.py` |
| L06 linking | taxonomy linking + NIL | Decorte et al. (2022) ESCO links of SkillSpan test spans | posting (cluster) | top-1 accuracy, MRR@10, NIL F1 | `linking_eval.py` |
| L07 compaction | compact profile vs full history | 120 seeded synthetic histories | profile | token reduction, ΔnDCG@10, gap Jaccard | `compaction.py` |
| L08 graph retrieval | graph vs hybrid retrieval | 24 held-out benchmark scenarios (+ 240 generated, + public prerequisite data) | scenario | nDCG@10, Recall@10, path validity | `graph_retrieval.py` |
| L09 course recommendation | relevance, gap coverage, feasibility | 24 held-out scenarios, 171-course synthetic catalogue | scenario | P@5, WSGC@5, prerequisite violation rate, nDCG@10 | `course_rec.py` |
| L10 significance | central re-testing | per-unit scores of all layers | as per layer | paired Δ, Holm/BH-adjusted p | `significance.py` |
| L11 human evaluation | validity of gold and labels | 7 blinded packets | item | κ, α, span F1; adjudicated rates | `human_eval_samples.py`, `agreement.py` |
| L12 systems | cost | GPU inference logs | sentence | latency p50/p95, peak VRAM | `systems.py` |

Every layer writes `reports/skillllm_v2/evals/<layer>/results.json` in one record schema (`common.record`). A record
holds the value, the 95 % CI and its method, the number and kind of independent units, the split, the direction, a
status, the sha256 of every input, and caveats. A number that cannot be computed is still written, with status
`not_available`, `pending_gpu` or `pending_human` and a reason. It is never omitted and never estimated. Smoke runs write
to `<layer>/smoke/` and are never quoted.

## 3. Data and splits

### 3.1 SkillSpan (extraction)

SkillSpan (Zhang et al., 2022; CC-BY-4.0; local HF export `data/source/skillspan_hf/`) contains English job-posting
sentences from two public sources, *tech* (StackOverflow) and *house* (STAR), annotated with two BIO layers: **skill**
components and **knowledge** components. These follow the ESCO definitions, and attitudes count as skills. Local
split sizes are 4,800 train, 3,174 dev and 3,569 test sentences (`data/source/skillspan_hf/README.md`). The repository
maps knowledge → `HARD_SKILL` and skill → `SOFT_SKILL`. Character offsets are computed on the space-joined token
string.

**The 974/3,569 caveat.** Every stored extraction prediction of the fine-tuned systems covers only the **974 test
sentences that contain at least one gold span**: 2,265 gold spans, of which 1,174 are knowledge/HARD and 1,091 are
skill/SOFT. These 974 sentences come from only 65 postings (D-A1-01). The remaining 2,595 gold-free test sentences were
never run through the anchor, so its false-positive behaviour on text without skills has never been measured. This
matters in three ways:
1. Precision on the 974-subset overstates precision on realistic input, because every sentence there contains a skill.
2. The legacy baselines (token-lookup, GLiNER zero-shot) were scored on all 3,569 sentences, so legacy tables compared
   different populations. L01 re-scores every system on exactly the 974 ids (paired) and reports the all-3,569 protocol
   separately wherever predictions exist (D-A1-01).
3. A3's GPU job `anchor_test_nospan` ran the anchor on the 2,595 sentences, so L01 reports a full-test score next to
   the subset score (D-C07).

A second caveat: "gold-free" does not mean "skill-free". SkillSpan leaves skills in posting titles and top headlines
untagged (guideline B.3.4). The human packet P2 measures how often humans find skills in gold-free sentences.

The resampling unit is the sentence (the repository's "document"). A cluster bootstrap over the 65 postings is the
sensitivity analysis for headline metrics, because sentences from one posting are correlated.

### 3.2 Seen and unseen skills (L02)

Every test gold span is assigned exactly one bucket relative to the train gold surfaces (D-A1-06). **S1 exact_seen**:
identical surface string. **S2 normalized_seen**: equal after NFKC, casefolding, whitespace collapse, edge-punctuation
stripping that keeps `+`, `#` and a leading `.`, and a last-token plural rule. **S3 canonical_seen_surface_unseen**: a
taxonomy maps it to a concept that some train surface also maps to. The 76 alias rows mined from test predictions are
excluded from this mapping. **S4 fully_unseen**: everything else. Seen = S1 ∪ S2, unseen = S4. Frequency strata are
secondary. The buckets are label-agnostic.

### 3.3 Robustness subset (L03)

120 of the 974 sentences, sampled proportionally over source × gold-span-count strata (seed 42), receive every
applicable perturbation from 14 families, plus an identity copy that serves as the clean reference and a determinism
control (D-A3-03). The perturbations act on the token/separator model, so gold offsets are recomputed exactly.
Context-only edits never touch span tokens. Surface-changing families (case, typo inside a span, alias substitution)
keep the skill and label but change the gold string, so part of their drop reflects a harder target (D-A3-04).
Inapplicable (sentence, family) pairs are listed, not dropped. Validity of the perturbations is checked by humans in
packet P7.

### 3.4 Linking gold (L06)

Decorte et al. (2022) label every SkillSpan TECH and HOUSE test span with its most specific ESCO v1.1.0 skill, or with
LABEL NOT PRESENT / UNDERSPECIFIED (= NIL) (HF `TechWolf/skill-extraction-{tech,house}`, CC-BY-4.0; D-A4-04). The
linker input is the gold span text without context. This isolates linking from extraction and gives an upper bound
for the pipeline. Abstention thresholds are chosen on the Decorte validation split only.

### 3.5 Recommendation benchmark (L08, L09)

- **Benchmark v2:** 48 synthetic scenarios (`inputs/courses/benchmark_scenarios.jsonl`: a learner profile, a target job
  and hidden gold needs), a synthetic 171-course catalogue (`inputs/courses/course_catalog_large.jsonl`) and rule-derived
  graded labels v1 (`inputs/courses/relevance_labels.jsonl`; relevant = grade ≥ 2). The split is 24 dev / 24 test, fixed
  before any tuning by sorting sha256(`skillllm-recsys-loop-v1:<id>`) (`reports/skillllm_v2/recommender_split.json`).
  The Full recommender was tuned on dev only, and the test half is reported once (D-A5-01).
- **Label-independence audit:** two label variants were defined before any system was scored on them. V-visible is the
  same rule applied to system-visible needs only. V-text is a text-mention rule that ignores `teaches_skills` and
  prerequisites (D-A5-05).
- **SG-v1:** 240 generated scenarios with labels from the unchanged rules. Held out entirely and never pooled with
  benchmark v2. They add power and populate the 1-gap stratum, which is empty in the 24 test scenarios (D-A5-07).
- **External validity:** public prerequisite annotations: CPR-Recover (654 real course descriptions, 861 course
  prerequisite edges), AL-CPL and LectureBank, pinned to commit SHAs in `data/external/MANIFEST.jsonl` (D-A5-06). They
  test the graph's prerequisite semantics on real courses, not the recommender's relevance.

### 3.6 Compaction histories (L07)

120 seeded synthetic career histories, 40 each of short, medium and long. A generator emits the truth: held skills with
evidence, goals, true gaps and planted contradictions. It also plants non-held mentions such as distractors and
aspirations, so that a representation that keeps every extracted mention can be wrong about held skills. Extraction is
simulated with an authored confidence rule. A sensitivity arm with uninformative confidence guards against that rule
favouring compaction (D-A4-02).

### 3.7 Human-evaluation packets (L11)

Seven blinded packets: extraction candidates, full-sentence spans (including gold-free sentences), linking,
prerequisite edges, learning paths, course recommendations and explanation faithfulness. A perturbation-validity packet
is built once A3's sample exists. Section 7 and `human_eval_protocol.md` describe them.

## 4. Systems compared

**Extraction.** The *anchor* is Mistral-Small-3.1-24B-Instruct with a LoRA adapter: r = 16, α = 32, 4-bit NF4, 80
epochs, seed 42, trained on SkillSpan train with dev evaluation (`artifacts/skillllm_v2/mistral_lora/model/`). It
decodes greedily with the `original` prompt. Comparators:
- the **safe wrapper**, which keeps the anchor and adds stages under source-aware protection, and its stage ladder
  (D-A2-02);
- the **legacy unconstrained post-processing** (`full_cag_graph`, `mistral_gliner_taxonomy`);
- QLoRA self-training loop adapters;
- the **token-lookup baseline**. It is stored as "BERT baseline" but memorises the most frequent BIO tag per lower-cased
  training token and never loads a transformer (Progress.md integrity flag, D-C11);
- **GLiNER zero-shot** (`urchade/gliner_medium-v2.1`, not fine-tuned on SkillSpan);
- the **fairly trained BERT-family baselines**: JobBERT and bert-base-cased, two-head BIO, dev early stopping, three seeds.
  The best on dev is the H1 comparator (D-A3-02).
H1 is judged primarily against the fair baselines. The token-lookup and zero-shot rows are weak reference points only.

**Linking (L06).** The repository's cascade linker loaded with ESCO, the shipped alias table bridged to ESCO without
gold, and exact-string, BM25, dense MiniLM and random baselines (D-A4-04).

**Retrieval and recommendation (L08, L09).** Random; BM25; dense MiniLM; canonical-skill overlap; graph multi-hop (fixed
2-hop discount 0.5); hybrid = RRF(BM25, dense) with k = 60; hybrid+graph; hybrid+skill-overlap (graph-free structured
control); the shipped `CourseRetriever`. Recommenders: Gap-Overlap Greedy with and without the prerequisite filter, the
tuned Full recommender, Full without prerequisite logic (the H6 ablation), and legacy Full (D-A5-02, D-A5-03). Nothing
in L08/L09 is fitted.

## 5. Metrics

Notation: G = gold span set, P = de-duplicated predicted span set (set semantics on (start, end, label)), per sentence d;
counts are pooled over sentences (micro) unless stated.

### 5.1 Span extraction (L01)

- **Exact-span (strict) P/R/F1.** TP = |P ∩ G| on (start, end, label); P = ΣTP/Σ|P|, R = ΣTP/Σ|G|,
  F1 = 2ΣTP/(2ΣTP + ΣFP + ΣFN). This equals nervaluate *strict*. Per-label F1 restricts to one label; macro-F1 averages
  HARD and SOFT.
- **Token-level F1 (repository "token_f1").** Character-level: overlap of (character offset, label) pairs. It is not
  word-level and not a SkillSpan metric. **Word-token F1:** the same over whitespace words labelled by the spans
  covering them.
- **Partial-span F1 (repository).** A prediction counts fully if it overlaps a not-yet-matched gold span of the same
  label (greedy one-to-one). It is not nervaluate *partial* and is reported only for continuity.
- **IoU.** Per sentence and label, Hungarian matching maximises total IoU = |p ∩ g| / |p ∪ g| (characters, and
  separately words) over same-label pairs with IoU > 0. We report mean IoU over matched pairs and **IoU≥0.5 F1** (a
  one-to-one matching maximising pairs with IoU ≥ 0.5) (D-A1-03).
- **SemEval-2013 / nervaluate modes.** Each gold/prediction pair is classified COR, INC, PAR, MIS or SPU, with
  POS = COR + INC + PAR + MIS and ACT = COR + INC + PAR + SPU. *strict* (boundary and type), *exact* (boundary only) and
  *type* (overlap and type) use P = COR/ACT and R = COR/POS. *partial* (overlap, type ignored) uses
  P = (COR + ½PAR)/ACT and R = (COR + ½PAR)/POS. The implementation re-implements nervaluate 1.2.1 semantics with
  canonical input ordering and is checked count-for-count against the package (D-A1-03, D-A1-05). A one-to-one
  max-matching variant is reported alongside.
- **Boundary-error taxonomy.** After removing exact matches, connected components of the overlap graph between unmatched
  predictions and gold are classified as spurious, missed, left/right extension or truncation, both-side shift, label
  confusion, split, merge or many-to-many, with FP/FN shares that sum exactly to the evaluator's FP/FN (D-A1-04).
- **Spurious-prediction rate on gold-free text** (measured from the `anchor_test_nospan` job): predicted spans per gold-free sentence, and the
  share of gold-free sentences with at least one prediction.

### 5.2 Generalization (L02)

For each bucket b: gold-side recall R_b = exact matches among bucket-b gold spans / |bucket-b gold|. Prediction-side
precision P_b = exact-matching predictions whose own surface falls in b / predictions whose surface falls in b.
F1_b is their harmonic mean. **H3 statistic:** Δ = F1(S1 ∪ S2) − F1(S4), computed on paired bootstrap draws (D-A1-07).
Per-label F1 and mean span length per bucket are reported as confounder checks.

### 5.3 Robustness (L03)

For perturbation family f, on the paired sentences where f applies: Δ_f = F1(perturbed_f) − F1(identity), and relative
degradation Δ_f / F1(identity). The **mean degradation** is the mean of Δ_f over families (unweighted, so frequent
families do not dominate). We also report prediction stability: the share of sentences whose predicted span set is
unchanged, after mapping offsets.

### 5.4 Calibration and selective prediction (L04)

The unit is a predicted span with correctness y ∈ {0, 1} (exact match) and confidence p (D-A2-03).
- **ECE** (equal-width, B = 10 and 15 bins): ECE = Σ_b (|B_b|/n)·|acc(B_b) − conf(B_b)|. **Adaptive ECE** uses
  equal-mass bins. **MCE** = max_b |acc(B_b) − conf(B_b)|.
- **Brier** = (1/n)Σ(p_i − y_i)². **NLL** = −(1/n)Σ[y_i log p_i + (1 − y_i) log(1 − p_i)]. **AUROC** of p for
  separating correct from incorrect spans.
- **Risk–coverage.** Rank spans by score. At coverage c the system keeps the top ⌈cn⌉. Selective risk r(c) = 1 −
  precision of the kept spans. **AURC** = mean of r(c) over all coverage levels (area under the risk–coverage curve).
  **e-AURC** = AURC − AURC of an oracle ranking. **Selective precision@c** is reported at c ∈ {100, 95, 90, 80, 70, 50}
  % with tie-aware expectation, as is **selective F1** with abstained spans counted as misses:
  F1_c = 2·TP_c / (k_c + |G|).
- Calibrators (temperature, Platt, isotonic, a logistic model on log-probability features) are fit and selected **on
  dev only**. The stored constant confidence (0.86) is the constant baseline for H4.

### 5.5 Wrapper contribution (L05)

Per sentence d, with anchor TP/FP sets A_T, A_F and stage TP/FP sets S_T, S_F (D-A2-02):
- **Anchor Preservation Rate** APR = Σ_d |A_T ∩ S_T| / Σ_d |A_T| (pooled).
- TP gained = |S_T \ A_T|, TP destroyed = |A_T \ S_T|, FP added = |S_F \ A_F|, FP removed = |A_F \ S_F|.
- **Net Wrapper Gain** NWG = 1000 · Σ_d[(gained − destroyed) − (added − removed)] / Σ_d |G_d|, i.e. net correct spans
  per 1,000 gold spans.
- Δ exact F1 against the anchor with non-inferiority margin −0.005 (H2). We also report false-rejection rates of the
  verifier and calibrator, merger gold drops, and SOFT-recall loss per rung of the stage ladder (R0 raw anchor … R7
  stored safe wrapper).

### 5.6 Linking (L06)

On linkable mentions: **top-1 accuracy**; **recall@k** (k ∈ {3, 5, 10}: any gold concept among the top-k); **MRR@10** =
mean of 1/rank of the first gold concept (0 if absent from the top 10); RP@5. On all mentions, with NIL as the positive
class: **NIL precision/recall/F1**, and **NIL-aware accuracy** (an abstention is correct exactly when the gold is NIL).
Strata: TECH/HOUSE, HARD/SOFT, seen/unseen surface, variant surface, alias-listed surface.

### 5.7 Profile compaction (L07)

**Token reduction** = mean over profiles of 1 − tokens(compact)/tokens(full), counted with the anchor's Mistral
tokenizer with truncation disabled (D-A4-05). **Downstream non-inferiority:** paired profile-bootstrap Δ of job and
course nDCG@10 (compact − full) against generator-authored relevance, margin −0.02. **Gap Jaccard** = mean over profiles
of |gaps(compact) ∩ gaps(full)| / |gaps(compact) ∪ gaps(full)|, where two empty sets give 1. Also: Kendall τ-b between
the score vectors from compact and full profiles, and retention of truly held skills.

### 5.8 Retrieval and graph paths (L08)

For a ranked list L and graded relevance g(c) ∈ {0..3}, with relevant meaning g ≥ 2:
**P@k** = |relevant ∩ L[1..k]| / k. **Recall@k** = |relevant ∩ L[1..k]| / |relevant|. **MRR** = 1/rank of the first
relevant item (within the top 20; 0 if none). **nDCG@k** = DCG@k / IDCG@k with DCG@k = Σ_{i≤k} (2^{g(L_i)} − 1) /
log₂(i + 1), where IDCG uses the ideal ordering of all judged items. **Path validity** is the share of returned path
edges that exist in the knowledge graph with the stated type and endpoint node types (structural; D-A5-02). Its
pedagogical validity is judged by humans (P4b).

### 5.9 Course recommendation (L09)

With G_s the open gold needs of scenario s, weights w_g ∈ {3 required, 2 implicit, 1 preferred}, T(c) the catalogue
skills taught by course c, P(c) its catalogue prerequisites, and K_s the learner's known and weak skills (D-A5-04):
- **WSGC@5** = Σ_{g∈G_s} w_g·1[g ∈ ∪_{c∈L[1..5]} T(c)] / Σ_{g∈G_s} w_g. It is computed from the catalogue, never from a
  system's own "covered" claims.
- **Precision@5** and **Recall@5** at grade ≥ 2; **nDCG@10**; **MRR**.
- **Prerequisite violation rate@5** = share of the top-5 courses c with P(c) ⊄ K_s ∪ ⋃_{earlier c'} T(c') (chained
  rule).
- **Redundant skill coverage@5** = the share of gap-skill teaching instances in the top-5 that re-teach an
  already-covered gap.
- **Learning-path validity** of the whole list: binary, plus "repairable" if some reordering of the same set is valid.
- **Explanation faithfulness.** Automatic: every claimed edge or skill in an explanation is checked against the
  catalogue and graph. Human: packet P6, with "faithful" meaning every factual claim is supported by the shown records.

### 5.10 Human-evaluation metrics (L11)

Cohen's κ = (p_o − p_e)/(1 − p_e). Weighted κ_w = 1 − Σw_ij O_ij / Σw_ij E_ij, with quadratic weights for ordinal scales.
Gwet's AC1. Krippendorff's α = 1 − (n − 1)·Σ_ck o_ck δ²_ck / Σ_ck n_c n_k δ²_ck, at the nominal, ordinal, interval or
ratio level, with missing data allowed. Span-level inter-annotator F1 = 2·matches/(|A| + |B|), exact and overlap. Full
definitions are in `human_eval_protocol.md` §6.

### 5.11 Systems (L12)

Per-sentence CUDA-synchronised latency (p50, p95, mean), sentences per second, prompt and generated token counts, and
per-job peak CUDA memory (GB), recorded during the anchor GPU jobs (D-A3-01). These are measured on a shared host, and
the host load is recorded with each job.

## 6. Statistical procedure

- **Uncertainty.** Every metric carries a 95 % percentile bootstrap CI over its independent unit: the sentence for
  extraction (posting-level as sensitivity), the scenario for retrieval and recommendation, the profile for compaction,
  the posting cluster for linking, and the item for human evaluation. Spans are never resampled as independent units.
  Micro-F1 CIs recompute pooled F1 from per-sentence TP/FP/FN in each resample. Resamples: 2,000; 10,000 for headline
  metrics and hypothesis contrasts; seed 42.
- **Paired comparisons** on identical units use a paired bootstrap of the difference (shared draws). The two-sided p
  inverts the percentile interval with +1 smoothing and counts ties on both sides (D-A2-01). Extraction comparisons also
  use **approximate randomization**: swap each unit's two outputs with probability ½, enumerated exactly when at most 16
  units differ, otherwise 10,000 seeded shuffles. Recommendation comparisons also use a paired sign-flip test.
- **Non-inferiority** holds when the lower bound of the two-sided 95 % CI of the difference exceeds the margin. This is
  equivalent to a one-sided α = 0.025 test.
- **Multiplicity.** Holm within each declared family (one family per layer per hypothesis), and Benjamini–Hochberg over
  all tests as a secondary view (L10).
- **Power.** Each comparison reports its minimum detectable effect at 80 % power from the bootstrap SE.
- **No tuning on test.** Thresholds, calibrators, fusion weights and model selection use SkillSpan dev, the Decorte
  validation split, or the recommender's dev half. Components known to have been tuned or mined on test are flagged
  rather than reused as clean (`threats_to_validity.md`).
- **Pre-registration.** H1–H7 and their decision rules were fixed before any result (`hypotheses_preregistration.md`).
  Definitions that each owning agent fixed before running are logged with timestamps in `Decisions.md` (D-A1-*, D-A2-*,
  D-A3-*, D-A4-*, D-A5-*, D-A6-*).

## 7. Human evaluation

Two independent annotators and one adjudicator label seven blinded packets: P1 extraction candidate spans in six
strata, P2 full-sentence spans including gold-free sentences, P3 linking, P4a prerequisite edges, P4b learning paths,
P5 course recommendations (pooled across two systems, re-blinded legacy sample included) and P6 explanation
faithfulness. A synthetic training round with guideline-cited answers comes first. System identity, gold, strata and
rule labels are hidden. Each annotator gets an independently shuffled order. Answer keys are stored in a separate tree.
Agreement is reported as κ, AC1, α and span F1 with bootstrap CIs. Substantive rates use adjudicated labels per stratum.
Sample sizes target ±0.10 on pooled proportions. An LLM may be used only as an extra, secondary annotator column. Full
protocol: `human_eval_protocol.md`. Every human metric is `pending_human` until annotations are returned.

## 8. Reproducibility

- **Runner:** `scripts/evals/run_research_evals.sh [--smoke]` runs L01 → L09, L12, L11, then L10, then the leaderboard
  (`--chapter`) and the evaluation card. Each step goes through the server-guardrails CPU launcher, and the runner stops
  at the first failure, naming the layer, its log and the resume command. GPU jobs are separate
  (`scripts/evals/gpu_*.sh`) and signal completion with `DONE.json`.
- **Evaluation card:** `reports/skillllm_v2/evals/repro/EVAL_CARD.md` (+ `eval_card.json`). It records the git commit
  and dirty state, sha256 of every dataset file and external download, base-model config and adapter hashes, Hugging Face
  cache revisions, every `predictions.jsonl` hash, the taxonomy, graph and index inputs, the extraction prompts (copied
  to `repro/prompts/` with hashes), seeds, pip freezes of both interpreters, CUDA/driver/torch versions, hardware, the
  exact command of every layer, and an index of every intermediate output.
- **Leaderboard:** `reports/skillllm_v2/evals/LEADERBOARD.md` and LaTeX tables `docs/thesis/tables/evaluation_*.tex`,
  generated from the layer records.

## 9. Results

> Everything in this section is generated. A block that says a layer "does not exist yet", or a value shown as `-` or
> `(pending_*)`, means the measurement has not been made. It must not be read as zero, as failure or as success.

### 9.1 Status of the layers

<!-- BEGIN AUTO:layer_status -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

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
<!-- END AUTO:layer_status -->

### 9.2 Headline metrics

<!-- BEGIN AUTO:headline -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

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
<!-- END AUTO:headline -->

### 9.3 Pre-registered hypotheses

<!-- BEGIN AUTO:hypotheses -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

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
<!-- END AUTO:hypotheses -->

### 9.4 Extraction (L01)

<!-- BEGIN AUTO:layer:L01_extraction -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| mistral_lora | exact_span_f1 | 0.6701 | [0.6465, 0.6938] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | token_f1 | 0.7879 | [0.7701, 0.8053] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | H1_verdict | 0.0370 | [0.0165, 0.0573] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| bert_baseline | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| gliner_baseline | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_boundary | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_gliner_taxonomy | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| full_cag_graph | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| safe_wrapper | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| safe_wrapper_boundary_refiner | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| safe_wrapper_mistral_gliner_taxonomy | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| safe_wrapper_full_cag_graph | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| qlora_loop_v1 | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| qlora_loop_v2 | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| qlora_loop_v3 | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| qlora_loop_v4 | reproduction_max_abs_diff_vs_stored | 0.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | exact_span_precision | 0.6735 | [0.6480, 0.6985] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | exact_span_recall | 0.6667 | [0.6406, 0.6926] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | partial_span_precision | 0.8180 | [0.7964, 0.8387] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | partial_span_recall | 0.8097 | [0.7884, 0.8301] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | partial_span_f1 | 0.8138 | [0.7962, 0.8307] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | token_precision | 0.7748 | [0.7534, 0.7966] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | token_recall | 0.8014 | [0.7798, 0.8227] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | hard_skill_f1 | 0.7265 | [0.6950, 0.7567] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | soft_skill_f1 | 0.6040 | [0.5716, 0.6356] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | macro_f1 | 0.6652 | [0.6423, 0.6882] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | micro_f1 | 0.6701 | [0.6465, 0.6938] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | word_token_f1 | 0.7809 | [0.7626, 0.7986] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | word_token_precision | 0.7684 | [0.7466, 0.7906] | 974 | sentence | test[974 sentences with >=1 gold span] |

_1340 further measured records: see `reports/skillllm_v2/evals/L01_extraction/REPORT.md`._

Source: `reports/skillllm_v2/evals/L01_extraction/results.json` (generated 2026-11-18T06:54:54+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L01_extraction -->

### 9.5 Generalization to unseen skills (L02)

<!-- BEGIN AUTO:layer:L02_generalization -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| mistral_lora | unseen_exact_f1 | 0.5874 | [0.5589, 0.6157] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | H3_verdict | 0.2063 | [0.1704, 0.2415] | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | bucket_size_S1_exact_seen | 780.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | bucket_size_S2_normalized_seen | 114.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | bucket_size_S3_canonical_seen_surface_unseen | 17.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | bucket_size_S4_fully_unseen | 1,354.0 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | frequency_stratum_size_0 | 1,371.0 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | frequency_stratum_size_1 | 205.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | frequency_stratum_size_2-5 | 297.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | frequency_stratum_size_>5 | 392.0000 | - | 974 | sentence | test[974 sentences with >=1 gold span] |
| dataset | train_test_mention_share_exact_seen | 0.3444 | - | 2265 | mention | test[974 sentences with >=1 gold span] |
| dataset | train_test_mention_share_normalized_seen | 0.3947 | - | 2265 | mention | test[974 sentences with >=1 gold span] |
| dataset | train_test_type_share_seen_normalized | 0.2104 | - | 2265 | mention | test[974 sentences with >=1 gold span] |
| dataset | train_test_type_jaccard_normalized | 0.0679 | - | 2265 | mention | test[974 sentences with >=1 gold span] |
| dataset | train_test_type_jaccard_exact | 0.0613 | - | 2265 | mention | test[974 sentences with >=1 gold span] |
| dataset | train_dev_mention_share_exact_seen | 0.2880 | - | 2163 | mention | dev[all gold spans] |
| dataset | train_dev_mention_share_normalized_seen | 0.3287 | - | 2163 | mention | dev[all gold spans] |
| dataset | train_dev_type_share_seen_normalized | 0.1908 | - | 2163 | mention | dev[all gold spans] |
| dataset | train_dev_type_jaccard_normalized | 0.0646 | - | 2163 | mention | dev[all gold spans] |
| dataset | train_dev_type_jaccard_exact | 0.0593 | - | 2163 | mention | dev[all gold spans] |
| dataset | alias_table_test_derived_rows | 76.0000 | - | 711 | alias_row | taxonomy |
| dataset | esco_concepts_loaded | 13,939.0 | - | 13939 | concept | taxonomy |
| mistral_lora | delta_seen_minus_unseen_recall_len1words | 0.1270 | [0.0470, 0.2061] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | delta_seen_minus_unseen_recall_len2words | 0.2088 | [0.1273, 0.2880] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | delta_seen_minus_unseen_recall_len3+words | 0.2364 | [0.1491, 0.3266] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | S1_exact_seen_exact_recall | 0.7962 | [0.7596, 0.8313] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | S1_exact_seen_exact_precision | 0.8107 | [0.7763, 0.8440] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | S1_exact_seen_exact_f1 | 0.8034 | [0.7735, 0.8313] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | S2_normalized_seen_exact_recall | 0.6930 | [0.6061, 0.7757] | 974 | sentence | test[974 sentences with >=1 gold span] |
| mistral_lora | S2_normalized_seen_exact_precision | 0.7596 | [0.6729, 0.8421] | 974 | sentence | test[974 sentences with >=1 gold span] |

_717 further measured records: see `reports/skillllm_v2/evals/L02_generalization/REPORT.md`._

Source: `reports/skillllm_v2/evals/L02_generalization/results.json` (generated 2026-11-18T06:58:38+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L02_generalization -->

### 9.6 Robustness (L03)

<!-- BEGIN AUTO:layer:L03_robustness -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| anchor | mean_degradation_exact_f1 | 0.0223 | [0.0035, 0.0417] | 120 | document | test |
| anchor | delta_exact_span_f1:whitespace | 0.0312 | [-0.0081, 0.0684] | 120 | document | test |
| anchor | delta_partial_span_f1:whitespace | -0.0255 | [-0.0562, 0.0004] | 120 | document | test |
| anchor | delta_token_f1:whitespace | 0.0159 | [-0.0243, 0.0543] | 120 | document | test |
| anchor | delta_exact_span_f1:bullets_prose | -0.0125 | [-0.0354, 0.0089] | 120 | document | test |
| anchor | delta_partial_span_f1:bullets_prose | -0.0033 | [-0.0228, 0.0147] | 120 | document | test |
| anchor | delta_token_f1:bullets_prose | 0.0125 | [-0.0080, 0.0386] | 120 | document | test |
| anchor | delta_exact_span_f1:case_lower | -0.0071 | [-0.0300, 0.0140] | 103 | document | test |
| anchor | delta_partial_span_f1:case_lower | -0.0099 | [-0.0315, 0.0094] | 103 | document | test |
| anchor | delta_token_f1:case_lower | -0.0026 | [-0.0206, 0.0126] | 103 | document | test |
| anchor | delta_exact_span_f1:case_upper | -0.0294 | [-0.0716, 0.0130] | 120 | document | test |
| anchor | delta_partial_span_f1:case_upper | -0.0271 | [-0.0568, 0.0018] | 120 | document | test |
| anchor | delta_token_f1:case_upper | -0.0203 | [-0.0527, 0.0077] | 120 | document | test |
| anchor | delta_exact_span_f1:case_title | -0.0201 | [-0.0602, 0.0197] | 113 | document | test |
| anchor | delta_partial_span_f1:case_title | -0.0270 | [-0.0564, -0.0010] | 113 | document | test |
| anchor | delta_token_f1:case_title | -0.0194 | [-0.0480, 0.0052] | 113 | document | test |
| anchor | delta_exact_span_f1:typo_outside | -0.0141 | [-0.0396, 0.0090] | 93 | document | test |
| anchor | delta_partial_span_f1:typo_outside | -0.0107 | [-0.0334, 0.0094] | 93 | document | test |
| anchor | delta_token_f1:typo_outside | -0.0005 | [-0.0216, 0.0175] | 93 | document | test |
| anchor | delta_exact_span_f1:typo_inside | -0.0544 | [-0.0954, -0.0166] | 118 | document | test |
| anchor | delta_partial_span_f1:typo_inside | -0.0380 | [-0.0684, -0.0129] | 118 | document | test |
| anchor | delta_token_f1:typo_inside | -0.0172 | [-0.0386, 0.0020] | 118 | document | test |
| anchor | delta_exact_span_f1:boilerplate | -0.0290 | [-0.0593, -0.0011] | 120 | document | test |
| anchor | delta_partial_span_f1:boilerplate | -0.0177 | [-0.0383, 0.0001] | 120 | document | test |
| anchor | delta_token_f1:boilerplate | -0.0187 | [-0.0427, 0.0018] | 120 | document | test |
| anchor | delta_exact_span_f1:distractor | -0.0497 | [-0.0862, -0.0181] | 120 | document | test |
| anchor | delta_partial_span_f1:distractor | -0.0473 | [-0.0776, -0.0204] | 120 | document | test |
| anchor | delta_token_f1:distractor | -0.0399 | [-0.0727, -0.0132] | 120 | document | test |
| anchor | delta_exact_span_f1:paraphrase_context | 0.0084 | [-0.0294, 0.0469] | 51 | document | test |
| anchor | delta_partial_span_f1:paraphrase_context | 0.0022 | [-0.0255, 0.0269] | 51 | document | test |

_294 further measured records: see `reports/skillllm_v2/evals/L03_robustness/REPORT.md`._

Not measured: not_available 5. Reasons:
- `not_available` x5: only one soft-skill span per sentence is annotated; other mentions are unlabelled, so precision/F1 are not measurable

Source: `reports/skillllm_v2/evals/L03_robustness/results.json` (generated 2026-11-18T07:35:14+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L03_robustness -->

### 9.7 Calibration and selective prediction (L04)

<!-- BEGIN AUTO:layer:L04_calibration -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| anchor_logprob:primary[sum_lp]_platt | sel_risk@80 | 0.2664 | [0.2394, 0.2950] | 974 | document (cluster of spans) | test |
| anchor_logprob:primary[sum_lp]_platt | ece_w10 | 0.0187 | [0.0133, 0.0422] | 974 | document (cluster of spans) | test |
| anchor_logprob | H4_verdict | 0.0587 | [0.0477, 0.0682] | 974 | document | test |
| anchor_stored:constant_0.86 | precision | 0.6735 | [0.6475, 0.6994] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | aurc | 0.3265 | [0.3006, 0.3525] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | aurc_oracle | 0.0604 | [0.0506, 0.0712] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | e_aurc | 0.2661 | [0.2500, 0.2814] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | auroc | 0.5000 | [0.5000, 0.5000] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_prec@100 | 0.6735 | [0.6475, 0.6994] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_risk@100 | 0.3265 | [0.3006, 0.3525] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_f1@100 | 0.6701 | [0.6464, 0.6944] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_prec@95 | 0.6735 | [0.6475, 0.6994] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_risk@95 | 0.3265 | [0.3006, 0.3525] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_f1@95 | 0.6528 | [0.6298, 0.6764] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_prec@90 | 0.6735 | [0.6475, 0.6994] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_risk@90 | 0.3265 | [0.3006, 0.3525] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_f1@90 | 0.6347 | [0.6122, 0.6575] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_prec@80 | 0.6735 | [0.6475, 0.6994] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_risk@80 | 0.3265 | [0.3006, 0.3525] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_f1@80 | 0.5954 | [0.5743, 0.6166] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_prec@70 | 0.6735 | [0.6475, 0.6994] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_risk@70 | 0.3265 | [0.3006, 0.3525] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_f1@70 | 0.5514 | [0.5319, 0.5715] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_prec@50 | 0.6735 | [0.6475, 0.6994] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_risk@50 | 0.3265 | [0.3006, 0.3525] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | sel_f1@50 | 0.4460 | [0.4301, 0.4625] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | ece_w10 | 0.1865 | [0.1606, 0.2125] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | ece_m10 | 0.1865 | [0.1606, 0.2125] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | mce_w10 | 0.1865 | [0.1606, 0.2125] | 974 | document (cluster of spans) | test |
| anchor_stored:constant_0.86 | ece_w15 | 0.1865 | [0.1606, 0.2125] | 974 | document (cluster of spans) | test |

_1379 further measured records: see `reports/skillllm_v2/evals/L04_calibration/REPORT.md`._

Source: `reports/skillllm_v2/evals/L04_calibration/results.json` (generated 2026-11-18T07:08:45+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L04_calibration -->

### 9.8 Safe-wrapper contribution (L05)

<!-- BEGIN AUTO:layer:L05_wrapper -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| R7_safe_wrapper_stored | exact_span_f1 | 0.6719 | [0.6477, 0.6954] | 974 | document | test |
| R7_safe_wrapper_stored | apr | 0.9987 | [0.9958, 1.0000] | 974 | document | test |
| R7_safe_wrapper_stored | H2_verdict | 0.0000 | [0.0000, 0.0000] | 974 | document | test |
| R0_raw_mistral | exact_span_f1 | 0.6701 | [0.6465, 0.6938] | 974 | document | test |
| R0_raw_mistral | exact_span_precision | 0.6735 | [0.6480, 0.6985] | 974 | document | test |
| R0_raw_mistral | exact_span_recall | 0.6667 | [0.6406, 0.6926] | 974 | document | test |
| R0_raw_mistral | token_f1 | 0.7879 | [0.7701, 0.8053] | 974 | document | test |
| R0_raw_mistral | hard_skill_f1 | 0.7265 | [0.6950, 0.7567] | 974 | document | test |
| R0_raw_mistral | soft_skill_f1 | 0.6040 | [0.5716, 0.6356] | 974 | document | test |
| R0_raw_mistral | soft_skill_recall | 0.5747 | [0.5389, 0.6095] | 974 | document | test |
| R0_raw_mistral | apr | 1.0000 | [1.0000, 1.0000] | 974 | document | test |
| R0_raw_mistral | soft_apr | 1.0000 | [1.0000, 1.0000] | 974 | document | test |
| R0_raw_mistral | nwg_per_1000_gold | 0.0000 | [0.0000, 0.0000] | 974 | document | test |
| R0_raw_mistral | gained_tp_per_1000_gold | 0.0000 | [0.0000, 0.0000] | 974 | document | test |
| R0_raw_mistral | destroyed_tp_per_1000_gold | 0.0000 | [0.0000, 0.0000] | 974 | document | test |
| R0_raw_mistral | added_fp_per_1000_gold | 0.0000 | [0.0000, 0.0000] | 974 | document | test |
| R0_raw_mistral | removed_fp_per_1000_gold | 0.0000 | [0.0000, 0.0000] | 974 | document | test |
| R0_raw_mistral | tp_gained (= recovered FN) | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | anchor_tp_destroyed | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | fp_added | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | fp_removed | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | verifier_false_rejections | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | calibrator_false_rejections | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | merger_dropped_gold_new_candidates | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | merger_dropped_gold_anchor_spans | 0.0000 | - | 974 | document | test |
| R0_raw_mistral | n_docs_changed_vs_anchor | 0.0000 | - | 974 | document | test |
| R1_boundary | exact_span_f1 | 0.6714 | [0.6473, 0.6951] | 974 | document | test |
| R1_boundary | exact_span_precision | 0.6748 | [0.6494, 0.7001] | 974 | document | test |
| R1_boundary | exact_span_recall | 0.6680 | [0.6414, 0.6938] | 974 | document | test |
| R1_boundary | token_f1 | 0.7880 | [0.7695, 0.8055] | 974 | document | test |

_796 further measured records: see `reports/skillllm_v2/evals/L05_wrapper/REPORT.md`._

Source: `reports/skillllm_v2/evals/L05_wrapper/results.json` (generated 2026-11-17T21:26:54+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L05_wrapper -->

### 9.9 Taxonomy linking (L06)

<!-- BEGIN AUTO:layer:L06_linking -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| shipped_bridge | top1_acc_linkable | 0.1573 | [0.1258, 0.1903] | 55 | posting (cluster); mentions in details | test |
| shipped_bridge | recall@5 | 0.1620 | [0.1295, 0.1968] | 55 | posting (cluster); mentions in details | test |
| shipped_bridge | mrr@10 | 0.1594 | [0.1280, 0.1930] | 55 | posting (cluster); mentions in details | test |
| cascade_esco@tau_val | top1_acc_linkable | 0.3239 | [0.2856, 0.3599] | 55 | posting (cluster); mentions in details | test |
| cascade_esco@tau_val | recall@5 | 0.5056 | [0.4604, 0.5505] | 55 | posting (cluster); mentions in details | test |
| cascade_esco@tau_val | mrr@10 | 0.3999 | [0.3619, 0.4380] | 55 | posting (cluster); mentions in details | test |
| legacy_money_table | legacy_linking_accuracy_is_coverage | 0.8904 | - | 0 | n/a | test |
| legacy:full_cag_graph | diagnostic_legacy_coverage_reproduced | 0.8904 | - | 2244 | predicted span | test |
| legacy:mistral_gliner_taxonomy | diagnostic_legacy_coverage_reproduced | 0.8903 | - | 2251 | predicted span | test |
| legacy:safe_wrapper | diagnostic_legacy_coverage_reproduced | 0.8898 | - | 2233 | predicted span | test |
| alias_table | diagnostic_test_gold_mentions_in_alias_table | 0.4137 | - | 2265 | test gold mention | test |
| shipped_linker | diagnostic_null_control_coverage[decorte_underspecified_mentions] | 0.8198 | - | 111 | string | test |
| shipped_linker | diagnostic_null_control_coverage[random_non_gold_ngrams] | 0.6458 | - | 861 | string | test |
| random | top1_acc_linkable | 0.0000 | [0.0000, 0.0000] | 55 | posting (cluster); mentions in details | test |
| random | recall@3 | 0.0000 | [0.0000, 0.0000] | 55 | posting (cluster); mentions in details | test |
| random | recall@5 | 0.0000 | [0.0000, 0.0000] | 55 | posting (cluster); mentions in details | test |
| random | recall@10 | 0.0000 | [0.0000, 0.0000] | 55 | posting (cluster); mentions in details | test |
| random | mrr@10 | 0.0000 | [0.0000, 0.0000] | 55 | posting (cluster); mentions in details | test |
| random | rp@5 | 0.0000 | [0.0000, 0.0000] | 55 | posting (cluster); mentions in details | test |
| random | nil_aware_accuracy | 0.0000 | [0.0000, 0.0000] | 55 | posting | test |
| random | nil_precision | 0.0000 | [0.0000, 0.0000] | 55 | posting | test |
| random | nil_recall | 0.0000 | [0.0000, 0.0000] | 55 | posting | test |
| random | nil_f1 | 0.0000 | [0.0000, 0.0000] | 55 | posting | test |
| random | top1_acc_linkable[tech] | 0.0000 | [0.0000, 0.0000] | 29 | posting (cluster); mentions in details | test |
| random | recall@5[tech] | 0.0000 | [0.0000, 0.0000] | 29 | posting (cluster); mentions in details | test |
| random | mrr@10[tech] | 0.0000 | [0.0000, 0.0000] | 29 | posting (cluster); mentions in details | test |
| random | nil_precision[tech] | 0.0000 | [0.0000, 0.0000] | 29 | posting | test |
| random | nil_recall[tech] | 0.0000 | [0.0000, 0.0000] | 29 | posting | test |
| random | nil_f1[tech] | 0.0000 | [0.0000, 0.0000] | 29 | posting | test |
| random | top1_acc_linkable[house] | 0.0000 | [0.0000, 0.0000] | 26 | posting (cluster); mentions in details | test |

_797 further measured records: see `reports/skillllm_v2/evals/L06_linking/REPORT.md`._

Not measured: pending_human 1. Reasons:
- `pending_human` x1: needs human annotation of annotation_sample.csv (A6 / L11)

Source: `reports/skillllm_v2/evals/L06_linking/results.json` (generated 2026-11-18T08:23:32+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L06_linking -->

### 9.10 Profile compaction (L07)

<!-- BEGIN AUTO:layer:L07_compaction -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| compact@0.5[informative] | token_reduction_vs_full | 0.0928 | [0.0110, 0.1674] | 120 | profile | synthetic |
| compact@0.5[informative] | job_ndcg10 | 0.2487 | [0.2007, 0.2993] | 119 | profile | synthetic |
| compact@0.5[informative] | course_ndcg10 | 0.7284 | [0.6939, 0.7604] | 117 | profile | synthetic |
| compact@0.5[informative] | gap_jaccard_vs_full | 0.3483 | [0.2961, 0.4014] | 120 | profile | synthetic |
| compact@0.5[informative] | H7_verdict | 0.1615 | [0.1004, 0.2241] | 117 | profile | synthetic |
| compact@shipped[informative] | H7_verdict | 0.0000 | [0.0000, 0.0000] | 117 | profile | synthetic |
| compact@0.5[uninformative] | H7_verdict | 0.1615 | [0.1004, 0.2241] | 117 | profile | synthetic |
| compact@shipped[uninformative] | H7_verdict | 0.0000 | [0.0000, 0.0000] | 117 | profile | synthetic |
| full[informative] | tokens | 323.8250 | [289.0323, 357.5100] | 120 | profile | synthetic |
| full[informative] | token_reduction_vs_full | 0.0000 | [0.0000, 0.0000] | 120 | profile | synthetic |
| full[informative] | skill_precision | 0.7016 | [0.6843, 0.7192] | 120 | profile | synthetic |
| full[informative] | skill_recall | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | skill_recall_vs_full | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | goal_retained | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | gap_precision_truth | 1.0000 | [1.0000, 1.0000] | 93 | profile | synthetic |
| full[informative] | gap_recall_truth | 0.5659 | [0.5000, 0.6317] | 120 | profile | synthetic |
| full[informative] | evidence_retention | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | contradiction_preservation | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | job_ndcg10 | 0.9137 | [0.8896, 0.9376] | 119 | profile | synthetic |
| full[informative] | course_ndcg10 | 0.5670 | [0.4974, 0.6359] | 117 | profile | synthetic |
| full[informative] | job_kendall_vs_full | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | job_top5_overlap_vs_full | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | course_kendall_vs_full | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | course_top5_overlap_vs_full | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | gap_jaccard_vs_full | 1.0000 | [1.0000, 1.0000] | 120 | profile | synthetic |
| full[informative] | contradiction_retraction | 1.0000 | [1.0000, 1.0000] | 58 | profile | synthetic |
| full[informative] | contradiction_label_conflict | 1.0000 | [1.0000, 1.0000] | 48 | profile | synthetic |
| full[informative] | contradiction_goal_change | 1.0000 | [1.0000, 1.0000] | 42 | profile | synthetic |
| full[informative] | contradiction_proficiency | 1.0000 | [1.0000, 1.0000] | 52 | profile | synthetic |
| compact@shipped[informative] | tokens | 2,778.0 | [2,520.2, 3,025.0] | 120 | profile | synthetic |

_530 further measured records: see `reports/skillllm_v2/evals/L07_compaction/REPORT.md`._

Source: `reports/skillllm_v2/evals/L07_compaction/results.json` (generated 2026-11-17T21:16:49+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L07_compaction -->

### 9.11 Graph-augmented retrieval (L08)

<!-- BEGIN AUTO:layer:L08_graph_retrieval -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| hybrid+graph | recall@10 | 0.8424 | [0.7578, 0.9199] | 24 | scenario | test |
| hybrid+graph | ndcg@10 | 0.8244 | [0.7777, 0.8691] | 24 | scenario | test |
| SkillLLM_v2 Full recommender (explanation paths) | path_validity | 1.0000 | [1.0000, 1.0000] | 24 | scenario | test |
| hybrid+graph vs hybrid(bm25+dense) | H5_verdict | 0.0855 | [0.0478, 0.1204] | 24 | scenario | test |
| kg_builder(as_wired) | teaches_edge_precision_vs_authored | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | teaches_edge_recall_vs_authored | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | prerequisite_edge_precision_vs_authored | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | prerequisite_edge_recall_vs_authored | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | enrichment_only_pairs_reaching_graph | 0.0000 | - | 171 | course | all |
| prereq_parser+linker | precision_vs_authored | 0.9274 | - | 171 | course | all |
| prereq_parser+linker | recall_vs_authored | 0.9274 | - | 171 | course | all |
| prereq_parser+linker | f1_vs_authored | 0.9274 | - | 171 | course | all |
| course_skill_extractor | precision_vs_authored | 0.6218 | - | 171 | course | all |
| course_skill_extractor | recall_vs_authored | 0.6258 | - | 171 | course | all |
| course_skill_extractor | f1_vs_authored | 0.6238 | - | 171 | course | all |
| taxonomy_linker | over_merge_groups_on_authored_skills | 19.0000 | - | 171 | course | all |
| taxonomy_linker | canonical_id_differs_from_authored_name_share | 0.3894 | - | 171 | course | all |
| kg_builder(as_wired) | over_merged_skill_ids | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | alias_duplicate_skill_node_groups | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | orphan_nodes | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | courses_without_teaches_edges | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | dangling_prerequisite_skills | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | infeasible_courses_from_empty_profile | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | prerequisite_cycles_nontrivial_scc | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | courses_teaching_own_prerequisite | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | edge_type_consistency | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | teaches_provenance_fields_present | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | teaches_evidence_nonempty | 1.0000 | - | 171 | course | all |
| kg_builder(as_wired) | teaches_evidence_informative | 0.0000 | - | 171 | course | all |
| kg_builder(as_wired) | requires_prerequisite_provenance_fields_present | 1.0000 | - | 171 | course | all |

_139 further measured records: see `reports/skillllm_v2/evals/L08_graph_retrieval/REPORT.md`._

Not measured: not_available 5. Reasons:
- `not_available` x2: no synthetic prerequisite pair falls on an annotated concept pair
- `not_available` x2: stratum '1-gap' is empty in the test half
- `not_available` x1: No relevance judgments exist for job-gap queries on a public catalogue; creating them needs human (or LLM, secondary-only) labels. CPR-Recov

Source: `reports/skillllm_v2/evals/L08_graph_retrieval/results.json` (generated 2026-11-18T07:08:07+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L08_graph_retrieval -->

### 9.12 Course recommendation (L09)

<!-- BEGIN AUTO:layer:L09_course_rec -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| SkillLLM_v2 Full | precision@5 | 0.6083 | [0.5333, 0.6833] | 24 | scenario | test |
| SkillLLM_v2 Full | ndcg@10 | 0.8854 | [0.8514, 0.9160] | 24 | scenario | test |
| SkillLLM_v2 Full | wsgc@5 | 0.8382 | [0.7735, 0.8994] | 24 | scenario | test |
| SkillLLM_v2 Full | prereq_violation_rate | 0.0000 | [0.0000, 0.0000] | 24 | scenario | test |
| SkillLLM_v2 Full vs Full-noPrereq (ablation) | H6_verdict | -0.1833 | [-0.2583, -0.1167] | 24 | scenario | test |
| random | precision@5 | 0.0167 | [0.0000, 0.0417] | 24 | scenario | test |
| random | recall@5 | 0.0153 | [0.0000, 0.0389] | 24 | scenario | test |
| random | ndcg@10 | 0.0453 | [0.0246, 0.0697] | 24 | scenario | test |
| random | mrr@10 | 0.0432 | [0.0152, 0.0768] | 24 | scenario | test |
| random | wsgc@5 | 0.0854 | [0.0362, 0.1450] | 24 | scenario | test |
| random | prereq_violation_rate | 0.5333 | [0.4583, 0.6083] | 24 | scenario | test |
| random | redundant_skill_coverage@5 | 0.0000 | [0.0000, 0.0000] | 24 | scenario | test |
| random | redundancy@5(existing) | 0.0000 | [0.0000, 0.0000] | 24 | scenario | test |
| random | learning_path_validity | 0.0000 | [0.0000, 0.0000] | 24 | scenario | test |
| random | learning_path_repairable | 0.0417 | [0.0000, 0.1250] | 24 | scenario | test |
| random | n_returned | 10.0000 | [10.0000, 10.0000] | 24 | scenario | test |
| lexical(BM25) | precision@5 | 0.3917 | [0.3000, 0.4833] | 24 | scenario | test |
| lexical(BM25) | recall@5 | 0.4172 | [0.3244, 0.5128] | 24 | scenario | test |
| lexical(BM25) | ndcg@10 | 0.6040 | [0.5334, 0.6744] | 24 | scenario | test |
| lexical(BM25) | mrr@10 | 0.7622 | [0.6337, 0.8854] | 24 | scenario | test |
| lexical(BM25) | wsgc@5 | 0.7188 | [0.6515, 0.7851] | 24 | scenario | test |
| lexical(BM25) | prereq_violation_rate | 0.3167 | [0.2167, 0.4167] | 24 | scenario | test |
| lexical(BM25) | redundant_skill_coverage@5 | 0.2648 | [0.1937, 0.3349] | 24 | scenario | test |
| lexical(BM25) | redundancy@5(existing) | 0.0351 | [0.0235, 0.0470] | 24 | scenario | test |
| lexical(BM25) | learning_path_validity | 0.0833 | [0.0000, 0.2083] | 24 | scenario | test |
| lexical(BM25) | learning_path_repairable | 0.1250 | [0.0000, 0.2500] | 24 | scenario | test |
| lexical(BM25) | n_returned | 10.0000 | [10.0000, 10.0000] | 24 | scenario | test |
| semantic(MiniLM) | precision@5 | 0.4250 | [0.3250, 0.5333] | 24 | scenario | test |
| semantic(MiniLM) | recall@5 | 0.4244 | [0.3253, 0.5226] | 24 | scenario | test |
| semantic(MiniLM) | ndcg@10 | 0.6013 | [0.5372, 0.6622] | 24 | scenario | test |

_355 further measured records: see `reports/skillllm_v2/evals/L09_course_rec/REPORT.md`._

Source: `reports/skillllm_v2/evals/L09_course_rec/results.json` (generated 2026-11-18T07:14:29+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L09_course_rec -->

### 9.13 Central significance testing (L10)

<!-- BEGIN AUTO:layer:L10_significance -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| anchor vs gliner_zero_shot | delta_exact_span_f1 | 0.5887 | [0.5599, 0.6164] | 974 | document | test |
| anchor vs gliner_zero_shot | delta_token_f1 | 0.6675 | [0.6411, 0.6925] | 974 | document | test |
| anchor vs gliner_zero_shot | delta_hard_skill_f1 | 0.6674 | [0.6311, 0.7024] | 974 | document | test |
| anchor vs gliner_zero_shot | delta_soft_skill_f1 | 0.5013 | [0.4610, 0.5395] | 974 | document | test |
| anchor vs token_lookup_bert | delta_exact_span_f1 | 0.5325 | [0.5110, 0.5541] | 974 | document | test |
| anchor vs token_lookup_bert | delta_token_f1 | 0.2689 | [0.2493, 0.2891] | 974 | document | test |
| anchor vs token_lookup_bert | delta_hard_skill_f1 | 0.4524 | [0.4214, 0.4821] | 974 | document | test |
| anchor vs token_lookup_bert | delta_soft_skill_f1 | 0.5576 | [0.5265, 0.5888] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s42 | delta_exact_span_f1 | 0.0504 | [0.0290, 0.0718] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s42 | delta_token_f1 | 0.0003 | [-0.0153, 0.0159] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s42 | delta_hard_skill_f1 | 0.0407 | [0.0105, 0.0723] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s42 | delta_soft_skill_f1 | 0.0636 | [0.0329, 0.0945] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s43 | delta_exact_span_f1 | 0.0553 | [0.0331, 0.0773] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s43 | delta_token_f1 | 0.0027 | [-0.0132, 0.0196] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s43 | delta_hard_skill_f1 | 0.0516 | [0.0211, 0.0823] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s43 | delta_soft_skill_f1 | 0.0618 | [0.0300, 0.0941] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s44 | delta_exact_span_f1 | 0.0586 | [0.0357, 0.0809] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s44 | delta_token_f1 | 0.0041 | [-0.0124, 0.0204] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s44 | delta_hard_skill_f1 | 0.0388 | [0.0076, 0.0696] | 974 | document | test |
| anchor vs fair_runs_bert-base-cased_s44 | delta_soft_skill_f1 | 0.0833 | [0.0521, 0.1149] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s42 | delta_exact_span_f1 | 0.0370 | [0.0167, 0.0574] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s42 | delta_token_f1 | -0.0148 | [-0.0287, -0.0012] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s42 | delta_hard_skill_f1 | 0.0404 | [0.0120, 0.0699] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s42 | delta_soft_skill_f1 | 0.0332 | [0.0051, 0.0613] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s43 | delta_exact_span_f1 | 0.0388 | [0.0170, 0.0605] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s43 | delta_token_f1 | -0.0157 | [-0.0303, -0.0010] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s43 | delta_hard_skill_f1 | 0.0263 | [-0.0039, 0.0570] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s43 | delta_soft_skill_f1 | 0.0524 | [0.0219, 0.0830] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s44 | delta_exact_span_f1 | 0.0502 | [0.0285, 0.0716] | 974 | document | test |
| anchor vs fair_runs_jjzha__jobbert-base-cased_s44 | delta_token_f1 | -0.0122 | [-0.0267, 0.0020] | 974 | document | test |

_1026 further measured records: see `reports/skillllm_v2/evals/L10_significance/REPORT.md`._

Source: `reports/skillllm_v2/evals/L10_significance/results.json` (generated 2026-11-18T08:28:22+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L10_significance -->

### 9.14 Human evaluation (L11)

<!-- BEGIN AUTO:layer:L11_human_eval -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| annotation_design | items_in_packet | 201.0000 | - | 201 | item | L11 packet P1 |
| annotation_design | items_in_packet | 100.0000 | - | 100 | item | L11 packet P2 |
| annotation_design | items_in_packet | 150.0000 | - | 150 | item | L11 packet P3 |
| annotation_design | items_in_packet | 80.0000 | - | 80 | item | L11 packet P4a |
| annotation_design | items_in_packet | 60.0000 | - | 60 | item | L11 packet P4b |
| annotation_design | items_in_packet | 213.0000 | - | 213 | item | L11 packet P5 |
| annotation_design | items_in_packet | 120.0000 | - | 120 | item | L11 packet P6 |
| annotation_design | items_in_packet | 50.0000 | - | 50 | item | L11 packet P7 |

Not measured: pending_human 49. Reasons:
- `pending_human` x15: no annotation files returned yet
- `pending_human` x6: awaiting human annotation: annotator completion A=0/201, B=0/201 (need >= 90% from both); drop files into reports/skillllm_v2/evals/L11_huma
- `pending_human` x6: awaiting human annotation: annotator completion A=0/213, B=0/213 (need >= 90% from both); drop files into reports/skillllm_v2/evals/L11_huma
- `pending_human` x6: awaiting human annotation: annotator completion A=0/50, B=0/50 (need >= 90% from both); drop files into reports/skillllm_v2/evals/L11_human_
- `pending_human` x4: awaiting human annotation: annotator completion A=0/100, B=0/100 (need >= 90% from both); drop files into reports/skillllm_v2/evals/L11_huma
- `pending_human` x4: awaiting human annotation: annotator completion A=0/60, B=0/60 (need >= 90% from both); drop files into reports/skillllm_v2/evals/L11_human_
- `pending_human` x4: awaiting human annotation: annotator completion A=0/120, B=0/120 (need >= 90% from both); drop files into reports/skillllm_v2/evals/L11_huma
- `pending_human` x2: awaiting human annotation: annotator completion A=0/150, B=0/150 (need >= 90% from both); drop files into reports/skillllm_v2/evals/L11_huma

Source: `reports/skillllm_v2/evals/L11_human_eval/results.json` (generated 2026-11-18T08:25:23+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L11_human_eval -->

### 9.15 Systems cost (L12)

<!-- BEGIN AUTO:layer:L12_systems -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

| system | metric | value | ci | n | unit | split |
|---|---|---|---|---|---|---|
| anchor | latency_p95 | 16.6614 | [15.1548, 18.3863] | 974 | document | test |
| anchor | peak_vram_gb | 18.4843 | - | 974 | document | test |
| anchor | rerun_output_identity_rate | 1.0000 | - | 974 | document | test |
| anchor | rerun_spanset_identity_rate_legacy_offsets | 1.0000 | - | 974 | document | test |
| anchor | rerun_spanset_identity_rate_current_offsets | 0.9969 | - | 974 | document | test |
| anchor | exact_span_f1_rerun_current_offsets | 0.6714 | [0.6473, 0.6959] | 974 | document | test |
| anchor | exact_span_f1_rerun_legacy_offsets | 0.6701 | [0.6464, 0.6944] | 974 | document | test |
| anchor | exact_span_f1_stored | 0.6701 | [0.6464, 0.6944] | 974 | document | test |
| anchor | exact_span_f1_delta_rerun_current_minus_stored | 0.0013 | [0.0000, 0.0030] | 974 | document | test |
| anchor_base | n_params | 24,011,361,280.0 | - | 1 | model | n/a |
| anchor_base | n_params_language_model | 23,572,403,200.0 | - | 1 | model | n/a |
| anchor_lora | n_params_trainable | 92,405,760.0 | - | 1 | model | n/a |
| base_model_hf_shards | disk_gb | 48.0228 | - | 1 | artifact | n/a |
| base_model_dir_total | disk_gb | 96.0778 | - | 1 | artifact | n/a |
| lora_adapter | disk_gb | 0.3697 | - | 1 | artifact | n/a |
| span_verifier_joblib | disk_gb | 0.0013 | - | 1 | artifact | n/a |
| token_lookup_table | disk_gb | 0.0001 | - | 1 | artifact | n/a |
| bert_fair_model_pt | disk_gb | 0.4333 | - | 1 | artifact | n/a |
| gliner_medium_v2.1_cache | disk_gb | 1.5615 | - | 1 | artifact | n/a |
| anchor | latency_p50 | 5.7546 | - | 974 | document | test |
| anchor | latency_p99 | 28.5082 | - | 974 | document | test |
| anchor | seconds_per_predicted_span | 3.0582 | - | 974 | document | test |
| anchor | throughput_docs_per_hour_per_gpu | 511.3960 | - | 974 | document | test |
| anchor | decode_seconds_per_generated_token | 0.1669 | - | 974 | document | test |
| qlora_v4_adapter_prior_run | mean_seconds_per_doc | 4.5842 | - | 974 | document | test |
| token_lookup | latency_p95 | 0.0002 | - | 974 | document | test |
| gliner_zeroshot | latency_p95 | 2.0843 | - | 974 | document | test |
| bert_fair | latency_p95 | 0.3394 | - | 974 | document | test |
| safe_wrapper | latency_p95 | 1.4431 | - | 974 | document | test |
| safe_wrapper:gliner | stage_latency_mean_s | 0.7941 | - | 974 | document | test |

_15 further measured records: see `reports/skillllm_v2/evals/L12_systems/REPORT.md`._

Not measured: not_available 3. Reasons:
- `not_available` x1: the safe wrapper's SpanVerifier (src/skillllm_v2/verify/span_verifier.py) is rule-based (confidence threshold + generic-phrase / length rule
- `not_available` x1: never trained: artifacts/skillllm_v2/taxonomy_reranker/metrics.json training_mode=retrieval_only (placeholder README)
- `not_available` x1: host process RSS was not recorded by the anchor GPU jobs (gpu_anchor_infer.py logs CUDA memory and nvidia-smi snapshots only); not re-measur

Source: `reports/skillllm_v2/evals/L12_systems/results.json` (generated 2026-11-18T07:42:00+00:00, git 54aab906d548 dirty).
<!-- END AUTO:layer:L12_systems -->

## 10. Interpretation

### 10.1 Verdicts as emitted by the layers

<!-- BEGIN AUTO:verdict_statements -->
_Auto-generated 2026-11-18T08:28:41+00:00 from `reports/skillllm_v2/evals/LEADERBOARD.md` — do not edit inside this block._

- **H1: NOT SUPPORTED** [mistral_lora] (layer L01_extraction, measured). primary effect = 0.0370 [0.0165, 0.0573]; p_raw = 0.0004, p_Holm = 0.0004. Pre-registered rule: H1 SUPPORTED iff anchor exact-span F1 minus baseline >= 0.10 AND Holm-adjusted p < 0.05 against EVERY baseline in family H1_anchor_vs_baselines (paired, 974 test sentences; p = approximate randomization, 10,000 shuffles; CI = paired bootstrap, 10,000 resamples).
- **H2: NOT SUPPORTED** [R7_safe_wrapper_stored] (layer L05_wrapper, measured). primary effect = 0.0000 [0.0000, 0.0000]; p_raw = 1.0000, p_Holm = 1.0000. Pre-registered rule: H2: APR(safe) - APR(legacy) 95% CI lower bound > 0 against BOTH legacy full_cag_graph and legacy mistral_gliner_taxonomy, AND exact-F1(safe) - exact-F1(anchor) lower 95% CI bound > -0.005 (D-A2-02). Caveat recorded by the layer: Single training run of the anchor and one test split: CIs reflect document sampling only.; SOFT_SKILL = SkillSpan 'skill' component (tags_skill), HARD_SKILL = 'knowledge' (tags_knowledge).; 974-doc subset with >=1 gold s
- **H3: SUPPORTED** [mistral_lora] (layer L02_generalization, measured). primary effect = 0.2063 [0.1704, 0.2415]; p_raw = 0.0002, p_Holm = 0.0002. Pre-registered rule: H3 SUPPORTED iff, for the anchor, exact F1 on seen mentions (S1 exact_seen + S2 normalized_seen) minus exact F1 on fully unseen mentions (S4) is >= 0.05 AND its 95% paired sentence-bootstrap CI (10,000) excludes 0. S3 is reported separately. Caveat recorded by the layer: Single SkillSpan test split and one training run per system: CIs reflect sentence sampling only, not seed/training variance.; Primary population = the 974 test sentences that contain gold spans; false positives on the 2,
- **H4: SUPPORTED** [anchor_logprob] (layer L04_calibration, measured). primary effect = 0.0587 [0.0477, 0.0682]; p_raw = 0.0002, p_Holm = 0.0006. Pre-registered rule: H4: selective precision@80% and @50% minus precision@100% > 0 (CI excl. 0) AND AURC(primary) - AURC(constant 0.86) < 0 (CI excl. 0); score + calibrator selected/fit on dev only (D-A2-03). Caveat recorded by the layer: Primary effect = selective precision@80% minus precision@100% under the dev-selected score.
- **H5: SUPPORTED** [hybrid+graph vs hybrid(bm25+dense)] (layer L08_graph_retrieval, measured). primary effect = 0.0855 [0.0478, 0.1204]; p_raw = 0.0002, p_Holm = 0.0006. Pre-registered rule: H5: hybrid+graph vs hybrid (no graph) nDCG@10 on multi-gap (≥2 gaps) held-out queries: Δ > 0 with Holm p < 0.05. Caveat recorded by the layer: All 24 test scenarios have ≥4 system-visible gaps: the 1-gap stratum is empty on the original benchmark.; Graph hop 1 equals flat skill overlap here because the KG's TEACHES edges are a 1:1 projection of the catalogue fi
- **H6: SUPPORTED** [SkillLLM_v2 Full vs Full-noPrereq (ablation)] (layer L09_course_rec, measured). primary effect = -0.1833 [-0.2583, -0.1167]; p_raw = 0.0004, p_Holm = 0.0015. Pre-registered rule: H6: prerequisite-aware vs same ranker without prerequisite constraints: violation rate lower with CI excluding 0, AND nDCG@10 non-inferior (lower CI bound of Δ > −0.02). Caveat recorded by the layer: The violation criterion is satisfied by construction when the strict mode enforces the same rule the metric checks; nDCG non-inferiority is the informative half.
- **H7: NOT SUPPORTED** [compact@0.5[informative]] (layer L07_compaction, measured). primary effect = 0.1615 [0.1004, 0.2241]; p_raw = 0.0005, p_Holm = 0.0010. Pre-registered rule: token reduction (mean, Mistral tokens) >= 0.50 AND job nDCG@10 and course nDCG@10 non-inferior to full (lower 95% CI of delta > -0.02, Holm over the two) AND mean gap Jaccard vs full >= 0.9.
- **H7: NOT SUPPORTED** [compact@shipped[informative]] (layer L07_compaction, measured). primary effect = 0.0000 [0.0000, 0.0000]; p_raw = 0.0005, p_Holm = 0.0010. Pre-registered rule: token reduction (mean, Mistral tokens) >= 0.50 AND job nDCG@10 and course nDCG@10 non-inferior to full (lower 95% CI of delta > -0.02, Holm over the two) AND mean gap Jaccard vs full >= 0.9.
- **H7: NOT SUPPORTED** [compact@0.5[uninformative]] (layer L07_compaction, measured). primary effect = 0.1615 [0.1004, 0.2241]; p_raw = 0.0005, p_Holm = 0.0010. Pre-registered rule: token reduction (mean, Mistral tokens) >= 0.50 AND job nDCG@10 and course nDCG@10 non-inferior to full (lower 95% CI of delta > -0.02, Holm over the two) AND mean gap Jaccard vs full >= 0.9.
- **H7: NOT SUPPORTED** [compact@shipped[uninformative]] (layer L07_compaction, measured). primary effect = 0.0000 [0.0000, 0.0000]; p_raw = 0.0005, p_Holm = 0.0010. Pre-registered rule: token reduction (mean, Mistral tokens) >= 0.50 AND job nDCG@10 and course nDCG@10 non-inferior to full (lower 95% CI of delta > -0.02, Holm over the two) AND mean gap Jaccard vs full >= 0.9.
<!-- END AUTO:verdict_statements -->

A verdict is a statement about the pre-registered rule (`hypotheses_preregistration.md`), not about the system as a whole.
Where the owning layer emitted several H7 records (primary arm compact@0.5 and the shipped default, each under an
informative and an uninformative confidence arm), all of them are listed. The "primary effect" is whatever the owning layer
chose as the effect of record, which is not always the criterion that decides the verdict: for H2 it is the difference in
Anchor Preservation Rate against the legacy stages, and for H7 the course-nDCG difference (the compact@0.5 arm fails
on token reduction, job nDCG and gap agreement and the shipped arm fails on token reduction, as the L07 block shows).

### 10.2 Reading the verdicts

- **H1 (NOT SUPPORTED).** The anchor beats the token-lookup baseline and GLiNER zero-shot by a very large margin and the
  difference to a fairly trained JobBERT is statistically significant, but it is far below the pre-registered bar. The
  large margins are properly attributed to weak comparators (see `threats_to_validity.md` A1, G1). The defensible claim is
  "a LoRA-tuned 24B decoder is modestly better than a fairly trained encoder on SkillSpan span extraction", not a large
  improvement.
- **H2 (NOT SUPPORTED as pre-registered).** The legacy ungated stages already ran the source-aware components, so the safe
  wrapper does not preserve more anchor true positives than they do. The aggregate F1 non-inferiority half holds, and the
  wrapper's own gain over the anchor is small and not significant. No wrapper component is verifiably dev-tuned, and the
  selection gate ran on test (G5), so the wrapper must not be presented as a held-out improvement.
- **H3 (SUPPORTED).** Unseen skills are harder than seen skills, and the gap persists within label and within span length
  (L02 block). This is the most robust finding of the programme.
- **H4 (SUPPORTED).** Abstaining on low-confidence anchor spans raises selective precision, and the dev-fit calibrated
  log-probability score is far better calibrated than a constant score. The effect is specific to the anchor's
  log-probability: GLiNER and the wrapper's dictionary confidences carry almost no ranking information (L04 block), and
  dev-tuned abstention yields no F1 gain.
- **H5 (SUPPORTED, internal-consistency evidence only).** The graph variant improves nDCG@10 on multi-gap scenarios of the
  project's synthetic benchmark. The layer notes that its graph hop 1 equals flat skill overlap because the TEACHES edges
  are a projection of the catalogue fields, and the benchmark has 24 scenarios (G11).
- **H6 (SUPPORTED on the v1 rule labels only).** The violation-rate half holds by construction (G10, B7 in the threats
  file). The informative half, nDCG non-inferiority, passes on the v1 labels but fails under the label-sensitivity variants,
  and the Full recommender is not significantly better than the strongest graph-free baseline after Holm correction (L09
  block). The claim is that prerequisite filtering removes violations at little ranking cost on this benchmark, not that
  the recommender is better than simpler systems.
- **H7 (NOT SUPPORTED, every arm).** The primary compact profile saves little over the full history and loses job-ranking
  decisions, and the shipped lossless packet uses more tokens than the history it summarises (L07 block). Exploratory arms
  in the layer report show what a compaction that preserves decisions would cost. They are labelled exploratory and do
  not change the verdict.

### 10.3 What this chapter does not establish

1. Human validity of any extraction, linking, path, recommendation or explanation judgement: all L11 metrics are
   `pending_human`; the blinded packets and answer keys are ready.
2. Real-world recommendation or compaction effectiveness: L07 to L09 run on project-authored synthetic data (D3, D4).
3. Performance on résumés or on job postings outside SkillSpan's English tech and house sources (D2).
4. Training-seed variance of the anchor (C2), and any hardware-independent latency (D5).
5. Trademark clearance of the proposed name (F3).

## 11. Threats to validity

See `threats_to_validity.md`. It lists every known threat with its evidence, its likely direction of bias, the
mitigation applied in this programme, and what remains open.
