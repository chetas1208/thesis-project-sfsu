# Decisions log — SkillGraphLM evaluation program

Append-only. Each decision: id, date, owner, decision, context, alternatives considered, consequences.
IDs: coordinator `D-C##`, agents `D-A<n>-##` (no global counter, so concurrent writers never collide).
Agents append with `cat >> Decisions.md <<'EOF' ... EOF` (never Edit/Write this file).
A decision that reverses an earlier one says so and names the id it supersedes.

---

### D-C01 — Evaluation becomes a 12-layer benchmark, not "SkillSpan F1 + money table"
- Date / owner: 2026-11-17, coordinator (user-directed)
- Decision: layers L01 extraction, L02 unseen-skill generalization, L03 robustness, L04 calibration/selective prediction,
  L05 safe-wrapper contribution, L06 taxonomy linking, L07 profile compaction, L08 graph+retrieval, L09 course
  recommendation, L10 significance, L11 human evaluation, L12 systems; plus naming audit, evaluation card, thesis chapter.
- Context: aggregate exact F1 alone cannot show generalization, robustness, calibration or downstream value, and recent
  NER work shows train/test entity overlap inflates scores.
- Consequences: each layer writes `reports/skillllm_v2/evals/<layer>/results.json` with one shared record schema.

### D-C02 — Keep `SkillLLM_v2` as the internal name; "SkillGraphLM" is a proposal
- Date / owner: 2026-11-17, coordinator
- Decision: no code/package/file renames. The research-facing name is adopted only after a naming-collision audit and
  explicit user approval.
- Alternatives: rename now (rejected: irreversible churn before the audit).

### D-C03 — One record schema with mandatory provenance
- Date / owner: 2026-11-17, coordinator
- Decision: `src/skillllm_v2/evals/research/common.py::record()` for every number: value, 95% CI + method, n independent
  units, unit, split, dataset, direction, status (measured | not_available | pending_human | pending_gpu | failed), sha256 of
  every input file. Unavailable numbers are explicit records with a reason, never omitted or estimated.
- Why: every thesis metric must be traceable to an artifact; missing results must be visible.

### D-C04 — Statistics conventions fixed up front
- Date / owner: 2026-11-17, coordinator
- Decision: resample independent units only (document for extraction, query/scenario for retrieval/recommendation, profile
  for compaction); 95% percentile bootstrap, seed 42, 2,000 resamples (10,000 for headline metrics); paired bootstrap +
  approximate randomization for paired comparisons; Holm within each hypothesis family, Benjamini-Hochberg across all as
  secondary.
- Why: spans within a document are not independent; many comparisons inflate false positives.

### D-C05 — Hypotheses H1-H7 and their decision rules are pre-registered
- Date / owner: 2026-11-17, coordinator
- Decision: thresholds written before any new result (contract, section "Pre-registered hypotheses"): H1 Δ≥0.10 vs every
  baseline with Holm p<0.05; H2 APR higher (CI excl. 0) and F1 non-inferior (margin −0.005); H3 seen−unseen Δ≥0.05, CI excl. 0;
  H4 selective precision at 80%/50% coverage above full coverage and AURC better than constant confidence, calibrator fit on
  dev; H5 hybrid+graph > hybrid on multi-gap queries, Holm p<0.05; H6 lower violation rate (CI excl. 0) and nDCG@10
  non-inferior (−0.02); H7 ≥50% token reduction, nDCG@10 non-inferior (−0.02), gap Jaccard ≥0.9.
- Why: rules chosen after seeing data are the easiest thing to attack in a defense.

### D-C06 — No tuning on test labels; test-tuned legacy components are flagged, not reused as clean
- Date / owner: 2026-11-17, coordinator
- Decision: thresholds, calibrators, fusion weights fit on SkillSpan dev or the recommender's dev half only. Existing
  logprob recalibration (cross-fitted on test) is reported only as labelled secondary.

### D-C07 — Report the 974/3,569 test-subset limitation and measure it
- Date / owner: 2026-11-17, coordinator
- Decision: all stored extraction predictions cover only the 974 test sentences that contain gold spans, so false positives
  on skill-free sentences have never been measured. Run the anchor on the other 2,595 test sentences (GPU) and report a
  full-test score next to the subset score.

### D-C08 — Public datasets live outside the repo
- Date / owner: 2026-11-17, coordinator (user authorized public downloads)
- Decision: `data/external/` is a symlink to `/usr/data/models/skillllm-v2-stage2/external_data` (root disk is 87% full),
  gitignored, with a per-download entry (url, version, license, sha256) in `data/external/MANIFEST.jsonl`. One owner per
  dataset family to avoid duplicate downloads.
- Why: licensing and the repo's no-real-job-text policy; reproducibility needs exact versions and hashes.

### D-C09 — GPU access for all agents, through guardrails and a reservation line
- Date / owner: 2026-11-17, coordinator (user granted GPU access)
- Decision: any agent may run GPU experiments, only via `server-guardrails run --ram 16 gpu0|gpu1`, after appending a
  `gpu-reserve` line to Progress.md, and a `gpu-release` line after. A3 has first claim on both GPUs for the anchor
  re-inference jobs (test+log-probs, dev+log-probs, skill-free test sentences, robustness).

### D-C10 — Leave the empty nested `.git` in SkillLLM_v2_stage2 untouched
- Date / owner: 2026-11-17, coordinator
- Decision: the directory is tracked by the outer repo; a stale empty `.git` (May 2026) sits inside it. Git metadata is
  read with `git -C "/home/923873155/Sem 4 Research"`; nothing deletes the nested `.git` without the user's say-so.

### D-A2-01 — Shared statistics module and p-value definitions (stats.py)
- Date / owner: 2026-11-17, A2
- Decision: (1) bootstrap two-sided p = min(1, 2·min((1+#{d*≤0})/(B+1), (1+#{d*≥0})/(B+1))) — percentile inversion, so p<0.05 ⇔ the 95% percentile CI excludes 0 (up to smoothing), same form as the existing `eval/significance.py`; ties at 0 count on both sides (identical systems → p=1). A null-shifted p (`p_shift`) is reported as secondary. (2) Approximate randomization: swap each unit's two outputs with prob ½; exact enumeration over the 2^d swaps of the d units that differ when d ≤ 16 (no Monte-Carlo error), otherwise 10,000 seeded shuffles with +1 smoothing. (3) Micro-F1 recomputed from pooled per-unit TP/FP/FN (2TP/(2TP+FP+FN)). (4) Holm within family, BH across all; None p-values are excluded from family size. (5) Non-inferiority = lower bound of the two-sided 95% CI > margin (≡ one-sided α=0.025). (6) MDE at 80% power from bootstrap SE (normal approx.).
- Context: the legacy `extraction_evals.paired_boot` computes p = 2·min(P(d≤0), 1−P(d≤0)), which returns p=0 for two identical systems (all d=0) and is biased when many replicates tie at 0 — relevant here because wrapper variants differ from the anchor on only ~10–20 of 974 documents.
- Alternatives: Berg-Kirkpatrick one-sided shifted test only; normal-approximation z-test; McNemar on spans (ignores document clustering).
- Consequences/why: one CI-consistent p-value convention across all layers; AR gives a distribution-free check that does not rely on bootstrap asymptotics when very few documents differ.

### D-A4-01 — L07/H7: which "compact profile" is tested, and how each H7 criterion is operationalised (pre-registered before any L07 result)
- Date / owner: 2026-11-17, A4
- Decision: H7 is evaluated on two configurations of the production path `build_cached_context_packet -> compact_context`:
  (a) PRIMARY "compact@0.5": budgeted mode with `budget_tokens = floor(0.5 x proxy-token count of the full history text)`
  (the system's own budget unit, `context.budget.count_tokens`); (b) "compact@shipped": `budget_tokens=None`, which is what
  `pipeline/inference.py::stage_profile` actually ships. Both get their own H7 verdict; (a) is the headline verdict.
  A budget sweep {None, 1.0, 0.5, 0.25, 0.1} x full-history proxy tokens gives the compression-vs-agreement curve.
- Criteria (fixed now): token reduction = mean over profiles of 1 - tokens(rep)/tokens(full), tokens counted with the
  Mistral tokenizer `artifacts/skillllm_v2/mistral_lora/model/tokenizer.json` on the exact serialisation (raw history text
  for full; `json.dumps(packet, sort_keys=True, ensure_ascii=False)` for compact; JSON list of skill names for minimal);
  passes if the point estimate >= 0.50. Non-inferiority: paired profile-bootstrap delta (rep - full) of nDCG@10 against
  generator-authored relevance, separately for job ranking and course ranking; passes if the lower 95% CI bound > -0.02
  (two tests, Holm within family "H7"; one-sided bootstrap p for H0: delta <= -0.02). Gap Jaccard = mean over profiles of
  Jaccard(gap set from rep, gap set from full) with gaps from `match.gap_detector.GapDetector.detect`; passes if mean >= 0.9
  (Jaccard of two empty sets = 1). H7 SUPPORTED only if all four hold.
- Downstream protocol: target job is supplied by the request (identical across representations); representations differ
  only in which held skills (with confidence) they expose. Full-history profile = every extracted skill fact (no
  compaction). Extraction is simulated by oracle mention spans from the generator (every surface mention, held or not,
  with an authored confidence rule) so that differences are attributable to the compaction step only.
- Alternatives: testing only the shipped default (rejected as sole test: it is lossless by design, so it can only fail the
  token criterion and says nothing about the budgeted compactor); budget chosen per profile to hit exactly 50% in Mistral
  tokens (rejected: would make the token criterion true by construction).
- Consequences: truth (skills, evidence sentences, goals, gaps, contradictions) comes only from the seeded generator;
  absolute nDCG is optimistic because ranker and authored relevance both reason about skill coverage — the paired deltas
  and agreement with the full-history ranking are the informative quantities.

### D-A1-01 — L01 evaluation population, unit and protocol alignment
- Date / owner: 2026-11-17, A1
- Decision: primary population = the 974 SkillSpan(HF) test sentences that carry >=1 gold span (2,265 gold spans: 1,174
  HARD_SKILL = SkillSpan "knowledge", 1,091 SOFT_SKILL = SkillSpan "skill"). Every system (incl. the two baselines) is
  re-scored on exactly these 974 ids so all comparisons are paired. The resampling unit is the repo "document", which is
  one SENTENCE (`test-<row>`); because the 974 sentences come from only 65 job postings, a posting-level cluster bootstrap
  (65 clusters) is reported as a sensitivity CI for headline metrics. Secondary protocol = all 3,569 test sentences,
  computed only where predictions exist for the 2,595 skill-free sentences (baselines now; anchor after A3's
  `anchor_test_nospan` job).
- Context: the stored baseline metrics (BERT 0.059150, GLiNER 0.075833) are computed over all 3,569 sentences while the
  anchor 0.670069 is over the 974-subset (verified by recomputation: baselines on the 974-subset = 0.137604 / 0.081400).
  The money table therefore compares different populations.
- Alternatives: keep the stored mixed-protocol numbers (rejected: not comparable); sentence-only CIs (kept as primary per
  contract, posting-level as sensitivity because sentences in one posting are correlated).
- Consequences: baseline numbers in L01 differ from money_table.md; both protocols are reported side by side.

### D-A1-02 — H1 test procedure (fixed before computing H1)
- Date / owner: 2026-11-17, A1
- Decision: family `H1_anchor_vs_baselines` = anchor (mistral_lora) minus each baseline on exact-span micro-F1, 974
  paired sentences. Effect = Δ exact F1; CI = paired percentile bootstrap (10,000 resamples, seed 42, shared draws,
  `research/stats.py::paired_bootstrap`); p for Holm = approximate-randomization p (10,000 per-sentence swaps,
  `stats.approximate_randomization`), bootstrap p reported as secondary. Holm across all baselines present at run time.
  Baselines = repo "BERT baseline" (a token-lookup table, not BERT), GLiNER zero-shot, plus any fairly trained BERT-family
  baseline A3 publishes under `artifacts/skillllm_v2/evals_gpu/` (dir name contains "bert", DONE.json present).
  Verdict SUPPORTED iff Δ>=0.10 and Holm p<0.05 vs EVERY baseline in the family; if A3 has announced a fair baseline
  that is not finished, the verdict carries `provisional=True` and the pending baseline is listed.
- Why: permutation p is exact under exchangeability and does not depend on bootstrap percentile asymmetry; Holm controls
  FWER within the family.

### D-A1-03 — L01 metric definitions added on top of the repo scorer
- Date / owner: 2026-11-17, A1
- Decision: (a) exact span, repo partial span, repo char-level token metrics and per-label/macro/micro come unchanged
  from `skillllm_v2.eval.extraction_eval` via `extraction_evals.doc_counts` (set semantics, duplicates removed).
  (b) word-level token P/R/F1 = whitespace-word x label sets (`extraction_evals._word_sets`). (c) char-IoU / word-IoU:
  per sentence and label, Hungarian matching (scipy `linear_sum_assignment`) that maximises total IoU over same-label
  pairs with IoU>0; report mean IoU over matched pairs, and IoU>=0.5 F1 = one-to-one matching maximising the number of
  pairs with IoU>=0.5 (P over de-duplicated predictions incl. unlocated ones, R over gold). (d) SemEval-2013 /
  nervaluate Strict/Exact/Partial/Type with COR/INC/PAR/MIS/SPU: re-implementation of nervaluate 1.2.1 semantics
  (greedy in prediction order, inclusive end offsets, min overlap 1% of the gold span), predictions de-duplicated by
  (start,end,label) keeping order; unlocated predictions (no offsets) are added as SPU. Cross-checked against the
  `nervaluate` 1.2.1 package (throwaway venv) - counts must match exactly. The repo's one-to-one max-matching variant
  (`extraction_evals.nervaluate_doc`) is reported alongside, because greedy nervaluate can under-count strict COR.
- Why: examiners know nervaluate/SemEval; IoU separates "slightly off" from "wrong"; the oracle check proves the
  implementation.

### D-A1-04 — Boundary-error taxonomy (component based)
- Date / owner: 2026-11-17, A1
- Decision: after removing exact (start,end,label) matches, build the overlap graph between unmatched located
  predictions and unmatched gold spans (any label) and classify each connected component: 0 gold -> spurious (subtype
  overlaps_correct_gold / no_overlap); 0 pred -> missed; 1:1 same label -> left_extension (pred starts earlier, same
  end), right_extension, left_truncation, right_truncation, both_side_shift (subtypes superset / subset / crossing);
  1:1 other label -> label_confusion_exact_boundary or overlap_wrong_label; 1 gold : >=2 preds -> split; >=2 gold :
  1 pred -> merge; many:many -> many_to_many; unlocated predictions -> spurious_unlocated. Each event carries its FP and
  FN share; invariant (tested): sum FP = evaluator FP, sum FN = evaluator FN. Extra diagnostic (not a partition):
  `mislocated_occurrence` = a FP whose surface string equals an unmatched same-label gold elsewhere in the sentence
  (offset resolution picked the wrong occurrence). Reports list example sentence ids only, never text.

### D-A6-01 — Pre-registration document is a verbatim copy, timestamped before any layer result
- Date / owner: 2026-11-17, A6
- Decision: `docs/thesis/evaluation/hypotheses_preregistration.md` copies the H1–H7 contract section verbatim (with the
  contract's sha256) at 2026-11-17T20:58:14Z, when `reports/skillllm_v2/evals/` contained no results.json; it lists the
  pre-existing (pre-programme) numbers each rule could have been fitted to, and allows only appended, dated amendments.
- Context: rules chosen after seeing data are the main attack surface in a defense; pre-programme numbers (anchor 0.670, recommender v2) were already known.
- Alternatives: paraphrased summary (rejected: invites drift); OSF registration (not available offline — can be done by the user later using this file).
- Consequences: leaderboard verdicts are rendered from the owning layer's H#_verdict record only; any amendment is visible.

### D-A2-02 — Wrapper-contribution metrics (APR, NWG) and per-rung accounting (L05)
- Date / owner: 2026-11-17, A2 (written before any L05 number was computed)
- Decision: per document, with set semantics on (start, end, label) exactly as `eval/extraction_eval.py`: anchor TP set A_T, anchor FP set A_F, stage TP set S_T, stage FP set S_F. TP gained = |S_T \ A_T| (≡ recovered anchor FN, identical by construction because the gold set is fixed); TP destroyed = |A_T \ S_T|; FP added = |S_F \ A_F|; FP removed = |A_F \ S_F|. **Anchor Preservation Rate APR = Σ_d |A_T ∩ S_T| / Σ_d |A_T|** (pooled, not a mean of per-doc ratios). **Net Wrapper Gain NWG = 1000 · Σ_d[(gained − destroyed) − (added − removed)] / Σ_d |gold|** (= 1000·(ΔTP − ΔFP)/N_gold). Verifier / calibrator false rejection = a candidate entering that stage whose key is a gold span and that the stage rejects; merger gold drop = an input candidate whose key is gold and that is absent from the merger output (split into anchor-sourced vs GLiNER/dictionary-only). Soft-skill recall loss = recall_SOFT(anchor) − recall_SOFT(stage). All CIs: document-cluster percentile bootstrap (2,000; 10,000 for headline F1/APR/H2), paired where a delta is reported; AR (10,000 shuffles or exact enumeration) for ΔF1.
- Ladder: each rung = the real pipeline order (merge → refine → verify → calibrate → link) with a growing set of components enabled, replayed on the stored anchor predictions: R0 raw anchor; R1 +boundary selector/refiner; R2 +GLiNER add-only merge; R3 +dictionary; R4 +taxonomy link; R5 +verifier; R6 +confidence calibrator (= replayed safe wrapper); R7 = stored gated safe wrapper output. Order-dependence is checked with single-component (alone on anchor) and leave-one-out variants. GLiNER candidates for the replay come from the stored zero-shot GLiNER predictions (`artifacts/skillllm_v2/gliner_baseline/predictions.jsonl`, same model/threshold/labels as the wrapper's GLiNERExtractor) because the wrapper never persisted its own GLiNER calls; the replay-vs-stored mismatch is measured and reported.
- H2 operationalisation (pre-committed): comparator set = the two artefacts named in the contract, `artifacts/skillllm_v2/full_cag_graph` and `artifacts/skillllm_v2/mistral_gliner_taxonomy` (legacy, ungated, Mistral regenerated live). Treatment = `artifacts/skillllm_v2/safe_wrapper/predictions.jsonl`. Condition (i): ΔAPR = APR(safe) − APR(legacy) has a 95% CI with lower bound > 0 against BOTH comparators (Holm over the two p-values also reported). Condition (ii): exact-F1(safe) − exact-F1(anchor) lower 95% CI bound > −0.005 (10,000 doc resamples). SUPPORTED iff (i) and (ii). A same-anchor "policy-off" counterfactual (source-aware protection disabled, all boundary edits allowed) is reported only as a labelled secondary, never as the verdict.
- Alternatives: APR as a mean of per-doc ratios (undefined for docs with no anchor TP; overweights short docs); span-level McNemar (ignores clustering).
- Consequences/why: APR isolates the failure mode the safe wrapper was built against (destroying spans the anchor already got right), which aggregate F1 hides when gains and losses cancel.

### D-A5-01 — L08/L09 definitions fixed before any new system run (pre-registration of A5's operationalisation)
- Date / owner: 2026-11-17, A5 (graph-rec). Written before running any retrieval/recommendation system in this program.
- Decision (units, splits): unit = scenario. Primary evaluation = the 24 TEST scenarios of `reports/skillllm_v2/recommender_split.json`
  (sha256 split fixed before the earlier tuning; verified: no test id appears in `recommender_loop_log.json`). Dev (24) is shown
  only as "tuned-on" context. Nothing in A5's layers is fitted: RRF uses the fixed k=60 with equal weights, the graph
  2-hop discount λ=0.5 is fixed a priori, BM25 k1=1.5 b=0.75 (repo defaults), dense = all-MiniLM-L6-v2 off the shelf.
- Labels: primary = existing `inputs/courses/relevance_labels.jsonl` (v1-rule), relevant = grade ≥ 2, graded gain 2^g−1
  (unchanged from the existing benchmark so numbers are comparable). Sensitivity label variants: D-A5-05.
- Statistics: per-system mean with 95% percentile scenario-bootstrap CI (10,000 resamples for headline metrics, 2,000 otherwise,
  seed 42); paired comparisons = paired bootstrap of the mean difference (95% CI, two-sided p from the bootstrap
  distribution, floored at 1/(B+1)) plus a paired sign-flip permutation test (10,000 flips) as a second p; Holm within a declared family.

### D-A5-02 — L08 retrieval systems, query and the controlled RAG-vs-GraphRAG design (H5)
- Date / owner: 2026-11-17, A5
- Decision: query q_gap = job title + names of the system-visible gap skills (job required ∪ preferred minus skills the user holds
  at non-beginner level; exactly the inputs the repo's CourseRetriever uses). All L08 systems see the same query and the same
  system-visible gap weights (required 1.0, preferred 0.7); none sees gold needs or labels. Systems: random; BM25(q_gap);
  dense MiniLM(q_gap); canonical-skill overlap (Σ gap weight over the course's catalogue `teaches_skills`, flat attribute match,
  no graph); graph multi-hop (hop 1: gap Skill ←TEACHES− Course, score Σ w_g; hop 2: hop-1 course −REQUIRES_PREREQUISITE→
  skill p the user lacks ←TEACHES− bridge course, score λ·parent score, λ=0.5 fixed); hybrid = RRF(BM25, dense), k=60;
  hybrid+graph = RRF(BM25, dense, graph), k=60; hybrid+skill-overlap = RRF(BM25, dense, skill-overlap) (the graph-free
  structured control); the repo's own `CourseRetriever` (as shipped, pool 40). Secondary rows: BM25/dense on q_job (title + job text).
- Controlled comparison: (a) H5 = hybrid+graph vs hybrid — identical query, identical k, identical fusion (RRF k=60, equal
  weights), the ONLY difference is the graph ranked list entering the fusion. (b) pool variant: identical candidate budget B=20
  and identical final scorer (MiniLM cosine to q_gap); RAG pool = hybrid top-20; GraphRAG pool = hybrid top-10 ∪ graph expansion
  of those 10 seeds (courses sharing a gap skill with a seed, or teaching a prerequisite of a seed that the user lacks), filled to 20
  in graph-score order, then hybrid order. (c) graph vs flat skill overlap inside the same fusion (hybrid+graph vs hybrid+skill-overlap).
- H5 (contract rule, unchanged): Δ nDCG@10 (hybrid+graph − hybrid) on multi-gap (≥2 system-visible gaps) TEST scenarios > 0 with
  Holm p < 0.05; Holm family "H5" = {primary (a) on multi-gap test, pool variant (b), graph-vs-flat (c)}; verdict from (a).
  Data fact recorded before results: all 24 test scenarios have 4–8 system-visible gaps, so the "1-gap" stratum of the original
  benchmark is EMPTY (n=0, not testable there); it is only populated in the generated set (D-A5-07).
- Strata: #gaps (1 vs ≥2, system-visible); "needs prerequisite hop" = the scenario has ≥1 course with content-only grade ≥ 2 whose
  catalogue prerequisites are NOT all known/weak for the user (computed from labels + catalogue only, never from system output).
- Metrics: P@5, Recall@10, Recall@20, MRR (first grade≥2 item within top-20), nDCG@5, nDCG@10. Path validity: every returned path
  (graph retriever paths, and the Full recommender's explanation paths) is decomposed into edges; an edge is valid iff it exists in
  the knowledge graph with the stated type and both endpoint node types match `evals/graph_evals.EDGE_SIGNATURE`.

### D-A5-03 — L09 systems and the H6 primary pair
- Date / owner: 2026-11-17, A5
- Decision: systems = random (sha256-seeded per scenario); lexical = BM25(q_job = title + job text, as in the existing benchmark);
  semantic = MiniLM(q_job); skill-overlap = user-agnostic job-skill overlap (existing); skill-gap ranking = Gap-Overlap Greedy (no
  prerequisite logic, existing); hybrid = RRF(BM25(q_gap), MiniLM(q_gap), gap-overlap) k=60, no prerequisite logic;
  prerequisite-aware = Gap-Overlap Greedy + prerequisite filter (existing); full = tuned `DEFAULT_CONFIG` pipeline; plus
  ablations Full-noPrereq (`DEFAULT_CONFIG` with prereq_mode="soft", w_prereq=0, w_unlock=0) and legacy Full (pre-tuning).
- H6 primary pair (fixed now): Full vs Full-noPrereq (same ranker, prerequisite constraints AND prerequisite score removed).
  Secondary pairs (reported, Holm family "H6" over violation-rate Δ): Gap-Overlap Greedy ± filter, BM25 ± filter, MiniLM ± filter,
  hybrid ± filter. Rule (contract): violation rate lower with 95% CI of Δ excluding 0 AND nDCG@10 non-inferior (lower 95% CI bound of
  Δ > −0.02). Caveat stated up front: the violation metric uses the same chained catalogue-prerequisite rule the strict mode / filter
  enforce, so the violation half of H6 is satisfied by construction; the informative half is nDCG non-inferiority.

### D-A5-04 — Formal metric definitions for L09
- Date / owner: 2026-11-17, A5
- G_s = gold open needs of scenario s = gold needs minus gold-known skills (weak skills stay open), weights w_g ∈ {3 required,
  2 implicit, 1 preferred}; T(c) = catalogue `teaches_skills` of course c (normalize_text), P(c) = catalogue `prerequisites`;
  K_s = known ∪ weak skills (weak counts as prerequisite exposure, the existing convention).
- WSGC@5(s, L) = Σ_{g∈G_s} w_g·1[g ∈ ∪_{c∈L[1..5]} T(c)] / Σ_{g∈G_s} w_g. Computed from the CATALOGUE field, never from a system's
  own "covered" claims; a list shorter than 5 gets no credit for empty slots. (Equals the existing weighted_gap_coverage_at_k on the
  catalogue view; asserted in code.)
- Precision@5 = #(grade≥2 in top-5)/5; Recall@5 = #(grade≥2 in top-5)/#(grade≥2 overall) (scenarios with none excluded);
  nDCG@10 graded (ideal over all judged courses); MRR = 1/rank of first grade≥2 item in top-10 (0 if none).
- Prerequisite violation rate@5 = share of the top-5 whose P(c) ⊄ K_s ∪ ∪_{earlier c'} T(c') (chained rule, existing metric).
- Redundant skill coverage@5 (RSC@5) = (Σ_{c∈top5}|T(c)∩G_s| − |∪_{c∈top5} T(c)∩G_s|) / Σ_{c∈top5}|T(c)∩G_s| (0 if denominator 0):
  the share of gap-skill teaching instances in the top-5 that re-teach an already covered gap skill. Existing Redundancy@5 kept for continuity.
- Learning-path validity (whole returned list, ≤10): valid iff every course's P(c) ⊆ K_s ∪ T(earlier courses in the list);
  "repairable" iff some reordering of the same set is valid (fixed-point closure). Reported per scenario (binary) and per course.

### D-A5-05 — Label-independence audit: label variants defined before scoring any system on them
- Date / owner: 2026-11-17, A5
- Decision: besides v1-rule and its existing content-only variant, two variants are defined now:
  (1) V-visible: the v1 rule run on SYSTEM-VISIBLE needs only (required = 3, preferred = 1, no implicit needs, user skills as given to
  the systems). Its agreement with v1 measures how much of the gold is reconstructible from the systems' own inputs with the same
  rule (high agreement = the benchmark mostly tests "implements the label rule").
  (2) V-text: an independently defined rule that does not read `teaches_skills` or `prerequisites` at all: a course's grade is the
  weighted count of open gold needs whose name (normalize_text, word-boundary match) or a SkillKeyResolver alias occurs in the course
  title + description + learning outcomes; grade 3 if mention mass ≥ 3 and ≥ 2 distinct needs, 2 if mass ≥ 2, 1 if mass ≥ 1, else 0.
  No prerequisite rule. It deliberately favours text systems and is a sensitivity probe, not a better gold.
- Agreement reported as Cohen's κ on grade≥2, quadratic-weighted κ on grades and Spearman ρ over all (scenario, course) pairs, and as
  Kendall τ between the system orderings (by mean nDCG@10) under v1 vs each variant.

### D-A5-06 — External validity: public prerequisite datasets and what is (and is not) tested on them
- Date / owner: 2026-11-17, A5
- Datasets (A5-owned, `data/external/`, MANIFEST entries): CPR-Recover (Liang et al., AAAI 2017; 654 real CS course descriptions
  from 11 US universities, 861 course-prerequisite edges, 3,544 concept pairs with 3 annotators; CC BY-NC-SA 4.0); AL-CPL (Liang et
  al., EAAI 2018; 4 domains, CC BY-NC-SA 4.0); LectureBank 1.0/2.0 topic prerequisite annotations (Li et al., AAAI 2019 / COLING 2020;
  no license file, research release with citation request). Pinned to commit SHAs.
- Test E1 (primary, CPR-Recover): the repo's graph semantics on REAL courses. Each course becomes a Course node; TEACHES edges =
  dataset concepts whose title (underscores→spaces, parenthetical dropped) occurs in the course description (word boundary,
  case-insensitive; this replaces Wikipedia-miner, which we do not have); a course-prerequisite edge B→A becomes
  A −REQUIRES_PREREQUISITE→ every concept B teaches, built with the repo's `CourseGraphBuilder`. Inferred concept prerequisite
  p ⇒ t iff ∃ course with TEACHES t and REQUIRES_PREREQUISITE p (support = #such courses); variant "minus co-taught" drops p when
  the same course also teaches p. Gold = majority vote of the annotations; evaluated on directed instances of every annotated pair.
  Baselines: prior-rate random, concept-frequency direction heuristic (more general concept = prerequisite), reversed graph rule
  (sanity), MiniLM name similarity with frequency direction. Metrics: P/R/F1 at support ≥ 1 (no tuning) and average precision;
  CIs by bootstrap over target concepts (pairs sharing a concept are not independent). Caveat: candidate pairs in CPR were generated
  from course dependencies, so recall is optimistic; precision is the informative number.
- Test E2: does the SYNTHETIC catalogue's prerequisite structure agree with independent annotations? Map catalogue skills to
  dataset concepts by normalised exact name; compare inferred catalogue skill-prerequisite pairs with annotations on overlapping pairs
  (expected to be few; reported as coverage + agreement, no significance claim).
- Not tested: the repo's `PrerequisiteParser` on external text (CPR descriptions carry prerequisite text for only ~10 courses and the
  gold is course→course, not skill strings), and retrieval on a public course catalogue (no relevance judgments exist for
  job-gap queries; producing them would need human or LLM labels; recorded as not_available rather than proxied).

### D-A5-07 — Larger-n generated scenario set (SG-v1), labels from rules fixed before any system is run on it
- Date / owner: 2026-11-17, A5
- Decision: generate 240 SYNTHETIC scenarios (seed 20261117) from the catalogue vocabulary only (no job-posting text): area chosen
  uniformly from the 8 catalogue areas; title from a fixed per-area list; required skills n_req ~ U{1..4} and preferred n_pref ~ U{0..2}
  sampled from skills taught in that area, implicit need n_impl ~ U{0,1} (not given to systems); user known skills = 1–4 skills taught
  by prerequisite-free courses of a random background area, plus with p=0.3 one of the job's required skills, and with p=0.2 one weak
  skill; job text = a fixed template naming title, required and preferred skills (favours lexical systems; stated). Labels = the
  unchanged v1 rule (`build_relevance_labels.label_scenario`), its content-only variant, and V-text (D-A5-05). SG-v1 is held out
  entirely: nothing is tuned on it (A5 tunes nothing anywhere). It populates the 1-gap and no-prerequisite-hop strata that the 24
  test scenarios lack. Reported separately from the primary test half and never pooled with it.
- Why: n=24 gives wide CIs; SG-v1 trades realism for power and stratum coverage. It is author-generated, so it cannot validate
  realism — only internal consistency of the H5/H6 conclusions across a broader synthetic distribution.

### D-A3-01 — Anchor re-inference uses the anchor's original decoding path, with log-probs recorded in every job
- Date / owner: 2026-11-17, A3
- Decision: all anchor GPU jobs (J1-J4, J6) run `scripts/evals/gpu_anchor_infer.py`, a thin driver over the unchanged
  `MistralExtractor` (config `configs/skillllm_v2_training_80ep_2gpu.yaml`, base `models/mistral-small-3.1-24b-instruct-2503`,
  adapter `artifacts/skillllm_v2/mistral_lora/model`, 4-bit NF4 + bf16 compute, `prompt_version="original"`, greedy
  `do_sample=False`, `max_new_tokens=512`, batch size 1, `allow_fallback=False`). `record_logprobs=True` in ALL jobs (not only
  J1/J2): it only adds `output_scores`, the greedy sequence is unchanged (verified on the smoke docs and by the J1
  reproduction check), and it gives confidence scores for spurious predictions (J3) and perturbed inputs (J4).
- Context: `scripts/eval_adapter_test.py` hard-codes the gold test subset and writes into `reports/`; existing files are read-only.
- Additions: per-doc CUDA-synchronised latency, prompt/generated token counts, per-doc peak CUDA memory, raw model output,
  `status=parse_error` (empty prediction list) instead of a crash when the model emits unparseable JSON (the original path
  would abort; all such docs are counted and reported), nvidia-smi sharing snapshots every 25 docs, DONE.json with sha256s.
- Alternatives: re-use eval_adapter_test.py via env vars (rejected: cannot select dev / no-span / custom inputs, overwrites
  report metrics).

### D-A3-02 — Fair BERT-family baseline (J5) design
- Date / owner: 2026-11-17, A3
- Decision: train jjzha/jobbert-base-cased (SkillSpan paper's JobBERT) and bert-base-cased as two-head BIO taggers (shared
  encoder; head 1 = SkillSpan knowledge tags -> HARD_SKILL, head 2 = skill tags -> SOFT_SKILL, so overlapping spans of the two
  types stay representable), first-sub-token labelling, AdamW lr 5e-5, wd 0.01, 10% linear warm-up, batch 16, <=20 epochs,
  early stopping on DEV exact-span F1 (patience 5), seeds 42/43/44. Spans decoded with the same function that builds gold
  spans from gold tags (round-trip verified exact on all dev/test sentences). THE fair baseline for H1 = the single run with
  the best DEV exact F1 across both encoders and all seeds (`bert_fair/SELECTED.json`); all runs reported (mean, sd).
- Why: the stored "BERT baseline" is a token-lookup table (integrity flag), so H1 vs it is a straw man. Selecting on dev
  avoids test tuning; reporting all seeds shows seed variance.
- Alternatives: per-type separate models as in the paper's single-task setting (2x compute; the paper also reports MTL);
  CRF head (not needed for a fair, standard baseline); lr sweep (not done: one standard setting, documented).

### D-A1-05 — SemEval/nervaluate input order is canonicalised (refines D-A1-03)
- Date / owner: 2026-11-17, A1
- Decision: before the greedy nervaluate scoring, predictions are de-duplicated and sorted by (start, end, label)
  (unlocated last); the oracle package receives the same order.
- Context: smoke run showed two systems with IDENTICAL span sets (anchor vs legacy Mistral+GLiNER+Taxonomy on 60
  sentences) getting different nervaluate strict F1 (0.740 vs 0.727) only because the wrapper re-orders its output -
  nervaluate 1.2.1 is order-dependent.
- Alternatives: file order (rejected: score depends on an irrelevant implementation detail); one-to-one max matching
  only (kept as a secondary column `semeval_<mode>_one_to_one_f1`).

### D-A4-02 — L07 synthetic histories: generator design, simulated extraction, baselines and a confidence-sensitivity arm (pre-registered before any L07 result)
- Date / owner: 2026-11-17, A4
- Decision: 120 seeded histories (40 short 3-6 events, 40 medium 8-15, 40 long 20-40) generated by
  `research/compaction.py` from authored templates over the skill universe `inputs/evals/skill_universe.json` (14 roles) plus
  extra hard/soft skills; fictional organisations; no real posting text. Truth (held skills + supporting sentences, current
  goal + goal history, true gaps = target-role required/preferred skills not held, planted contradictions of 4 types:
  retraction, HARD/SOFT label conflict, goal change, proficiency conflict) is emitted by the generator only.
  Non-held mentions are planted too (distractors "team used X but I did not", aspirations "would like to learn X") so that a
  representation that keeps every extracted mention can be WRONG about held skills.
- Simulated extractor: one extraction per surface mention with an authored confidence rule (held evidence U(0.75,0.99),
  retraction statements U(0.5,0.8), distractors U(0.3,0.6), aspirations U(0.35,0.65)). Because the budgeted compactor ranks by
  confidence, this rule favours compaction; a SENSITIVITY ARM with uninformative confidence (every mention U(0.5,0.99))
  is run and reported next to the primary arm. The H7 verdict is taken on the primary arm; if the sensitivity arm
  disagrees, that is reported as a threat to validity, not hidden.
- Extra baselines at equal proxy-token budget: "recent@b" (keep the most recent history lines that fit the budget, i.e.
  naive recency truncation), plus a truth-oracle profile as an upper reference. Course relevance (authored): 2 if the course
  teaches a true missing required skill, 1 if a true missing preferred skill, +1 if its prerequisites are all truly held;
  job relevance = `match_evals.grade_pair` on TRUE skills over `match_evals.make_jobs(n_variants=3)`.
- Kendall tau-b between representation and full-history score vectors (identical vectors -> 1.0; undefined -> 0.0).

### D-A2-03 — L04 calibration design and H4 operationalisation (pre-committed before any anchor log-prob result)
- Date / owner: 2026-11-17, A2 (written while A3's J1/J2 anchor log-prob jobs were still running; no anchor log-prob metric had been computed)
- Unit: a predicted span; correct = exact (start, end, label) match to SkillSpan gold of its document. Document universe = docs with ≥1 gold span (test 974; dev = the docs A3's J2 covers). CIs: document-cluster percentile bootstrap (2,000; 10,000 for the H4 contrasts and headline ECE / selective risk@80%).
- Scores: anchor log-probs from A3 (metadata.logprob {sum, mean, min, n_tokens}) → raw probabilities exp(mean lp), exp(sum lp), exp(min lp); logit of a log-prob score computed stably as lp − log(−expm1(lp)). Stored anchor confidence (constant 0.86) is the "constant-confidence baseline". GLiNER: model score. Dictionary: extractor confidence (0.82 exact alias, ≤0.72 fuzzy). Wrapper: final safe-wrapper confidence.
- Calibrators, fit on DEV ONLY and frozen before touching test: temperature σ(g/T), Platt σ(a·g+b), isotonic (PAV on g, clipped), and for the anchor a logistic model on [logit(exp mean lp), logit(exp sum lp), logit(exp min lp), log n_tokens, log n_words, SOFT label, # other sources agreeing (dictionary / GLiNER exact-key agreement)]. Model/score selection also on dev: H4's primary ranking score = the candidate with the lowest dev AURC (logistic scored by 5-fold document-grouped CV on dev); the reported calibrator = lowest dev NLL by 5-fold document-grouped CV on dev. Test is used once, for reporting.
- Metrics: ECE equal-width 10 and 15 bins, adaptive (equal-mass) ECE 10 and 15, MCE, Brier, NLL, AURC, e-AURC, AUROC, reliability diagrams (PNG + CSV), risk-coverage curve; selective precision AND resulting micro-F1 (abstained spans counted as missed: F1_c = 2·TP_c/(k_c + G)) at 100/95/90/80/70/50 % coverage, tie-aware (expected value under random tie-breaking).
- H4 (anchor): (a) selective precision@80 % − precision@100 % > 0 with 95% CI excluding 0 AND the same for @50 %; (b) AURC(primary score) − AURC(constant 0.86) < 0 with CI excluding 0; (c) the score/calibrator were fit/selected on dev only. SUPPORTED iff (a), (b), (c). If A3's dev or test log-probs are absent → `pending_gpu`.
- Secondary (labelled, never the verdict): v3/v4 test-only log-probs (no dev predictions exist for those adapters) with 5-fold document-grouped cross-fitting on TEST; the legacy logprob_confidence_v3/v4 numbers are re-derived and compared.
- Why: calibration claims are only meaningful if the mapping is learned on data disjoint from the reporting set; selecting the score on test (e.g. "exp(sum lp) was best on test") would be test-set model selection.

### D-A1-06 — L02 seen/unseen bucket rules (fixed before any L02 result)
- Date / owner: 2026-11-17, A1
- Decision: every TEST gold span (974 sentences, 2,265 spans) gets exactly one bucket relative to the TRAIN gold spans
  (4,800 SkillSpan(HF) train sentences); surface = `text[start:end]`; buckets are label-agnostic (a label-aware S1 share
  is reported as a secondary number):
  - S1 exact_seen: the case-sensitive surface string equals some train gold span surface.
  - S2 normalized_seen: not S1, and norm(surface) equals norm(some train surface), where norm = (1) Unicode NFKC,
    (2) casefold, (3) collapse whitespace runs to one space + strip, (4) strip edge characters in
    `.,;:!?"'`()[]{}<>*•·…–—-/\|` repeatedly from both ends (NOT `+` or `#`, so C++/C# survive; a leading "." is
    kept, so ".NET" survives, a trailing "." is stripped), (5) simple plural rule on the LAST whitespace token only, if it
    has >=4 characters: `-ies` -> `-y`; `-sses/-shes/-ches/-xes/-zes` -> drop `es`; else a final `s` is dropped unless the
    token ends in `ss`, `us` or `is`.
  - S3 canonical_seen_surface_unseen: not S1/S2, and a taxonomy maps the surface to a canonical id that at least one
    train surface also maps to. Mappings: (a) repo alias table `data/skillllm_v2/taxonomy/skill_aliases.jsonl` matched
    with the repo's own `schema.normalize_text`, using ONLY rows with source `manual_curated` or
    `skillspan_gold_train_dev` - the 76 `mistral_high_confidence` rows are mined from the anchor's TEST predictions
    (`normalize/skill_alias_builder.py`, default input `mistral_lora/predictions.jsonl`) and are excluded as test-derived;
    (b) ESCO preferred/alt labels (exact normalize_text match) once A4's ESCO download is in MANIFEST.jsonl.
  - S4 fully_unseen: everything else.
  Frequency strata (secondary): train mention count of norm(surface): 0, 1, 2-5, >5.
- Alternatives: label-aware buckets (rejected as primary: the question is lexical memorisation, the label is scored by
  the exact-match metric anyway); fuzzy/embedding similarity (rejected: threshold choice would be post hoc).
- Consequences: S3 via the alias table is small because 601 of its 711 rows are single-alias gold surfaces.

### D-A1-07 — L02 metrics and H3 statistic
- Date / owner: 2026-11-17, A1
- Decision: per bucket and system, gold-side recall = exact (start,end,label) matches of the bucket's gold spans /
  bucket gold spans (primary); prediction-side precision = de-duplicated predictions whose OWN surface falls in the
  bucket (same rule; unlocated predictions use their surface_form) that exactly match a gold span / all such predictions;
  bucket F1 = harmonic mean of the two. A true positive has the same surface on both sides, so it lands in the same
  bucket. CIs: sentence-level cluster bootstrap (shared draws across buckets; 10,000 for the anchor, 2,000 otherwise),
  posting-level sensitivity. H3 (anchor): Δ = F1(seen = S1∪S2) - F1(unseen = S4), paired over the same bootstrap draws
  (10,000), two-sided percentile-inversion p; SUPPORTED iff Δ >= 0.05 and the 95% CI excludes 0. S3 reported separately.
  Confounder check reported alongside (not part of the rule): per-label seen/unseen F1 and mean span length per bucket.

### D-C11 — Relabel the legacy baselines; H1 hinges on a fair trained baseline
- Date / owner: 2026-11-17, coordinator (after A3 integrity flag 21:03Z)
- Decision: `artifacts/skillllm_v2/bert_baseline` is reported everywhere as "token-lookup baseline (memorised most-frequent
  BIO tag per lower-cased train token)", never as "BERT"; GLiNER as "GLiNER zero-shot (urchade/gliner_medium-v2.1)". H1 is
  judged primarily against A3's fine-tuned JobBERT / bert-base-cased baselines (dev early stopping, 3 seeds); the legacy
  baselines stay in tables as weak reference points.
- Context: `src/skillllm_v2/baselines/bert_baseline.py` never loads a transformer; its config names bert-base-uncased.
- Alternatives: drop the legacy rows (rejected: the old money table cites them; correcting is better than hiding).
- Consequences: earlier claims of "BERT baseline 0.059" in reports/docs are wrong as labelled; A1's claims reconciliation
  and A6's threats-to-validity list them. Pre-registered H1 wording already included "any fairly-trained JobBERT baseline",
  so no rule changes.

### D-C12 — Stored anchor predictions stay official; offset-resolution drift reported separately
- Date / owner: 2026-11-17, coordinator (after A3 integrity flag 21:03Z)
- Decision: the July anchor predictions (old `find_offsets`, first substring match) remain the official anchor artifact for
  all comparisons with existing systems. The J1 re-run reports (a) model-output identity vs the stored raw outputs and
  (b) F1 under the old vs current offset code, so the thesis can quote the size of the offset effect.
- Why: changing the official artifact mid-program would break comparability with every stored wrapper/QLoRA prediction.

### D-A4-03 — L07: exploratory budgets 2x and 4x added after the smoke run (does not change H7)
- Date / owner: 2026-11-17, A4
- Decision: the smoke run (9 profiles, never quoted) showed the budgeted compactor needs roughly 4x the raw-history proxy
  tokens to keep every skill (each JSON skill fact costs ~35-45 proxy tokens vs ~12-15 for the raw sentence), so at the
  pre-registered 0.5x budget it keeps ~0-5 skills. Budgets 2x and 4x are added to the sweep as EXPLORATORY points so the
  compression-vs-agreement curve shows where agreement recovers. The H7 verdict configurations (compact@0.5 primary,
  compact@shipped) and the rule are unchanged (D-A4-01).
- Also fixed in my own code before the full run: top-5 overlap now divides by max(|top-k|) because the course
  recommender can return fewer than 5 courses (the smoke showed full-vs-full overlap 0.58, an artefact).

### D-A4-04 — L06 linking gold, label space, mention unit, NIL handling and thresholds
- Date / owner: 2026-11-17, A4
- Decision: gold = Decorte et al. (2022) TECH + HOUSE (HF `TechWolf/skill-extraction-{tech,house}`, CC-BY-4.0), which
  label every SkillSpan TEST skill span of those postings with the most specific ESCO v1.1.0 skill, or LABEL NOT PRESENT /
  UNDERSPECIFIED. Label space = ESCO v1.1.0 skills pillar (the gold's version; v1.2.1 kept as "latest" for others). Gold
  label strings -> URIs by exact preferred label (fallbacks: normalised preferred, normalised alternative; unresolved
  counted and excluded from the linkable set). TechWolf subset (sentence-level, no spans, not SkillSpan) is downloaded but
  not used for span linking.
- Mention = unique (subset, sentence, span) in Decorte rows; gold set = union of its ESCO labels (multi-label spans: a
  hit if top-k contains any gold concept). NIL = no ESCO label (LABEL NOT PRESENT or UNDERSPECIFIED).
- Metrics: top-1 accuracy / recall@{3,5,10} / MRR@10 / RP@5 on linkable mentions; NIL precision/recall/F1 (positive class
  NIL, predicted NIL = abstain) and NIL-aware accuracy on all mentions. Linking input = gold span text, context-free
  (isolates linking from extraction; reported as an upper bound for the pipeline).
- Thresholds: abstention threshold per score-based linker chosen on Decorte VALIDATION by maximising NIL-aware accuracy
  (ties -> lowest threshold); every linker also reported "@never" (never abstains). No test tuning.
- Uncertainty: cluster bootstrap over SkillSpan postings (`source:idx`, the independent unit; mentions/sentences of one
  posting are correlated); paired cluster bootstrap for linker comparisons, Holm within each metric family.
- Strata: TECH/HOUSE; HARD (SkillSpan knowledge tag) / SOFT (skill tag) from the aligned SkillSpan gold span; seen =
  normalised surface is a SkillSpan TRAIN gold span surface (A1 had not posted buckets when L06 ran); variant_surface =
  surface differs from every gold preferred label; alias_listed = surface equals an ESCO alternative label of a gold concept.
- The system's own taxonomy is not ESCO: (a) `cascade_esco` = the repo's TaxonomyLinker algorithm loaded with ESCO as its
  taxonomy; (b) `shipped_bridge` = shipped 696-entry alias table + a deterministic repo-entry -> ESCO bridge by exact
  normalised label match (name/aliases vs ESCO preferred/alternative labels), built without any gold; a top-1 whose entry
  has no bridge is NIL. Baselines: random, exact-string, BM25 (k1 1.5, b 0.75, untuned), dense all-MiniLM-L6-v2 (max cosine
  over a concept's labels).
- Alternatives: evaluating only coverage on predicted spans (rejected: label-free); mapping repo ids to ESCO using test gold
  (rejected: leakage); using TechWolf sentence-level set (rejected: not span linking).

### D-A3-03 — L03 robustness subset: 120 stratified test docs (seed 42), all perturbations on the same docs
- Date / owner: 2026-11-17, A3
- Decision: proportional stratified sample of 120 of the 974 gold-bearing test docs over strata (source tech/house x
  gold-span count 1 / 2-3 / 4+), seed 42, largest-remainder allocation (`perturbations.stratified_sample`). Every
  applicable perturbation family is applied to every sampled doc (paired design), plus an `identity` copy run in the same
  GPU job as the clean reference (determinism control; also cross-checked against J1).
- Why 120: measured anchor speed on this shared host is ~7 s/doc (other projects keep the 20-core host at load ~38; 4-bit
  decoding is CPU-launch-bound), vs 4.6 s/doc in an earlier quiet run. 120 docs x ~12 applicable variants = ~1,430
  inferences = ~2.8 GPU-hours = ~1.4 h wall on two GPUs, inside the 2-3 GPU-hour budget. 160 docs would need ~3.7 GPU-hours.
- Consequences: Δ CIs are paired doc-bootstrap over 120 docs; families applicable to fewer docs (alias_sub, reorder,
  paraphrase, punctuation) get their own n and wider CIs; inapplicable (doc, family) pairs are listed with reasons in
  `inputs/evals/research/L03_robustness/not_available.jsonl`, never dropped silently.

### D-A3-04 — Perturbation families and validity rules
- Date / owner: 2026-11-17, A3
- Decision: token+separator document model (SkillSpan gold is token-aligned) so offsets are recomputed exactly and
  asserted for every span. 14 families + identity (see `research/perturbations.py` docstring). Label-preserving by
  construction: context-only edits never touch span tokens or separators inside spans; surface-changing families
  (case_*, typo_inside, alias_sub) keep the same skill and label but change the gold string; insertions (boilerplate,
  distractor) are synthetic sentences written for this evaluation (no real postings).
- Observations that shaped it: SkillSpan HF sentences contain NO comma tokens (commas were removed upstream), so
  "reorder clauses" splits only at sentence-internal ; : . ! ? / mid-text bullet markers and otherwise swaps two
  coordinated gold spans ("A and B" -> "B and A"); splitting at every "and" produced ungrammatical text in a dry run and
  was rejected. "punct_detok" therefore mostly glues the final " ." / ":" / ")" as in natural text.
- Validity risks (to be checked by the 50-item human validity sample, L11): distractor names (e.g. "Ruby", "Spark",
  "Agile" as names) could be read as skills by an annotator; alias_sub depends on the alias source (repo taxonomy has only
  12 multi-alias concepts; ESCO alt labels are added if A4's ESCO download is complete when the set is frozen);
  case_upper/typo_inside change the gold surface, so exact-match drops there partly reflect a harder target, not only
  brittleness.
- Alternatives: LLM paraphrasing (rejected: not deterministic, can change spans); character-noise at random positions
  (rejected: unrealistic).

### D-A4-05 — L07: Mistral token counting must disable the tokenizer's stored truncation
- Date / owner: 2026-11-17, A4
- Decision: `artifacts/skillllm_v2/mistral_lora/model/tokenizer.json` ships with `truncation.max_length=512`. My first
  full L07 run counted packet tokens through it and every lossless packet came out at exactly 512 tokens (caught by the
  suspicious constant mean). The counter now calls `no_truncation()`/`no_padding()`; a test guards it. The first full run's
  outputs are overwritten by the corrected rerun and were never quoted.

### D-A5-08 — Explanation-faithfulness claim taxonomy (fixed in code before the first full run)
- Date / owner: 2026-11-17, A5
- Decision: claims are read from the STRUCTURED fields the explanation text is rendered from (covered_missing_skills, evidence[],
  prerequisite_status, unlocks[], limitations[]), not by regex over prose. Claim types: covers_gap_skill (strict = the field read
  literally: the course teaches X and X is a gap; lenient = honours the packet's own evidence_type qualifier: text-related /
  enables-via-bridge / teaches), job_requires_skill (X in job required ∪ preferred), user_gap_classification (missing_critical ⇔
  not held & required; missing_preferred ⇔ not held & preferred-only; weak ⇔ held at beginner), outcome_quote_mentions_skill (the
  quoted learning outcome mentions X by name or SkillKeyResolver alias — catches the generator's fallback to outcomes[0]),
  prerequisite claims (each "satisfied by learner / earlier course Y", "all satisfied", "missing", "none required"), unlock claims
  (T requires p; this course teaches p; placement in list), limitation claims (workload > 40 h, overlap with earlier picks, bridge,
  prerequisite gap). Faithfulness = supported / total (micro) with a per-scenario bootstrap CI; unsupported claims per scenario.
- Tautology probes for the existing Explanation Completeness metric: (a) random courses with placeholder list fields; (b) Full's own
  packets rotated onto the wrong course of the same list (completeness should stay 1.0 while faithfulness drops).
- Why: a field-presence metric cannot distinguish a faithful explanation from a fabricated one; the claim-level check can.

### D-A5-09 — Smaller implementation choices (recorded for examiners)
- Date / owner: 2026-11-17, A5
- E1 average precision with score ties (the graph rule's support is an integer, mostly 0): positives inside a tie group are
  placed at evenly spaced expected ranks (expected-precision approximation); P/R/F1 at the a-priori threshold support ≥ 1 is the
  primary E1 number. CPR concept pairs without a 2-of-3 majority are excluded.
- L09 "hybrid" fuses BM25(q_gap), MiniLM(q_gap) and the gap-overlap greedy list (length 40) with RRF k=60; "+filter" variants apply
  the existing `apply_prereq_filter` (same chained rule as the violation metric).
- Reproduction check: L09 recomputes the stored benchmark-v2 test numbers with an independent harness and records the maximum
  absolute difference (expected 0).
- Graph QA compares graph edges with the authored catalogue fields (by-construction 1.0, reported as such) and, separately, the
  enrichment outputs (CourseSkillExtractor / PrerequisiteParser / TaxonomyLinker) with the authored fields, because the
  enrichment is computed by the pipeline but does not reach the graph.

### D-A1-08 — ESCO version for L02 S3, and how H1/L01 pick up A3's GPU outputs
- Date / owner: 2026-11-17, A1
- Decision: (a) S3 uses the ESCO skills pillar only once A4's download is registered in `data/external/MANIFEST.jsonl`;
  the latest version listed there is used (v1.2.1 if present), matching `schema.normalize_text` of preferred + alt
  labels exactly; the version and file hash are written to `results.json` meta. (b) A3's fair BERT-family baseline enters
  the H1 family only as the single dev-selected model (`evals_gpu/bert_fair/SELECTED.json` + `predictions.jsonl`); its
  individual seed runs are reported descriptively (mean/sd) and never used for selection. (c) The anchor's
  skill-free-sentence predictions (`evals_gpu/anchor_test_nospan`, DONE.json) are merged with the STORED 974-sentence
  anchor predictions for the full-test protocol; caveat: the two halves come from different inference runs (A3 reports
  offset-code drift between them).

### D-A3-05 — Cross-dataset (out-of-distribution) evaluation: datasets, label mapping, metrics (pre-registered before any run)
- Date / owner: 2026-11-17, A3
- Decision: two public English span-annotated job-text corpora, test splits, seeded samples of 300 sentences each (seed 42):
  Green et al. 2022 (jjzha/green, CC BY 4.0; tags SKILL / DOMAIN / QUALIFICATION / OCCUPATION / EXPERIENCE) and
  Sayfullina et al. 2018 soft skills (jjzha/sayfullina; one SOFT span per sentence). Scoring is LABEL-AGNOSTIC (any
  predicted HARD_SKILL/SOFT_SKILL span counts as a skill; the anchor's two-way label has no counterpart there).
  Green primary target = SKILL spans; secondary = SKILL ∪ DOMAIN ∪ QUALIFICATION (SkillSpan "knowledge" partly covers domains
  and degrees). OCCUPATION / EXPERIENCE are never targets. Sayfullina: only recall (exact + overlap) is valid, because only
  one soft skill per sentence is annotated; precision/F1 are recorded as not_available with that reason.
- Why: tests generalisation beyond SkillSpan's annotation guideline and sources. Guideline mismatch (Green's SKILL also
  covers tasks such as "managing all aspects of the kitchen"; Sayfullina sentences are lemmatised) depresses all systems,
  so results are read relative across systems (anchor vs fair BERT vs GLiNER vs token lookup), not as absolute quality.
- Not used: jjzha/jobstack (gated, manual approval), jjzha/skillskape (no spans: sentence -> ESCO label), non-English sets
  (kompetencer/gnehm/fijo; the anchor is English-only by training).

### D-A2-04 — L10 central significance: families, references, ingestion rules
- Date / owner: 2026-11-17, A2
- Decision: (1) Central extraction re-test from stored predictions on the 974-doc test subset with four declared families: `H1_anchor_vs_baselines` (anchor − zero-shot GLiNER, anchor − token-lookup "BERT", anchor − each fair BERT-family run of A3 once its DONE.json exists), `H2_wrapper_vs_anchor` (stored safe-wrapper variants and legacy stages − anchor), `QLoRA_vs_anchor` (v1..v4 − anchor), `ensemble_vs_anchor` (span-key majority of anchor/v1/v2, reconstructed exactly as scripts/ensemble_eval.py − anchor). Exact-span F1 is primary; token, HARD and SOFT F1 form separate secondary families `<family>:<metric>`. Paired document bootstrap 10,000 + AR 10,000 (exact when ≤16 documents differ). (2) Every `reports/skillllm_v2/evals/*/per_unit_scores.jsonl` is ingested; rows are grouped by (layer, family, metric, label, bucket, population); comparison = system − reference where the reference is the row's `reference` key if declared, else the first of [R0_raw_mistral, mistral_lora, anchor, full, hybrid, baseline] present, else all pairs (≤6 systems) or vs the first system; paired on shared unit ids only (unpaired units counted and reported). Count rows → pooled micro-F1 (and token micro-F1 when token counts exist); bucket rows → pooled recall gold_tp/gold_n; value rows → mean. Layer families use 2,000 bootstrap resamples. (3) Holm within each declared family (`<layer>:<family>` for ingested files), BH across all non-cross-check tests; AR p-values are adjusted the same way in parallel columns. (4) Comparisons computed only to cross-check legacy files are excluded from multiplicity. A legacy result is flagged when |Δ| differs by >1e-6, when the raw-p<0.05 decision differs (bootstrap or permutation), when a CI bound differs by >0.003, or when the Holm decision differs (families differ, so this flag is informational).
- Alternatives: re-testing only the owners' primary metrics (would hide secondary-metric multiplicity); Bonferroni (too conservative across hundreds of tests).
- Consequences/why: one command reproduces every p-value in the thesis from per-unit data; owners keep verdict authority, L10 reports agreement.

### D-A6-02 — L11 packet design: strata, blinding, what annotators see
- Date / owner: 2026-11-17, A6 (written before any annotation exists)
- Decision: seven packets (P1 extraction candidates, P2 full-sentence spans, P3 linking, P4a prerequisite edges, P4b learning
  paths, P5 course recommendations, P6 explanation faithfulness; P7 perturbation validity when A3's sample exists) + a
  synthetic training round T0 whose items cite SkillSpan Appendix B rules (read from the paper PDF A1 downloaded). Two
  annotators (separately shuffled orders, seeds 1001/2002 + build seed), one adjudicator for disagreements only. Item ids
  are assigned after a seeded shuffle; answer keys live in `inputs/evals/research/L11_human_eval/answer_keys/` (separate
  tree); a test fails if a hidden field (system, gold, stratum, rank, rule label, ids) appears in any packet header.
  P1 strata: anchor_fp 50, anchor_fn 50, boundary 50, added_by_wrapper (census: only 11 such spans exist), baseline_only 20,
  agreed_tp_control 20; proportional by tech/house. P2: 60 gold-bearing + 40 gold-free test sentences, with a headline
  flag (SkillSpan B.3.4 leaves posting titles untagged, so "gold-free" is not "skill-free").
  P5: pooled top-5 of tuned Full + the pre-specified baseline (Gap-Overlap Greedy + prereq filter) on the 24 held-out
  scenarios + the 60 legacy rows; annotators see job text + listed required/preferred skills, NOT `gold.needs` (the hidden
  rule-label input, which the legacy CSV exposed together with the automatic grade). P6 cannot be system-blind (only Full
  explains) and is labelled so. P3 fallback (A4's sample absent at build) = safe-wrapper links stratified by method, with
  links via the 76 test-mined alias rows flagged in the key (I12 circularity).
- Context: the existing 60-row review sample is unblinded and leaks the label inputs; extraction "false positives" may be
  gold omissions; the 2,595 gold-free sentences were never scored.
- Alternatives: random (unstratified) span sample (rejected: wrapper additions and anchor FPs are rare and would be
  missed); showing gold needs in P5 (rejected: would measure agreement with the rule's inputs, not relevance).
- Consequences: human metrics are pending_human; packet counts are measured records; no hypothesis verdict depends on L11.

### D-A6-03 — Agreement statistics and sample-size targets for L11
- Date / owner: 2026-11-17, A6
- Decision: Cohen's κ (quadratic-weighted for ordinal scales), Gwet's AC1 (nominal; kappa-paradox check), Krippendorff's α at
  the scale's level (coincidence matrix, missing data allowed), pairwise span F1 (exact/overlap, label-sensitive) for free
  spans; 95% percentile bootstrap over items (2,000, seed 42). No verbal κ labels; α ≥ 0.667 used only as a planning
  reference. Targets: ±0.10 for pooled proportions (n ≥ 97), ±0.14 within a 50-item stratum; κ half-width ≈ 0.11–0.17 at
  the built packet sizes (Monte-Carlo at κ=0.6, prevalence 0.5). Substantive rates use adjudicated labels, per stratum.
- Validation: `agreement.py` reproduces Krippendorff (2011)'s published example and matches `krippendorff` 0.8.1 and
  scikit-learn to ≤ 2.3e-16 on 74 cases.
- Alternatives: Fleiss' κ (only for > 2 raters; not needed), simple percent agreement (reported, not primary).

### D-A2-05 — Resample count grows with family size (Holm vs the bootstrap p-value floor)
- Date / owner: 2026-11-17, A2
- Decision: for every test in a family of m tests, bootstrap resamples and AR shuffles are B = max(base, 40·m + 1) (cap 50,000), so the smallest attainable smoothed two-sided p, 2/(B+1), times m stays below 0.05.
- Context: the first L10 run used 2,000 resamples for ingested layer families; with m = 53 (L01) or 244 (L07) tests, Holm could not reject even anchor − GLiNER (Δ = 0.59 F1, p at the floor 0.001 × 53 = 0.053). That was an artefact of resampling resolution, not of the data.
- Alternatives: normal-approximation p from the bootstrap SE (unfloored but assumes normality); Bonferroni-free reporting.
- Consequences/why: Holm decisions in large declared families reflect the effects, not the Monte-Carlo resolution; runtime grows linearly with family size (seconds per test).

### D-A3-06 — Alias-substitution source restricted to true aliases
- Date / owner: 2026-11-17, A3
- Decision: alias groups = repo taxonomy groups with source `manual_curated` (12 concepts; the 76 aliases mined from anchor
  test output are excluded) + ESCO v1.2.1 preferred/alt labels, where an ESCO alternative is accepted only if it is an
  acronym/expansion of the gold surface, a spelling variant (identical after lower-casing and removing non-alphanumerics),
  or the same head words plus generic words (programming, language, software, development, skills, system, database, ...).
- Why: a dry run with unfiltered ESCO alt labels produced non-aliases (PowerShell->Tcl, SQL->Adaptive Server Anywhere,
  machine learning->semi-supervised learning, Agile->structured methodology) that would change the skill and invalidate the
  label. Consequence: alias_sub applies to 15/120 docs; reported with its own n and a wide CI.

### D-A3-07 — L12 systems metric definitions
- Date / owner: 2026-11-17, A3
- Decision: latency = CUDA-synchronised wall time of `MistralExtractor.extract` per document (batch 1, greedy, logprob
  recording on), J1 run on the 974 test docs = headline population; p50/p95/p99 over documents; CI of p95 by bootstrapping
  documents (1,000). Per-span latency = doc latency / #predicted spans (docs with >= 1 span) plus the aggregate
  total-time / total-spans. Headline `peak_vram_gb` = `torch.cuda.max_memory_reserved` over the run (allocator footprint;
  allocated reported alongside; nvidia-smi adds the CUDA context). Parameter counts from safetensors headers (no weights
  loaded); LoRA trainable = all tensors in `adapter_model.safetensors`. CPU systems (token lookup, GLiNER, fair BERT for
  latency, wrapper stages) are timed on CPU, batch 1, guardrails-pinned threads; wrapper stage times exclude the anchor call
  and the wrapper's end-to-end latency is anchor + wrapper per document.
- Context: the host is shared (other projects' CPU jobs, load average 38-47 on 20 cores during the runs); bitsandbytes 4-bit
  decoding is CPU-launch-bound (GPU util ~20%), so latency is reported with the recorded load and with the earlier quieter
  measurement of the same decoding path (4.58 s/doc, qlora_loop_v4_eval) as context. Not a deployment benchmark.
- Alternatives: vLLM / batched decoding (would measure a different system than the one evaluated).

### D-A3-08 — Anchor re-run reproduction check separates model output from offset resolution (coordinator ruling)
- Date / owner: 2026-11-17, A3
- Decision: stored `artifacts/skillllm_v2/mistral_lora/predictions.jsonl` stays the official anchor artifact. J1 is compared on
  (a) model-output identity: ordered (surface_form, raw label) lists per doc; (b) offset resolution: J1 surfaces re-resolved
  with the legacy `find_offsets` of commit 3e86a1b (first plain substring match, same `used` bookkeeping) vs the stored
  (start, end, label) sets; J1 is scored with both the current and the legacy offset code; Δ(current - stored) exact F1 with
  paired doc bootstrap + approximate randomization. Output: `artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/REPRODUCTION.json`
  and L12 records.

### D-A3-09 — J5 sanity target fixed before training finishes: published SkillSpan numbers
- Date / owner: 2026-11-17, A3
- Decision: the J5 fair baseline is judged "fairly trained" if its exact span F1 on the FULL 3,569-sentence test set (labels
  required, both types pooled = the paper's COMBINED) falls near the published values (Zhang et al. 2022, Table 5, test,
  5 seeds): BERT-base 58.16±0.47 (STL) / 57.73±0.66 (MTL); JobBERT 59.73±0.38 (STL) / 59.18±0.37 (MTL)
  (data/external/skillspan_paper/skillspan_naacl2022.pdf). Our setup is MTL-like (shared encoder, two heads). A result far
  below would mean our training is still not fair and is reported as such; we do not tune on test to close any gap.
- Note: the anchor's headline is on the 974-doc gold-bearing subset; A1 reports both populations. Published numbers are
  full-test only.

### D-C13 — Continuation agents replace the lost A2-A6; ownership unchanged
- Date / owner: 2026-11-18, coordinator
- Decision: the original agents were lost to an API rate limit and a session restart. New agents B1..B5 take over the unfinished layers with the same ownership and the same contract (reports/skillllm_v2/evals/CONTRACT.md); they must re-read Progress.md/Decisions.md and not redo finished work. B1=A3 remainder (bert_fair selection + test inference, L03, L12), B2=A2 remainder (L04, L10 rerun), B3=A4 remainder (L06), B4=A5 remainder (L08 verify, L09), B5=A6 remainder + A1 reruns (docs, repro, leaderboard, L01/L02 pending_gpu rerun).
- Why: pre-registered hypotheses and definitions remain in force; the continuation changes who executes, not what is tested.

### D-B4-01 — L09: pre-declared additions (written BEFORE the first full L09 run; supplements D-A5-03, does not change any D-A5 definition)
- Date / owner: 2026-11-18, B4 (continuing A5). Only a 4-scenario smoke subset of L09 (not a result) had been seen when this was written.
- Decision (1, descriptive family "L09-vs-baselines"): on the 24 held-out TEST scenarios, SkillLLM_v2 Full (tuned DEFAULT_CONFIG) is compared with each of the other 7 primary systems on nDCG@10, Precision@5, WSGC@5 and prerequisite violation rate@5 (paired bootstrap B=10000 + sign-flip permutation, p = max of the two; Holm over the 7 baselines separately per metric — the same scheme as the stored benchmark `course_recommendation_benchmark_v2`). This is a descriptive family: it does not alter H6 (H6 = Full vs Full-noPrereq per D-A5-03) and no verdict is attached except the plain sentence "significantly better / not distinguishable / significantly worse".
- Decision (2, "strongest graph-free baseline"): the comparator pre-specified by the stored benchmark, `prerequisite_aware(greedy+filter)` (Gap-Overlap Greedy + prerequisite filter), is reported as the named strongest graph-free baseline; in addition the baseline with the highest mean nDCG@10 on the DEV half (tuning split) among the 7 is determined from dev only, and if it differs it is reported too. Test labels play no part in the choice. Reproducing the stored result (Δ nDCG@10 +0.035, Holm p 0.094) independently is a check of the harness, not a new claim.
- Decision (3, tautology wording rule): "Explanation Completeness is tautological" is asserted in the report only if completeness(random courses + placeholder fields) ≥ 0.99 AND completeness(Full packets rotated onto wrong courses) ≥ 0.99 AND faithfulness(rotated) < faithfulness(Full) − 0.05; otherwise the report says it is not demonstrated.
- Decision (4, learning-path validity unit): per-scenario binary, whole top-10 list in returned order (D-A5-04); "repairable" reported alongside.
- Alternatives: pooling SG-v1 with test (rejected, D-A5-07); selecting the strongest baseline on test (rejected: test-label peeking).
- Why: the reviewer-facing question is "does Full beat the best graph-free alternative", which D-A5-03 only answered indirectly (via ± filter pairs).

### D-B1-01 — bert_fair selection executed as pre-registered; H1 reported as NOT SUPPORTED
- Date / owner: 2026-11-18, B1 (continuing A3)
- Decision: SELECTED.json was produced by the pre-existing `gpu_bert_fair.py --select` (max DEV exact F1 over all six runs = jjzha/jobbert-base-cased seed 42, dev 0.5907). No test number was consulted; no new training or test inference was needed because each run had already tagged all 3,569 test sentences at training time with the dev-best checkpoint. The H1 verdict is whatever extraction_full computes against it: NOT SUPPORTED (Δ +0.037, below the fixed 0.10 margin) although significant.
- Context: bert-base-cased runs stopped at the 20-epoch cap (best epoch 20/20/17), so that encoder may be slightly under-trained; it is not the selected model, and JobBERT converged earlier (best epochs 19/18/9). No lr sweep was run (one standard setting, D-A3-02).
- Alternatives: re-select using test (rejected: contract rule 2); lowering the margin (rejected: pre-registered).
- Consequences/why: the thesis claim becomes "the anchor beats a fairly trained JobBERT by about 4 F1 points (CI 1.7-5.7), not by 10", and "beats the repository's stored baselines by 0.53-0.59 because those are a lookup table and a zero-shot model".

### D-C14 — Token-saving and server-safety operating mode for the rest of the program
- Date / owner: 2026-11-18, coordinator (user request)
- Decision: caveman style for chat output; context-ledger (`ledger.py check/digest/note/resume`) for re-reads; tail-only reads of Progress.md and logs; every heavy job via server-guardrails launcher, one heavy job per agent; paused/queued jobs are waited on, never relaunched. Code, commits and thesis docs stay in normal prose.
- Why: two rate-limit cutoffs already cost ~9 h of agent time; re-reading large files is the main avoidable spend.

### D-C15 — Push evaluation program to `main` after the evals are final
- Date / owner: 2026-11-18, coordinator (explicit user instruction: "once you have the best evals over that, push on main")
- Decision: single commit (or a small series) pushed to origin/main by the repo owner account only after: all layers measured or pending_human, leaderboard/chapter/eval card regenerated, research tests pass, leak scanner clean, secret/size check clean. Stage explicit paths, never `git add -A`; exclude data/external, model weights, logs and anything large.
- Why: the user owns the ruleset bypass; the ruleset's PR gate applies to collaborators, not to this push. Nothing is pushed before the results are final.

### D-C16 — No AI attribution on the main push (supersedes the trailer mention in D-C15)
- Date / owner: 2026-11-18, coordinator (explicit user instruction: "drop it, it should be just me")
- Decision: the evaluation commit(s) and any PR text carry no Co-Authored-By, Claude-Session or "Generated with" lines; author is chetas1208 as configured in git.

### D-B4-02 — L09 post-hoc additions after the first full run (flagged as post-hoc; verdicts of D-A5/D-B4-01 NOT changed)
- Date / owner: 2026-11-18, B4. Written after seeing the first full L09 results (test half + SG-v1 means), hence exploratory.
- Observations that motivated it: (i) H6 primary nDCG Δ is +0.114 on v1 labels but -0.023 [-0.056,+0.006] on content-only and -0.033 [-0.062,-0.005] on V-text labels: non-inferiority (lower CI > -0.02) holds only under the v1 rule that itself caps bridge-only/not-startable courses (the label rule shares the prerequisite-feasibility logic the strict mode implements). (ii) On SG-v1 Full has much lower WSGC@5 than greedy (0.718 vs 0.886) while higher nDCG@10.
- Decision: (a) the report states per label variant whether the H6 nDCG non-inferiority holds (computed flag + sentence) and the H6_verdict record carries `label_sensitivity` in details; the formal verdict stays the pre-registered one (primary v1 labels). (b) SG-v1 (held out, never pooled) Full vs the pre-specified strongest baseline `prerequisite_aware(greedy+filter)` and vs `skill_gap(greedy)` on nDCG@10, P@5, WSGC@5, violation: paired bootstrap + sign-flip, Holm over the 2 comparisons per metric, labelled exploratory/post-hoc.
- Why: examiners will ask whether the H6 nDCG half is an artefact of the label rule; stating the sensitivity result plainly is cheaper than leaving it in a table cell.

### D-B2-01 — L04 executed unchanged; one labelled post-hoc supplement (base-rate constant)
- Date / owner: 2026-11-18, B2 (continuing A2)
- Decision: the real anchor dev/test log-prob predictions (A3 J1/J2, both DONE.json) were run through calibration_selective.py with the D-A2-03 design UNCHANGED (score selected on dev AURC -> sum_lp; calibrator by dev 5-fold CV NLL -> Platt; test used once; H4 contrasts paired, 10,000 document-cluster replicates). H4 rule and baseline (stored constant 0.86) were not altered. Because a constant equal to the dev base rate has ECE ~ 0 by construction, a secondary comparison against that constant is reported by research/calibration_supplement.py (supplement_base_rate.json). It is post-hoc and labelled secondary; it does not feed the verdict.
- Context: headline ECE improvement over 0.86 would otherwise overstate calibration value (0.86 is simply mis-set vs precision 0.675). The informative evidence is AURC / selective precision, which a constant cannot have.
- Alternatives: re-define the H4 baseline after seeing results (rejected: post-hoc); omit (rejected: examiner would ask).
- Consequences: deployable abstention (threshold tau fixed on dev for max dev micro-F1) gives test micro-F1 +0.0001, i.e. H4 as pre-registered measures ranking ability (selective precision), not an end-to-end F1 gain; micro-F1 with abstained = missed falls at every coverage < 100%.

### D-B2-02 — L10: byte-identical fair-baseline files counted once; headline forest
- Date / owner: 2026-11-18, B2
- Decision: significance.py de-duplicates BERT-family baseline prediction files by sha256 (bert_fair/predictions.jsonl is a byte copy of the dev-selected run jjzha__jobbert-base-cased_s42), so the H1 family has 8 comparisons instead of 9 and no system is double-counted in Holm/BH. Adds forest_headline.png (pre-registered primary contrasts only: central exact-span F1 families, H5, H6) next to the full forest_plot.png.
- Why: a duplicated comparison inflates family size m (slightly conservative Holm) and presents one baseline as two independent ones. No hypothesis decision changed (H1 stays NOT SUPPORTED on the effect-size criterion).

### D-B5-01 — Naming audit: evidence standard, adjudication rule and verdict (continues D-A6 work)
- Date / owner: 2026-11-18, B5 (continuation of A6)
- Decision: per (term, source) records are produced by public APIs (arXiv, OpenAlex, Semantic Scholar, DBLP, Crossref, ACL Anthology bibliography, GitHub, PyPI, npm, Hugging Face) with a screening rule fixed in the module docstring, plus manual web-search records (WebSearch/WebFetch, dated, with URLs). The predecessor's 14 manual records carry an estimated timestamp and were not time-stamped by a tool, so B5 re-ran the web queries on 2026-11-18 and stored them as separate dated records; where the two disagree, neither is deleted. Final risk is adjudicated per term (never below screening without override_reason). Verdict rule: adopt / adopt_with_caveat (no exact collision but same-field near-name stem) / pick_alternative (exact same-field collision or blocking mark). Result: ADOPT WITH CAVEAT for SkillGraphLM; 'SkillLLM', 'Skill-LLM', 'SkillGPT' are high-risk (same task) and unusable as research names.
- Context: web content is untrusted data; only titles/names/URLs are stored. DBLP (bot challenge), Google Scholar (consent wall), trademark registries (JS apps, Justia 403) are unreachable from this host, and Semantic Scholar rate-limited some terms; these are listed as unreachable rather than treated as "no hit". The trademark status of SKILLGRAPHS (serial 99842770) is UNVERIFIED.
- Alternatives: automated-only audit (rejected: misses products/web); renaming now (rejected: contract forbids renaming, author decides).
- Consequences: nothing renamed; thesis must carry a disambiguation sentence and the full descriptive title; a formal trademark search is the user's task. Alternative names (SkillAnchorLM, AnchorPathLM, GapGraphLM low risk; SkillTrellisLM medium) are candidates only.

### D-B1-02 — L03/L12 completion choices
- Date / owner: 2026-11-18, B1
- Decision: (a) the fair BERT (dev-selected JobBERT seed 42) and the recomposed safe wrapper (full_cag_graph composition on the perturbed-input ANCHOR output; verifier/calibrator not re-fit, no new tuning, CPU) were run on the same 120-doc perturbation set and both cross-dataset samples; they enter L03 as extra systems. (b) CPU RSS was missing from earlier test974 DONE.json files, so a fresh-process 100-doc measurement per CPU system (scripts/evals/gpu_cpu_rss.sh, stored under evals_gpu/smoke/rss_*, a resource measurement not a result) fills it; anchor host RSS stays not_available. (c) Host load for J1/J2 was never recorded; the L12 caveat cites the J3/J4/J6 medians (35) and the coordinator's ~38 note instead of inventing a value. (d) Headline degradation keeps the pre-set definition (unweighted mean over all 14 families); the surface-preserving subset (gold string unchanged) is reported next to it because case/typo/alias families change the target string.
- Why: keeps every number traceable to a file and shows that the degradation headline is mostly driven by surface-changing families.

### D-B3-01 — L06 continuation: circularity verdict, L02-aligned strata, NIL strata, annotation sample design
- Date / owner: 2026-11-18, B3 (continuing A4)
- Decision: (1) The legacy "Linking Accuracy" (0.890374) is judged NOT an accuracy and CIRCULAR on the evidence of label-free
  diagnostics computed from the stored linked_skills files (reproduced exactly): coverage is 0.895 on predicted spans that
  exactly match a gold span vs 0.879 on the rest, 52% of links are BM25 fallbacks, 76% of linked spans carry the taxonomy tag
  SKILLSPAN_GOLD, 67/76 test-mined alias rows are test gold surfaces, and the shipped linker links ~66% of random non-gold
  1-3-grams. These are labelled `diagnostic_*` records (status measured, direction none); no accuracy is derived from them.
  (2) seen/unseen strata use the L02 buckets by calling A1's `generalization.Bucketer` unchanged (seen = S1+S2, unseen = S4,
  S3 reported separately, same alias + ESCO indices), replacing A4's simpler "exact train surface" definition so L02 and L06 agree.
  HARD/SOFT = SkillSpan tag of the aligned gold span; HARD x seen/unseen and SOFT x seen/unseen cross cells added.
  (3) Alias-resolution accuracy = top-1 on mentions whose surface is an ESCO ALTERNATIVE label of the gold concept
  (`alias_listed`), with `pref_label_surface` and `variant_surface` as contrasts; flagged as dictionary-style (trivially high for
  lookup/dense). (4) NIL precision/recall/F1 reported overall and per stratum (tech, house, HARD, SOFT, seen, unseen) with posting-cluster CIs.
  (5) Annotation sample (300 items) = stratified over PREDICTED spans of the safe-wrapper output (label x link-method confidence bin x
  predicted-NIL x seen), min 4 per cell then sqrt-proportional; candidates = cascade_esco top-5 shown in per-item shuffled order,
  no scores; mention highlighted with [[...]]; 205 items are "control" items whose span equals a Decorte gold span (the key holds
  the Decorte label so a human-vs-Decorte agreement can be computed); the blinded key is a separate file.
  (6) Thresholds still tuned on Decorte validation only; nothing tuned on test.
- Context: Decorte validation and test are both drawn from SkillSpan test (alignment 99.8%), so "validation" is a 20% split of the
  same postings, used for thresholds only. Gold is a single annotation pass (no IAA published).
- Alternatives: dropping the circularity audit and only citing the Decorte results (rejected: examiners will ask about 0.890);
  re-deriving buckets with a different normaliser (rejected: would make L02/L06 strata disagree).
- Consequences: the 0.890 can only be cited as "linking coverage (diagnostic)"; cascade_esco@tau_val (tau=1.0) is effectively a
  lookup and is reported as such; the shipped linker cannot be evaluated in ESCO terms beyond the bridge (15.5% of repo entries map).

### D-B5-02 — L11 P3 switches to ESCO candidate-choice mode; P4b/P7 use the new samples
- Date / owner: 2026-11-18, B5
- Decision: when `L06_linking/annotation_sample.csv` exists (A4/B3: 300 items, sentence + mention + 5 shuffled ESCO candidates, blinded; key in `annotation_answer_key.jsonl`), P3 is built from it instead of the safe-wrapper proposed-concept fallback. Annotators choose one of A-E, NONE_OF_THESE or NOT_A_SKILL (`q1_choice`; the more specific candidate when two are equally right, the other named in `q2_note`). A seeded random 150 of the 300 rows are shown (workload; drawn from a separate RNG stream so other packets are unchanged); all 300 stay in the key. P3 metrics in this mode: agreement on q1_choice, human-vs-Decorte-gold rate on control items, per-linker top-1 vs adjudicated human choice. P4b now uses A5's `paths_for_human_eval.jsonl` (pool 350) and P7 uses A3's validity sample (50 items; only original/perturbed text shown, family and gold spans in the key). T0-P3 rewritten in the same format.
- Context: the proposed-concept format cannot be applied to a candidate list without a proposal, and a candidate list is the better design (blind to systems, yields independent ESCO gold).
- Alternatives: keep the fallback (rejected: it measures the circular legacy linker); multi-select A-E (rejected: complicates kappa); show all 300 (rejected: workload, total packets already ~820 items).
- Consequences: human metrics remain pending_human; the P3 sample has 16 strata so half-sampling leaves about 9 items per stratum, so stratum-level claims are not supported (pooled claims only).

### D-B5-03 — Chapter section 10 is an auto-generated verdict list plus number-free prose; naming/threat refresh rules
- Date / owner: 2026-11-18, B5
- Decision: `chapter_evaluation.md` §10.1 is an AUTO block (`verdict_statements`, written by `leaderboard.py --chapter`) that lists every `H#_verdict` record with verdict, effect of record, CI, p-values, the layer's pre-registered rule text and recorded caveats, verbatim; no verdict is re-labelled. §10.2 is hand-written but number-free (it points to the layer blocks); §10.3 lists what the chapter does not establish. `threats_to_validity.md` gained a register (Section G) of every `integrity-flag` line in Progress.md with its effect and handling, and its status columns were refreshed (H1 now NOT SUPPORTED, offset drift quantified, gold-free population measured, linking measured).
- Context: the earlier rule "no number typed by hand" is kept; negative verdicts (H1, H2, H7) are stated as emitted.
- Alternatives: leaving §10 for the author (rejected: the task requires the chapter to state every verdict); typing numbers into prose (rejected).
- Consequences: after any layer rerun (e.g. L10), rerun `leaderboard --chapter` and `repro_card`; the G register must be extended when a new integrity-flag is logged.

### D-C17 — Rename new test file to avoid a pytest basename collision
- Date / owner: 2026-11-18, coordinator
- Decision: `tests/evals/research/test_significance.py` renamed to `test_research_significance.py`. Without it, `pytest tests/evals tests/skillllm_v2` failed at collection ("import file mismatch") because the tracked `tests/skillllm_v2/test_significance.py` has the same basename and neither directory has `__init__.py`. Checked all other new test basenames: no further collisions. No tracked file changed.

### D-C18 — Commit only compact eval outputs; leave regenerable bulk out of git history
- Date / owner: 2026-11-18, coordinator
- Decision: the main push stages 278 files / 35.8 MB (code, tests, scripts, results.json, REPORT.md, figures, small CSVs, thesis docs and tables, synthetic inputs, Progress.md, Decisions.md). Excluded: `**/smoke/**`, `**/per_unit_scores.jsonl`, `L06_linking/per_mention_outputs.jsonl`, `L05_wrapper/ladder/per_doc_counts.csv` (about 110 MB). They stay on disk and are regenerated by `scripts/evals/run_research_evals.sh`; `reports/skillllm_v2/evals/README.md` says so.
- Alternatives: commit everything (rejected: 146 MB into a public repo's history is irreversible); gzip the dumps (rejected: L10 reads plain jsonl).
- Consequence: a fresh clone cannot rerun L10 without rerunning the earlier layers first; eval-card hashes of excluded files cannot be checked from the repo alone. Secret scan (token/key patterns) found nothing; leak scanner clean over 993 files; no model weights or binaries staged.
