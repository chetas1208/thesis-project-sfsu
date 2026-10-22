# SkillLLM_v2: master plan, status, and what to claim

Status as of 2026-10-22, about 09:05 UTC. Written for the thesis advisor and committee. Detail lives in the four drafts: `docs/thesis_drafts/training.md`, `pretraining.md`, `evals.md`, `inference.md`; the master audit is `docs/INFERENCE_AND_EVALS.md`. Numbers are from repository files (cited in the drafts), **derived** (arithmetic shown there) or **pending**. Status vocabulary: **done**, **running**, **pending** (planned or awaiting an input), **rejected** (with reason), **open** (known defect, not fixed).

## 1. In one paragraph

The thesis system is a skill-span extractor (Mistral-Small-3.1-24B with LoRA adapters) followed by rule-based post-processing and a rule-based downstream (profile, graph, matching, gaps, course recommendation, explanations). The extractor is the only component evaluated on real, public, human-annotated data: exact-span F1 0.670 (95% CI [0.646, 0.694]) on 974 SkillSpan test documents, single seed. A cheap quantised loop reaches within 0.9 to 1.4 points of it at 0.9 to 1.8% of the GPU-hours with no significant difference detected; a three-adapter majority vote adds about 2 points (suggestive, not confirmed after multiplicity correction). The post-processing wrapper adds at most 0.0024. Everything downstream is verified on synthetic, self-authored data, which found and fixed real defects but does not establish real-world quality. Human review (0 of 60 rows), resume-domain evaluation, gold canonical links and multi-seed variance are the missing evidence. Pretraining of the 24B was assessed and **rejected** on feasibility and measurability grounds.

## 2. Status by area

**Table B1.**

| Area | Status | Where it stands (evidence) |
|---|---|---|
| Extraction, 80-epoch anchor | **done** | Run 2 on disk: exact F1 0.670069; run 1 scored 0.675034 (log only, adapter overwritten); 180.1 GPU-h per run; dev loss minimum 0.2340 at epoch 3 against 0.3974 at epoch 80 (`training.md` T3 to T5) |
| QLoRA loop v1, v2 | **done** | 1.59 and 3.17 GPU-h; test exact F1 0.6565 and 0.6614; both not significantly different from the anchor (CIs about 3 points wide) |
| Three-adapter ensemble and vote-count confidence | **done** | +0.0198 [+0.0041, +0.0356], raw p 0.016, Holm-5 0.081; precision 0.29 / 0.55 / 0.80 by vote count (estimated on test) |
| Loop v3, short prompt (GPU Experiment 1) | **running** | Started 08:54; interim round-0 training cost 0.095 GPU-h against v2's 0.197; dev and test results pending (`training.md` T13) |
| Loop v4, ensemble distillation (Experiment 2) | **pending** | Corpus prepared (1,832 of 1,887 postings survive the SkillSpan overlap filter; 2,400 chunks); training not started; cap 2.4 GPU-h |
| Log-probability confidence (Experiment 3) | **pending** | Descriptive analysis planned after Experiments 1 and 2 generate test predictions |
| Domain- or task-adaptive pretraining of the 24B | **rejected** | Only 3.2 to 4.9 M tokens of postings exist and 0 resume tokens, 400 to 2,500 times below remembered DAPT corpora (from memory, verify); 4 to 20 GPU-h per pass-count before re-fine-tuning, exceeding the 8 GPU-h cap; a 1-point gain is undetectable at the MDE of about 3 points (`pretraining.md`) |
| Smaller-base pilot (Llama-3.1-8B, in `docs/loop_training.md`) | **rejected for the thesis result** (not run) | Would break comparability with the Mistral numbers; only a hyper-parameter pilot, untested |
| 80-epoch re-runs for seed variance | **rejected** | 180 GPU-h per run; the loop recipe (3 GPU-h) is the affordable replicate unit |
| Wrapper (merge, refine, verify, calibrate, link) | **done** | At most +0.0024 exact F1, significant only for offset cleanup on 10 documents; below the 0.005 seed noise floor; non-regressive |
| Learned span verifier at inference | **rejected for anchor spans** | On real candidates AUROC 0.579; applying it to all spans costs 5.5 points exact F1; kept off by default (`evals.md` M13) |
| Evaluation suite (711 results, 37 controls, 10 modules) | **done, re-run pending** | Verdict PASS; report is a mixed-state snapshot (context fixes in, graph, matcher, gap and e2e fixes not); baseline refresh pending (`evals.md` 5.1) |
| Bug fixes found by the suite | **done** (agent-reported), re-confirmation **pending** | 21 defects tabulated; 6 remain **open** (`evals.md` Table E8: date conflicts, `trace_explanation_path`, `CourseGraphBuilder` idempotence and ids, course linkage, `GapEngine` weak-held weight) |
| Human review of recommender labels | **pending** | `human_review_sample.csv` 0 of 60 filled |
| Resume-domain / out-of-distribution evaluation | **pending** (not started) | No resume text exists locally; `Resume+Job Data/` is empty |
| Gold canonical links and linking accuracy | **pending** (not started) | Only coverage (0.8898) and an authored 127-item probe exist; no ESCO, O*NET or Lightcast export |
| Course recommender benchmark | **done** | Held-out half nDCG@10 0.885 [0.851, 0.916]; difference from the strongest baseline +0.035 [-0.006, +0.072], Holm p 0.094; synthetic catalogue, rule-derived labels |
| 12-stage inference pipeline | **done, partly verified** | CPU stages verified with a stub extractor on 12 synthetic cases; real 24B model through the whole pipeline **not run** (`inference.md` T6) |
| Python library and model card | **done (unpublished)** | `SkillLLM` API, CLI, ensemble layer; real model loading unit-tested with fakes only; `MODEL_CARD.md` is a template with placeholders |
| Serving optimisation (merge, vLLM, AWQ, prefix cache) | **pending** (described, not done) | No benchmark; any change needs re-evaluation on the 974 documents |
| Thesis document | **open** | PDF and chapters dated 2026-09-02; contain none of the loop, ensemble, vote or suite results; still quote 0.671102 and say no nDCG or P@5 exists; administrative fields missing (`docs/thesis_missing_admin_info.md`) |

## 3. What to claim

**Table B2. Claims table.** "Supported" claims carry the stated qualifier; "unsupported" claims must not appear without new evidence.

| # | Claim | Verdict | Wording that is safe, and why |
|---|---|---|---|
| C1 | The fine-tuned extractor reaches useful span-extraction quality on SkillSpan | **Supported** | "Exact-span F1 0.670 [0.646, 0.694], partial 0.814, character-level 0.788, HARD 0.727, SOFT 0.604 on the 974 gold-bearing test documents; one seed; a repeat training gave 0.675." Not comparable with published SkillSpan numbers |
| C2 | The quantised loop matches the 80-epoch run at about 1 to 2% of the compute | **Supported with qualifier**; "matches" is **unsupported** | "Within 0.9 to 1.4 exact-F1 points; no significant difference detected (CIs [-3.4, +0.7] and [-2.9, +1.1] points; MDE about 3 points; one seed)" |
| C3 | The loop is better than, or at least as good as, the anchor | **Unsupported** | Point estimates are lower; CIs allow a 3-point deficit |
| C4 | Hard-example mining, loss masking or the curriculum cause the efficiency | **Unsupported** | Nothing is ablated; anchor and loop differ in 8 factors |
| C5 | Long training over-fits | **Partly supported** | Dev loss rose from 0.2340 (epoch 3) to 0.3974 (epoch 80); dev F1 per epoch was never computed |
| C6 | Three-adapter majority vote improves extraction | **Suggestive only** | "+1.98 exact-F1 points [+0.41, +3.56], raw p 0.016, not significant after Holm correction over five metrics (p 0.081); single seeds; the ensemble idea arose after seeing single-adapter test scores; 3x inference cost" |
| C7 | Agreement count is a usable confidence signal | **Supported as a signal** | "Precision 0.29, 0.55, 0.80 for 1, 2, 3 votes, estimated on the test split itself" while the models' own confidence is constant (AUROC 0.5, ECE 0.186) |
| C8 | The wrapper (GLiNER, dictionary, taxonomy, "CAG/graph") improves extraction | **Unsupported** | At most +0.0024, significant only for offset cleanup on 10 documents; no effect from GLiNER or the graph; the variant name "full CAG/graph" involves no context or graph |
| C9 | The safe wrapper "falls back to the anchor" at run time | **Unsupported** | The fallback is an offline gate decided on the test set; none triggered |
| C10 | "Linking accuracy 0.89" | **Unsupported** | It is coverage; there are no gold links; 46.6% of links are BM25 guesses with precision 0.60 on an authored probe (n = 20) |
| C11 | The span verifier and calibrator are learned components | **Unsupported** | Rule filters; the trained verifier is off and weak on real errors |
| C12 | Job matching, gap detection, explanations are accurate | **Unsupported** | Only synthetic, self-authored data (nDCG@10 0.99 is a ceiling effect; gap F1 1.0 is by construction); explanation faithfulness checked structurally only |
| C13 | The recommender outperforms strong baselines | **Partly supported** | Significantly above BM25, dense and user-agnostic baselines on a synthetic catalogue; **not** distinguishable from the strongest user-aware gap-greedy baseline with prerequisite filtering (+0.035, p 0.094); labels rule-derived; 0 of 60 human grades |
| C14 | "Course recommendation P@5 = 1.0" | **Unsupported** | Stale money-table entry; current P@5 is 0.617 (v1) and 0.608 (v2 held-out half) |
| C15 | Results hold on resumes | **Unsupported** | Not evaluated |
| C16 | The system is fair, or suitable for automated hiring | **Unsupported** and out of scope | No fairness evaluation; model card prohibits such use |
| C17 | Software defects were found and fixed by systematic evaluation | **Supported** | 21 defects tabulated with before and after values on synthetic data; independent re-run pending |
| C18 | The pipeline degrades gracefully | **Supported on synthetic stub tests** | Not tested with the real model or under load |
| C19 | Pretraining the 24B would help | **Unsupported**, assessed and rejected | `pretraining.md` |
| C20 | The BERT baseline is BERT | **Unsupported** | It is a token-memorisation table; rename |

## 4. Top ten next steps, ranked by expected value per cost

Cost estimates are mine unless a file is cited; "human h" are rough judgement. GPU steps respect the 8 GPU-h cap in `docs/GPU_EXPERIMENTS_LOG.md` only where stated.

**Table B3.**

| Rank | Step | Cost | Expected value | Notes |
|---:|---|---|---|---|
| 1 | **Run the real 24B model through the full 12-stage pipeline on the 12 e2e cases and a handful of real postings and resumes**; record per-stage latency, memory, and whether extraction errors propagate | under 0.1 GPU-h (about 24 generations of about 4 s plus a 20 s model load, derived); allow up to 0.3 for debugging | Very high: closes the largest verification gap in `inference.md` T6 for almost no cost | Needs a free GPU slot; sequence after Experiment 1 |
| 2 | **Re-run the suite on the fixed tree, refresh `baseline.json`, fix stale constants and tests** (anchor 0.675034 in the gate config, docs and docstrings; money-table P@5; GLiNER `config.json`; `loop_training.md` 0.6726; two stale eval tests) | about 5 CPU minutes plus 1 to 2 h editing; edits to `src/` and `reports/` need owner approval | High: removes headline inconsistencies (I1, I3, I6, I11) before the thesis quotes them | Not done by me: I was told not to touch `src/`, `reports/` or existing docs |
| 3 | **Reconcile the thesis with current evidence**: anchor 0.670069 with CI, loops, ensemble, vote confidence, drop 0.671102, rename "BERT baseline", state coverage not accuracy, update the "no nDCG or P@5" limitation, add the claims in Table B2 | 2 to 4 days writing; no GPU | Very high: the PDF is six weeks stale and contradicts current results | Needed before any deployment or review |
| 4 | **Annotate an out-of-distribution evaluation set**: 200 to 300 sentences from resumes and 2026 postings (finish the 250-row `review_sheet.csv`, 60 corrected so far) and score the anchor, v2 and the ensemble | about 6 to 12 human h; inference about 0.3 GPU-h per adapter | Very high: resumes are the deployment domain and are unmeasured; also gives gold-free false-positive rates | Resume text must be sourced (the local folder is empty) with consent |
| 5 | **Finish Experiments 1 to 3** (running or pending) and report results as pre-declared, including negative results | within the 8 GPU-h cap | High: tests prompt compression, distillation to 1x cost, and log-probability confidence | Interim v3 round 0 already 2.1x cheaper in training time |
| 6 | **Human-grade the 60 recommender rows** (`human_grade_0to3`) and compute agreement with `auto_grade_0to3` | about 1 to 2 human h | High for research question 2: tests label validity of the only downstream benchmark | |
| 7 | **Replicate the loop recipe with two more seeds** (v2 settings) and evaluate on the 974 documents | about 6.4 GPU-h training plus about 2.4 GPU-h evaluation (derived from 3.17 and 1.20 per run); outside the 8 GPU-h cap, needs a separate decision | High: first estimate of seed variance; makes the compute claim and the ensemble claim defensible | Alternatively two seeds only for the cheapest recipe |
| 8 | **Gold canonical links**: annotate or import about 300 span-to-skill links (the TechWolf datasets, CC BY 4.0, per `docs/EVAL_LANDSCAPE.md`) and install `ir_measures`, `statsmodels`, `nervaluate` so the six skipped oracle checks run | 1 to 2 human days; CPU only | Medium to high: replaces "coverage" with real linking accuracy; unlocks any encoder adaptation | Needs a taxonomy export (ESCO, O*NET) |
| 9 | **Safety hardening**: make `trust_remote_code` default `False` in the research path, document the joblib trust rule, add a prompt-injection smoke test | under 1 day; needs `src/` edit approval | Medium: removes the one default that executes model-directory code | Verify the Mistral checkpoint loads without remote code |
| 10 | **Batched generation and a serving benchmark** for the three adapters (precondition for distillation at scale and for any latency claim) | about 1 to 2 GPU-h | Medium: replaces extrapolated ensemble latency with a measurement | Then decide whether merge plus vLLM is worth it |

Dropped from the list on purpose: further wrapper tuning (effects below the noise floor); any pretraining; re-running the 80-epoch recipe.

## 5. Timeline against the 1 November and 5 December deadlines

The two dates come from the task brief. **I could not find them in `Project Desc.md`** (it contains no dates; I searched every `.md` and `.tex` under `Sem 4 Research`), and nothing I read defines what "deployment" on 1 November comprises. I assume it means a runnable demonstration or library hand-off, and the plan below treats it as such. Today is Thursday 22 October: 10 days to 1 November, 44 days to 5 December (derived).

**Table B4.**

| Window | Work | Gate or output |
|---|---|---|
| 22 to 25 Oct (4 days) | Experiments 1 to 3 within the GPU cap; suite re-run and baseline refresh (rank 2); real-model pipeline smoke run (rank 1) once the GPU is free | Interim results logged; honest e2e latency; consistent constants |
| 26 to 31 Oct (6 days) | Freeze the deployable configuration: single anchor or three-adapter ensemble (decide from rank 1 timing and rank 5); safety hardening (rank 9); library "doctor" check on the target machine; write the deployment limitations (not for hiring decisions; confidence meaning; offline mode is a gazetteer) | **1 Nov deployment gate**: only Table B2 "supported" claims shown; stub or degraded modes clearly flagged in output; model card draft with licence review items listed |
| 2 to 15 Nov | Out-of-distribution annotation and scoring (rank 4); human grading (rank 6); gold links (rank 8); seed replicates if budget is approved (rank 7) | New evidence tables; updated claims |
| 16 to 30 Nov | Thesis rewrite (rank 3) with the four drafts as source chapters: training, evaluation framework, inference and limitations; regenerate figures and tables from artifacts; advisor review round | Full draft to the advisor by about 23 Nov, revisions by 30 Nov |
| 1 to 5 Dec | Freeze; reproducibility appendix with exact commands; fill administrative fields (`docs/thesis_missing_admin_info.md`: course designation, committee, approval page); final QA | **5 Dec final submission** |

Risks to the schedule: GPU contention (the server runs other jobs; GPU0 shows about 4 GB in use and all GPU work goes through GPU1 under guardrails); resume text is unavailable locally and needs sourcing before rank 4; steps that need edits to `src/` or `reports/` are blocked on owner approval; the thesis text is the long pole (rank 3), so it should start in parallel from 26 Oct rather than after the experiments finish.

## 6. Compute ledger (derived from cited reports)

| Item | GPU-hours |
|---|---:|
| Anchor training, run 1 and run 2 (about 180 each; 3.237e5 s and 3.242e5 s on 2 GPUs) | about 360 |
| Loop v1 and v2 training | 4.76 (1.591 + 3.173) |
| Loop v1 and v2 test generation | about 2.2 (0.98 + 1.20) |
| New experiments cap (Experiments 1 to 3; excludes the rows above) | 8.0 |

## 7. Decisions that need the advisor

1. Accept the compute result as "no significant difference detected, about 1 to 2% of the cost" rather than "equivalent", and the ensemble as "suggestive".
2. Approve a budget for seed replicates (rank 7), or accept that all single-run results carry the 0.005 noise floor in the thesis text.
3. Decide whether the thesis scope is an evidence-preserving systems study with extraction as the measured result and downstream recommendation as a synthetic, bounded evaluation (the framing already in `docs/thesis/NEXT_RESEARCH_UPDATE.md`), or whether real matching and recommendation data must be collected before 5 December (not feasible in 44 days without a data source).
4. Approve edits to `src/` and `reports/` for the consistency and safety fixes (ranks 2 and 9).
