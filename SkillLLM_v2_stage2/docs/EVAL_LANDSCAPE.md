# Evaluation landscape for SkillLLM_v2

Research date: 2026-10-22. Repo metadata (licence, stars, last push, open issues) was read with `gh api` on that date; release dates and runtime dependency lists come from the PyPI JSON API; HF licences from the HF API. Claims that I ran myself are marked **[tested]**; everything else is read from docs/metadata and is marked **[docs]** or listed under caveats (section 9).

## 1. Bottom line

| Role | Pick | Why in one line |
|---|---|---|
| Harness backbone for model-in-the-loop evals (extraction runs, explanations, end-to-end agent) | [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) (MIT) | Only candidate that is MIT, very actively maintained, runs local models (HF, vLLM, Ollama, llama.cpp), has custom scorers/metrics, per-sample logs and an agent/tool/sandbox model, and needs no API key. |
| Ranking-metric oracle (linking, job match, course ranking) | [ir_measures](https://github.com/terrierteam/ir_measures) + [pytrec_eval-terrier](https://pypi.org/project/pytrec-eval-terrier/) | trec_eval is the reference implementation; tiny install; our hand-written nDCG@10/P@5/Recall@10 matched it to 1e-16 **[tested]**. |
| Span-extraction oracle | [nervaluate 1.x](https://github.com/MantisAI/nervaluate) (MIT, zero runtime deps) | strict/exact/partial/ent_type; strict P/R matched our `evaluate_extraction` **[tested]**. |
| Statistics oracle | [scipy.stats](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) (`bootstrap(paired=True)`, `permutation_test`, `wilcoxon`) + optional [statsmodels](https://github.com/statsmodels/statsmodels) `multipletests` (Holm) | Matched our bootstrap, sign-flip and Holm code **[tested]**. |
| Domain benchmark/data for ESCO linking | [WorkRB](https://github.com/techwolf-ai/workrb) (Apache-2.0) + TechWolf HF datasets (CC BY 4.0) | Only maintained benchmark toolbox for exactly this domain, incl. graded ESCO linking tasks. Heavy (torch + sentence-transformers), so use it in a separate env. |

**Single best repo to build the harness around: Inspect AI**, with a strong caveat. The repo already has an in-house, stdlib-only `EvalContext`/`EvalResult` contract (`src/skillllm_v2/evals/contract.py`) and per-module suites under `src/skillllm_v2/evals/` (these were being written while I researched; I did not audit them). Keep that contract and the hand-written metrics as the source of truth for deterministic CPU evals. Use Inspect only where a model, tool loop or judge is in the loop (modules a-rerun, e, h), and have its scorers emit values that are then wrapped into `EvalResult`. No candidate is a good "framework to adopt wholesale" for the deterministic metric modules (a, b, c, g); there the right move is oracle cross-checks, not a new framework.

## 2. What the project already has (so recommendations are not redundant)

* `eval/ranking_metrics.py`: nDCG (exponential gain, ideal over all judged courses), P@k (divides by k), Recall@k, truncated AP@k with denominator min(R,k), paired bootstrap, Holm. `eval/significance.py`: document-level paired bootstrap, sign-flip permutation, McNemar, Holm. `eval/extraction_eval.py`: exact/token/partial span F1, per-label F1.
* `evals/` package: shared `EvalResult` (value, n, CI, caveats, status), suites for extraction (already nervaluate-style modes), linking, calibration, verifier, context, graph/explanations, match, gap, e2e. `gap_evals.py` already cross-checks Kendall/Spearman against scipy, which is the pattern this document extends to the other metrics.
* `requirements.txt` pins `numpy==1.26.3`, `pandas==2.2.1`, `torch==2.6.0`, `nervaluate==0.2.0`, `spacy==3.7.4`. These pins matter for compatibility (section 5).

## 3. Empirical cross-checks I ran **[tested]**

Throwaway venvs under the session scratchpad (not the project venv): `venv_eval` (ranx 0.3.21, ir_measures 0.4.3, pytrec_eval-terrier 0.5.11, nervaluate 1.2.1, seqeval 1.2.2, scipy 1.18.1, statsmodels, scikit-learn 1.9.1; 800 MB, dominated by numba/pandas/matplotlib pulled in by ranx) and `venv_small` (ir_measures + nervaluate only; 222 MB, almost all numpy/scipy). I imported `ranking_metrics.py` and `extraction_eval.py` read-only from the repo and compared on seeded synthetic data (200 scenarios x 40 judged courses x 25 ranked; grades 0-3; 10 scenarios with no relevant course).

| Our metric | Oracle | Result |
|---|---|---|
| `ndcg_at_k(k=10)` (2^rel-1 gain) | ir_measures/trec_eval with gains pre-transformed to 2^g-1; ranx `ndcg_burges@10` | max per-scenario abs diff 1.1e-16 vs both |
| `precision_at_k(k=5, rel>=2)` | ir_measures `P(rel=3)@5` on transformed gains; ranx with `Qrels.set_relevance_level(2)` | 0.0 diff |
| `recall_at_k(k=10, rel>=2)` | same | 0.0 diff |
| `average_precision_at_k(k=10)` | trec_eval/ranx AP@10 | **up to 0.249 per-scenario diff**, caused purely by convention: ours divides by min(R,k) (stated in our docstring), trec_eval/ranx divide by R. After rescaling ours by min(R,k)/R, diff is 5.6e-17. Our MAP is therefore not comparable with trec_eval-style MAP in papers unless stated. |
| Scenarios with no relevant course | trec_eval/ranx/ir_measures score them 0.0 and keep them in the mean; ours returns `None` and drops them | nDCG@10 mean 0.2799 (ours, 10 dropped) vs 0.2659 (trec-style, all 200). Report the number dropped and the policy. |
| `evaluate_extraction` exact (strict, span+label) P/R/F1 | nervaluate 1.2.1 `strict` | 0.6502/0.5778/0.6119 (ours) vs 2309 correct, 1105 incorrect, 137 spurious, 582 missed, i.e. P 0.6503 / R 0.58 / F1 0.61 (nervaluate). Agree. nervaluate `exact` (boundary only, type ignored) gave F1 0.77, a useful "type-agnostic span F1" number. Our `partial_span_f1` (0.7433) does not equal nervaluate `partial` (0.84): the two define partial credit differently, so they cannot be compared directly. |
| `paired_bootstrap` mean-diff CI | `scipy.stats.bootstrap((b,a), paired=True, method="percentile")` | CI [0.0013, 0.0488] vs [0.0014, 0.0486] |
| `bootstrap_mean_ci` | `scipy.stats.bootstrap` percentile | [0.5575, 0.6184] vs [0.557, 0.6178] |
| sign-flip p-value | `scipy.stats.permutation_test(permutation_type="samples")`, `wilcoxon` | our bootstrap p 0.039; scipy permutation 0.035; Wilcoxon 0.034 (different tests, same conclusion) |
| `holm_adjust` | `statsmodels.multipletests(method="holm")` | identical on a 6-p-value example |
| ECE (a 15-line equal-width implementation, binary confidence) | `sklearn.calibration.calibration_curve` + bin counts | identical to 6 decimals (0.068035). sklearn has `calibration_curve`, `brier_score_loss`, `CalibrationDisplay` but **no ECE function** (checked `sklearn.metrics`/`sklearn.calibration` in 1.9.1). |

Findings about the oracles themselves **[tested]**:

* `ir_measures.nDCG(dcg='exp-log2')` is routed to the `gdeval` provider, which shells out to Perl, and crashed here ("format error on line 1", exit 25). Workaround that worked: pre-apply the gain transform (grade g -> 2^g-1) to the qrels and use the default linear `nDCG`, which goes through pytrec_eval.
* ranx issue [#84](https://github.com/AmenRa/ranx/issues/84) reproduced: passing plain dicts whose query-insertion orders differ makes `evaluate()` pair the wrong queries (P@1 0.0 instead of 0.5 in my reproduction). Passing `Qrels`/`Run` objects gave the right answer. ranx issue [#71](https://github.com/AmenRa/ranx/issues/71) reproduced: `Qrels({"q": {"d": 1.9}})` silently truncates to 1, so non-integer graded relevance is unsupported (our grades 0-3 are ints, so this is harmless today). Other open ranx issues I read but did not reproduce: [#85](https://github.com/AmenRa/ranx/issues/85) (bpref NaN and Fisher test reporting p=0 when a score is NaN), [#70](https://github.com/AmenRa/ranx/issues/70) (Fisher randomization non-deterministic even with fixed seed).

## 4. Ranked shortlist per module

Verdict key: **Adopt** (use in the eval harness or as the cross-check oracle), **Reference** (read it / copy definitions / use as optional extra; do not make it a dependency), **Skip**.

### (a) Skill span extraction (exact / token / partial F1, per label, paired stats)

1. **nervaluate 1.2.1** (MIT; PyPI release 2026-03-12; zero runtime deps; py>=3.11; [repo](https://github.com/MantisAI/nervaluate), 225 stars, pushed 2026-03-12). Adopt as oracle for strict/exact/partial/ent_type. Note the repo pins `nervaluate==0.2.0`; 1.x changed the API (`from nervaluate.evaluator import Evaluator`, loaders `list`/`dict`/`prodigy`) and needs Python >= 3.11. The older `davidsbatista/NER-Evaluation` repo (last push 2024-07) is the origin and is superseded.
2. **scipy.stats / statsmodels** for the paired tests (section 1). Adopt as oracle.
3. **seqeval** (MIT; PyPI 1.2.2 from 2020-10; [repo](https://github.com/chakki-works/seqeval), last push 2024-08). Reference: conlleval-style BIO chunk F1 equals strict span F1, and SkillSpan reports span-level P/R/F1 via MaChAmp, so it is a cheap third cross-check for BIO-formatted baselines (BERT, GLiNER). Frozen but stable.
4. **[SkillSpan repo](https://github.com/kris927b/SkillSpan)** (MIT code, data CC BY 4.0, 69 stars, last push 2025-02). Reference only: its `metrics/` folder is empty (just `.gitkeep`), so there is no official standalone scorer; the paper reports span-level P/R/F1, five seeds, and almost-stochastic-dominance tests (read in the [paper PDF](https://arxiv.org/pdf/2204.12811)). Use it for the data format and protocol, not code. The data in this repo (SkillSpan HF export, [CC BY 4.0](https://huggingface.co/datasets/jjzha/skillspan)) is already checked in.
5. **[deep-significance](https://github.com/Kaleidophon/deep-significance)** (Almost Stochastic Order test, the test SkillSpan used). **GPL-3.0**, PyPI release 2023-10, so Reference only; if multi-seed comparisons are needed, re-implement ASO from the paper rather than importing GPL code into a permissively-licensed repo.
6. **Skip**: lm-evaluation-harness, HELM, OpenAI evals, promptfoo for this module (they score generations against references; no span-offset F1, and our offsets/alignment logic is the hard part).

### (b) Canonical skill linking (retrieval/ranking against ESCO/O*NET)

1. **ir_measures + pytrec_eval-terrier** for P/R/nDCG/MAP/MRR (Apache-2.0, ir_measures pushed 2026-10-20, PyPI 0.4.3 2025-11; runtime deps: only `pytrec-eval-terrier`, plus numpy/scipy transitively). Adopt as oracle. Gotchas in section 3.
2. **WorkRB 0.6.0** (Apache-2.0; [repo](https://github.com/techwolf-ai/workrb), 44 stars, pushed 2026-07-22, 3 open issues). Adopt as an optional benchmark adapter and sanity baseline: tasks include ESCO skill normalisation (`ESCOSkillNormRanking`, 72k queries x 13.9k targets), skill extraction ranking on TECH/HOUSE/TECHWOLF/SkillXL/SkillSkape and **graded** variants (`TechGradedSkillExtractRanking` etc., 61-100 English queries), job-to-skill, MELO/MELS; baselines BM25, TF-IDF, edit distance, random, plus sentence-transformer bi-encoders. It computes MAP, MRR, recall/hit/rp@k and graded nDCG itself in `src/workrb/metrics/ranking.py` and reports CIs. Cost: core dependencies include `torch>=2.6`, `sentence-transformers>=5.1`, `datasets`, `pydantic>=2.11`; so install it in a separate env and exchange rankings/qrels files, not imports. Small, young repo (bus-factor risk), v0.x API.
3. **Data**: TechWolf HF datasets, all [CC BY 4.0](https://huggingface.co/TechWolf) (`skill-extraction-tech/house/techwolf`, `Skill-XL`, `Skill-extraction-*-graded`, `Skill-normalisation-ESCO-graded`), and the [Skill-Extraction-benchmark repo](https://github.com/jensjorisdecorte/Skill-Extraction-benchmark) (no licence file declared in the repo; use the HF CC BY 4.0 copies). ESCO labels are tagged to ESCO v1.1.0, so pin the same ESCO version in `normalize/esco_loader.py` or report the mismatch. Adopt as the supervised linking gold that the evaluation plan currently lacks (the README says linking is only a coverage signal).
4. **ranx** (MIT, 701 stars, last push 2025-08-07, 19 open issues). Reference: nice `compare()` and fusion, but three reproduced/confirmed correctness hazards (dict-ordering, int-truncation of grades, Fisher non-determinism), numba JIT (slow import, serverless issues), and it pulls pandas/seaborn/fastparquet/ir-datasets. If used, only via `Qrels`/`Run` objects, never raw dicts. TalentCLEF's official scripts ([2025](https://github.com/TalentCLEF/talentclef25_evaluation_script), [2026](https://github.com/TalentCLEF/talentclef26_evaluation_script)) are thin ranx wrappers and carry no licence file; reference only.
5. **BEIR** (Apache-2.0, PyPI 2.2.0 2025-06, uses pytrec_eval-terrier + sentence-transformers) and **MTEB** (Apache-2.0, very active, but py>=3.11 and a huge extras tree): Skip. They benchmark retrieval models on public corpora, which is not our question; ESCO-specific tasks in WorkRB are closer.
6. **ESCOXLM-R** ([mainlp/escoxlmr](https://github.com/mainlp/escoxlmr), Apache-2.0, last push 2023-08) and JobBERT models: Reference as candidate encoders/baselines for retrieval, not eval tools.

### (c) Confidence calibration (ECE etc.)

1. **In-house ECE/MCE/Brier + sklearn `calibration_curve`/`brier_score_loss` as the cross-check** (BSD-3, already a dependency). Adopt. ECE depends on binning choices (equal-width vs equal-mass, n_bins, top-label vs per-class), so publish those choices; report ECE with a bootstrap CI.
2. **[torchmetrics](https://github.com/Lightning-AI/torchmetrics) `CalibrationError`** (Apache-2.0, pushed 2026-10-07). Optional second oracle; its core deps are only numpy, packaging, torch, lightning-utilities, and torch is already pinned here. Not tested by me (torch is not in my throwaway venv).
3. **[netcal](https://github.com/EFS-OpenSource/calibration-framework)** (Apache-2.0, 381 stars, PyPI 1.4.0 2026-04): rich (ECE, ACE, MCE, MMCE, many calibrators) but its declared hard dependencies include `torch>=2.3`, `pyro-ppl`, `gpytorch`, `tensorboard` and `numpy>=2.0` (conflicts with the `numpy==1.26.3` pin). I started a trial install and it began downloading torch plus the CUDA wheels, so I aborted it. Skip as a dependency; Reference for metric definitions (debiased/adaptive ECE).
4. HELM uses `uncertainty-calibration` (verified_calibration, last release 2022-11): Reference for debiased ECE only.

### (d) Profile/context compaction

No off-the-shelf evaluator matches "evidence-preserving structured packet". Adopt in-house retention/groundedness/determinism metrics (already in `evals/context_evals.py`). External: **[LLMLingua](https://github.com/microsoft/LLMLingua)** (MIT, 6.8k stars, pushed 2026-09-10): Reference as a compression baseline to compare token budget vs skill recall. `bert-score` (MIT, torch+transformers, PyPI 2023-02) and `rouge-score` (2022): Skip; they reward fluency overlap, not retained skill ids. Token counting via `tiktoken` (an Inspect dependency) is enough.

### (e) KG reasoning and explanation faithfulness/groundedness

1. **Deterministic graph-grounding checks, in-house** (every cited node/edge/path exists in the NetworkX graph and supports the stated gap). This is the faithfulness definition used by path-reasoning recommenders such as PEARLM's path-faithfulness rate (share of generated k-hop paths with no triplet missing from the KG; see [arXiv 2310.16452](https://arxiv.org/pdf/2310.16452)). Adopt; no library exists (my search found no maintained open-source explanation-quality library, and no standard LIR/SEP/ETD package).
2. **Offline entailment scorers for free-text explanations** (optional layer): [Vectara HHEM-2.1-open](https://huggingface.co/vectara/hallucination_evaluation_model) (Apache-2.0, HF, updated 2026-10-06) and [MiniCheck](https://github.com/Liyan06/MiniCheck) (Apache-2.0 code, MIT HF checkpoints, last push 2025-08). Reference/optional; I did not run either, so CPU latency is unverified.
3. **[RAGAS](https://github.com/vibrantlabsai/ragas)** (repo moved from explodinggradients; Apache-2.0; 16k stars; last commit 2026-02-24, release 0.4.3 2026-01-13): metrics are RAG-shaped (faithfulness, context precision/recall) and need an LLM judge; hard deps include langchain, langchain-community, langchain_openai, openai, datasets. Reference only.
4. **[DeepEval](https://github.com/confident-ai/deepeval)** (Apache-2.0, 19k stars, release 4.3.1 2026-10-16): Reference/Skip. Hard dependencies include `openai`, `posthog`, `grpcio`, `opentelemetry-*` and several pytest plugins; telemetry is on by default and is disabled with `DEEPEVAL_TELEMETRY_OPT_OUT=1` per the [FAQ](https://deepeval.com/docs/faq) (I could not confirm from source whether the backend is Sentry or PostHog; the `posthog` dependency is verified). Judge-model configuration was not verified. Violates "offline, no keys" by default.
5. **TruLens** (MIT, release 2.16.0; hard dependency on `trulens-dashboard[full]`), **openevals** (MIT, hard deps langchain/langchain-openai/langsmith), **UpTrain** (Apache-2.0, last push 2024-08, last release 2024-05): Skip.
6. If an LLM judge is wanted later, use Inspect's `model_graded_qa`/`model_graded_fact` (see (h)) with a local judge so the judge prompt and verdicts are logged per sample.

### (f) Job-user matching and skill-gap detection

1. **ir_measures** for nDCG/P/R/MAP against graded labels (same as g). Adopt as oracle.
2. **WorkRB** job-to-skill and candidate-matching tasks (`ESCOJob2SkillRanking`, `SearchQueryCandidateRanking`, `ProjectCandidateRanking`) as an external reality check that the matcher is not worse than BM25/TF-IDF on a public ESCO-based task. Optional.
3. scipy for rank correlations on gap-priority orderings (already cross-checked in `gap_evals.py`).
4. [TechWolf RecSys-HR challenge repo](https://github.com/techwolf-ai/workrb-recsys-hr-challenge-2026): not opened (no licence verified); Reference at most.

### (g) Course recommendation (nDCG@10, P@5, Recall@10, MAP, prerequisite violation, redundancy, coverage)

1. **ir_measures + pytrec_eval-terrier** as the oracle for all accuracy metrics (verified, section 3). Adopt.
2. **Beyond-accuracy metrics stay in-house**; definitions to copy from **[RecBole](https://github.com/RUCAIBox/RecBole)** (MIT; `evaluator/metrics.py` contains ItemCoverage, AveragePopularity, ShannonEntropy, GiniIndex, TailPercentage; last push 2025-02-24; hard deps torch, ray, tensorboard, colorlog==4.7.2): Reference only, do not install, its evaluator is welded to its own data pipeline. **[LensKit](https://github.com/lenskit/lkpy)** (MIT per PyPI and `LICENSE.md`; GitHub shows NOASSERTION; release 2026.4.0; ranking metrics incl. NDCG, MAP, RBP, Gini, entropy, intra-list similarity): good definitions, but requires Python >= 3.12.5, `numpy>=2.0`, `torch~=2.4`, so Reference only here.
3. **[Elliot](https://github.com/sisinflab/elliot)** (Apache-2.0, pushed 2026-10-08): metrics span accuracy, diversity, novelty, coverage, bias, fairness and Wilcoxon/paired-t significance, but `requirements.txt` pins `tensorflow==2.3.2`, `numpy==1.18.0`, README targets Python 3.6-3.8. Not installable on Python 3.12/3.13 as-is. Reference for metric definitions only. **[RecList](https://github.com/RecList/reclist)** (MIT; last push 2023-08): stale, Skip.
4. **Prerequisite-violation and redundancy metrics: no library found.** A web search for prerequisite-aware course-recommendation benchmarks returned only papers reporting standard nDCG/MRR/MAP and datasets without graded labels; I found no maintained evaluation package. Keep these custom, and test them with property tests (inject a known violation, assert the rate moves) as `eval/course_recommendation_eval.py` already invites.

### (h) End-to-end LLM-agent behaviour

1. **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** (MIT, 3.1k stars, pushed 2026-10-21, 444 open issues, release 0.3.283; py>=3.11). Adopt. Local providers documented: `hf/`, `vllm/`, `ollama/`, `llama-cpp-python/`, `sglang/` ([providers](https://inspect.aisi.org.uk/providers.html)); built-in scorers report accuracy/stderr, `exact()`/`f1()` mean/stderr, and `model_graded_qa()`/`model_graded_fact()`; custom scorers/metrics are first-class ([scorers](https://inspect.aisi.org.uk/scorers.html)). Core dependencies are pure Python but numerous (boto3, fastapi, uvicorn, textual, s3fs, ...), no torch. Not verified: bootstrap_stderr/clustered stderr semantics (docs page deferred to "Scoring Metrics" which I did not read), sandbox requirements (Docker) for tool-using agents, and how cleanly a custom pooled-count micro-F1 metric fits its reducer API. Pin the version; the 0.3.x API is still moving.
2. **[inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)** (MIT, pushed 2026-10-22): Reference for how to package datasets/scorers as Inspect tasks.
3. **[tau-bench](https://github.com/sierra-research/tau-bench)** (MIT, 1.5k stars, last push 2026-03) and **[agentevals](https://github.com/langchain-ai/agentevals)** (MIT, pushed 2026-07; trajectory matching, LangChain stack): Reference for tool-call/trajectory metric designs (pass^k, trajectory match). Not our domain tasks.
4. **lm-evaluation-harness** (MIT, 14.3k stars, pushed 2026-09-14, release 0.4.13): best-in-class for static benchmarks on local HF/vLLM models with YAML task configs and bootstrap stderr (`bootstrap_stderr` in `lm_eval/api/metrics.py`), but it is built for single-turn LL/generation tasks. Skip for (h); possible for re-scoring extraction prompts. **HELM** ([Apache-2.0](https://github.com/stanford-crfm/helm), release 0.5.16): scenario framework, requires torch/torchvision/spacy~=3.5/datasets~=3.1; Skip. **OpenAI evals** (MIT in `LICENSE.md`, GitHub reports NOASSERTION; last push 2026-04-14): README directs users to the hosted dashboard, requires an OpenAI key and Git-LFS registry; Skip. **promptfoo** (MIT, 26k stars, Node/TypeScript; telemetry on by default, `PROMPTFOO_DISABLE_TELEMETRY=1`): excellent prompt-regression CLI but a second language runtime and not metric-level; Skip.

### Error analysis

* **[CheckList](https://github.com/marcotcr/checklist)** (MIT; 2.1k stars; last push 2024-01; PyPI 0.0.11 from 2021-05): Reference. The idea (minimum-functionality, invariance, directional tests) is exactly what `evals/match_evals.py` property/invariance checks already do; the library itself is unmaintained.
* **[Errudite](https://github.com/uwdata/errudite)**: **GPL-2.0**, last push 2021-07. Skip.
* Keep `eval/wrapper_error_analysis.py` and `eval/linking_error_analysis.py` as the error-analysis home.

## 5. Licence and compatibility table

Install weight is as measured in my venvs where marked, otherwise from PyPI `requires_dist` (non-extra) **[docs]**. "Pins" = interaction with `requirements.txt` (numpy==1.26.3, pandas==2.2.1, torch==2.6.0, nervaluate==0.2.0, Python 3.12/3.13).

| Package | Licence | Python | Hard deps / weight | Key/offline | Pin interaction | Verdict |
|---|---|---|---|---|---|---|
| ir_measures 0.4.3 + pytrec-eval-terrier 0.5.11 | Apache-2.0 / MIT | >=3.9 | numpy+scipy via pytrec_eval; 222 MB venv incl. nervaluate **[tested]** | offline, no key | I resolved numpy 2.5.4; behaviour under numpy 1.26.3 not tested | **Adopt** |
| nervaluate 1.2.1 | MIT | >=3.11 | none | offline | requirements pins 0.2.0 with an older API: bump | **Adopt** |
| scipy 1.18.1 / statsmodels 0.15.0 | BSD | >=3.12 / >=3.10 | numpy | offline | I resolved scipy 1.18.1 with numpy 2.5.4; the minimum scipy version providing `bootstrap(paired=True)` was not verified | **Adopt (oracle)** |
| scikit-learn | BSD-3 | any | already in requirements | offline | none | **Adopt (ECE cross-check)** |
| Inspect AI 0.3.283 | MIT | >=3.11 | ~40 pure-Python deps, no torch (docs) | offline with local providers | pydantic>=2.13 | **Adopt (harness)** |
| WorkRB 0.6.0 | Apache-2.0 | >=3.10 | torch>=2.6, sentence-transformers>=5.1, datasets, pydantic>=2.11, rank-bm25, rapidfuzz | offline after dataset download | torch 2.6 pin OK; pandas/numpy ranges looser than pins | **Adopt (separate env)** |
| TechWolf HF datasets | CC BY 4.0 (MIT for SkillMatch-1K, JobBERT-eval; Apache-2.0 for vacancy-job-to-skill) | n/a | n/a | download once | n/a | **Adopt (data)** with attribution |
| ranx 0.3.21 | MIT | >=3.8 | numba, pandas, seaborn, fastparquet, ir-datasets, orjson, lz4, cbor2; 800 MB venv (with statsmodels etc.) **[tested]** | offline | numba caps numpy version; not tested vs 1.26.3 | **Reference** |
| seqeval 1.2.2 | MIT | any | scikit-learn | offline | none | **Reference** (cheap BIO cross-check) |
| torchmetrics 1.9.0 | Apache-2.0 | >=3.10 | torch, numpy, packaging, lightning-utilities | offline | none beyond torch | Reference/optional |
| netcal 1.4.0 | Apache-2.0 | >=3.10 | torch, pyro-ppl, gpytorch, tensorboard, matplotlib, numpy>=2 | offline | **conflicts with numpy==1.26.3**; pulls CUDA wheels | **Skip** |
| RecBole 1.2.1 | MIT | n/a | torch, ray, tensorboard, colorlog==4.7.2 | offline | likely conflicts (ray, numpy) | Reference |
| LensKit 2026.4.0 | MIT | >=3.12.5 | torch~=2.4, numpy>=2, pyarrow, pydantic~=2.12 | offline | **conflicts with numpy==1.26.3** | Reference |
| Elliot | Apache-2.0 | 3.6-3.8 per README | tensorflow==2.3.2, numpy==1.18.0 | offline | uninstallable here | Reference |
| RecList | MIT | n/a | n/a | n/a | stale (2023) | Skip |
| deep-significance 1.2.8 | **GPL-3.0** | >=3.7 | numpy, scipy, pandas, joblib, dill | offline | licence incompatible with a permissive repo | Reference only |
| DeepEval 4.3.1 | Apache-2.0 | >=3.9 | openai, posthog, grpcio, opentelemetry, pytest plugins | telemetry on by default; judge defaults unverified | none | Reference/Skip |
| RAGAS 0.4.3 | Apache-2.0 | >=3.9 | langchain stack, openai, datasets | needs an LLM judge | none | Reference |
| TruLens 2.16.0 / openevals 0.2.2 / UpTrain 0.7.1 | MIT / MIT / Apache-2.0 | n/a | dashboard stack / langchain / stale since 2024 | n/a | n/a | Skip |
| lm-eval 0.4.13 | MIT | >=3.10 | datasets, evaluate, sacrebleu, rouge-score, scikit-learn | offline with local models | datasets pin interplay unknown | Skip (not for this task shape) |
| HELM 0.5.16 | Apache-2.0 | >=3.10 | torch, torchvision, spacy~=3.5, datasets~=3.1, nltk | offline | heavy | Skip |
| openai/evals | MIT (`LICENSE.md`) | >=3.9 | OpenAI SDK, Git-LFS data | needs key | n/a | Skip |
| promptfoo | MIT | Node | Node/TypeScript | telemetry default on | n/a | Skip |
| BEIR 2.2.0 / MTEB 2.26 | Apache-2.0 | >=3.9 / >=3.11 | sentence-transformers+pytrec_eval / ~200 requires entries incl. extras | offline after download | heavy | Skip |
| CheckList 0.0.11 | MIT | n/a | stale (2021 PyPI) | n/a | n/a | Reference |
| Errudite | **GPL-2.0** | n/a | stale (2021) | n/a | licence | Skip |
| arch (bootstrap library) | NOASSERTION on GitHub (I did not read its licence file) | n/a | n/a | n/a | n/a | Skip (scipy suffices) |
| pingouin 0.7.0 | **GPL-3.0** | >=3.11 | n/a | n/a | licence | Skip |

## 6. Concrete wiring plan

Principle: **our code stays the source of truth; external packages are oracles in a test/CI "controls" suite, plus Inspect as the runner for LLM-in-the-loop evals.** Oracles go in an optional `requirements-eval-oracles.txt` (or a separate venv) so the production environment is untouched.

| Metric | Computed by (source of truth) | Cross-check oracle | Tolerance / policy |
|---|---|---|---|
| Exact span P/R/F1 (strict) | `eval/extraction_eval.py` | nervaluate `strict` | equal to 1e-9 on counts |
| Type-agnostic span F1 | add as new metric | nervaluate `exact` | new thesis column |
| Partial span F1 | `eval/extraction_eval.py` | none equivalent; document that nervaluate `partial` gives half credit and differs | do not compare |
| BIO token/chunk F1 for BERT/GLiNER baselines | existing | seqeval | 1e-9 |
| Per-label HARD/SOFT F1 | existing | nervaluate `evaluator.results_by_tag` / `summary_report(mode="entities")` | 1e-9 |
| Bootstrap CI, paired bootstrap | `eval/significance.py` | `scipy.stats.bootstrap(paired=True, method="percentile")` | CI endpoints within about 2e-3 at 10k resamples (observed ~2e-4) |
| Sign-flip permutation p | `eval/significance.py` | `scipy.stats.permutation_test(permutation_type="samples")` | same significance decision, |dp| < 0.01 |
| Holm | `ranking_metrics.holm_adjust` | `statsmodels.multipletests(method="holm")` | exact |
| nDCG@k (exp gain), P@k, R@k | `eval/ranking_metrics.py` | ir_measures with gains pre-transformed to 2^g-1 (default linear `nDCG`; avoid `dcg='exp-log2'`, which routes to Perl gdeval) | 1e-9 per scenario |
| AP@k / MAP | `eval/ranking_metrics.py` | ir_measures `AP(rel=3)` on transformed gains | compare after rescaling by min(R,k)/R, and **also report the trec_eval-convention MAP** in the thesis table |
| Scenarios without relevant items | drop (current) | trec-style counts as 0 | report n dropped and the alternate mean |
| ECE / MCE / Brier | `evals/calibration_evals.py` | sklearn `calibration_curve` + `brier_score_loss`; optionally torchmetrics `CalibrationError` | 1e-9 for same binning |
| Linking accuracy / ranking | `evals/linking_evals.py` | ir_measures on the same qrels; optional WorkRB run on TechWolf graded sets for an external number | 1e-9 vs ir_measures |
| Job match nDCG@10 | `evals/match_evals.py` | ir_measures | 1e-9 |
| Beyond-accuracy (coverage, Gini, entropy, redundancy, prerequisite violation) | in-house | hand-computed tiny fixtures; formulas checked against RecBole/LensKit definitions by reading, not importing | property tests |
| Explanation groundedness | in-house graph check (`evals/graph_evals.py`) | optional HHEM/MiniCheck entailment on a 50-100 explanation audit subset | report agreement as a secondary column |
| Agent e2e | Inspect task + custom scorer wrapping the existing e2e checks | n/a | log dir pinned, seed and model revision recorded |

Minimal oracle snippet (the exact pattern that passed; use `Qrel`/`ScoredDoc` rather than ranx dicts):

```python
import ir_measures
from ir_measures import nDCG, P, R, AP, Qrel, ScoredDoc

qrels = [Qrel(q, d, 2**g - 1) for q, dd in grades.items() for d, g in dd.items()]   # exp gain pre-applied
run   = [ScoredDoc(q, d, s)   for q, dd in scores.items() for d, s in dd.items()]
per_query = {}
for r in ir_measures.iter_calc([nDCG@10, P(rel=3)@5, R(rel=3)@10, AP(rel=3)@10], qrels, run):
    per_query.setdefault(str(r.measure), {})[r.query_id] = r.value   # rel=3 <=> grade >= 2
```

Suggested order of work: (1) add `tests/evals/test_oracles.py` that skips when oracles are not importable and asserts the table above; (2) bump `nervaluate` in `requirements.txt` from 0.2.0 to 1.2.x after checking Python >= 3.11 (the README says 3.12/3.13, so fine) and the 1.x API; (3) load the TechWolf graded ESCO sets into `evals/linking_evals.py` as the supervised linking gold; (4) only then wrap LLM-in-the-loop evals as Inspect tasks.

## 7. Statistical-rigor checklist (what to implement or report, with sources)

* Resample the unit that is independent (document or scenario), not tokens; recompute pooled-count F1 per resample (already done in `significance.py`).
* Paired tests for system deltas; Holm across the family (variants x metrics), as already done.
* Report the policy for undefined cases (scenarios without relevant courses).
* For multi-seed training comparisons SkillSpan used five seeds and almost stochastic dominance; reproduce the seed count, and if ASO is wanted, implement it rather than importing GPL code.
* Pin versions and record them in each `EvalResult.details`; Inspect logs and `snakemake` (already in the repo) cover provenance.

## 8. Risks

* **Convention drift**: AP@k denominator and no-relevant-scenario handling silently change numbers by 0.01-0.02 absolute in my synthetic test (nDCG mean 0.280 vs 0.266). State them in the thesis table notes.
* **Oracle bugs**: ranx has open correctness issues (section 3); ir_measures has a Perl-dependent path for exp-log2 nDCG. Always cross-check two oracles or one oracle plus a hand-computed fixture.
* **Dependency conflicts**: numpy>=2 libraries (netcal, LensKit, current scipy/ir_measures resolution) against `numpy==1.26.3`; solve with a separate oracles venv.
* **Young/small repos** (WorkRB: 44 stars, v0.x): treat as an external check, not infrastructure.
* **Concurrent in-house changes**: `src/skillllm_v2/evals/` was changing during this research; check for duplicated functionality before wiring.

## 9. What I could not verify

* **Inspect AI details**: did not install or run it. Not confirmed: `bootstrap_stderr`/clustering behaviour, Docker sandbox needs, fit of pooled micro-F1 into its metric API, install size/time. Local-provider list and scorer overview are from its docs pages only.
* **WorkRB**: did not install (torch + sentence-transformers) and did not run any task; metric details read from `ranking.py` source and the README only. Dataset licences were checked on HF for the TechWolf copies only; ESCO's own reuse licence was not re-checked.
* **ir_measures/pytrec_eval-terrier under numpy 1.26.3**: tested only with numpy 2.5.4, py3.13. The `gdeval` Perl failure may be specific to my environment's Perl or to my qrels format (grade-0 rows), and I did not investigate further.
* **torchmetrics `CalibrationError`**: not run. **netcal**: not completed (aborted when it began pulling torch/CUDA); its dependency list is from PyPI metadata.
* **DeepEval**: telemetry backend (Sentry vs PostHog) conflicts between sources; default judge model and local-judge configuration not verified. **RAGAS, TruLens, openevals, UpTrain, promptfoo, OpenAI evals, HELM, lm-eval**: assessed from repo metadata, PyPI dependency lists and a few doc pages, not installed or run. For OpenAI evals the licence is MIT per `LICENSE.md` though GitHub reports NOASSERTION. LensKit likewise (PyPI MIT, `LICENSE.md` header only read to its first lines).
* **SkillSpan**: confirmed the paper's protocol (span-level P/R/F1, five seeds, almost stochastic dominance) from the PDF text and that the repo's `metrics/` folder is empty. I did not locate a partial-span or token-F1 definition in the official materials, so our `partial_span_f1` and `token_f1` cannot be called "SkillSpan-official".
* **HHEM-2.1-open / MiniCheck / LLMLingua**: licences and update dates from HF/GitHub metadata only; not run, CPU speed unknown.
* **Prerequisite-violation/redundancy benchmarks and KG-explanation metric libraries**: absence of a library is based on a small number of web searches, not an exhaustive survey. LIR/SEP/ETD-style metrics did not surface at all.
* **`davidsbatista/NER-Evaluation` vs `MantisAI/nervaluate`**: PyPI's maintained project is under MantisAI; I treated the davidsbatista repo as the legacy origin.
* **Elliot, RecBole**: I did not attempt installs; unusability for Elliot is inferred from its pinned `requirements.txt` and README, and RecBole's incompatibility is a guess from its dependency list.
* **Not opened**: [jensjorisdecorte/JobBERT-evaluation-dataset](https://github.com/jensjorisdecorte/JobBERT-evaluation-dataset) (no licence; job-title normalisation data). `microsoft/SkillLens` surfaced in a search for "skill extraction" but its README shows it studies LLM *agent* skills, not job skills, so it is irrelevant here. Also not read beyond abstracts: the Skill-XL / MELO / ESCO-EQF linking papers beyond their abstracts ([Skill-XL](https://arxiv.org/pdf/2505.24640), [MELO](https://arxiv.org/pdf/2410.08319), [arXiv 2512.03195](https://arxiv.org/abs/2512.03195v1)).

## 10. Appendix: repo metadata snapshot (GitHub API, 2026-10-22)

| Candidate | Licence | Stars | Last push | Open issues |
|---|---|---|---|---|
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | MIT | 14.3k | 2026-09-14 | 1228 |
| [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | MIT | 3.1k | 2026-10-21 | 444 |
| [UKGovernmentBEIS/inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) | MIT | 708 | 2026-10-22 | 259 |
| [stanford-crfm/helm](https://github.com/stanford-crfm/helm) | Apache-2.0 | 3.0k | 2026-09-01 | 117 |
| [openai/evals](https://github.com/openai/evals) | MIT (GitHub: NOASSERTION) | 19.6k | 2026-04-14 | 348 |
| [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo) | MIT | 26.3k | 2026-10-22 | 834 |
| [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | Apache-2.0 | 19.0k | 2026-10-19 | 766 |
| [vibrantlabsai/ragas](https://github.com/vibrantlabsai/ragas) | Apache-2.0 | 16.1k | 2026-02-24 | 635 |
| [uptrain-ai/uptrain](https://github.com/uptrain-ai/uptrain) | Apache-2.0 | 2.4k | 2024-08-18 | 66 |
| [truera/trulens](https://github.com/truera/trulens) | MIT | 3.6k | 2026-10-19 | 116 |
| [langchain-ai/openevals](https://github.com/langchain-ai/openevals) | MIT | 1.2k | 2026-10-22 | 13 |
| [langchain-ai/agentevals](https://github.com/langchain-ai/agentevals) | MIT | 751 | 2026-07-14 | 30 |
| [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) | MIT | 1.5k | 2026-03-18 | 55 |
| [AmenRa/ranx](https://github.com/AmenRa/ranx) | MIT | 701 | 2025-08-07 | 19 |
| [cvangysel/pytrec_eval](https://github.com/cvangysel/pytrec_eval) (original) | MIT | 354 | 2023-10-10 | 35 |
| [terrierteam/ir_measures](https://github.com/terrierteam/ir_measures) | Apache-2.0 | 103 | 2026-10-20 | 20 |
| [beir-cellar/beir](https://github.com/beir-cellar/beir) | Apache-2.0 | 2.3k | 2025-10-16 | 88 |
| [embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb) | Apache-2.0 | 3.5k | 2026-10-22 | 332 |
| [MantisAI/nervaluate](https://github.com/MantisAI/nervaluate) | MIT | 225 | 2026-03-12 | 11 |
| [chakki-works/seqeval](https://github.com/chakki-works/seqeval) | MIT | 1.2k | 2024-08-28 | 35 |
| [EFS-OpenSource/calibration-framework](https://github.com/EFS-OpenSource/calibration-framework) | Apache-2.0 | 381 | 2026-04-16 | 4 |
| [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics) | Apache-2.0 | 2.5k | 2026-10-07 | 269 |
| [RUCAIBox/RecBole](https://github.com/RUCAIBox/RecBole) | MIT | 4.6k | 2025-02-24 | 359 |
| [sisinflab/elliot](https://github.com/sisinflab/elliot) | Apache-2.0 | 298 | 2026-10-08 | 16 |
| [lenskit/lkpy](https://github.com/lenskit/lkpy) | MIT (PyPI) / NOASSERTION (GitHub) | 312 | 2026-10-21 | 135 |
| [RecList/reclist](https://github.com/RecList/reclist) | MIT | 477 | 2023-08-09 | 4 |
| [marcotcr/checklist](https://github.com/marcotcr/checklist) | MIT | 2.1k | 2024-01-09 | 11 |
| [uwdata/errudite](https://github.com/uwdata/errudite) | GPL-2.0 | 109 | 2021-07-22 | 5 |
| [Kaleidophon/deep-significance](https://github.com/Kaleidophon/deep-significance) | GPL-3.0 | 342 | 2026-09-21 | 0 |
| [techwolf-ai/workrb](https://github.com/techwolf-ai/workrb) | Apache-2.0 | 44 | 2026-07-22 | 3 |
| [kris927b/SkillSpan](https://github.com/kris927b/SkillSpan) | MIT | 69 | 2025-02-13 | n/a |
| [jensjorisdecorte/Skill-Extraction-benchmark](https://github.com/jensjorisdecorte/Skill-Extraction-benchmark) | none declared | 17 | 2024-07-18 | n/a |
| [mainlp/escoxlmr](https://github.com/mainlp/escoxlmr) | Apache-2.0 | 10 | 2023-08-18 | n/a |
| [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) | MIT | 6.8k | 2026-09-10 | 125 |
| [Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck) | Apache-2.0 (HF checkpoints: MIT) | 229 | 2025-08-27 | 5 |
| [huggingface/lighteval](https://github.com/huggingface/lighteval), [huggingface/evaluate](https://github.com/huggingface/evaluate) | MIT / Apache-2.0 | 2.6k / 2.5k | 2026-10-06 | 455 / 325 |

Not evaluated in depth (seen in results): lighteval and HF `evaluate` (general LLM/NLP metric wrappers; nothing here that scipy/ir_measures/nervaluate do not already cover), Evidently, `microsoft/recommenders`.
