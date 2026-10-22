# SkillLLM_v2 wiring audit

Scope: every module under `src/skillllm_v2/{extract,verify,normalize,context,graph,match,policy,retrieval,baselines,eval,pipeline}`
plus the root files (`wrapper.py`, `source_aware_policy.py`, `schema.py`, `schemas.py`, `config.py`).
Course ingestion (`course/`) and the course ranker (`recommend/`, `pipeline/course_recommendation_pipeline.py`) are
out of scope and untouched; they are only listed where other modules depend on them.

Method: an AST pass over `src/` resolved `import` / `from ... import` statements (including names re-exported through
package `__init__.py` files) into a module-level import graph. Reachability was then computed from each entry point,
and every "suspect" module was re-verified by `grep` for actual instantiation / call sites (an import or a re-export in
an `__init__.py` is not a call). Tests were excluded from "reachable" and are reported separately.

Entry points considered: `wrapper.py` (`SkillLLMv2Pipeline.extract`), `eval/run_safe_wrapper.py`,
`pipeline/smoke_runner.py`, `pipeline/pipeline_runner.py` (-> `stage_registry` -> `smoke_runner`), and `scripts/*.sh`
(`python -m ...` modules plus inline `python - <<PY` heredocs).

Definitions used in the table:

* **Real inference path (before)**: executed when a caller does `SkillLLMv2Pipeline().extract(text)`
  (`wrapper.py`) -- the only inference API that existed. It covered extraction -> linking only.
  Nothing for profile/context, graph, matching, gaps, explanation or courses existed as an API; those modules
  were only reachable from `smoke_runner` / demo scripts with hand-built toy inputs.
* **Smoke/eval only**: reachable from `smoke_runner`, `run_safe_wrapper`, `ablation_runner`, `research_refresh`,
  or demo shell heredocs, but not from `wrapper.py`.
* **Unreachable**: no importer outside tests / re-export shims, or only re-exported from an `__init__.py` and never called.

## BEFORE (state at the start of this task)

| Module | Public API | Reachable from an entry point? | Real inference path, or only smoke/eval? | Notes |
|---|---|---|---|---|
| `wrapper.py` | `SkillLLMv2Pipeline(mistral, gliner, dictionary, linker).extract(text, include_gliner, include_dictionary, normalize)` | Yes (library; `pipeline/__init__`, `smoke_runner.eval_full_cag_graph` / `end_to_end_demo_smoke`, `eval/ablation_runner`, `scripts/run_skillllm_v2_smoke.sh`, `run_skillllm_v2_end_to_end_demo.sh`; also used by `course/extract_course_skills.py` and `pipeline/course_recommendation_pipeline.py`) | Real (extraction only) | Only stages 1-6 (extract, merge, refine, verify, calibrate, link). No gate, no trace, no downstream stages. Public attributes `.mistral/.gliner/.dictionary/.merger/.linker` are read by `ablation_runner`. |
| `config.py` | `load_yaml`, `repo_root_from` | Yes (9 importers incl. `candidate_merger`, `confidence_calibrator`, `boundary_selector`, `non_regression_gate`) | Real (tuned-config loading) | Not used by `wrapper.py` itself: the wrapper ignores every YAML config. |
| `schema.py` | `SkillCandidate`, `validate_skill`, `normalize_text`, `find_offsets`, `bio_spans`, `ensure_skills` | Yes (29 importers) | Real | Core candidate contract. |
| `schemas.py` | `SkillCandidate`, `MergedSkillCandidate`, `VerifiedSkill`, `TaxonomyLink`, `SystemBenchmarkResult`, `JobMatchResult`, `CourseRecommendation`, `validate_*` | Yes: `pipeline/smoke_runner.schema_smoke`, `baselines/bert_baseline`; `tests/skillllm_v2/test_schemas.py` | Smoke/eval only | **Duplicates `schema.SkillCandidate`** (identical fields, different `SkillSource` literal ordering). `MergedSkillCandidate`, `VerifiedSkill`, `TaxonomyLink`, `JobMatchResult`, `CourseRecommendation` are instantiated nowhere. |
| `source_aware_policy.py` (root) | `ANCHOR_SOURCES`, `candidate_sources`, `is_anchor_candidate`, `source_key` | Yes: imported by `extract/candidate_merger`, `verify/span_verifier`, `verify/confidence_calibrator` | Real | This is the file the real code imported. |
| `policy/source_aware_policy.py`, `policy/__init__.py` | same four names | **No** (0 importers anywhere, src/scripts/tests) | Unreachable | Both files were pure re-exports of the root file. The duplication was already one-way, but the "policy" package that looks like the canonical home was dead code and the implementation lived in a root-level file. |
| `extract/mistral_extractor.py` | `MistralExtractor.extract`, `.load`, `parse_skill_json` | Yes (wrapper, smoke_runner, `eval/generate_split_predictions`, `train/loop_trainer`) | Real | When weights are missing and `allow_fallback=True` (the default) `extract()` silently returns alias-matched candidates with `source="mistral"`: they look like model output and are treated as the protected anchor. Not flagged in output. |
| `extract/gliner_extractor.py` | `GLiNERExtractor.extract`, `.load`, `resolve_gliner_device` | Yes (wrapper, run_safe_wrapper, smoke_runner) | Real | `allow_fallback=True` default emits four hard-coded `SOFT_SKILL` candidates (`source="gliner"`, conf 0.48) when the `gliner` package is missing (it is missing in this env) -- again unflagged. Device defaults to `cuda:0` when torch sees a GPU. |
| `extract/dictionary_extractor.py` | `DictionaryExtractor.extract` | Yes (wrapper, run_safe_wrapper, smoke_runner, course pipeline) | Real | Falls back to 22 built-in aliases when `alias_paths` is empty/missing (configured path `data/taxonomies/custom_skill_aliases.jsonl` does not exist). |
| `extract/candidate_merger.py` | `CandidateMerger.merge`, `load_merger_config` | Yes | Real | Loads tuned config by CWD-relative path (silently ignored when CWD is not the repo root). Uses root `source_aware_policy`. |
| `verify/boundary_refiner.py` | `BoundaryRefiner.refine`, `.generate_variants` | Yes | Real | Calls `BoundarySelector` (default instance) internally. |
| `verify/boundary_selector.py` | `BoundarySelector.select`, `load_selector_config` | Yes (via refiner; `train/tune_boundary_refiner`) | Real | Verified: `refiner.refine` -> `BoundarySelector()` at `boundary_refiner.py:59`. |
| `verify/span_verifier.py` | `SpanVerifier.verify`, `.filter` | Yes (wrapper `extract`, run_safe_wrapper, ablation_runner, smoke_runner) | **Real** | Suspect cleared: instantiated in `wrapper.__init__` and `.filter()` called in `extract`. |
| `verify/confidence_calibrator.py` | `ConfidenceCalibrator.calibrate`, `.accept`, `load_calibrator_config` | Yes | **Real** | Suspect cleared: `calibrate()` called in `wrapper.extract`. Thresholds in `configs/skillllm_v2_stage2.yaml:calibration` are NOT read by the wrapper (it uses tuned artifact / defaults). |
| `normalize/taxonomy_linker.py` | `TaxonomyLinker.link`, `.link_candidates` | Yes | Real | `link_candidates` overwrites `canonical_skill_id/name` for every match including bm25/dense (confidence as low as 0.55); never removes a skill. |
| `normalize/skill_alias_builder.py` | `load_alias_records`, `build_alias_index` (read helpers); `build_alias_table`, `main` (offline builder) | Read helpers: yes (merger, refiner, selector, linker). Builder: only `python -m skillllm_v2.normalize.skill_alias_builder` (no script or stage invokes it) | Read helpers real; builder is offline | Builder needs the SkillSpan dataset. |
| `normalize/alias_resolver.py` | `AliasResolver.resolve` | Re-exported by `normalize/__init__` only; **never instantiated** (grep `AliasResolver(`: 0 hits) | Unreachable | Dead. |
| `normalize/esco_loader.py`, `onet_loader.py`, `lightcast_loader.py` | `load_esco`, `load_onet`, `load_lightcast` -> `TaxonomyLinker` | **No importers, no callers, no tests** | Unreachable | No config key points at them either. |
| `context/profile_summarizer.py` | `summarize_profile` | `research_refresh.write_context_outputs` (artifact writer), tests | Smoke/eval only | Not on any inference path; the demo reads canned artifacts. |
| `context/profile_schema.py` | `Evidence`, `ProfileSkillFact`, `ProfileSummary` | via summarizer | Smoke/eval only | |
| `context/skill_memory_builder.py` | `build_skill_memory` | via `context_compactor` | Smoke/eval only | |
| `context/context_compactor.py` | `compact_context` | via `cached_context_builder`; test | Smoke/eval only | |
| `context/cached_context_builder.py` | `build_cached_context_packet` | `research_refresh`; test | Smoke/eval only | Content-hashed CAG packet. |
| `context/context_retriever.py` | `ContextRetriever.retrieve_skills` | **No importers, no callers, no tests** (not even re-exported by `context/__init__`) | Unreachable | Dead. |
| `retrieval/contextual_retrieval.py` | `ContextualRetriever.search`, `contextualize_record` | Only `retrieval/__init__` re-export. **0 callers, 0 tests** | Unreachable | Dead. |
| `graph/graph_builder.py` | `SkillGraphBuilder` | `smoke_runner.graph_smoke` (hard-coded toy graph), tests | Smoke only | Never built from pipeline output. |
| `graph/graph_retriever.py` | `SkillGraphRetriever` | `graph/__init__` re-export; used only by `tests/test_skillllm_v2_stage2.py` | Unreachable from src (tests only) | |
| `graph/skill_graph_explainer.py` | `explain_skill_path` | **No importers, no callers, no tests** | Unreachable | Dead. |
| `graph/schema.py` | `NodeType`, `EdgeType` | graph builders | Smoke + course side | |
| `graph/course_graph_builder.py` | `CourseGraphBuilder` | course pipeline / recommend modules | Course side (out of scope) | |
| `match/job_matcher.py` | `JobMatcher.score` | `smoke_runner` (toy dicts), demo scripts, tests | Smoke/demo only | Expects `profile["skills"]` as list of names. |
| `match/gap_detector.py` | `GapDetector.detect`, `.detect_vector` | `smoke_runner`, demo scripts, tests | Smoke/demo only | Imports `recommend.gap_vector.GapEngine`. |
| `match/explanation_generator.py` | `generate_recommendation_explanation` | **No importers, no callers, no tests** | Unreachable | Dead. |
| `match/course_recommender.py` | `CourseRecommender.recommend` | `smoke_runner`, demo scripts, tests | Smoke/demo only | Depends on `recommend/` + `course/`; belongs to the course side. |
| `baselines/*` | `BertBaseline`, `GLiNERBaseline`, `evaluate_prediction_file` | `smoke_runner` baseline stages; `train_*` have `__main__` and shell scripts | Eval/benchmark only (by design) | Comparison systems for the money table; not part of inference. `baselines/train_bert_baseline.py` and `train_gliner_baseline.py` have no importers (invoked as scripts). |
| `eval/run_safe_wrapper.py` | `run_safe_wrapper`, `main` | Yes (`scripts/run_skillllm_v2_safe_wrapper.sh`, `smoke_runner.money_table_generation`, `research_refresh`) | Eval only | Re-implements the whole wrapper chain inline over cached anchor predictions; does not use `wrapper.py`. `pipeline/run_safe_wrapper.py` is a one-line re-export of it. |
| `eval/non_regression_gate.py` | `evaluate_gate`, `select_extraction`, `GateResult`, `load_guardrails` | `run_safe_wrapper`, `ablation_runner` | Eval only | The only implementation of the guardrail floors; needs gold-derived metrics, so it cannot run per request. Its decision is persisted to `artifacts/skillllm_v2/safe_wrapper/non_regression_status.json`. |
| `eval/ablation_runner.py` | `run_ablation` | `smoke_runner.extraction_ablation_full` | Eval only | Duplicates the wrapper chain a second time. |
| other `eval/*` | metrics / reports / significance / recommender benchmark | offline CLIs and `research_refresh` | Eval only | Not touched. `eval/linking_eval.py`, `matching_eval.py` are 5-7 line stubs with no importers. |
| `evals/` (new package from another agent) | `EvalContext`, `EvalResult` | none yet | n/a | Not in scope. |
| `pipeline/smoke_runner.py`, `stage_registry.py`, `pipeline_runner.py`, `artifact_tracker.py`, `gates.py` | stage runners | `scripts/*.sh` (`pipeline_runner` x50) | Smoke/eval orchestration | The orchestration of *experiments*; not an inference pipeline. |
| `pipeline/gpu_preflight.py`, `diagnose_stale_artifacts.py` | CLIs | scripts only | Ops tooling | |
| `pipeline/course_recommendation_pipeline.py` | `CourseRecommendationPipeline.run` | `eval/recommender_benchmark`, `recommend/loop_eval`, `scripts/run_skillllm_v2_course_recommendation.sh` | Course side (out of scope) | Constructs its own `SkillLLMv2Pipeline`; has no seam in the end-to-end flow. |

### Unreachable-module list BEFORE (no entry point calls them; tests/re-exports excepted)

1. `policy/__init__.py`, `policy/source_aware_policy.py` (re-export shims, 0 importers)
2. `retrieval/contextual_retrieval.py` (+ `retrieval/__init__.py`)
3. `context/context_retriever.py`
4. `graph/skill_graph_explainer.py`
5. `graph/graph_retriever.py` (tests only)
6. `match/explanation_generator.py`
7. `normalize/alias_resolver.py` (re-exported, never instantiated)
8. `normalize/esco_loader.py`, `normalize/onet_loader.py`, `normalize/lightcast_loader.py`

### On the real inference path BEFORE

`wrapper.py`, `config.py` (indirectly), `schema.py`, root `source_aware_policy.py`, `extract/{mistral,gliner,dictionary}_extractor.py`,
`extract/candidate_merger.py`, `verify/{boundary_refiner,boundary_selector,span_verifier,confidence_calibrator}.py`,
`normalize/{taxonomy_linker,skill_alias_builder(read helpers)}.py`.

The suspects `verify/span_verifier.py` and `verify/confidence_calibrator.py` are **not** dead: both run in `wrapper.extract`.
`context/skill_memory_builder.py` is reachable only through the artifact writer (`research_refresh`) and unit tests;
`match/job_matcher.py` and `match/gap_detector.py` only through smoke/demo code with toy inputs.

### Other findings

* No inference-time equivalent of the safe-wrapper non-regression behaviour: `eval/non_regression_gate.py` operates on gold-derived
  metrics over the test split; `wrapper.py` always returned the fully-processed candidates and never fell back to the anchor.
* Stub/fallback outputs are silent (`MistralExtractor`, `GLiNERExtractor` fallbacks) -- nothing in the candidate dicts or any trace
  says "this was a deterministic stub".
* Component defaults depend on CWD (`artifacts/skillllm_v2/*/tuned_config.yaml`, `data/skillllm_v2/taxonomy/skill_aliases.jsonl`).
* The YAML configs (`skillllm_v2_stage2.yaml`: extractors, taxonomy, calibration) are read by `smoke_runner` and training only; nothing in the inference API consumed them.

## AFTER (this task)

New entry point: `pipeline/inference.py` (`SkillLLMV2Pipeline`, CLI `python -m skillllm_v2.pipeline.inference`,
`scripts/run_skillllm_v2_inference.sh`). `wrapper.py` now delegates to it. Reachability was recomputed with the same AST
pass (root = `skillllm_v2.pipeline.inference`) and grep-verified for call sites.

| Module | Status now | How it is wired |
|---|---|---|
| `policy/source_aware_policy.py`, `policy/__init__.py` | **Canonical implementation**; imported by merger, verifier, calibrator, `inference.py` | Root `source_aware_policy.py` is now a pure re-export shim (test asserts identity of the objects) |
| `schemas.SkillCandidate` | Re-exported from `schema.SkillCandidate` | Duplicate dataclass removed (`schemas.py` keeps the other result types) |
| `retrieval/contextual_retrieval.py` | Stage 8 (`ContextualRetriever.search` over the supplied jobs) | |
| `context/context_retriever.py` | Stage 8 (`ContextRetriever.retrieve_skills` for the target job) | |
| `context/{profile_summarizer,skill_memory_builder,context_compactor,cached_context_builder}.py` | Stage 8, on every run (one pass through `build_cached_context_packet`) | previously only `research_refresh` |
| `graph/graph_builder.py`, `graph/graph_retriever.py`, `graph/skill_graph_explainer.py` | Stage 9 (graph built from the run's skills/jobs; retriever + explainer used; course `TEACHES` edges added after stage 11 and read back via `courses_for_skill`) | |
| `normalize/alias_resolver.py` | Instantiated in `_build_alias_maps`; canonicalizes skills for stages 8-11 | |
| `normalize/{esco,onet,lightcast}_loader.py` | Called by the new `normalize/taxonomy_registry.build_taxonomy_linker` when `taxonomy.esco_path/onet_path/lightcast_path` exist (`ChainedTaxonomyLinker`) | Dormant until a taxonomy export is configured; none exists in this repo |
| `match/job_matcher.py`, `match/gap_detector.py` | Stage 10 on real pipeline output | |
| `match/explanation_generator.py` | Stage 12 | |
| `verify/span_verifier.py`, `verify/confidence_calibrator.py`, `verify/boundary_*.py` | Stages 3-5 (already real before; now traced, with per-stage dropped lists) | |
| `verify/learned_verifier.py` (new) | Stage 4, optional (`verify.learned.enabled`), loads the previously-unused `verifier_model.joblib`; absent model degrades gracefully | off by default (never evaluated as an inference component) |
| `eval/non_regression_gate.py` | Stage 7 reuses `evaluate_gate` / `load_guardrails` and the persisted `non_regression_status.json` | |
| `config.py` | `load_config` adds the `extends:` resolution that the YAML configs already declared but nothing implemented | |
| `extract/*` | Stage 1-2; `MistralExtractor`/`GLiNERExtractor` gained `is_model_loaded()` so stubs can be detected and flagged | |

Still NOT on the inference path, by design or because it is outside the allowed scope:

* `match/course_recommender.py`, `course/`, `recommend/`, `graph/course_graph_builder.py`, `pipeline/course_recommendation_pipeline.py`: course side; reached only through `CourseRecommenderPort` (adapter) at run time.
* `baselines/*`, `eval/*` (other than the gate), `train/*`, `research_refresh.py`, `pipeline/{smoke_runner,stage_registry,pipeline_runner,...}`: experiment/benchmark/ops tooling.
* `normalize/skill_alias_builder.build_alias_table`: offline builder (its read helpers are on the path).
* `TaxonomyLinker`'s dense branch (`dense_model_name`): never enabled by any config; `inference.py` does not expose it.
* `schemas.py` result dataclasses (`JobMatchResult`, `CourseRecommendation`, …): smoke-only.
* `pipeline/run_safe_wrapper.py`: one-line re-export of `eval/run_safe_wrapper.py`.

Unreachable-module list AFTER: none of the BEFORE list remains unreachable from `pipeline.inference` (the esco/onet/lightcast loaders
are reachable but only exercised when a taxonomy path is configured; they are covered by a test with a temporary export).
Still unreachable from the inference entry point: `match.course_recommender` (course side), `schemas` (smoke), root `source_aware_policy` (compat shim).

Defects found while wiring (fixed or flagged): silent stub anchors (`MistralExtractor`, `GLiNERExtractor` fallbacks) are now tagged and surfaced in
trace/warnings (`wrapper.extract` emits a `RuntimeWarning`); dictionary-only extraction produced zero skills under the default add-only merger
(dictionary confidence 0.82 < gate 0.85), so the no-anchor mode opens those gates (flagged, unevaluated); `TaxonomyLinker.link_candidates` overwrote
canonical ids regardless of link confidence (now optionally suppressed via `taxonomy.attach_min_confidence`, default parity).
