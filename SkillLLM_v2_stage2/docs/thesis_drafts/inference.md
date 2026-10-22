# Inference: the 12-stage pipeline, the library, the ensemble layer, cost, degradation, safety, and what is verified

Draft for the thesis committee. Status as of 2026-10-22 (about 09:00 UTC). Sources: `docs/INFERENCE_PIPELINE.md`, `docs/LIBRARY.md`, `docs/WIRING_AUDIT.md`, `docs/INFERENCE_AND_EVALS.md`, `docs/BUGFIX_LOG_MATCH.md`, `configs/skillllm_v2_inference.yaml`, `MODEL_CARD.md`, and spot checks of `src/skillllm_v2/pipeline/inference.py`, `src/skillllm_v2/api.py` and `src/skillllm_v2/extract/mistral_extractor.py` (read, not run). Numbers are from files (paths given) or **derived** (arithmetic shown) or **unmeasured**. I executed no model and no pipeline for this draft.

Companion drafts: `training.md`, `evals.md`, `pretraining.md`; plan: `docs/BEST_PLAN.md`.

## 1. Two entry points, one system

1. **Research pipeline** `skillllm_v2.pipeline.inference.SkillLLMV2Pipeline` (note the capital V; 1,538 lines): one config-driven path from a profile text (and optionally a job) to a JSON result with a per-stage trace. Config `configs/skillllm_v2_inference.yaml` (extends `configs/skillllm_v2_stage2.yaml`). The older `wrapper.SkillLLMv2Pipeline.extract` (lower-case v) now delegates to stages 1 to 6 and keeps its return type (`docs/INFERENCE_PIPELINE.md`).
2. **Packaged library** `skillllm_v2.SkillLLM` (`src/skillllm_v2/api.py`): a typed, model-optional API (`from_pretrained`, `offline`, `extract`, `extract_batch`, `analyze`, `recommend`, `save_config`, `from_config`), a CLI (`skillllm version | doctor | extract`) and the span-voting ensemble. It does not import torch until a model is used (`docs/LIBRARY.md`). `analyze` calls the research pipeline's `JobMatcher` and `GapDetector` on the library's own extraction when `skillllm_v2.pipeline.inference` is importable; otherwise it falls back to set overlap and says so.

Nothing has been published (no PyPI package, no Hub upload), per `docs/LIBRARY.md` and `MODEL_CARD.md` (a template with placeholders).

## 2. The 12 stages

**Table I1. Stages of `SkillLLMV2Pipeline`** (method names verified in `src/skillllm_v2/pipeline/inference.py`; component details from `docs/INFERENCE_PIPELINE.md`, `docs/INFERENCE_AND_EVALS.md` section 1.3).

| # | Stage method | What it does | Model or rule | Learned? |
|---|---|---|---|---|
| 1 | `stage_extract` | Anchor extraction: Mistral-Small-3.1-24B plus a LoRA adapter, 4-bit NF4, greedy decoding, `max_new_tokens` 512, strict-JSON parse, offsets recovered by first-unused string match; optional GLiNER (`urchade/gliner_medium-v2.1`, zero-shot, threshold 0.35, CPU by default) and a dictionary extractor (22 built-in aliases, or the 711-row alias table in the course path) | LLM plus rules | yes (adapter) |
| 2 | `stage_merge` | `CandidateMerger` with the anchor-protected add-only policy: every anchor span is kept; non-anchor candidates need a per-source confidence bar, an exact alias-table hit and no overlap with an anchor span (thresholds in `artifacts/skillllm_v2/candidate_merger/tuned_config.yaml`, provenance doubtful, `evals.md` I10) | rules | no |
| 3 | `stage_refine` | `BoundaryRefiner` and `BoundarySelector`: keep the original span unless a variant beats it by a margin (0.08 hard, 0.15 soft); allowed edits `trim_leading_filler`, `taxonomy_normalized`, `realign_word_boundary` | rules | no |
| 4 | `stage_verify` | `SpanVerifier`: rule filter (drops spans with no offsets; never rejects anchor-derived spans); optional `LearnedSpanVerifier` (TF-IDF plus logistic regression, `verify.learned.enabled: false` by default) | rules (optional model) | optional |
| 5 | `stage_calibrate` | `ConfidenceCalibrator`: per-source confidence thresholds; it only filters, the Mistral confidence stays the constant 0.86; no tuned config exists, built-in defaults apply | rules | no |
| 6 | `stage_link` | `TaxonomyLinker`: alias, normalised, fuzzy (SequenceMatcher at least 0.88), BM25 fallback; dense branch disabled; optional chained ESCO, O*NET, Lightcast loaders (no export exists in the repository) | rules | no |
| 7 | `stage_gate` | Non-regression gate: wrapper output or untouched anchor (section 6) | rule | no |
| 8 | `stage_profile` | Cached context packet (compaction, summary, skill memory), context and contextual retrievers | rules | no |
| 9 | `stage_graph` | `SkillGraphBuilder` plus retriever and path explainer; course `TEACHES` edges added after stage 11 | rules | no |
| 10 | `stage_match` | `JobMatcher` (fixed formula) and `GapDetector` (plus gap vector), skill keys and alias canonicalisation | rules | no |
| 11 | `stage_courses` | `CourseRecommenderPort` only (adapter to `CourseRecommendationPipeline`, or a labelled null implementation) | rules | no |
| 12 | `stage_explain` | Template explanations over graph facts; limitations computed from the run (stub, gate, port status) | templates | no |

Stages 1 to 7 are "extraction"; 8 to 12 are "downstream". `run_extraction_only` runs 1 to 7; `extract_candidates` runs 1 to 6 (what `wrapper.extract` uses). Every stage can be called alone, switched off (`inference.stages.*`), and writes a `StageRecord` (status, component, timing, counts, provenance) into one `PipelineTrace`.

Only one stage contains a trained neural model that matters for quality: stage 1. The remaining stages are rule-based (the "span verifier" and "calibrator" of earlier thesis text are not learned models; `evals.md` section 3). The wrapper stages 2 to 6 add at most +0.0024 exact-F1 on SkillSpan (`reports/skillllm_v2/significance_f1.md`), which is below the 0.005 gap between two identical anchor trainings (`training.md` Table T5).

## 3. Library API and the ensemble layer

**API surface** (`docs/LIBRARY.md`): `SkillLLM.from_pretrained(base, adapter=... | adapters={name: path}, min_votes=2, quantization="4bit", calibration=...)`; context-manager use; `extract` returns an `ExtractionResult` (`skillllm.extraction/1`: `skills` with `surface_form`, `label`, `start`, `end`, `confidence`, `votes`, `n_systems`, link fields and `provenance`; `metadata` with mode, offline flag, base model, adapters, quantisation, latency, warnings). `SkillLLM.offline()` is a 290-alias, 162-skill gazetteer with no model and no GPU; it was **not evaluated** on SkillSpan and must not inherit any number in this thesis. Errors are typed (`MissingExtraError`, `ModelNotFoundError`, `DeviceError`, `GPUMemoryError`, `FeatureUnavailableError`, `ConfigError`).

**Ensemble layer** (`src/skillllm_v2/ensemble.py`, pure Python): `vote_spans` keeps a span `(start, end, label)` if at least `min_votes` distinct systems predict it exactly. All adapters attach to one loaded base model and are switched with PEFT `set_adapter`, system by system. `min_votes` is a precision/recall dial; the default 2 is the majority rule that was evaluated:

**Table I2. The dial (test, 974 documents; `reports/skillllm_v2/ensemble_eval.md`).**

| `min_votes` | Precision | Recall | Exact F1 | Latency factor vs one adapter |
|---|---:|---:|---:|---|
| 3 (unanimous) | 0.802 | 0.506 | 0.620 | about 3x |
| **2 (default)** | **0.722** | **0.661** | **0.690** | about 3x |
| 1 (union) | 0.587 | 0.781 | 0.670 | about 3x |
| single anchor | 0.674 | 0.667 | 0.670 | 1x |

Default span confidence under a three-adapter vote is the precision of spans with that many votes (1 of 3: 0.2901; 2 of 3: 0.5450; 3 of 3: 0.8018), **estimated on the test split itself** and labelled so in the result metadata; `fit_vote_calibration` re-fits on user data. With one adapter or with a different number of adapters the confidence remains the raw constant prior (0.86 Mistral, 0.82 dictionary), which is explicitly **not a probability**. The ensemble gain (+0.0198 exact F1, Holm-5 p 0.081) is suggestive, not confirmed (`training.md` section 4).

## 4. Cost model

### 4.1 Measured generation cost (24B, 4-bit, one document at a time, greedy)

**Table I3. What was measured.**

| Quantity | Value | Source |
|---|---|---|
| Loop v1 test generation, 974 docs | 3,524.4 s (3.62 s per doc, derived) + 16.6 s load; peak allocated 16.13 GB, reserved 18.51 GB | `artifacts/skillllm_v2/qlora_loop_eval/timing_segments.jsonl` (via `reports/skillllm_v2/qlora_loop_summary.md`) |
| Loop v2 test generation, 974 docs | 4,303.1 s (4.42 s per doc, derived) + 20.4 s load; peak allocated 16.32 GB, reserved 18.38 GB | `qlora_loop_v2_summary.md`; `docs/LIBRARY.md` quotes "about 4.4 s per document per adapter" |
| Anchor evaluation stage, 974 docs | 4,257.5 s including model load (4.37 s per doc, derived); two visible GPUs with `device_map=auto` | `artifacts/skillllm_v2/status/eval_raw_mistral_lora.success.json` (via the audit) |
| Per-document `latency_seconds` (anchor predictions, 901 documents with at least one span) | mean 4.65 s, median 3.79 s, min 2.04 s, max 35.06 s (first-call outlier includes model load) | audit, from `metadata.latency_seconds` |
| Free memory required | the loop trainer refuses to load below 19 GB free; the library's note says "about 16 GB" for 4-bit inference | `configs/skillllm_v2_loop_qlora.yaml`, `docs/LIBRARY.md` |

Discrepancy to keep visible: the "3.6 s per document" figure quoted in the project brief is loop v1's generation time; loop v2 and the library documentation say about 4.4 s; the anchor stage is 4.37 s. The numbers belong to different adapters and hardware placements (two GPUs against one shared GPU1) and were not separated by a controlled experiment. Use "about 3.6 to 4.4 s per document per adapter on one shared RTX 3090 for SkillSpan-length text".

### 4.2 Derived cost of the ensemble and of a request

The ensemble runs every adapter on every document: three generation passes (`docs/LIBRARY.md` section 5; adapters share one 4-bit base, so memory is about one base plus small adapters, **but the peak memory with three adapters loaded was not measured**; the 16.3 GB figure is for a single-adapter run).

**Table I4. Derived wall time (one GPU, no batching; derived from Table I3, not measured for these configurations).**

| Scenario | Passes | Seconds | Notes |
|---|---:|---|---|
| One SkillSpan-length document, single adapter | 1 | 3.6 to 4.4 | measured |
| Same, three-adapter ensemble | 3 | about 11 to 13 | `docs/LIBRARY.md` says "roughly 13 s"; extrapolated, not measured |
| One request = profile text plus one job text, single adapter | 2 | about 7 to 9 | stage 1 runs on both texts (`job_extract` record) |
| Same request, ensemble | 6 | about 22 to 27 | |
| 974-document test pass, single adapter | 974 | 0.98 to 1.20 GPU-h | measured (v1 and v2) |
| 974-document test pass, ensemble | 2,922 | about 2.9 to 3.6 GPU-h | derived |
| Pseudo-labelling 2,400 chunks with three adapters, unbatched | 7,200 | about 7.2 to 8.8 GPU-h | planned in GPU Experiment 2 with batched generation; speed-up unmeasured |

Two caveats on Table I4. (a) SkillSpan rows are short (about 16 tokens on average in the test split, my count); a resume is hundreds to thousands of tokens, which increases prefill and, because the JSON answer lists every skill, decode length too. The code caps generation at 512 new tokens, and the library tells users to chunk long documents; resume-length latency and the effect of chunking on exact-span F1 are **unmeasured**. (b) The one-document-at-a-time, no-padding protocol reproduces how the reported numbers were obtained; `extract_batch` runs system by system to avoid adapter thrashing but is not batched generation.

### 4.3 Cost of everything except the LLM

The module suite's end-to-end evaluation measured the CPU stages with the **stub** extractor (`reports/skillllm_v2/eval_suite/REPORT.md`, `e2e.latency.*`, milliseconds): profile extraction 15.1, job extraction 15.9, matching 0.017, gap detection 0.008, gap vector 0.042, recommendation 0.36; total 31.4 ms per case; pipeline initialisation 1,327 ms; first call 623 ms. After the matcher fixes the alias lookups raise match, detect and vector to 0.18, 0.14 and 0.20 ms per call (`docs/BUGFIX_LOG_MATCH.md`). Compared with 3.6 to 4.4 s for one generation (Table I3), the non-model stages are two to three orders of magnitude cheaper (derived: 4.4 s against about 31 ms is about 140x), so **latency is dominated by stage 1 and by the number of adapters**. The 24B model load takes about 17 to 20 s (v1 and v2 timing segments). GLiNER, if installed, adds a CPU or GPU pass of unmeasured cost.

### 4.4 Options to cut cost (all untested here)

Described, not done; each needs re-evaluation on the 974 documents, because any change of numerics or prompt changes the evaluated system:

1. **Prompt compression.** The fixed prompt is 78 to 86% of training tokens (`training.md` Table T2 and discrepancy A5 in `evals.md`); the compact-prompt loop v3 is **pending** (GPU Experiment 1). The log predicts little latency gain because generation is decode-bound.
2. **Prefix caching.** The ~300-token system and user prefix is identical across requests; caching its key-value state would remove the repeated prefill. Not implemented; gain unmeasured.
3. **Distillation of the ensemble into one adapter** (GPU Experiment 2, **pending**): target 1x cost at ensemble-like quality.
4. **Merge the LoRA into the base and serve with a batched inference server** (for example vLLM), possibly with AWQ or GPTQ weight quantisation. Merging into a 4-bit base is lossy, and merging into bf16 needs about 48 GB, more than one 24 GB card (derived: 24B parameters at 2 bytes). Multi-adapter serving could hold the three ensemble adapters. Greedy decoding should be near-deterministic but a different kernel or quantiser can change spans; the accepted-metric check (974 documents, paired bootstrap) is mandatory before substituting. No benchmark exists in the repository.
5. **Fewer adapters.** Two of the three would lose the majority-vote structure; no evaluation of two-adapter rules exists.

## 5. Degradation and fallbacks (never silent)

**Table I5. Behaviour under failure** (`docs/INFERENCE_PIPELINE.md`; extractor failures per `docs/BUGFIX_LOG_MATCH.md`).

| Situation | Behaviour |
|---|---|
| Mistral loads | real anchor, `trace.anchor_mode="model"` |
| `anchor.mode: auto`, no weights or GPU | **no anchor**, `trace.degraded`; dictionary (and GLiNER) candidates only; merge gates are opened because add-only gating against a missing anchor would reject everything (flagged, unevaluated) |
| `anchor.mode: stub` or `--no-mistral` | deterministic alias-match stub (opt-in); candidates carry `source="mistral"` so policies treat them as the anchor, but `metadata.stub=True`, trace lists `anchor:deterministic_stub`, stage status `degraded`, and the final explanation's limitations say so |
| `anchor.mode: mistral`, weights missing | raises (explicit request, no fallback) |
| Any extractor raises (Mistral, GLiNER, dictionary) | caught per extractor; stage `degraded` with the error text; remaining extractors used; if all fail, empty extraction with `status="failed"`; no exception unless `strict_extractors=True` or `continue_on_error: false`. `wrapper.extract` emits a `RuntimeWarning` naming the failed extractor. (Before this fix the exceptions escaped; the e2e graceful-degradation share was 0.8182 and is reported as 1.0 by the fix pass, **not re-confirmed by a committed suite re-run**.) |
| GLiNER missing | skipped with a note (`auto`); no hard-coded fallback candidates unless `allow_fallback: true` |
| Optional stage raises | recorded `failed`, data passed through, run continues; `extract` and `merge` always raise |
| Unparsable model JSON | the library reports a warning and counts no skills for that document and system; the research `MistralExtractor.parse_skill_json` raises `ValueError`, now absorbed by the per-extractor handling above |
| Course recommender missing | `NullCourseRecommender`: empty list, "NULL IMPLEMENTATION" note, stage `degraded` |
| Learned verifier missing | rule verifier only, `degraded` |
| Anchor confidence | constant 0.86; `anchor.confidence: logprob` without a hook warns **NOT IMPLEMENTED**; a log-probability capture option is being added for GPU Experiment 3 (**pending**) |

## 6. The "safe wrapper" fallback is an offline decision

Stage 7 reuses `eval.non_regression_gate.evaluate_gate` (accept the wrapper only if exact-F1 is at least anchor - 0.010034 and soft-F1 at least anchor - 0.03). Those floors need gold labels, so they cannot be computed per request. The decision order: (1) metrics passed in, (2) forced mode, (3) for a real-model anchor, the **persisted offline decision** in `artifacts/skillllm_v2/safe_wrapper/non_regression_status.json` (missing file: `fallback_to_anchor`), (4) for a stub anchor, structural guard only. A **structural guard** (`gate.min_anchor_retention` 0.8, "an untuned safety heuristic") rejects wrapper output that dropped or moved more than 20% of anchor spans. On fallback the extraction output is a deep copy of the anchor's candidates, but downstream stages still use the enriched wrapper output by default (`gate.downstream_from: enriched`). Consequences: the "fallback" has never triggered on disk (all three variants were accepted), the decision was taken on the same 974 test documents it is reported on, and a runtime per-request fallback on quality does not exist.

## 7. Safety and security

1. **`trust_remote_code`.** The packaged library defaults to `trust_remote_code=False` (`src/skillllm_v2/api.py` lines 421 and 701). The research path does the opposite: `MistralExtractor.load` passes `"trust_remote_code": True` when loading the base model (`src/skillllm_v2/extract/mistral_extractor.py` line 159), the tokenizer loader defaults to `True` ("to keep the research scripts' behaviour on local weights"), and both trainers hard-code `True`. `SkillLLMV2Pipeline` uses `MistralExtractor`. `trust_remote_code=True` executes Python files shipped inside the model directory when the model class requires them, so **pointing the research pipeline at an untrusted or Hub-supplied model directory is arbitrary code execution**. With a local, project-controlled copy the practical risk is small, and I did not test whether the Mistral 3.1 checkpoint needs remote code at all (I believe the architecture is natively supported in recent `transformers`, but I did not verify). Recommendation (not done, `src/` is out of scope for this draft): make the research default `False`, and require an explicit flag for any directory not under version-controlled project paths.
2. **Other code-execution surfaces.** `LearnedSpanVerifier` loads `verifier_model.joblib`, and joblib files are pickles, so only trusted files should be loaded (off by default). GLiNER downloads `urchade/gliner_medium-v2.1` from the Hugging Face hub at first use when installed; the library itself makes no network calls unless `allow_hub=True`.
3. **Prompt injection and untrusted text.** The extractor reads resumes and postings, which are untrusted. The model returns JSON and the pipeline validates it against the candidate schema, which limits the blast radius to wrong or inflated skill lists; it does not make the output trustworthy (an applicant could embed skill strings). `MODEL_CARD.md` says not to feed outputs to a downstream agent without validation. No adversarial-text evaluation exists. The repository's guard skill (`laya-guard`) is separate and is not part of this pipeline.
4. **Personal data.** Resumes contain personal data. Run locally, do not log inputs, follow applicable data-protection law (`MODEL_CARD.md`). The job-posting pool used for experiments is licensed or private material kept local (`Extract_jobDesc/README.md`).
5. **Fairness and use limits.** No evaluation across demographic groups, languages or resume formats; out-of-scope uses include automated hiring or screening decisions; the match score is "a rule-weighted heuristic with no validation study behind it" (`docs/LIBRARY.md`). Thesis text must not present the matcher or the recommender as validated.

## 8. Reproducing and running

```bash
# CPU only, no weights, flagged stub anchor (demo inputs) -> reports/skillllm_v2/demo/inference_result.json
bash scripts/run_skillllm_v2_inference.sh
# real 24B anchor, re-launched through the shared-server guardrails (heavy)
bash scripts/run_skillllm_v2_inference.sh --mistral
# library CLI
skillllm doctor --base-model models/mistral-small-3.1-24b-instruct-2503
skillllm extract --offline --text "Python and SQL"
skillllm extract --base-model models/mistral-small-3.1-24b-instruct-2503 \
    --adapters anchor=artifacts/skillllm_v2/mistral_lora/model v1=artifacts/skillllm_v2/qlora_loop/best \
               v2=artifacts/skillllm_v2/qlora_loop_v2/best --min-votes 2 --file resume.txt
```

## 9. What is verified and what is not

**Table I6. Verification status (as of 2026-10-22).**

| Claim | Status | Evidence |
|---|---|---|
| The anchor adapter extracts skills from SkillSpan test sentences with exact F1 0.670 (single seed) | **Verified** (measured, reproduced from saved predictions to 1e-6) | `reports/skillllm_v2/eval_suite/REPORT.md` (`extract.*.reproduction` 0.0) |
| Loop adapters and the ensemble metrics | **Verified** from saved predictions; significance computed | `training.md` Tables T9 and T10 |
| 4-bit 24B generation takes about 3.6 to 4.4 s per document and peaks near 16.1 to 16.3 GB allocated | **Verified** for single-adapter, SkillSpan-length text on one shared RTX 3090 | `timing_segments.jsonl` (via the loop reports) |
| The CPU stages are deterministic, produce schema-valid output and are fast (about 31 ms per case) | **Verified on 12 synthetic cases with the stub extractor** | suite e2e module (determinism 1.0 in process and across two hash seeds; schema validity 1.0; traceability 1.0) |
| Stage logic after the bug-fix passes | **Reported by the fix agents, with new unit tests; not re-confirmed by a committed suite re-run** (pending); I did not run the tests | `docs/BUGFIX_LOG.md`, `docs/BUGFIX_LOG_MATCH.md` |
| The full 12-stage pipeline with the **real** model end to end | **Not verified.** The e2e evaluation and the demos use the stub or no anchor; extraction errors have never been propagated through to recommendations | `docs/EVAL_SUITE.md` limitations |
| `SkillLLM.from_pretrained` loading the real model through the packaged library | **Not verified** (unit tests use fake backends; "Real-model loading ... has been unit-tested with fakes only") | `MODEL_CARD.md` |
| Memory with three adapters attached; ensemble latency | **Not measured** (single-adapter peak only; 3x is an extrapolation) | `docs/LIBRARY.md` |
| Behaviour on resume-length or multi-page text, chunking, `max_new_tokens` truncation | **Not measured** | audit section 4 item 12 |
| Offline (gazetteer) extractor quality | **Not evaluated** | `docs/LIBRARY.md` section 4 |
| Taxonomy linking correctness | **Not evaluated** (coverage 0.8898 only; no gold ids; authored-probe recall@1 0.925 on 107 items) | `evals.md` M14, M15 |
| Matching, gap detection, explanations, recommendation on real users | **Not evaluated**; only synthetic data and one synthetic-catalogue benchmark | `evals.md` Table E3 |
| Quality of the gate at inference time | **Not verified**; the decision is a persisted offline test-set result; the structural guard threshold 0.8 is untuned | section 6 |
| Quantised/merged/vLLM serving equivalence | **Not done** | section 4.4 |
| Concurrency, throughput under load, GPU contention, failure recovery of the 24B process | **Not measured** | |
| Safety against hostile input | **Not evaluated** | section 7 |
| Tests pass | **Not run by me**; fix passes report 438 passed, 1 skipped, 2 failed (the two eval tests that encode fixed bugs) | `docs/BUGFIX_LOG_MATCH.md` |
