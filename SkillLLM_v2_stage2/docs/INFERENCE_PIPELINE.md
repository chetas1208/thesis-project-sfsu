# SkillLLM_v2 end-to-end inference pipeline

`skillllm_v2.pipeline.inference.SkillLLMV2Pipeline` is the single, config-driven inference path. It wires the
extraction chain, the safe-wrapper non-regression behaviour, profile/context, graph, matching, course recommendation
(through a port) and the final explanation into twelve individually callable stages. Each stage returns a typed
artifact (`pipeline/inference_types.py`) carrying a `StageRecord` (status, component, timing, counts, provenance) and
appends to one `PipelineTrace`.

`wrapper.SkillLLMv2Pipeline.extract` (the old API, used by eval/course code) now delegates to stages 1-6 of this
pipeline and keeps its public attributes and return type. The old class name differs only by case
(`SkillLLMv2Pipeline` = legacy extraction wrapper, `SkillLLMV2Pipeline` = full pipeline); `wrapper.SkillLLMv2Pipeline().pipeline()`
returns the full pipeline sharing the wrapper's components.

## Stage diagram

```
 profile_text ──► 1 extract ─► 2 merge ─► 3 refine ─► 4 verify ─► 5 calibrate ─► 6 link ─► 7 gate ──► skills
                  │ anchor:     CandidateMerger  Boundary-   SpanVerifier  Confidence-   TaxonomyLinker  non-regression:
                  │ Mistral     (source-aware    Refiner +   (+ optional   Calibrator     attach-only     accept wrapper
                  │ | stub      policy:          Boundary-   learned                                     | untouched anchor
                  │ | none      anchor-protected Selector    LogReg)                                     (enriched kept
                  │ + GLiNER    add-only)                                                                 for downstream)
                  │ + dictionary
 job_text ─(text? stages 1-6 → job_extract)─┐
                                            ▼
        8 profile ──► 9 graph ──► 10 match ──► 11 courses ──► 12 explain
        summary,      SkillGraph   JobMatcher    CourseRecommender   generate_recommendation_
        skill memory, Builder +    + Gap-        Port ONLY           explanation + gate/stub/
        compaction,   Retriever +  Detector      (adapter | null)    port limitations
        ContextRetr.  skill-path   (+ gap vector)   └► course nodes/TEACHES edges
        Contextual-   explainer                        added to the graph
        Retriever
```

| # | Stage method | Components | Artifact |
|---|---|---|---|
| 1 | `stage_extract` | `MistralExtractor` / `DeterministicStubExtractor`, `GLiNERExtractor`, `DictionaryExtractor` | `ExtractArtifact` |
| 2 | `stage_merge` | `CandidateMerger` + `policy.source_aware_policy` | `CandidateArtifact` |
| 3 | `stage_refine` | `BoundaryRefiner`, `BoundarySelector` | `CandidateArtifact` |
| 4 | `stage_verify` | `SpanVerifier`, optional `LearnedSpanVerifier` | `CandidateArtifact` (+`dropped`) |
| 5 | `stage_calibrate` | `ConfidenceCalibrator` | `CandidateArtifact` (+`dropped`) |
| 6 | `stage_link` | `TaxonomyLinker` / `ChainedTaxonomyLinker` (ESCO, O*NET, Lightcast loaders) | `CandidateArtifact` |
| 7 | `stage_gate` | `eval.non_regression_gate.evaluate_gate` + persisted offline decision | `GateArtifact` |
| 8 | `stage_profile` | `build_cached_context_packet` (→ `compact_context` → `summarize_profile` + `build_skill_memory`), `ContextRetriever`, `ContextualRetriever` | `ProfileArtifact` |
| 9 | `stage_graph` | `SkillGraphBuilder`, `SkillGraphRetriever`, `explain_skill_path` | `GraphArtifact` |
| 10 | `stage_match` | `JobMatcher`, `GapDetector` (`detect` + `detect_vector`), `AliasResolver` canonicalization | `MatchArtifact` |
| 11 | `stage_courses` | `CourseRecommenderPort` only | `CourseArtifact` |
| 12 | `stage_explain` | `match.explanation_generator.generate_recommendation_explanation` | `ExplanationArtifact` |

`job_extract` is an extra trace record, present only when the job is raw text (stages 1-6 are run on it; HARD spans become
required skills and SOFT spans preferred skills, exactly as the course pipeline does).

## Running

```bash
# CPU only, no weights: the demo inputs with the flagged deterministic stub anchor
scripts/run_skillllm_v2_inference.sh                      # -> reports/skillllm_v2/demo/inference_result.json
PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.pipeline.inference \
    --profile inputs/demo/user_profile.json --job inputs/demo/jobs.jsonl --out out.json --no-mistral

# real Mistral anchor (heavy, 24B 4-bit): launched through the shared-server guardrails
scripts/run_skillllm_v2_inference.sh --mistral            # re-execs: guardrails run cpu -- ...  (GUARD_CLASS=gpu1 for a GPU)
```

CLI flags: `--profile FILE` (.json `{"text","user_id"}` or text), `--job FILE` (.jsonl/.json structured job(s), or text),
`--out`, `--config`, `--no-mistral` (stub anchor), `--anchor auto|mistral|stub|none`, `--stages a,b`, `--skip a,b`,
`--courses auto|adapter|null`, `--catalog`, `--target-job`, `--top-k`.

Python:

```python
from skillllm_v2.pipeline.inference import SkillLLMV2Pipeline
p = SkillLLMV2Pipeline()                                   # configs/skillllm_v2_inference.yaml
res = p.run(profile_text, job)                             # job: text | job dict | list of job dicts | None
res = p.run(profile_text, job, skip=["graph", "gliner"])   # stage names and component switches
res = p.run(profile_text, job, stages=["extract", "merge", "match"])
ext = p.run_extraction_only(text)                          # stages 1-7, ext.skills = gated output, ext.trace
ext = p.extract_candidates(text)                           # stages 1-6, no gate (what wrapper.extract uses)
art = p.stage_merge(text, p.stage_extract(text).candidates)  # every stage is callable on its own
res.trace.to_dict()                                        # per-stage status/timing/provenance, stub flags, warnings
res.to_dict()                                              # JSON-serializable
```

Every skill dict carries `metadata.extractor_mode` (`model` | `deterministic_stub` | `fallback_stub` | `dictionary_rules`).

## Graceful degradation (never silent)

| Situation | Behaviour |
|---|---|
| `anchor.mode: auto`, Mistral loads | real anchor, `trace.anchor_mode="model"` |
| `auto`, no weights/GPU | **no anchor** (`anchor_mode="none"`, `trace.degraded`), dictionary (+GLiNER) candidates only; merge gates are opened because add-only gating against a missing anchor would reject everything (recorded in the merge record; unevaluated) |
| `anchor.mode: stub` / `--no-mistral` / `allow_stub: true` | deterministic alias-match stub, **opt-in**; `source="mistral"` so policies treat it as the anchor, but every candidate has `metadata.stub=True`, `trace.stub_components=["anchor:deterministic_stub"]`, the status is `degraded`, and the final explanation's `limitations` say so |
| `anchor.mode: mistral`, weights missing | raises (explicit request, no fallback) |
| GLiNER missing / fails | `gliner.enabled: auto` skips with a note; `true` records `degraded`; fallback stub candidates only with `allow_fallback: true` (flagged) |
| Optional stage raises | recorded `failed`, data passed through, run continues (`continue_on_error`); `false` re-raises. `extract`/`merge` always raise |
| Course recommender missing | `NullCourseRecommender`: empty list with a `NULL IMPLEMENTATION` note, stage `degraded` |
| Learned verifier model missing | rule verifier only, stage `degraded` |
| Anchor confidence | Mistral assigns a constant 0.86, so calibration is not informative for anchor spans. `confidence_hook=` (a `(text, candidates) -> candidates` callable) is the seam for token-logprob confidence; `anchor.confidence: logprob` without a hook warns **NOT IMPLEMENTED** |

## Stage 7: non-regression gate at inference time

The guardrail floors (`exact_span_f1 >= anchor - 0.010034`, `soft_skill_f1 >= anchor - 0.03`) need gold labels, so they cannot be
recomputed per request. The decision is taken in this order:

1. `candidate_metrics` + `anchor_metrics` passed to `run_extraction_only` → `eval.non_regression_gate.evaluate_gate` (same code, same guardrails file);
2. `gate.mode: wrapper|anchor` forces a side;
3. real-model anchor → the persisted offline decision in `gate.status_path` (`non_regression_status.json: final_decision`); missing file → `gate.on_missing_status` (default `fallback_to_anchor`);
4. stub anchor → the offline decision does not transfer (it was measured on the real model): structural guard only; no anchor → not applicable.

Then a **structural guard** (`gate.min_anchor_retention`, default 0.8, an untuned safety heuristic) rejects wrapper output that dropped or moved too many
anchor `(start, end, label)` spans. On fallback `ExtractionResult.skills` is a deep copy of the untouched anchor candidates (only the provenance tag
`extractor_mode` was added at extraction); `enriched_skills` (wrapper output) is still produced and, with `gate.downstream_from: enriched` (default,
the documented gate behaviour), feeds stages 8-12.

## Course seam (`pipeline/course_port.py`)

`CourseRecommenderPort` (a `typing.Protocol`): `name`, `is_available()`, `recommend(CourseRecommendationRequest) -> CourseRecommendationResponse`.
Implementations: `CourseRecommendationPipelineAdapter` (lazy; delegates to `CourseRecommendationPipeline.run`, never modifies it, normalizes `title`→`course_name`,
returns an UNAVAILABLE response instead of raising when the catalog/import is missing) and `NullCourseRecommender` (labelled no-op). `inference.py` imports nothing from
`course/`, `recommend/` or `course_recommendation_pipeline` (a test enforces it). To swap: `SkillLLMV2Pipeline(course_port=MyPort())` or `courses.port: adapter|null|auto`.
The request carries canonical skill ids on both user skills and job skills so the course side's gap engine sees one id space.

## Config keys (`configs/skillllm_v2_inference.yaml`, `extends: configs/skillllm_v2_stage2.yaml`)

`extends` is resolved by the new `skillllm_v2.config.load_config`. Inherited: `model`, `extractors`, `taxonomy`, `calibration`. New `inference:` keys:

| Key | Default | Meaning |
|---|---|---|
| `stages.<name>` | all `true` | per-stage switch (`extract` mandatory) |
| `continue_on_error` | `true` | optional stage failures are recorded, not fatal |
| `anchor.mode` | `auto` | `auto` \| `mistral` \| `stub` \| `none` |
| `anchor.allow_stub` | `false` | let `auto` fall back to the flagged stub |
| `anchor.confidence` | `constant` | `constant` \| `hook` \| `logprob` (not implemented) |
| `gliner.enabled/device/threshold/allow_fallback` | `auto/cpu/0.35/false` | optional GLiNER; never auto-selects a GPU |
| `dictionary.enabled` | `true` | |
| `verify.learned.enabled/model_path/threshold` | `false/…joblib/0.5` | optional trained verifier (anchor spans are never rejected) |
| `taxonomy.attach_min_confidence` | `0.0` | links below it are kept as suggestions (`taxonomy_link.applied=false`), canonical fields untouched |
| `taxonomy.alias_table_paths`, `esco_path`, `onet_path`, `lightcast_path` | alias table, null×3 | external taxonomies are wired through the loaders into a `ChainedTaxonomyLinker` |
| `gate.mode/status_path/guardrails_path/on_missing_status/min_anchor_retention/downstream_from` | see above | stage 7 |
| `profile.user_id/top_k_skills/top_k_jobs`, `matching.uncertain_confidence_below`, `graph.max_hops` | | |
| `courses.port/catalog_path/top_k` | `auto`, `inputs/courses/course_catalog.jsonl`, 5 | `auto` = adapter if importable and the catalog exists, else null |

Relative paths resolve from the repo root (not the CWD), including the tuned merger/selector configs and the alias table.

## How to swap a component

Constructor injection: `SkillLLMV2Pipeline(mistral=, gliner=, dictionary=, merger=, refiner=, selector=, verifier=, calibrator=, linker=, job_matcher=, gap_detector=, course_port=, learned_verifier=, confidence_hook=)`.
An injected anchor that reports `is_model_loaded() == False` is treated as (and flagged as) a stub. Anchor candidates must be `source="mistral"` dicts that satisfy `schema.validate_skill` (violations raise `PipelineStageError`).

## Notes and limits

* Skills are canonicalized for matching by exact alias evidence only (`AliasResolver` over the linker's entries), not by fuzzy/BM25 links.
* `JobMatcher` fit fields (`semantic_role_fit`, …) are taken from the job record; for text jobs they are 0.0 and the match record says so. `weak_skills` are not derived; `uncertain_skills` = extraction confidence below `matching.uncertain_confidence_below`.
* Verified wiring is exercised by `tests/skillllm_v2/test_inference_pipeline.py` (CPU only, no weights).
