# SkillLLM_v2 Thesis Codebase Audit

Generated on 2026-09-02 from the local working tree. This audit is the claim boundary for the accompanying thesis: statements about the implementation and results must be traceable to the paths cited here. It does not treat a stage marker, a demo artifact, or a module name as evidence of a completed evaluation.

## 1. Repository state and scope

The primary implementation is `SkillLLM_v2_stage2/`. At audit time it is on branch `feature/skillllm-v2-research-refresh-20260709`, which has **no commits**: all project files are untracked. Consequently, the thesis can identify the branch name and the artifact snapshot, but cannot claim that a particular committed revision is reproducible from Git alone.

The audited Stage-2 tree contains 87 Python modules under `src/skillllm_v2/`, 8 YAML configurations, 38 shell scripts, 23 focused test files under `tests/skillllm_v2/`, and one legacy aggregate test file. The project also includes earlier, differently scoped experiments under `Experiments/`; they are not folded into the Stage-2 SkillSpan result table because their data and metric protocols differ.

The original upstream-style `SkillLLM_v2_stage2/README.md` still describes an older Skill-LLM layout and LLaMA-oriented commands. It is not a reliable description of the audited Mistral-focused Stage-2 system. The source, configuration, persisted artifacts, and reports cited below take precedence.

## 2. Data and preparation

The local SkillSpan export is at `data/source/skillspan_hf/`. Its README identifies the corpus as English job-posting sentences under CC BY 4.0 and gives the paper citation. The local line counts are 4,800 training examples, 3,174 development examples, and 3,569 test examples. Records contain tokens, `tags_skill`, `tags_knowledge`, document delimiters (`idx`), and a `source` field (`tech` or `house`).

`src/skillllm_v2/train/build_mistral_sft_data.py` reads the three local splits through `load_skillspan_split`, constructs text and gold spans, and writes JSONL prompt--target records. Each output row contains an identifier, a fixed extraction instruction, the original text, and a JSON object with gold skills. `mistral_lora_trainer.py` subsequently renders the system/user prompt used by `MistralExtractor` and trains causal language modelling targets containing only `text` and `label`. It does not train a BIO tagger or a token-fusion model.

## 3. Extraction architecture actually implemented

The system’s shared record is `SkillCandidate` in `src/skillllm_v2/schema.py`. It requires a surface form, hard/soft label, character offsets (or both offsets null), source, confidence, evidence, optional canonical fields, and metadata. Validation rejects invalid labels, source values, offsets, confidence ranges, and missing fields. This explicit schema is the primary provenance mechanism.

### Primary extractor

`extract/mistral_extractor.py` implements a Mistral Small 3.1 24B Instruct extractor. It prompts for strict JSON skill lists, performs deterministic generation (`do_sample=False`), parses fenced or embedded JSON, and aligns output strings using exact then case-insensitive substring search. Its configured 4-bit path uses NF4 quantization and bfloat16 compute; its runtime can attach a PEFT adapter when one is present. A deterministic keyword fallback exists **only when fallback is allowed**. The primary full-test metric artifact explicitly states `Model load fallback used: False`; the thesis may therefore describe that artifact as a model-backed rather than a fallback result.

### Auxiliary candidates and guarded composition

`extract/gliner_extractor.py` provides GLiNER candidates and `extract/dictionary_extractor.py` provides alias-based exact/fuzzy candidates. The reported standalone GLiNER baseline is explicitly zero-shot rather than fine-tuned. `extract/candidate_merger.py` merges candidate groups while preserving all sources in metadata. Its add-only policy protects Mistral candidates: candidates containing Mistral are retained; non-anchor additions must meet source-dependent confidence thresholds, satisfy alias-evidence requirements when configured, and must not overlap an anchor span. This is a concrete implementation of the non-destructive design, not merely a proposed principle.

`verify/boundary_refiner.py`, `verify/span_verifier.py`, and `verify/confidence_calibrator.py` implement rule-based boundary, validity, and confidence decisions. A serialized verifier artifact exists, but `reports/skillllm_v2/linking_eval.md` states that supervised taxonomy-reranker labels were unavailable; no thesis claim should imply a fully evaluated learned verifier or reranker. Tests verify that the boundary refiner retains a candidate without taxonomy evidence and never drops all candidates.

## 4. Non-regression wrapper and evidence conflict

The current wrapper entry point is `eval/run_safe_wrapper.py`. It loads cached predictions from the full-test 80-epoch Mistral run (974 documents for this evaluation path) rather than retraining Mistral. It evaluates three variants: boundary refiner, Mistral+GLiNER+taxonomy, and the full CAG/graph wrapper. For every variant, `eval/non_regression_gate.py` compares candidate exact-span and soft-skill F1 with the Mistral anchor and selects either candidate predictions or the untouched anchor predictions.

The active full-test anchor artifact is `artifacts/skillllm_v2/mistral_lora/metrics.json`:

| Measure | Verified value |
|---|---:|
| Exact-span F1 | 0.670069 |
| Token F1 | 0.787886 |
| Partial-span F1 | 0.813845 |
| Hard-skill F1 | 0.726450 |
| Soft-skill F1 | 0.604046 |

The safe-wrapper artifact records 0.671111 exact-span F1 for its boundary variant and 0.671102 for the full variant. These are tiny increases over the 0.670069 anchor under the full-test gated run, but they are below the older improvement target of 0.675034. All three variants pass the configured tolerance floors, so their decisions are `accept`; “accepted as non-regressive” must never be rewritten as “demonstrably improves extraction.”

There is a historical conflict worth reporting rather than hiding. `reports/skillllm_v2/wrapper_error_analysis.md` reports another 974-document analysis in which an anchor scored 0.675034 and older boundary/full processing substantially reduced exact and soft F1. The newer artifact timestamps show a subsequent safe-wrapper evaluation on 2026-07-13. The current thesis result table uses the later persisted `metrics.json` files and the safe-wrapper gate, while the earlier report is retained as diagnosis motivating the anchor-protection redesign. Its older figures are not pooled with the later table.

## 5. Normalization, linking, and taxonomy boundary

`normalize/taxonomy_linker.py` attaches canonical fields to a candidate. The companion `linking_error_analysis.md` reports an alias table with 696 canonical entries and an **attach-only** linker: it never changes a candidate’s surface form, offsets, label, or source. It reports linked-candidate coverage of 0.890271 for Mistral+GLiNER+taxonomy and 0.890374 for full CAG/graph. Those values are coverage-style signals, not gold-label entity-linking accuracy. In contrast, `reports/skillllm_v2/linking_eval.csv` says `taxonomy_linking_accuracy,0.0`, and the prose report says supervised reranker labels are unavailable. The thesis must call the 0.890 values *linking coverage* and state that supervised canonical-link accuracy was not evaluated.

The taxonomy loaders include ESCO, O*NET, Lightcast, and custom interfaces, but the current configured stage uses local custom alias paths. It is accurate to describe external-taxonomy support as an interface and local alias-backed linking as the reported implementation; it is not accurate to claim a completed full ESCO, O*NET, or Lightcast ingestion/evaluation.

## 6. Context, graph, matching, and recommendation

The context subsystem is implemented in `context/`. `profile_summarizer.py` aggregates only supplied extracted spans, retains evidence, source offsets, labels, confidence, canonical fields, user-supplied roles/history/preferences, and reports label conflicts. It intentionally does not infer hidden skills or sensitive attributes. `context_compactor.py` separates (a) stable cached profile/taxonomy/course summaries, (b) a dynamic policy for job postings and scraped descriptions, and (c) graph-path retrieval policy. `cached_context_builder.py` writes a deterministic SHA-256 content hash. The code therefore supports evidence-preserving aggregation and bounded structured packets; it does not implement temporal decay, semantic contradiction resolution, or a learned profile summarizer.

`graph/graph_builder.py` builds an in-memory NetworkX `MultiDiGraph` with User, Profile, Job, Course, Skill, SkillCluster, and Occupation node types. Implemented edge types include `HAS_SKILL`, `MISSING_SKILL`, `REQUIRES`, `PREFERS`, `TEACHES`, `REQUIRES_PREREQUISITE`, `SIMILAR_TO`, and `BELONGS_TO`. `graph_retriever.py` returns direct user-skill, job-requirement, and course-skill relationships; `skill_graph_explainer.py` produces shortest-path text when one exists. This is a lightweight in-memory graph, not a deployed graph database or a learned GraphRAG system.

`match/job_matcher.py`, `gap_detector.py`, and `course_recommender.py` are deterministic, explainable scoring modules. Job matching combines required coverage (0.45), preferred coverage (0.20), supplied semantic-role fit (0.15), experience fit (0.10), preference fit (0.10), and a missing-required penalty (0.25). Course scoring combines gap coverage (0.45), prerequisite fit (0.25), supplied level fit (0.15), goal fit (0.15), and a workload penalty. These exact formulae are implementation details; they have not been calibrated against labelled rankings. Both `matching_eval.csv` and `course_recommendation_eval.csv` explicitly mark nDCG@10 and P@5 as `NA` because relevance labels are unavailable. Demo outputs and smoke stages prove data-flow execution, not recommendation effectiveness.

## 7. Training configuration and compute evidence

`configs/skillllm_v2_training_80ep_2gpu.yaml` configures seed 42, full mode, the local Mistral model path, 4-bit loading, two visible GPUs (`0,1`), DDP/torchrun-style settings, 80 epochs, batch size 1, gradient accumulation 32, learning rate 0.00012, maximum sequence length 512, LoRA rank 16, LoRA alpha 32, dropout 0.05, gradient checkpointing, and fp16-oriented distributed settings. The source trainer selects bfloat16 if CUDA supports it, otherwise float16.

`artifacts/skillllm_v2/mistral_lora/model/training_summary.json` records 4,800 train samples, 3,174 development samples, world size 2, 324,217.825 seconds training runtime, training loss 0.143082, development loss 0.397449, and epoch 80. `reports/skillllm_v2/research_refresh/run_manifest.json` reports Python 3.12.3, PyTorch 2.6.0+cu124, and two NVIDIA GeForce RTX 3090 GPUs with 24 GiB each. These artifacts support a single, persisted two-GPU 80-epoch run. They do not establish multi-seed variance, hyperparameter-search coverage, or generalization beyond this dataset.

## 8. Baselines and result provenance

The local full-test artifacts report the following values. These are the thesis’s primary quantitative table because they are directly persisted metric JSON files and are summarized consistently by `metric_sanity_report.md` and `money_table.md`.

| System | Exact F1 | Token F1 | Hard F1 | Soft F1 | Status |
|---|---:|---:|---:|---:|---|
| BERT baseline | 0.059150 | 0.274922 | 0.129799 | 0.018796 | CPU-fast baseline artifact |
| GLiNER | 0.075833 | 0.114409 | 0.054559 | 0.096571 | zero-shot/few-shot; fine-tuning skipped |
| Raw Mistral LoRA | 0.670069 | 0.787886 | 0.726450 | 0.604046 | model-backed anchor |
| Mistral + GLiNER + taxonomy | 0.670062 | 0.787575 | 0.728020 | 0.602305 | full-test legacy wrapper artifact |
| Full CAG/graph | 0.671102 | 0.787575 | 0.728618 | 0.603755 | full-test wrapper artifact; ranking metrics unavailable |

`eval/prediction_provenance_check.py` checks that downstream artifacts are newer than Mistral predictions and rejects a Mistral metric note indicating fallback. The persisted report states that no stale or missing downstream artifacts were detected. It is a useful artifact-ordering check, not a cryptographic provenance system.

## 9. Tests, smoke work, and reproducibility limits

The test suite covers schema validation and offsets, candidate merging, boundary preservation, calibration, taxonomy linking, graph construction, job/gap/course flow, Mistral data preparation, profile evidence retention, cached-context hash stability, pipeline gates, smoke stages, and non-regression selection/fallback. On 2026-09-02, the focused suite was rerun with `PYTHONPATH=src python -m pytest -q tests/skillllm_v2 tests/test_skillllm_v2_stage2.py`: 50 tests passed in 1.34 seconds under Python 3.13.12. Two legacy aggregate-test assertions were aligned with the current conservative boundary-selector contract: unsupported trims/splits preserve the original span, while an exact alias-backed normalization may replace it. Artifact status markers record successful component tests and full evaluation stages, plus explicitly named smoke stages for graph, matching/recommendation, taxonomy linking, and end-to-end demonstrations.

The audit did not treat “success” status files as quality claims. In particular, graph and matching success records coexist with unavailable ranking metrics, the GLiNER baseline remains zero-shot, the supervision required for a taxonomy reranker is unavailable, and the Stage-2 working tree has no commit. The thesis must present the code as a reproducible local workflow with persisted artifacts and explicit limitations, not as a production deployment or a fully evaluated job-matching system.

## 10. Thesis-safe conclusion

The defensible central finding is that the 80-epoch Mistral LoRA extractor is the practical extraction anchor in this artifact snapshot. Additional components are useful only when they preserve its provenance and are gated against material extraction regression; their demonstrated downstream contribution is structured normalization, explainable data flow, and a testable architecture, not verified improvement in job-match or course-recommendation quality. The strongest future work is a labelled taxonomy-linking set, labelled job/course relevance evaluation, multi-seed extraction experiments, and versioned source control.
