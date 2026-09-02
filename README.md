# SkillLLM v2 — Evidence-Preserving Skill Intelligence

An end-to-end research repository for extracting skills from employment text, preserving source evidence, attaching canonical skill identities, and exposing explainable career-intelligence workflows.

The project is deliberately evidence-first. A downstream module may enrich a prediction with taxonomy, profile, graph, or matching metadata, but it must not silently rewrite a reliable source span. Quantitative claims in this repository are tied to persisted artifacts and documented evaluation boundaries.

## What is here

| Area | Location | Purpose |
| --- | --- | --- |
| Stage-2 system | [`SkillLLM_v2_stage2/`](SkillLLM_v2_stage2/) | Primary implementation: extraction, guarded composition, normalization, context, graph, matching, recommendation, pipeline gates, and tests |
| Earlier Skill-LLM implementation | [`SkillLLM_v2/`](SkillLLM_v2/) | Upstream-style Skill-LLM code, legacy baselines, evaluation utilities, and the job-description extraction companion |
| Benchmark lab | [`Experiments/`](Experiments/) | Dataset preparation and comparison workflows for BERT/JobBERT, NNOSE, GLiNER, Skill-LLM, Gemma, GPT, and ConTeXT experiments |
| Thesis report | [`docs/thesis/`](docs/thesis/) | Reproducible LaTeX source, figures, tables, bibliography, QA record, and compiled PDF |
| Research notes | [`Project Desc.md`](Project%20Desc.md), [`skillExtract.md`](skillExtract.md), [`skillMatch.md`](skillMatch.md) | Early problem framing and research notes |

## Current research status

The current claim boundary is documented in [`docs/thesis/codebase_audit.md`](docs/thesis/codebase_audit.md). The most important verified results are:

| Result | Value | Interpretation |
| --- | ---: | --- |
| Raw Mistral LoRA exact-span F1 | `0.670069` | Primary extraction anchor on the persisted Stage-2 full-test artifact |
| Raw Mistral LoRA token F1 | `0.787886` | Token-overlap measure; not a replacement for exact-span evaluation |
| Full wrapper exact-span F1 | `0.671102` | Accepted as non-regressive under configured floors; not evidence of a general improvement |
| BERT baseline exact-span F1 | `0.059150` | CPU-oriented baseline artifact |
| GLiNER exact-span F1 | `0.075833` | Zero-shot/few-shot baseline; fine-tuning was not completed |
| Linking coverage | `0.890374` | Attach-only canonical linking coverage, not supervised linking accuracy |
| Job ranking / course ranking | unavailable | nDCG@10 and P@5 labels are not persisted |
| Focused tests | `50 passed` | Python unit/smoke suite for the Stage-2 vertical slice |

These numbers describe one stored artifact snapshot. They do not establish multi-seed variance, production readiness, hiring validity, or generalization beyond the evaluated SkillSpan-derived data.

## System overview

```text
source text
    │
    ▼
Mistral extraction anchor ──┐
    │                        ├─ guarded candidate merge/refinement ── validation
    ├─ GLiNER candidates ───┘          │
    └─ dictionary candidates           ▼
                                attach-only taxonomy linking
                                         │
                         ┌───────────────┼────────────────┐
                         ▼               ▼                ▼
                 profile/context     skill graph      job/course flows
                         │               │                │
                         └───────────────┴────────────────┘
                         evidence-backed explanations
```

The safe-wrapper gate compares candidate variants with the cached Mistral anchor. If a variant violates its configured exact-span or soft-skill floors, the untouched anchor is selected. This is a research safeguard, not a claim that every auxiliary component improves extraction.

## Repository map

### `SkillLLM_v2_stage2/` — primary implementation

- `src/skillllm_v2/extract/` — Mistral, GLiNER, dictionary extraction, and candidate merging.
- `src/skillllm_v2/verify/` — boundary preservation, span validation, and confidence calibration.
- `src/skillllm_v2/normalize/` — alias-backed canonical attachment and taxonomy interfaces.
- `src/skillllm_v2/context/` — evidence-preserving profiles, context compaction, and deterministic cache packets.
- `src/skillllm_v2/graph/` — in-memory NetworkX graph construction and path explanations.
- `src/skillllm_v2/match/` — deterministic job matching, gap detection, and course recommendation.
- `src/skillllm_v2/pipeline/` — stage registry, smoke gates, artifact tracking, GPU preflight, and orchestration.
- `src/skillllm_v2/eval/` — extraction metrics, provenance checks, wrapper selection, error analysis, and report generation.
- `configs/` — smoke, Stage-2, ablation, wrapper-guardrail, and 80-epoch two-GPU configurations.
- `scripts/` — reproducible entry points for data preparation, evaluation, smoke tests, and training.
- `tests/` — focused unit and integration tests.
- `data/source/skillspan_hf/` — checked-in SkillSpan train/dev/test export and its CC BY 4.0 notice.

### `Experiments/` — benchmark and data-preparation lab

The benchmark normalizes data into a canonical schema, exports model-specific inputs, runs selected baselines, and writes comparable span/document metrics. The self-contained implementation lives in [`Experiments/skill_extraction_benchmark/`](Experiments/skill_extraction_benchmark/). The outer `Experiments/src/` tools support broader dataset preparation and model adapters, including the private resume-data workflow.

### `SkillLLM_v2/` — earlier implementation and ingestion companion

This directory retains the upstream-style Skill-LLM implementation and legacy evaluation utilities. [`Extract_jobDesc/`](SkillLLM_v2/Extract_jobDesc/) is an asynchronous Playwright-based job-description ingestion pipeline with optional LLM-assisted navigation and cleaning. It is useful research infrastructure, but it is not the Stage-2 extraction anchor.

### `docs/thesis/` — report source and PDF

The current entry point is [`SkillLLM_v2_Thesis.tex`](docs/thesis/SkillLLM_v2_Thesis.tex). The compiled report is [`SkillLLM_v2_Thesis.pdf`](docs/thesis/SkillLLM_v2_Thesis.pdf). [`THESIS_QA.md`](docs/thesis/THESIS_QA.md) records compilation, citation, visual, and claim-boundary checks.

## Quick start: Stage-2 smoke path

The smoke path uses deterministic fallback behavior where configured and does not require downloading the large local Mistral model.

```bash
cd SkillLLM_v2_stage2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

PYTHONPATH=src python -m pytest -q \
  tests/skillllm_v2 tests/test_skillllm_v2_stage2.py

# Run all registered smoke stages.
PYTHONPATH=src python -m skillllm_v2.pipeline.pipeline_runner \
  --config configs/skillllm_v2_smoke.yaml --mode smoke

# Or run the small end-to-end demo.
bash scripts/run_skillllm_v2_end_to_end_demo.sh
```

Smoke reports are written under the checked-in `reports/skillllm_v2/` snapshot tree; the standalone demo JSON is runtime output and is ignored by Git.

## Full Stage-2 evaluation

Full evaluation requires the local model and adapter paths declared by the selected YAML file. A model-backed run must not silently fall back to keyword extraction.

```bash
cd SkillLLM_v2_stage2
source .venv/bin/activate
export PYTHONPATH=src

# Build a dataset for the Mistral SFT path.
bash scripts/build_mistral_sft_data.sh

# Run a raw cached-anchor evaluation when the trained adapter is available.
bash scripts/eval_raw_mistral_lora.sh

# Apply the non-regression-gated wrapper to cached anchor predictions.
bash scripts/run_skillllm_v2_safe_wrapper.sh

# Generate the consolidated report table.
bash scripts/generate_skillllm_v2_money_table.sh
```

For the durable 80-epoch, two-GPU workflow, use `scripts/run_skillllm_v2_80epoch_2gpu_pipeline.sh`. It performs GPU preflight, tests, smoke gates, training, evaluation, report generation, and failed-status-marker checks. It expects two visible GPUs and a compatible local PyTorch environment; do not start it casually on a CPU-only machine.

## Benchmark workflow

The benchmark lab is separate from the Stage-2 system because its datasets, model protocols, and metrics are not interchangeable with the thesis table.

```bash
cd Experiments/skill_extraction_benchmark
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

# Supply credentials through the shell or a local, ignored .env file.
export HF_TOKEN='your-token-in-the-environment'
bash scripts/40_run_all.sh
```

The benchmark scripts prepare canonical data, export model inputs, train or invoke selected models, generate predictions, and build comparison reports. Read [`BENCHMARK_GUIDE.md`](Experiments/skill_extraction_benchmark/BENCHMARK_GUIDE.md) before launching a long run.

## Build the thesis PDF

```bash
cd docs/thesis
TECTONIC_BIN=/path/to/tectonic bash build.sh
```

The script uses Tectonic when available, then falls back to `latexmk` or `pdflatex` plus BibTeX. It writes `SkillLLM_v2_Thesis.pdf` beside the entry-point `.tex` file. The title page intentionally retains administrative fields that must be confirmed with the department; do not invent an author, committee, or official template.

## Data, artifacts, and reproducibility

Checked-in data is limited to the public SkillSpan-derived exports and small demo inputs needed by the source tree. The following remain local and are ignored by [`.gitignore`](.gitignore):

- API keys and all `.env` files.
- Python virtual environments, caches, bytecode, and runtime logs.
- Downloaded model weights, adapters, checkpoints, and generated predictions.
- The private/raw resume corpus and its derived files.
- External model/artifact symlinks that would not work in a fresh clone.

This policy keeps the repository cloneable and prevents credentials or personal data from entering Git history. Persisted reports document the expected artifact paths and the evidence available for the audited snapshot; regenerate local artifacts with the scripts when needed.

## Security and contribution rules

- Never place a token, password, private key, or personal dataset in a tracked file.
- Copy an `.env.example` file to `.env` and populate it locally; `.env` is ignored.
- Run the focused tests and syntax checks before submitting changes.
- Keep extraction provenance and claim boundaries intact. Do not turn smoke/demo outputs into full-test claims.
- Do not commit model weights, virtual environments, logs, checkpoints, or generated result trees.

See [`SECURITY.md`](SECURITY.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the detailed workflow.

## License and third-party data

The repository code and documentation are distributed under the root [`LICENSE`](LICENSE). The checked-in SkillSpan export carries its own CC BY 4.0 notice. Upstream model weights, datasets, and hosted APIs retain their separate licenses and access requirements; consult the corresponding project documentation before redistribution.

## Citation

For the research context and complete bibliography, use the report source and bibliography under [`docs/thesis/`](docs/thesis/). The repository intentionally does not invent author or committee metadata; complete those fields only after departmental confirmation.
