# SkillLLM v2 Stage-2

This directory contains the primary, audited SkillLLM v2 implementation. It is a provenance-aware research vertical slice for skill extraction and downstream career-intelligence data flow.

The system is intentionally conservative: the Mistral extractor is the source-of-truth anchor, auxiliary candidates are admitted only through explicit policies, and the safe wrapper falls back to the untouched anchor when configured non-regression floors are not met.

## Capabilities

- Mistral Small 3.1 24B Instruct extraction with exact source-span alignment.
- Optional GLiNER and dictionary candidates with source-aware merging.
- Boundary preservation, span validation, and confidence calibration.
- Attach-only canonical linking through local aliases and taxonomy interfaces.
- Evidence-preserving profile summaries and deterministic context packets.
- In-memory NetworkX graph construction and path explanations.
- Deterministic job matching, skill-gap detection, and course recommendation.
- Pipeline stage gates, artifact tracking, GPU preflight, and non-regression evaluation.

## Use as a Python library

The final model ships as an installable, typed package (`pip install .`; extras `[model]`, `[eval]`, `[recommend]`, `[test]`). The core install and the model-free offline mode need only the standard library.

```python
from skillllm_v2 import SkillLLM
SkillLLM.offline().extract("Python, SQL and communication skills")            # no model, flagged offline
SkillLLM.from_pretrained(base_dir, adapters=[a, b, c], min_votes=2).extract(text)  # 3-adapter majority vote (3x generation cost)
```

CLI: `skillllm extract --offline --text "..."`, `skillllm doctor`, `skillllm version`. Install, result schema, the `min_votes` precision/recall dial (held-out P 0.80/R 0.51 at 3 votes, P 0.72/R 0.66/F1 0.690 at 2 votes versus anchor F1 0.670), the honest performance table with caveats and the compute table are in [`docs/LIBRARY.md`](docs/LIBRARY.md); the adapter card template is [`MODEL_CARD.md`](MODEL_CARD.md).

## Layout

```text
src/skillllm_v2/
├── extract/       Mistral, GLiNER, dictionary, and candidate merging
├── verify/        boundary, span, and confidence controls
├── normalize/     alias-backed canonical attachment
├── context/       profile evidence and cache compaction
├── graph/         in-memory graph and explanations
├── match/         job, gap, and course flows
├── eval/          metrics, provenance, wrapper gate, and reports
├── pipeline/      stage registry, smoke gates, and orchestration
└── train/         SkillSpan data builders and training adapters
configs/           smoke, ablation, guardrail, and training YAML files
scripts/           reproducible shell entry points
tests/             focused unit and integration tests
data/source/       checked-in SkillSpan train/dev/test export
inputs/demo/       small demo jobs and course records
reports/           checked-in report snapshots; runtime outputs are ignored
```

## Setup

Use Python 3.12 or 3.13 and install the repository requirements in an isolated environment:

```bash
cd SkillLLM_v2_stage2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The checked-in SkillSpan export is sufficient for tests and smoke execution. The large Mistral weights and adapters are not stored in Git; set their local paths in the selected config before a model-backed run.

## Tests and smoke execution

```bash
PYTHONPATH=src python -m pytest -q \
  tests/skillllm_v2 tests/test_skillllm_v2_stage2.py

PYTHONPATH=src python -m skillllm_v2.pipeline.pipeline_runner \
  --config configs/skillllm_v2_smoke.yaml --mode smoke

bash scripts/run_skillllm_v2_end_to_end_demo.sh
```

The focused suite covers schemas, extraction data builders, merging and refinement, calibration, linking, context, graph/matching/course flows, pipeline gates, and safe-wrapper selection. Smoke stages update the checked-in `reports/skillllm_v2/` snapshots; the standalone demo JSON is ignored as generated runtime state.

## Reproducible stages

Useful entry points include:

```bash
# Dataset builders
bash scripts/build_mistral_sft_data.sh
bash scripts/build_skillspan_bio_dataset.sh
bash scripts/build_skillllm_v2_verifier_dataset.sh

# Cached-anchor evaluation and guarded wrapper
bash scripts/eval_raw_mistral_lora.sh
bash scripts/run_skillllm_v2_safe_wrapper.sh
bash scripts/generate_skillllm_v2_money_table.sh
```

The safe-wrapper command consumes cached Mistral predictions and does not retrain Mistral. It is the correct path for evaluating boundary, GLiNER/taxonomy, and full CAG/graph variants against the persisted anchor.

For the complete 80-epoch, two-GPU workflow:

```bash
bash scripts/run_skillllm_v2_80epoch_2gpu_pipeline.sh
```

That launcher runs GPU preflight, tests, smoke gates, training, evaluation, report generation, and failed-status-marker checks. It expects two visible GPUs and a compatible local PyTorch environment. Use a lease-aware or otherwise durable session for a long run.

## Evidence boundary

The current report snapshot records raw Mistral exact-span F1 `0.670069` and full-wrapper exact-span F1 `0.671102`. The wrapper is accepted as non-regressive under configured floors; the small difference is not a general improvement claim. Linking values are coverage-style signals, not supervised entity-linking accuracy. Job-ranking nDCG@10 and course-ranking P@5 are unavailable because labelled relevance data is not persisted.

Read [`../docs/thesis/codebase_audit.md`](../docs/thesis/codebase_audit.md) for the full code-to-artifact mapping and claim rules.

## Data and secrets

Do not commit credentials, private resume data, model weights, checkpoints, logs, or generated output trees. Use local environment variables and the root `.gitignore`. The SkillSpan export includes its own CC BY 4.0 notice; model and dataset licenses remain separate from this repository.
