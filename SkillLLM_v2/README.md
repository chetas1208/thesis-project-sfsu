# SkillLLM v2 — Earlier Implementation

This directory preserves the earlier SkillLLM implementation that accompanies the original skill-extraction work. It is useful for provenance, comparison, and reproducing the upstream-style experiments; new end-to-end development should generally use [`SkillLLM_v2_stage2/`](../SkillLLM_v2_stage2/).

## What is here

- `src/`: model training, validation, generation, and GLiNER experiment code.
- `configs/`: experiment configuration files.
- `tests/`: tests for the legacy implementation.
- `Extract_jobDesc/`: the companion asynchronous job-description collection tool.
- `data/`, `models/`, and run outputs: local runtime state; large/private artifacts are intentionally excluded from Git.

The implementation targets the original SkillSpan/Skill-LLM workflow and may require the exact model access and dependency versions used by the associated experiment. It is not the source of the audited Stage-2 metrics in the root README.

## Setup

Create an isolated environment and install the pinned requirements:

```bash
cd SkillLLM_v2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model downloads may require an authenticated Hugging Face account. Keep credentials in environment variables or a local, ignored `.env` file; never paste tokens into scripts, configs, notebooks, commits, or issues.

## Experiment entry points

Inspect the scripts and configuration selected for a particular reproduction before running it. Typical components are:

```text
src/
├── data/          dataset preparation helpers
├── gliner/        GLiNER experiments and notebooks
└── ...            training, generation, and evaluation modules
```

Because this is a preserved earlier implementation, paths and checkpoints are intentionally not hard-coded in this README. Prefer the reproducible Stage-2 commands documented in the repository root when you need a supported smoke test or full pipeline.

## Related documentation

- [Repository README](../README.md) — project map, current evidence, and supported workflows.
- [Stage-2 README](../SkillLLM_v2_stage2/README.md) — primary audited implementation.
- [Thesis artifacts](../docs/thesis/README.md) — build and quality-assurance instructions.
