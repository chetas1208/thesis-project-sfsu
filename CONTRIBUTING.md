# Contributing

Contributions should preserve the repository’s evidence-first research contract: source spans remain auditable, generated claims remain tied to artifacts, and unavailable evaluations are reported as unavailable.

## Before changing code

1. Read the relevant subsystem README and [`docs/thesis/codebase_audit.md`](docs/thesis/codebase_audit.md).
2. Keep changes scoped to the subsystem being improved.
3. Do not add credentials, personal data, model weights, virtual environments, logs, or generated output trees.

## Local checks

From `SkillLLM_v2_stage2/`:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/skillllm_v2 tests/test_skillllm_v2_stage2.py
```

For shell or Python changes, also run `bash -n` on affected shell launchers and compile the affected Python files without writing bytecode. For thesis changes, run `bash docs/thesis/build.sh` with an available TeX engine and update the QA record when the document structure or evidence changes.

## Evidence and evaluation discipline

- A smoke or demo result proves execution, not model quality.
- The Mistral full-test artifact is the extraction anchor for the current Stage-2 snapshot.
- Wrapper variants must be evaluated through the non-regression gate.
- Linking coverage must not be described as supervised linking accuracy.
- Job-ranking nDCG@10 and course-ranking P@5 remain unavailable until labelled relevance data exists.

## Pull requests and commits

Use a focused commit with a descriptive message. Explain which files changed and why, list validation commands, and call out any assumptions or remaining warnings. Keep unrelated generated changes out of the commit. Do not force-push shared branches.
