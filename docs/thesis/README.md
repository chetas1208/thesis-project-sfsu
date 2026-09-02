# SkillLLM_v2 Technical Research Report

LaTeX source for the thesis-quality SkillLLM_v2 research report prepared for Professor Chen. The title page intentionally leaves author, final course designation, committee, and official-template fields for departmental confirmation.

## Contents

- `SkillLLM_v2_Thesis.tex` — current report entry point (11 chapters, 3 appendices)
- `SkillLLM_v2_Thesis.pdf` — current compiled report
- `codebase_audit.md` — code-to-artifact audit and claim boundaries
- `THESIS_QA.md` — compilation and document QA record
- `NEXT_RESEARCH_UPDATE.md` — advisor-facing one-page progress update
- `figures/` and `tables/` — reproducible TikZ/PGFPlots figures and LaTeX tables
- `references.bib` — thesis bibliography (69 entries; use with `\bibliography{references}`)
- `SkillLLM_reference_index.md` — readable index of all citation keys
- `build.sh` — local PDF build script

Optional copies: `Ref.bib` and `SkillLLM_references.bib` mirror `references.bib`.

`main.tex` and the non-`_v2` modular files are retained as legacy drafts. The `_v2` modular files are inputs to `SkillLLM_v2_Thesis.tex` and should be used for revisions.

## Companion files (outside PDF)

- `../thesis_evidence_report.md` — claim-to-artifact audit trail
- `../thesis_missing_admin_info.md` — administrative checklist

## Build

Requires Tectonic, or a TeX distribution with `latexmk`, or `pdflatex` plus `bibtex`:

```bash
cd docs/thesis
bash build.sh
```

Output: `docs/thesis/SkillLLM_v2_Thesis.pdf`

Set `TECTONIC_BIN=/path/to/tectonic` to use a non-default Tectonic installation.

## Format compliance

This source follows Graduate Studies guidance for margins and structure but is **not** a department-distributed official template. Complete preliminary and final format checks with the Graduate Program Specialist before library submission.

## Evidence policy

The main report's quantitative results trace to:

- `SkillLLM_v2/results/skillspan_baselines/` (legacy multi-LLM benchmark)
- `SkillLLM_v2_stage2/artifacts/skillllm_v2/` (Stage-2 system metrics)

See `codebase_audit.md` for the full Stage-2 mapping and the explicit separation between primary, historical, smoke, and unavailable evidence.
