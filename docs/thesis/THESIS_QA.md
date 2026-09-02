# Thesis QA Record

## Build result

- [x] Entry point: `SkillLLM_v2_Thesis.tex`.
- [x] PDF: `SkillLLM_v2_Thesis.pdf`.
- [x] Build command: `TECTONIC_BIN=/path/to/tectonic bash build.sh`.
- [x] Build completed successfully with Tectonic and BibTeX on 2026-09-02.
- [x] PDF metadata and renderer inspection report 75 letter-size pages.

## Document checks

- [x] 11 numbered chapters and 3 appendices are present.
- [x] 15 reproducible TikZ/PGFPlots figures and 5 LaTeX tables are present.
- [x] All 53 cited bibliography keys resolve in `references.bib`; the bibliography source contains 69 records.
- [x] Static input-path and citation-key checks passed.
- [x] Focused Stage-2 suite passed: 50 tests in 1.34 seconds with Python 3.13.12.
- [x] No unresolved citation or cross-reference messages were found in the final log.
- [x] Visual inspection covered the title page, system diagram, formulas, training diagram, results/context pages, conclusion, appendix paths, artifact index, and bibliography end.
- [x] The initially overlapping artifact-index table was redesigned with breakable paths and re-inspected.

## Scope and claim checks

- [x] Primary extraction values are copied from persisted Stage-2 metric files.
- [x] Historical wrapper diagnostics are explicitly separated from current primary metrics.
- [x] Linking is labelled coverage, not supervised accuracy.
- [x] Job and course ranking effectiveness are labelled unavailable because nDCG@10 and P@5 are `NA`.
- [x] The title page does not invent an author, course designation, committee, or approval-page format.

## Remaining submission inputs

- [ ] Replace the administrative placeholders only after the department confirms the author, final course route, committee, and required official template.
- [ ] If a formal library submission is required, complete its template-specific formatting review and enrich any bibliography records whose venue metadata is not yet available locally.

The TeX log retains non-fatal box warnings from dense tables and long technical text, plus BibTeX warnings about missing venue fields in imported bibliography records. The final rendered pages sampled above contain no clipped table content, overlapping objects, unresolved references, or placeholder text.
