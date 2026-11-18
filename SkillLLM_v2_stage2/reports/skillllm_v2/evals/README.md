# Layered research evaluation (L01-L12)

Entry points: `LEADERBOARD.md` (headline metrics, H1-H7 verdicts, layer status), `<layer>/REPORT.md` and `<layer>/results.json`
per layer, `repro/EVAL_CARD.md` (what was run, on what, with which hashes), `naming_audit/REPORT.md`, `CONTRACT.md` (rules every
layer followed). Methodology and results chapter: `docs/thesis/evaluation/`. Event log and design decisions: `Progress.md`,
`Decisions.md` at the repository subtree root.

## Not committed (regenerable bulk)

To keep repository history small these files are produced locally but not committed:

- `<layer>/smoke/` (smoke runs, never results),
- `*/per_unit_scores.jsonl`, `L06_linking/per_mention_outputs.jsonl`, `L05_wrapper/ladder/per_doc_counts.csv`
  (per-document / per-mention score dumps, about 110 MB in total).

Regenerate with `scripts/evals/run_research_evals.sh` (every layer, then significance, leaderboard and evaluation card), or one layer
with `PYTHONPATH=src python -m skillllm_v2.evals.research.<module>`. The significance layer (L10) reads the per-unit dumps, so rerun
the layers before it. Model predictions and weights live under `artifacts/` and public datasets under `data/external/`; neither is
tracked (see `data/external/MANIFEST.jsonl` locally for versions, licenses and sha256).
