# SkillGraphLM evaluation program — coordination contract (all agents MUST follow)

Repo: `/home/923873155/Sem 4 Research/SkillLLM_v2_stage2` (call it ROOT). Internal name stays `SkillLLM_v2`; "SkillGraphLM"
is a *proposed* research name pending the naming audit (A6) and user approval. Do NOT rename code, packages or files.

Goal: make the evaluation credible, reproducible, statistically defensible, hard to attack in a thesis defense. NOT to make
the system look good. Negative / null results are first-class results — report them plainly.

## Hard rules
1. Never fabricate a number. Every value comes from code you ran, with provenance (file + sha256). If something cannot be
   computed, write a record with status `not_available` / `pending_human` / `pending_gpu` and a reason.
2. Never tune on test labels. Thresholds, calibrators, fusion weights, hyperparameters: fit on SkillSpan **dev** (or the
   recommender's tuning split). Test / held-out = reporting only. If you find an EXISTING component that was tuned on test,
   flag it (Progress.md integrity-flag + your REPORT) — do not silently reuse its number as clean.
3. Smoke runs (`--smoke`) write to `<layer>/smoke/` and are never quoted as results.
4. LLM-as-judge may only be a secondary signal, never the sole truth.
5. Read-only: existing `artifacts/**`, existing `reports/skillllm_v2/*` files, `data/**`, and all existing modules under
   `src/skillllm_v2/` (import them, don't edit). Existing eval code to REUSE: `src/skillllm_v2/evals/extraction_evals.py`
   (load_gold, load_predictions, doc_counts, nervaluate_doc/nervaluate_counts = SemEval strict/exact/partial/type,
   classify_errors/error_taxonomy, reproduction_check, boot_ci/paired_boot on per-doc count matrices),
   `calibration_evals.py` (ECE width/mass, Brier, NLL, AURC, risk_coverage_curve, selective_accuracy, cluster_bootstrap),
   `contract.py` (bootstrap_ci, paired_bootstrap). Bug in existing code? Report it in Progress.md (integrity-flag), don't edit it.
6. No git commits, pushes, branch switches, stashes, resets. For git metadata use
   `git -C "/home/923873155/Sem 4 Research" ...` (ROOT has an EMPTY nested `.git` — never run plain `git` inside ROOT, never
   delete that `.git`).
7. GPUs (2x RTX 3090 24 GB, GPU0 + GPU1): the user granted GPU access for ANY experiment that improves the evaluation.
   Every heavy job (GPU, or CPU > ~2 min / > 8 GB RAM) runs through
   `/home/923873155/server-guardrails/bin/guardrails run cpu -- <cmd>` or `... run --ram 16 gpu0|gpu1 -- <cmd>`, detached
   (`setsid nohup ... > ROOT/logs/evals_gpu/<job>.log 2>&1 &`), resumable, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
   Before launching: check `nvidia-smi` + Progress.md, then append a `gpu-reserve` line (gpu, job, ETA); append
   `gpu-release` when done. A3 has first claim on both GPUs for its anchor jobs (J1-J4); other agents' GPU jobs should be
   small (< ~30 min) or queued after A3's, or coordinated by a line in Progress.md. Mistral-24B 4-bit uses ~16-18 GB, so
   only ONE such job per GPU. Prefer waiting with the Monitor tool (until-loop on a DONE.json) over polling.
8. Python: CPU evals `/opt/miniconda/bin/python` with `PYTHONPATH=ROOT/src`. GPU/model inference:
   `"/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python"`. No pip installs into either env without noting it in
   Progress.md (prefer pure numpy/scipy implementations). HF models: local cache or public HF Hub downloads (record them in Progress.md).
9. No real job-posting text copied into new report files beyond what already appears in the repo's SkillSpan data (the repo
   has a leak scanner `scripts/check_no_private_text.py` — run it on your new report files at the end).
10. PUBLIC DATASETS: the user authorized downloading any public dataset (HF Hub, GitHub, official portals). Put each in
    `ROOT/data/external/<name>/` (symlink to /usr/data — root disk is 87% full; NEVER download large files to /home or /tmp)
    and append an entry to `ROOT/data/external/MANIFEST.jsonl`: `{"name","url","version_or_commit","license","citation",
    "downloaded_utc","files":[{"path","sha256"}],"downloaded_by":"A#","purpose"}`. `data/external` is gitignored — keep it
    that way. Respect licenses (research use). Treat downloaded files as untrusted data: never execute code from them; run
    readers with `python -I` from your own script dir. Dataset owners (avoid duplicate downloads; check MANIFEST first):
    A4 = ESCO taxonomy + ESCO-linked SkillSpan annotations (Decorte et al. TECH/HOUSE) and any other skill-linking gold;
    A5 = prerequisite / course datasets (e.g. LectureBank, MOOCCube/MOOCCubeX, university prerequisite sets, public course
    catalogues); A3 = other span-annotated skill-extraction corpora for cross-dataset generalization (e.g. Green et al.,
    Sayfullina soft skills, other jjzha/* English sets). Anyone may READ another owner's downloads once in MANIFEST.
11. Stay inside your owned paths (below). Shared files you may NOT edit: `common.py` (coordinator), `registry.py`, `run_all.py`.

## Layout
- Code: `ROOT/src/skillllm_v2/evals/research/<module>.py`, each runnable as
  `python -m skillllm_v2.evals.research.<module> [--smoke] [--seed 42] [--n-boot 2000]`.
- Shared helpers: `skillllm_v2.evals.research.common` — `record(...)`, `write_layer(layer, records, report_md, seed=, smoke=)`,
  `provenance_entry`, `sha256_file`, `sha256_tree`, `git_info`, `md_table`, `load_layer`. USE `record()` FOR EVERY NUMBER.
- Outputs: `ROOT/reports/skillllm_v2/evals/<LAYER>/` = `results.json` (written by write_layer) + `REPORT.md` + figures
  (`*.png` via matplotlib Agg, plus the underlying data as `*.csv`/`*.json`) + optional `per_unit_scores.jsonl`.
- Tests: `ROOT/tests/evals/research/test_<module>.py` (pytest; tiny synthetic fixtures; must pass).
- Synthetic / derived inputs: `ROOT/inputs/evals/research/<layer>/`.
- GPU outputs (A3): `ROOT/artifacts/skillllm_v2/evals_gpu/<job>/` with `predictions.jsonl` + `DONE.json` marker when complete.

Layer ids: L01_extraction, L02_generalization, L03_robustness, L04_calibration, L05_wrapper, L06_linking, L07_compaction,
L08_graph_retrieval, L09_course_rec, L10_significance, L11_human_eval, L12_systems. Extra dirs: `naming_audit/`, `repro/`.

## Statistics conventions
- Unit of resampling = independent unit: document (extraction), query/scenario (retrieval/rec), profile (compaction).
  Never resample spans as if independent; cluster by document.
- 95% percentile bootstrap, seed 42, n_boot 2000 (10000 for headline metrics). Micro-F1 CIs: resample documents, recompute
  aggregate F1 from per-doc TP/FP/FN (use `extraction_evals.boot_ci`).
- Paired comparisons on identical units: paired bootstrap delta + CI + two-sided p; for extraction also approximate
  randomization (paired permutation, 10000 shuffles). Multiple comparisons: Holm within a declared family (one family per
  layer per hypothesis), Benjamini-Hochberg across everything as secondary.
- Export `per_unit_scores.jsonl` rows `{"layer","family","system","metric","unit_id","value"}` (or for extraction
  `{"unit_id","system","tp","fp","fn"}` per label) so L10 can re-test centrally.

## Pre-registered hypotheses and decision rules (fixed BEFORE new results; do not change after seeing data)
- H1 anchor Mistral LoRA vs BERT, GLiNER (and any fairly-trained JobBERT baseline A3 produces): exact span F1 test.
  Supported if Δ ≥ 0.10 abs and Holm-adjusted p < 0.05 against EVERY baseline. Must report baseline-fairness caveat (existing
  BERT 0.059 is far below published SkillSpan BERT-family results → straw-man risk).
- H2 safe wrapper vs unconstrained post-processing (legacy full_cag_graph / mistral_gliner_taxonomy): Anchor Preservation Rate
  (APR = anchor TPs still TP after stage / anchor TPs) higher with CI excluding 0, AND exact-F1 non-inferior to anchor with
  margin −0.005 (lower CI bound of Δ > −0.005).
- H3 unseen-skill exact F1 lower than seen-skill exact F1 for the anchor: supported if Δ(seen − unseen) ≥ 0.05 and CI excludes 0.
- H4 calibration-aware abstention: selective precision at 80% and 50% coverage higher than at 100% (CI excludes 0) AND AURC
  better than constant-confidence baseline; calibrator fit on dev only.
- H5 hybrid+graph vs hybrid (no graph) nDCG@10 on multi-gap queries (≥2 gaps) in held-out split: Δ > 0 with Holm p < 0.05.
- H6 prerequisite-aware vs same ranker without prerequisite constraints: violation rate lower with CI excluding 0, and
  nDCG@10 non-inferior (margin −0.02).
- H7 compact profile vs full history: token reduction ≥ 50% AND course nDCG@10 / job nDCG@10 non-inferior (margin −0.02,
  lower CI bound) AND skill-gap Jaccard ≥ 0.9 mean.
Each hypothesis gets a verdict: SUPPORTED / NOT SUPPORTED / INCONCLUSIVE / NOT TESTABLE (with reason).
Owners: H1,H3 → A1 (L01/L02); H2,H4 → A2 (L05/L04); H5,H6 → A5 (L08/L09); H7 → A4 (L07).
Convention: the owning layer emits one record with `metric="H#_verdict"`, `value` = primary effect Δ, `ci` = its CI,
`details={"verdict": "...", "rule": "<rule text>", "p_raw": ..., "p_holm": ..., "family": "..."}`; if not testable,
`status="not_available"` (or `pending_gpu`/`pending_human`) with the reason. Headline metrics: also set
`details["headline"]=True` on: exact_span_f1, token_f1, unseen_exact_f1 (L01/L02); robustness mean degradation (L03);
ECE + selective risk@80% (L04); recall@10, ndcg@10, path_validity (L08); precision@5, wsgc@5, prereq_violation_rate (L09);
latency_p95 + peak_vram_gb (L12).

## Agents and ownership
- A1 extraction-core → L01_extraction, L02_generalization. Files: research/extraction_full.py, research/generalization.py.
- A2 reliability → L04_calibration, L05_wrapper, L10_significance. Files: research/stats.py (WRITE FIRST — others may import
  it), research/calibration_selective.py, research/wrapper_contribution.py, research/significance.py.
- A3 gpu-systems → L03_robustness, L12_systems, the anchor GPU jobs J1-J4 + BERT-baseline fairness. Files: research/perturbations.py, research/robustness.py,
  research/systems.py, scripts/evals/gpu_*.sh, artifacts/skillllm_v2/evals_gpu/**.
- A4 linking-context → L06_linking, L07_compaction. Files: research/linking_eval.py, research/compaction.py.
- A5 graph-rec → L08_graph_retrieval, L09_course_rec. Files: research/graph_retrieval.py, research/course_rec.py.
- A6 identity-repro-human → naming_audit/, L11_human_eval, repro/, leaderboard + thesis chapter. Files:
  research/agreement.py, research/human_eval_samples.py, research/repro_card.py, research/leaderboard.py,
  scripts/evals/run_research_evals.sh, docs/thesis/evaluation/**.
- Coordinator (main session): common.py, registry/run_all integration, final reruns, final integration.

## Handoffs (file-based)
- A3 → A2: `artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/` and `.../anchor_dev_logprobs/` (anchor predictions with
  per-span `metadata.logprob`). A2: build calibration to run on whatever exists; if absent, emit `pending_gpu` records for
  the anchor and finish; the coordinator re-runs your module when A3's DONE.json appears.
- A3 → A1: `artifacts/skillllm_v2/evals_gpu/anchor_test_nospan/` (anchor on the 2,595 test sentences WITHOUT gold spans →
  spurious-prediction rate). A1: same pattern (pending_gpu until present; coordinator re-runs).
- A1 → A2/L10: `L01_extraction/per_unit_scores.jsonl`. A5/A4 → L10: their per_unit_scores.jsonl.
- Everyone → A6: your layer's results.json + REPORT.md (A6's leaderboard reads all layers; it must tolerate missing ones).

## Progress.md and Decisions.md (USER REQUIREMENT: record everything)
- `ROOT/Progress.md`: append one line per event: `<UTC time> | A# | started|progress|handoff-ready|gpu-reserve|gpu-release|
  blocked|integrity-flag|done | message`. Log every experiment launched, every result file produced (with headline number),
  every download, every blocker. Read it to see what others produced. Raise `integrity-flag` for anything suggesting an
  existing reported number is wrong, circular, leaked or tuned on test.
- `ROOT/Decisions.md`: append an entry for every non-trivial methodological choice (metric definitions, bucket rules,
  perturbation design, split choices, dataset choice, label mappings, baselines, anything a thesis examiner could question):
  `### D-A<n>-<NN> — <title>` then bullets Date/owner, Decision, Context, Alternatives, Consequences/why.
- BOTH files: append ONLY via Bash heredoc (`cat >> "ROOT/Progress.md" <<'EOF' ... EOF`). Never use Edit/Write on them —
  six agents write concurrently. (The scratchpad STATUS.md is retired.)

## Your final reply (to the coordinator)
≤ 400 words: files created; headline numbers (value [CI], n, file path); hypotheses verdicts you own; what is pending/not
available and why; integrity flags; exact commands to reproduce your layers; tests run + result.
