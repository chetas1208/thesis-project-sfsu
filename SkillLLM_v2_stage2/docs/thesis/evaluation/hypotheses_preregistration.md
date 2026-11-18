# Pre-registration of hypotheses H1–H7 and their decision rules

- **Registered:** 2026-11-17T20:58:14Z (UTC), by agent A6 on behalf of the evaluation coordinator.
- **State of evidence at registration:** `reports/skillllm_v2/evals/` held only `CONTRACT.md`. No layer `results.json`
  existed, so no result of the new evaluation programme had been computed when these rules were written down. Results
  that predate the programme are listed below under *Known before registration*, so a reader can see what the rules
  could have been fitted to.
- **Source of the rules:** `reports/skillllm_v2/evals/CONTRACT.md`, section "Pre-registered hypotheses and decision rules"
  (file sha256 `b0b4d69e835fc9e935f2303c1ce01993dff82a2861d59aa9950562010a9f5fd4`, mtime 2026-11-17 20:49:44 UTC).
  The block below copies that section verbatim. The coordinator's decision D-C05 in `Decisions.md` records the same rules.
- **Repository state:** outer git repository `/home/923873155/Sem 4 Research`, branch `main`, HEAD
  `54aab906d548d871ad855bdbb09b7eea18ddf0d5`. The Stage-2 sources and evaluation code are uncommitted (dirty tree).
  `reports/skillllm_v2/evals/repro/EVAL_CARD.md` records the hashes of every input.
- **Change policy:** the rules are frozen. A later correction can only be appended as a dated *amendment* under
  "Amendments" below, giving the reason and saying whether it was written before or after the relevant data were seen. An
  amended hypothesis is reported under both the original and the amended rule. The original rule's verdict is the one
  quoted as confirmatory.

## Verbatim decision rules (copied from CONTRACT.md)

```text
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
```

## Shared statistical conventions (also fixed in the contract)

- 95 % percentile bootstrap with seed 42 and 2,000 resamples (10,000 for headline metrics). The resampling unit is the
  independent unit: the document for extraction, the query or scenario for retrieval and recommendation, the profile for
  compaction. Spans are never resampled as if they were independent.
- Paired comparisons on identical units use a paired bootstrap delta with CI and a two-sided p-value. Extraction
  comparisons also get approximate randomization (10,000 paired shuffles).
- Multiplicity: Holm within each declared family (one family per layer per hypothesis). Benjamini–Hochberg across all
  tests is reported as a secondary view.
- No tuning on test labels. Thresholds, calibrators and fusion weights are fit on SkillSpan dev or on the recommender's dev
  half only.

## How each verdict is reached

| Verdict | Meaning |
|---|---|
| SUPPORTED | Every condition in the rule holds on the pre-specified population and test. |
| NOT SUPPORTED | The data are informative and at least one condition fails, e.g. Δ < threshold with a CI that excludes the threshold, or the non-inferiority bound fails. |
| INCONCLUSIVE | The CI straddles a decision boundary, or the test is underpowered (the CI includes both "supported" and "not supported" values). |
| NOT TESTABLE | A required input does not exist (for example no GPU predictions yet, no human labels, or no dev split for a calibrator). The reason is stated, and the record status is `not_available`, `pending_gpu` or `pending_human`. |

Each verdict record carries `details.rule` (the text above), `p_raw`, `p_holm` and `family`. The leaderboard
(`reports/skillllm_v2/evals/LEADERBOARD.md`) renders those records unchanged. It never recomputes or re-labels a verdict.

## Known before registration (results that predate the programme)

These numbers were in the repository before the rules were written. They are listed so a reader can judge whether a
threshold could have been chosen to fit them. All of them come from files under `reports/skillllm_v2/` and
`artifacts/skillllm_v2/` as described in `docs/thesis_drafts/evals.md`.

| Hypothesis | Pre-existing evidence | Risk that the rule was fitted to it |
|---|---|---|
| H1 | Anchor exact F1 0.670069 on the 974-document gold-bearing test subset. BERT-lookup 0.0592 and GLiNER zero-shot 0.0758, both scored on all 3,569 rows. | The 0.10 margin is far below the observed gap. The rule mainly guards against a fairly trained baseline (A3) closing it. The straw-man caveat is part of the rule. |
| H2 | Destructive legacy stages 0.5704 / 0.6515 / 0.5616 against the run-1 anchor. Safe variants 0.6724 / 0.6695 / 0.6719. | The direction was known. The APR metric and the −0.005 margin are new and had not been computed. |
| H3 | No seen/unseen split had been computed. | None known. |
| H4 | Constant anchor confidence (AUROC 0.5). A log-probability study existed only as a test-set cross-fit. | The rule requires a dev-fitted calibrator, which did not exist. |
| H5, H6 | Recommender v2 on the held-out half: Full vs strongest baseline nDCG@10 +0.035 [−0.006, +0.072], Holm p 0.094. Prerequisite violation 0.000. | H5 restricts to multi-gap queries and to a hybrid-vs-hybrid+graph ablation that had not been run. For H6 the near-zero violation rate was known. |
| H7 | Context packet larger than the raw text (ratio 10.55 on synthetic profiles). | The rule asks for ≥ 50 % reduction against full history, a comparison not previously made. |

## Amendments

_None as of registration._ Append below with date, author, reason, and whether the data had been seen.
