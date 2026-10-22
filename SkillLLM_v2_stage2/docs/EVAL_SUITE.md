# SkillLLM_v2 evaluation suite

One command runs every module-level evaluation, writes a master table, checks that the evaluations themselves behave (controls),
cross-checks our hand-written metrics against reference libraries when available, and fails if something regressed.
Everything runs offline on CPU; no GPU, no model weights.

## Running it

```bash
scripts/run_eval_suite.sh                       # full run via the guardrails CPU launcher (~5 min); creates the baseline on the first full run
scripts/run_eval_suite.sh --smoke               # tiny subsets (~20 s); writes to reports/skillllm_v2/eval_suite/smoke/, never touches the baseline
scripts/run_eval_suite.sh --only match,gap,e2e  # a subset (names or *_evals stems); only those modules are gated
scripts/run_eval_suite.sh --update-baseline     # accept the current numbers as the new baseline (full run only)

# without the wrapper:
PYTHONPATH=src python -m skillllm_v2.evals.run_all [--only mod,mod] [--smoke] [--seed N] [--out DIR] [--n-boot N]
                                                   [--baseline PATH] [--tolerance 0.03] [--no-gate] [--no-crosscheck] [--list]
```

Outputs in `reports/skillllm_v2/eval_suite/`:

| file | content |
|---|---|
| `REPORT.md` | verdict, run metadata (git commit, seed, timestamps, python), per-module summary, **coverage vs pipeline stages**, **integrity** (every control), **oracle cross-check**, master table (`module / eval / metric / value [95% CI] / n / dataset / status / caveats`), regression gate, failures and skips |
| `results.json` | the same, machine readable, with full caveats and `details` per result |
| `baseline.json` | gated metric values of the last accepted full run |

Exit codes: `0` clean; `1` a control failed, an oracle cross-check disagreed, or a gated metric regressed past tolerance;
`2` an eval module crashed or failed to import; `3` bad usage (unknown module, `--update-baseline` on a smoke or partial run).
A module whose file does not exist yet is reported `not_available` and does not fail the run. A crashing module never stops
the others (its exception and traceback are recorded).

### Regression gate

A metric is gated when its direction is known: `details["direction"]` in `{"higher","lower","none"}` if the eval sets it,
otherwise a name heuristic (f1/recall/precision/ndcg/mrr/... are higher-is-better; violation/error/ece/brier/latency/... are
lower-is-better; anything else is *not gated*). A gated metric regresses when it is worse than baseline by more than
`--tolerance` (absolute, default 0.03). Latency (`hardware_dependent`) regresses only when it exceeds 3x baseline + 5 ms.
Controls are not gated as metrics; they have their own pass/fail. A metric that was `ok` in the baseline and now `failed` is a
regression; one that is merely missing is listed under "dropped". Smoke runs are compared with nothing (the gate is skipped when
`smoke` differs from the baseline's). Different seeds change the synthetic datasets, so a seed change is flagged in the notes.
The tolerance is a guard against code regressions, not a significance test: it ignores the CI of either run.

## What counts as a control

A control is an experiment with a known answer (oracle ranking = nDCG 1, shuffled labels = chance, schema validator must reject a
corrupted document, ...). Convention, any of:

* `details = {"control": True, "passed": bool, "expected": "..."}` (used by match/gap/e2e);
* `details["control_passed"]` or `status="failed"` with a reason (used by extraction/context);
* a bare value, for which `run_all.CONTROL_RULES` supplies a pass criterion (calibration random/constant controls, random baselines,
  false-alarm controls).

A control reported with none of these shows as "no verdict" in the integrity table (it is listed but cannot fail the run);
add a rule to `CONTROL_RULES` or set `details["passed"]`.

## Adding an eval

1. Create `src/skillllm_v2/evals/<name>_evals.py` with `run(ctx: EvalContext) -> list[EvalResult]` (see `contract.py`).
   Honour `ctx.smoke` (tiny subsets), `ctx.seed` (all randomness derived from it, never `hash()`), `ctx.n_boot`, and `ctx.root` for data.
   Use `bootstrap_ci` / `paired_bootstrap` from `contract.py`; give `n` = number of *independent* units; put synthetic data in
   `inputs/evals/` and say "synthetic" in `dataset`.
2. Always include baselines (random, trivial heuristic) and at least one control with `details["passed"]`.
3. Put the direction in `details["direction"]` when the metric name would not reveal it, and `details["hardware_dependent"]=True` for timings.
4. Register it in `registry.REGISTRY` with the pipeline stages it evaluates (`stages`) or touches only indirectly (`partial_stages`);
   the coverage table in the report is computed from this.
5. Add a test under `tests/evals/`. Run `--only <name> --smoke`, then a full run, then `--update-baseline` if the new numbers are accepted.

## The match / gap / e2e modules

All three use *authored synthetic* data (`inputs/evals/skill_universe.json`, `e2e_cases.json`, `e2e_catalog.jsonl`); ground truth is
derived from authored skill sets and authored grading rules, never from the system's own output.

**match** (`match_evals.py`): 112 synthetic candidates ranking 42 jobs (14 role archetypes x 3 variants). Authored relevance 0..3 =
tier-weighted coverage of true skills with half credit for adjacent skills (PyTorch~TensorFlow, ...) and a cap when a core skill is
entirely missing. Metrics: nDCG@5/10, MAP@10 and recall@5 (relevant = grade >= 2), hit@1; baselines keyword-Jaccard, keyword-count,
random; paired bootstrap deltas; tie analysis (the matcher clamps to 0, so ties are common); robustness (alias substitution, lower-cased
skills, noise skills, missing skills, random job-fit fields) as paired deltas versus the clean run; invariance to user id, protected
attributes, job title and skill order; seeded random property checks (score in [0,1], adding a needed skill never lowers the score, ...).

**gap** (`gap_evals.py`): 240 synthetic (user, job) pairs with an authored per-skill state (strong/weak/outdated/uncertain/transferable/absent).
Precision/recall/F1 of detected gaps (micro with bootstrap over pairs, macro per pair), critical-gap recall, gap-type accuracy, baselines
(all job skills; plain set difference); severity rank agreement with the authored severity (Kendall tau-b and Spearman, hand implemented,
checked against scipy); transferable handling; differential agreement between `GapDetector.detect` and `GapEngine`; and monotonicity
properties over 400 seeded random cases (adding a held skill never increases the gap, raising importance never lowers the weight, raising
user strength never raises it, weights bounded by importance, ...).

**e2e** (`e2e_evals.py`): 12 synthetic (profile text, job text) cases through the real offline components (stub Mistral anchor,
dictionary extractor, `JobMatcher`, `GapDetector`/`GapEngine`, `CourseRecommender`) in three wiring modes: `pipeline` (extractor both
sides), `demo` (extractor user skills, authored job names, as `smoke_runner.end_to_end_demo_smoke` does), `oracle` (authored names, isolates
downstream stages). Measures extraction vs truth, gap F1 per mode, recommendation precision@3 vs a random baseline, **traceability**
(each recommendation's covered skill must be a detected gap, a job skill, taught by the course, and the prerequisite status must be
consistent with the user's skills and earlier recommended courses), **schema validity** of the final JSON (authored from the documented
minimum contract; strict JSON, ranges, contiguous ranks), **latency per stage** (CPU wall-clock), **determinism** (in process and across two
`PYTHONHASHSEED` values in fresh interpreters), **graceful degradation** (extractor raises, empty catalog, empty text, unicode, malformed
catalog rows, 20k-character input, ...), and a generic probe of `pipeline/inference.py` (`SkillLLMV2Pipeline`, stub anchor) if it imports.

### Metric glossary (only what the three modules use)

* **nDCG@k**: exponential gain `2^rel - 1`, `log2(rank+1)` discount, ideal ranking over all judged items; candidates with no relevant item are *dropped*.
* **MAP@k**: AP@k divides by `min(R, k)` (repo convention, `eval/ranking_metrics.py`). trec_eval / ranx divide by `R` and count scenarios with no
  relevant item as 0. We report both: `match.map_at_10` and `match.map_at_10_trec_convention`; on this benchmark they differ a lot
  because many candidates have no grade >= 2 job. Every ranking result carries `dropped_undefined_scenarios` / `dropped_rule` in `details`.
* **recall@k / hit@1**: share of relevant items in the top k / whether the top item has the best authored grade (undefined, hence dropped, when no job reaches grade 2).
* **micro / macro P, R, F1**: micro pools gap counts over pairs; macro averages per-pair scores (empty vs empty = 1).
* **Kendall tau-b / Spearman rho**: rank agreement of authored severity vs system gap weight, ties handled, per pair then averaged; pairs with
  fewer than 3 shared gaps or a constant side are dropped, not scored 0.
* **violation rate**: share of seeded random cases in which a property is violated (target 0). Some properties build the condition in every case
  (`det_dup`, `det_trans_held`): there 1.0 means "the behaviour exists", not "1 in 1 random cases".
* **traceable share**: share of recommendations whose whole chain user gap -> job skill -> course skill -> prerequisite status checks out.
* **95% CI**: percentile bootstrap over the units counted in `n` (2000 resamples, seeded). It describes resampling noise over those units only.

## Oracle cross-check and verified statistics

`run_all` recomputes a sample of metrics through independent reference implementations *if they are importable* (never hard dependencies, nothing is
installed): `ir_measures`+`pytrec_eval` for nDCG@10 / P@5 / Recall@10 / AP@10 (gains pre-transformed to `2^g-1`, the workaround for the Perl `gdeval`
provider, see `docs/EVAL_LANDSCAPE.md` section 3), `scipy.stats` (`bootstrap(paired=True)`, `kendalltau`, `spearmanr`), `statsmodels.multipletests` (Holm),
and `nervaluate` (importable check only, no adapter). Max absolute differences go into the "Oracle cross-check" section; a difference above tolerance
fails the run; missing libraries are `skipped` with the import error. AP differences from the divisor convention are recorded as information
(scenarios with `R <= k` must match exactly; the others are expected to differ) and are not failures. The scipy checks run in the default environment;
the ir_measures/statsmodels adapters were exercised in the throwaway venv described in `docs/EVAL_LANDSCAPE.md` and agree to 1e-16 on nDCG, 0 on P/R/AP (R <= k).

Per `docs/EVAL_LANDSCAPE.md`, our bootstrap CI, sign-flip permutation test and Holm adjustment were verified against `scipy.stats.bootstrap(paired)`,
`scipy.stats.permutation_test` and `statsmodels.multipletests`; the cross-check section re-verifies the bootstrap and Holm parts on every run when the libraries are present.
Libraries that were evaluated and not adopted (netcal, DeepEval, RAGAS, ranx as a dependency, ...) and why are in `docs/EVAL_LANDSCAPE.md`; they are not repeated here.

## Honest limitations

* **Synthetic data.** match, gap and e2e use authored data. The matcher and the grading rule both reason about required-skill coverage, so the matcher's
  absolute nDCG (about 0.99) is a ceiling effect of the benchmark, not evidence of real-world quality. Informative are the baselines, robustness deltas,
  tie behaviour and properties. The gap engine scores F1 = 1.0 because the authored states use the engine's own gap vocabulary; the comparison that
  carries information is detector vs engine vs baselines and the severity correlation (tau-b about 0.81, unaffected by that agreement).
* **Authored judgements.** What counts as a gap (a weak preferred skill, a transferable cover) and how severe it is are author choices; changing them moves
  the detector's recall. No human validation has been done.
* **Tiny e2e n.** 12 cases; the CIs are wide and several shares have n < 10.
* **Stub extraction.** Offline, Mistral is the deterministic stub (8 alias strings) and GLiNER is absent. Extraction numbers inside e2e describe the stub,
  not the trained model. Job text is split into required/preferred by a cue regex in the eval harness (there is no job parser in the system).
* **Units and independence.** CIs resample candidates / pairs / cases; skills inside a pair are not independent (pooled critical-gap recall and
  transferable recall CIs are optimistic and say so). Bootstrap CIs are percentile intervals, which undercover for small n or boundary values (a CI of
  [1, 1] means "no variation seen", not certainty).
* **Latency** is wall-clock on whatever machine runs the suite, includes Python overhead, and uses stub extractors; it is gated loosely.
* **Coverage gaps.** Course-ranking quality on labelled scenarios lives in the older `eval/recommender_benchmark.py`, outside this suite. The `verifier` module
  currently reports no control. Controls without an explicit verdict are listed but cannot fail the run.
* **Regression tolerance** is absolute and ignores CIs; a real effect smaller than the tolerance can slip through, and noise inside the tolerance is accepted.

## Findings the suite currently surfaces (not fixed here; `match/`, `recommend/`, `pipeline/` are out of scope for this suite)

| finding | evidence (result name) |
|---|---|
| `JobMatcher` and `GapDetector.detect` compare skill strings exactly: `python` != `Python`. Lower-casing the user's skills drops matching nDCG@10 by 0.71 | `match.robust.lowercase_user_skills.*`, `match.prop.case_invariance`, `gap.robust.detector.*`, `gap.prop.detector.case_invariance` |
| No alias resolution in the matcher (alias surface forms count as missing); 50% alias substitution costs 0.10 nDCG@10 | `match.robust.alias_*`, `match.prop.alias_invariance` |
| The matcher clamps scores to 0: about half of all job scores are exactly 0, so weak matches tie | `match.score_zero_clamp_rate`, `match.tie_rate_top10`, `match.tie_sensitivity_spread` |
| `GapDetector.detect` relabels a *held* weak skill as `covered_by_transferable_skill` when a transferable mapping exists, emits duplicate gaps for a skill listed both required and preferred, and ignores weak/outdated/uncertain preferred skills | `gap.prop.detector.held_skill_never_labelled_covered_by_transferable`, `...one_gap_per_skill...`, `gap.detector.micro_recall` |
| `GapEngine`: a skill held weakly (strength < 0.4) gets a larger gap weight than the same skill merely transferable-covered (fixed 0.4), so acquiring it can raise the gap | `gap.prop.engine.weakly_held_skill_not_worse_than_transferable_cover` |
| `SkillLLMv2Pipeline.extract` does not catch extractor exceptions although its docstring says missing dependencies never block the pipeline | `e2e.degradation.graceful_share` (details.scenarios) |
| The extractor emits `canonical_skill_name` (`structured query language`, `Amazon Web Services`) while jobs and course catalogs use surface names (`SQL`, `AWS`): the exact-string matcher/gap detector then misses held skills and course linkage drops | `e2e.name_consistency.*`, `e2e.gap_demo.*` vs `e2e.gap_oracle.*`, `e2e.recommend_pipeline.gap_to_course_linkage_at_5` |
