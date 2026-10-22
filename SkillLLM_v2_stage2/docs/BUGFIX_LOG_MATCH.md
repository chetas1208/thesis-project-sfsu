# Bugfix log: match / gap / extractor degradation

Found by the eval suite (`match`, `gap`, `e2e` modules, see `docs/EVAL_SUITE.md`). Each bug was reproduced first, fixed, covered by a
regression test in `tests/skillllm_v2/test_match_fixes.py`, and re-measured with the same evals
(`PYTHONPATH=src python -m skillllm_v2.evals.run_all --only match,gap,e2e`, seed 42, full run, same machine).
No commits, no baseline update: `reports/` and `src/skillllm_v2/evals/` were not touched.

## Summary

| # | bug | status | where |
|---|---|---|---|
| 1 | exact-string skill comparison (case, whitespace, punctuation, no alias resolution) | fixed | `match/skill_keys.py` (new, shared), `match/job_matcher.py`, `match/gap_detector.py` |
| 2 | `GapDetector.detect`: held weak skill relabelled `covered_by_transferable_skill`; duplicate gap for a skill listed required + preferred; weak / outdated / uncertain PREFERRED skills ignored | fixed | `match/gap_detector.py` |
| 3 | canonical name (extractor) vs surface name (jobs, catalogs) never meet | fixed in matcher, detector, `detect_vector` and the pipeline match stage; course ranking part not fixable in scope (see Not fixed) | `match/*`, `pipeline/inference.py` |
| 4 | extractor exceptions escape `SkillLLMv2Pipeline.extract` and `SkillLLMV2Pipeline` although docs say missing dependencies never block the pipeline | fixed | `wrapper.py`, `pipeline/inference.py` |
| 5 | `GapEngine`: weak-held skill outweighs a transferable-covered one, so acquiring a skill can raise the gap | NOT fixed: `recommend/` is out of scope; fix proposed below | `recommend/gap_vector.py` |

## Before / after (eval suite)

"Before" = the first run of this session on the unmodified tree; "after" = final tree. Lower is better for rates / violation / delta magnitudes.

| metric (eval name) | before | after |
|---|---|---|
| match nDCG@10, clean (`match.ndcg_at_10`) | 0.9905 | 0.9905 (unchanged) |
| match nDCG@10 delta when user skills are lower-cased (`match.robust.lowercase_user_skills`) | -0.7119 | 0.0000 |
| match case-invariance violation rate (`match.prop.case_invariance`) | 0.2767 | 0.0000 |
| match alias-invariance violation rate (`match.prop.alias_invariance`) | 0.1400 | 0.0533 |
| match nDCG@10 delta, 25% / 50% of held skills replaced by an alias | -0.0334 / -0.0962 | -0.0133 / -0.0342 |
| detector case-invariance violation rate (`gap.prop.detector.case_invariance`) | 0.2275 | 0.0000 |
| detector spurious gaps per job skill under lower-cased user skills | 0.4516 | 0.0000 |
| detector micro recall / F1 vs authored truth (`gap.detector.*`) | 0.8045 / 0.8917 | 1.0000 / 1.0000 |
| detector macro F1 per pair | 0.8710 | 1.0000 |
| detector vs engine gap-set Jaccard | 0.8011 | 1.0000 |
| held skill labelled covered-by-transferable (`gap.prop.detector.held_skill_never_labelled_covered_by_transferable`) | 1.0000 | 0.0000 |
| one gap per skill when listed required + preferred (violation rate) | 1.0000 | 0.0000 |
| e2e gap F1, demo wiring (extractor canonical names vs authored job names) | 0.6147 | 0.7171 |
| e2e gap precision, demo wiring | 0.5667 | 0.6722 |
| e2e gap F1, pipeline wiring / oracle wiring | 0.8942 / 1.0000 | 0.8942 / 1.0000 |
| e2e false gaps on skills that were both held and extracted (demo wiring) | 0.2500 | 0.0000 |
| e2e graceful-degradation pass rate (`e2e.degradation.graceful_share`) | 0.8182 (9/11) | 1.0000 (11/11) |
| e2e gap-to-course linkage@5, pipeline / oracle | 0.5185 / 1.0000 | 0.5185 / 1.0000 (not addressable in scope) |
| e2e match / detect / vector latency (ms per call, hardware dependent) | 0.016 / 0.007 / 0.047 | 0.18 / 0.14 / 0.20 (alias lookups; still far below extraction at ~20 ms) |

All 0-valued controls still pass (0 control failures). Match properties "score in [0,1]", "adding a needed skill never lowers the score",
"irrelevant skill invariance" and the attribute/order invariance checks are unchanged at 0.

Reading the residuals:

* Remaining alias-invariance violations (0.0533) and alias nDCG deltas are alias-table coverage gaps, not logic: the shipped tables
  (`data/skillllm_v2/taxonomy/skill_aliases.jsonl` and the packaged `resources/offline_aliases.jsonl`) do not know
  `py`/`python3` -> Python, `PowerBI` -> Power BI, `TF` -> Terraform, `TS` -> TypeScript, `version control` -> Git, `REST` -> REST APIs,
  `PM` -> Project Management, `automated testing` -> Test Automation. Adding those rows fixes them without code changes.
* Demo-wiring gap F1 stays below the oracle (0.7171 vs 1.0) because the offline extractor only recovers 62% of the held skills
  (`e2e.extraction.profile_recall`); the naming loss itself is gone (false-gap rate on held-and-extracted skills 0.25 -> 0.0).
* The regression gate against `reports/skillllm_v2/eval_suite/baseline.json` reports two e2e regressions
  (`recommend_pipeline.gap_to_course_linkage_at_5` 0.6204 -> 0.5185 and `precision_at_3` 0.6296 -> 0.5185). They are identical in the
  untouched "before" run of this session (0.5185), i.e. they come from other uncommitted changes in the tree (extraction / merger /
  course code), not from this work.

## 1. Case / whitespace / punctuation / alias-blind matching

`JobMatcher.score` and `GapDetector.detect` used `set` intersection of raw strings.

Fix: one shared module `match/skill_keys.py`.

* `skill_key(name)`: NFKC + casefold + whitespace/punctuation normalisation that keeps symbol-bearing names distinct (`c++`, `c#`, `f#`,
  `.net`, `node.js` vs `c`, `net`, `node js`). It imports `skillllm_v2.schema.graph_key` (the graph key function that exists now) and falls
  back to an identical local implementation, so graph node identity and matcher identity agree.
* `SkillKeyResolver`: a skill (string or dict) becomes a set of keys: literal key, alias-resolved key (existing `normalize.AliasResolver`),
  and for dicts the canonical id / canonical name; two skills are the same when their key sets intersect. For a dict that declares a
  canonical id/name, a surface form whose alias resolves to a *different* skill is ignored (canonical takes precedence).
  Alias lookups are skipped for symbol-bearing surfaces and rejected when the hit differs from the surface only by erased symbols
  (`AliasResolver` normalises with `normalize_text`, so the table row `C++` -> alias `c` would otherwise merge `C` and `C++`).
  Lossy ids (`c` for `C++`) are ignored.
* Alias table: `JobMatcher(alias_resolver=..., aliases=..., use_default_aliases=True)` and the same on `GapDetector`. With no argument the
  repo alias table plus the packaged `resources/offline_aliases.jsonl` are loaded once (cached per working directory); if neither loads,
  matching still works on literal keys. `JobMatcher()` / `GapDetector()` keep their signatures; output keys are unchanged.
* Duplicate job skills that are the same skill (`Python`, `python`) now count once in coverage.

## 2. `GapDetector.detect` semantics

* A held weak / outdated / uncertain skill keeps its own gap type. `covered_by_transferable_skill` is assigned only to a skill the user does
  not hold (missing required or preferred).
* One entry per distinct job skill. A skill that is both required and preferred is reported once, as required, with the added field
  `also_listed_as_preferred: true`.
* Preferred skills that are held but weak / outdated / uncertain are now reported (`weak_skill` / `outdated_skill` / `uncertain_skill`).
  Legacy gap types have no preferred variants, so the added field `requirement_strength` (`required` / `preferred`) carries it.
* Legacy schema kept: `{"skill", "gap_type"[, "transferable_from"]}` plus the added `requirement_strength`, `skill_key`,
  `also_listed_as_preferred`. The user's `skills` may now also be dicts (previously a `TypeError`); `skill_entries` is merged in.

## 3. Canonical name vs surface name

* Matcher / detector: both spellings resolve to the same key (`SQL` == `structured query language`, `AWS` == `Amazon Web Services`)
  through the alias table; dict skills match on canonical id / name or surface.
* `GapDetector.detect_vector` (feeds `GapEngine`, which matches exact ids): the user's held skills, weak / outdated / uncertain lists and
  transferable map are re-keyed onto the job's skill ids before the engine runs. The job's ids are not changed (`normalize_text(surface)`
  stays the vector key, so id-based course linkage is unaffected). Exact-input results are identical to the old behaviour (tested).
* `SkillLLMV2Pipeline`: default `JobMatcher` / `GapDetector` are built with the pipeline's alias resolver; `_matching_profile` still has
  `skills` (canonical names) and now also `skill_entries` (canonical id + canonical name + surface form + confidence). Injected
  `job_matcher` / `gap_detector` are used as given.

## 4. Extractor failures

`stage_extract` (and therefore `SkillLLMv2Pipeline.extract`, the job-text sub-run and `run()`):

* Mistral/anchor (resolution and `.extract`), GLiNER (construction and `.extract`) and dictionary failures are each caught per extractor.
  The `extract` stage record becomes `status="degraded"` with the error text in `provenance["extractor_failures"][name]` and in `notes`;
  the trace gets a warning and `degraded=True`. The remaining extractors' candidates are used; nothing is fabricated for the failed one
  (a failed anchor sets `anchor_mode="none"`).
* If every extractor that was tried fails, the extraction is empty, the record is `status="failed"` with `error` set, and no exception is raised.
* Strict: `SkillLLMV2Pipeline(strict_extractors=True)` (or `inference.continue_on_error: false`) re-raises; `SkillLLMv2Pipeline.extract(strict=True)`
  does the same. An explicit `anchor.mode: mistral` that cannot load always raises (documented behaviour, unchanged).
* `SkillLLMv2Pipeline.extract` returns the same list type; it additionally sets `self.last_trace` and emits a `RuntimeWarning` naming each failed
  extractor (so it is never silent). `job_extract` records inherit the failure and warnings from their sub-run.
* The previously cached GLiNER failure is now also recorded as degraded on later calls.

## 5. NOT fixed: `GapEngine` weak-held vs transferable-covered (out of scope, `recommend/gap_vector.py`)

Reproduced: `GapEngine().compute({"skills": []|[Python weak], "transferable_skills": {"python": "Ruby"}}, job(["Python"]))` gives weight 0.60
for the not-held, transferable-covered skill (fixed `user_str = 0.4`) but 0.72 once the user holds Python weakly (confidence 0.8 x
beginner 0.35 = 0.28 strength, `weight = importance x (1 - 0.28)`), so acquiring the skill raises the gap. The held branch never consults
the transferable map. Measured violation rate: 0.8775 (`gap.prop.engine.weakly_held_skill_not_worse_than_transferable_cover`).

Proposed fix (owner of `recommend/`): in the held branch compute the same `transferable_from` lookup as the missing branch and use
`effective_strength = max(user_str, 0.4)` when a transferable cover exists (equivalently `weight = min(weight, importance x
strength_multiplier x 0.6)`), keeping the gap type `weak_skill`. Without a cover nothing changes (a weak-held skill stays below the
missing weight 1.0). The eval property above should drop to 0; add a regression test next to the existing gap-vector tests.

## Not fixed / notes

* Gap-to-course linkage@5 (0.5185 vs oracle 1.0): the loss is in how `recommend/course_ranker` / `course/` link gap ids to catalog skills
  (`normalize_text` equality, no alias resolution: catalog `SQL` vs gap `structured query language`). Out of scope. The shared
  `SkillKeyResolver` can be reused there (`resolver.keys(skill_a) & resolver.keys(skill_b)`).
* Eval expectations that encode the fixed bugs (reported, not edited): `tests/evals/test_match_gap_e2e.py::test_match_property_checks_flag_case_sensitivity_and_pass_when_fixed`
  (`match.prop.case_invariance > 0.05`) and `::test_gap_property_checks_flag_broken_engine` (`gap.prop.detector.case_invariance > 0` and
  `held_skill_never_labelled_covered_by_transferable == 1.0`) now fail because the code is fixed. The "FINDING" caveat strings in
  `match_evals.py` / `gap_evals.py` are stale too. `reports/skillllm_v2/eval_suite/baseline.json` should be refreshed (`--update-baseline`) once
  the owner accepts the numbers above; it also predates the two unrelated e2e linkage / precision@3 drifts.
* `tests/evals/test_context_graph_explanation.py::test_graph_run_smoke` and `::test_match_audit_clean_mutations_and_completeness` fail on the
  concurrent graph / explanation work (not this task).
* Informational e2e scenarios `job_without_job_id` (`KeyError`) and `null_skill_lists` (`TypeError`, now only on `skills=None` for the
  engine path) remain input-contract violations; they are not scored.
* Alias-table coverage (see residuals above) is data work.

## Tests

`tests/skillllm_v2/test_match_fixes.py` (31 tests): skill keys (case, whitespace, punctuation, symbol preservation), matcher case /
alias / canonical precedence / duplicates / unchanged exact scores, detector case / alias / weak-not-transferable / required+preferred
duplicate / preferred weak-outdated-uncertain / precedence / legacy schema / dict skills, `detect_vector` re-keying and exact-input
parity with the engine, extractor degradation (dictionary, anchor, GLiNER, all-failed, strict, full `run()`), wrapper warning / trace /
strict, and the pipeline match stage meeting canonical name and surface. Full suite: 449 passed, 1 skipped, 2 failed (the two eval tests above
that encode the fixed bugs); the other failures seen mid-session in `test_context_graph_explanation.py` were resolved by the concurrent work.
