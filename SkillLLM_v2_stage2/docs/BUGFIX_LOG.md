# Bug-fix log: context, graph and match-explanation modules

Source: the context / graph / explanation evals (`src/skillllm_v2/evals/{context,graph,explanation}_evals.py`). Each bug
below was reproduced first, fixed, and covered by a regression test. Before/after numbers come from the same evals
(`run(EvalContext(root=<repo>))`, seed 42, full size) plus a few probes the evals cannot reach through their default calls
(marked *probe*). All data is synthetic (the evals' seeded generators); nothing here is a real-world accuracy number.

Fixed: 1, 2, 3 (labels), 4, 5, 6, 7, 9 (SkillGraphBuilder), 10, 11, 12.
Not fixed (out of the allowed file set or no robust method): 3 (date conflicts), 8, 9 (CourseGraphBuilder part), 13 (not in scope).

## Before / after

| Metric (eval name) | Before | After | Note |
|---|---|---|---|
| Budget adherence of production packet, budgets 128/256/512/1024/2048 (*probe*, `build_cached_context_packet(budget_tokens=B)`) | 0.0 at every budget (no parameter) | 0.0 / 1.0 / 1.0 / 1.0 / 1.0 | 128 is infeasible: the smallest valid packet (wrapper + hash + zero skills) is ~129 proxy tokens; reported as `budget.met = False`. `context.production_budget_adherence` in the eval stays 0.0 because the eval calls without a budget |
| Important-skill recall of production packet under budget (*probe*) | 1.0 (nothing is ever dropped, nothing fits) | 0.00 / 0.21 / 0.59 / 0.83 / 0.90 | The reference compactor in the eval (which ships no wrapper, ~15 tokens of header) gets 0.55 / 0.77 / 0.88 at 256 / 512 / 1024: the production packet pays ~130 tokens of unavoidable structure |
| `context.determinism_across_clock_ticks` (whole packet bytes) | 0.0 | 0.0 | unchanged by design: `generated_at` is still in the packet; only the hash must be stable |
| content_hash equal when the clock ticks (`details.content_hash_equal_rate`, *probe*) | 0.0 | 1.0 | volatile fields excluded from the hash |
| content_hash equal under permuted extraction order (*probe*) | 0.0 | 1.0 | |
| `context.order_invariance_modulo_timestamps` | 0.0 | 1.0 | canonical evidence / offsets / fact order |
| `context.label_conflict_detection` | 0.474 | 1.0 | same_canonical_id 0.03 -> 1.0, different_canonical_id 0.97 -> 1.0 |
| `context.date_conflict_detection` | 0.0 | 0.0 | not fixed (documented limitation) |
| `context.duplicate_evidence_rate` | 0.067 | 0.033 | residual 3.3% is the eval's injected diff_id conflicts: the same surface and evidence under two different canonical ids, i.e. two separate facts (correct); duplicates within one fact are now 0 |
| `context.skill_storage_multiplicity` (counts entries) | 2.0 | 2.0 | still counts one reference per skill in `skill_memory`; tokens of skill_memory / tokens of the skill list: 1.07 -> 0.097 (*probe*) |
| `context.packet_over_raw_tokens` (default, no budget) | 18.9 | 10.5 | |
| `context.retriever_mrr` | 0.778 | 0.783 | evidence de-duplication, same retrieval code |
| `graph.explainer_connects_user_to_job` (path connectivity) | 0.0 | 1.0 | |
| `graph.path_toy_exact_match` | 0.857 | 1.0 | |
| `graph.path_ends_at_target_beyond_hops` | 0.0 | 1.0 | |
| `graph.path_shortest_correct_within_hops` | 1.0 | 1.0 | no regression vs the independent BFS oracle |
| `graph.retriever_unknown_node_leak_rate` | 1.0 | 0.0 | |
| `explain.trace_path_faithful_when_edge_missing` (fabricated-relation rate = 1 - this) | 0.0 (rate 1.0) | unchanged for `CourseGraphBuilder.trace_explanation_path`; new `narrate_skill_trace`: fabricated rate 0.0 (*probe*) | see bug 8 |
| `graph.ingest_idempotence` (CourseGraphBuilder, eval) | 0.5 | 0.5 | not fixed (file is off limits). SkillGraphBuilder (*probe*): edges once / edges twice 0.5 -> 1.0 |
| `graph.skill_builder.provenance_complete` | 0.0 | 0.578 (1.0 when a clock/`last_seen` is supplied, *probe*) | `HAS_SKILL` edges lack `last_seen` unless a time is known |
| `graph.builder_rejects_dangling_edge` | 0.0 | 0.0 (default) / 1.0 with `strict=True` (*probe*) | default still creates a flagged placeholder, see bug 9 |
| add_node silently retypes (probe, 1 = bug) | 1.0 | 0.0 | |
| `explain.match.evidence_relevance` | 0.147 | 1.0 | |
| `explain.match.missing_skill_has_evidence` | 0.0 | 0.0 | by design: a missing skill has no profile evidence; stated in `missing_skill_evidence`, not faked |
| `explain.match.completeness` | 1.0 | 0.729 | **eval artifact, see below**: 13/48 explanations have zero matched skills, so they now have an empty `evidence` list (before it was padded with unrelated skills) |
| `explain.match.static_boilerplate_share` | 0.085 | 0.112 | **eval artifact**: the audit labels every limitation "static" regardless of content, so more (specific) limitations raise the share. The old boilerplate strings no longer appear (`boilerplate_limitation_present_rate` 1.0 -> 0.0, *probe*) |

## Bugs

### 1. `compact_context` has no budget; skill_memory stores every skill twice (fixed)
* `compact_context(..., budget_tokens=None)` and `build_cached_context_packet(..., budget_tokens=None)`. Without a budget the packet keeps every skill (default behaviour unchanged except fixes 2-4 and the memory dedupe below).
* With a budget the whole packet (wrapper, payload, hash) serializes to at most `budget_tokens` proxy tokens (`context/budget.py::count_tokens`, the same `\w+|[^\w\s]` proxy as the evals; real BPE counts are ~1.2-1.5x larger).
* Importance (`budget.skill_importance`): `confidence * (1 + ln(#distinct evidence)) * mean source weight * (1.25 if the skill is named in the job target)`. Source weights are neutral 1.0 unless the caller passes `source_reliability` (nothing in the repo measures per-source reliability). **Recency is not used**: evidence timestamps are ingestion time and extraction output carries no dates. **HARD > SOFT is not used**: nothing justifies it (`label_weights` is available if a caller wants it).
* Packing: cheapest form per skill admitted in importance order, then leftover budget upgrades the most important skills to more evidence; catalog summaries and aliases not tied to a kept skill only if budget remains; conflict notes get first claim but at most 30% of what the skeleton leaves (`budget.notes_omitted` counts the rest). Evidence text is only ever truncated to a substring (never rewritten); the cheapest form drops the default label `HARD_SKILL` and per-evidence confidence.
* Budget mode also drops static policy boilerplate (`cache_policy`, `dynamic_rag_policy.reason`, ...), `skill_memory` indexes (only `skill_count` stays; rebuild with `build_skill_memory(profile_summary)`), empty containers and per-evidence timestamps, and adds `payload["budget"] = {limit, used, kept, total, met}`. `met=False` only if even zero skills do not fit.
* `build_skill_memory`: `skills_by_label[label]` now lists skill *keys* (canonical id, else normalized surface) referencing `profile_summary.skills` instead of copying every skill with its evidence. `inline_skills=True` restores the old layout.

### 2. content_hash changes every second (fixed)
* `content_hash` is computed over the packet minus `generated_at`, `timestamp` and `content_hash` (`cached_context_builder.compute_content_hash`).
* `summarize_profile`, `compact_context`, `build_cached_context_packet` accept `timestamp=` (fixed string) and `clock=` (zero-arg callable) for tests. `generated_at` is still in the packet (it says when it was built), so whole-packet bytes still differ across seconds unless a timestamp/clock is injected.
* Canonical order: extractions mapping to one skill fact are sorted before the fact is built (representative surface / label / canonical name = highest confidence, ties by source, surface, offsets); evidence, `metadata.source_offsets` and facts are sorted. Permuting the extraction list gives identical bytes. Behavioural change: the representative of a group is now the highest-confidence extraction, not the first one seen.

### 3. Label conflicts lost (fixed for labels; dates not fixed)
* Conflicts are detected on the raw extractions before merging, grouped by normalized surface and by canonical id (different surfaces of one id), explicit labels only.
* Contradictory date ranges are **not** surfaced: free-text ranges ("2018 to 2021", "between 2013 and 2015", "since 2019", month names, "present") are ambiguous without a dependable parser and no date library is bundled; the eval's date check also inspects raw text no extraction points at. Documented in `summarize_profile`.

### 4. Repeated (skill, evidence text) pairs (fixed)
Merged into one `Evidence` (max confidence, sources joined with `+`); `metadata.mentions` keeps the number of supporting extractions so the repetition signal is not lost.

### 5. `explain_skill_path` never connects a user to a job/course (fixed)
BFS over an explanatory view: forward edges always, edges that point into a skill (HAS_SKILL, MISSING_SKILL, REQUIRES, PREFERS, TEACHES, REQUIRES_PREREQUISITE, BELONGS_TO) also backwards; RELATED_TO / SIMILAR_TO / unknown types stay directed (so the eval's directed toy cases still hold). Result adds `edges` (`from`, `to`, `relation`, `direction` = forward/reverse/undirected), `hops`, `found`, `endpoint`, `reason`; `explanation` renders `u1 -[HAS_SKILL]-> python <-[REQUIRES]- j1`. Works on DiGraph/MultiDiGraph/Graph/MultiGraph and subgraph views. `pipeline/inference.py` now passes the directed subgraph instead of the undirected copy.

### 6. Paths longer than `max_hops` truncated but labelled with the original target (fixed)
A path is never cut. Beyond `max_hops` it is returned whole with `exceeds_max_hops=True`, `confidence=0.7` and a note in `explanation`; `strict_max_hops=True` returns no path (`path == []`, `reason`). `target` is the requested target, `endpoint` the real last node, `truncated` is always False.

### 7. SkillGraphRetriever leaks other users' skills for unknown ids (fixed)
Every method checks `has_node` first (networkx iterates a str id character by character); unknown or unhashable ids return `[]`. Results are de-duplicated, order preserved.

### 8. `trace_explanation_path` fabricates 'required' / 'teaches' (NOT fixed in place)
The method lives in `graph/course_graph_builder.py`, which this task excluded. Provided instead: `graph/skill_graph_explainer.py::narrate_skill_trace(graph, user_id, job_id, course_id, skill_id)`, same output keys plus `relations_present`, narrating only edges that exist and saying "The graph has no REQUIRES or PREFERS / TEACHES edge ..." otherwise (fabricated-relation rate 1.0 -> 0.0 on the eval's 60 random quadruples per world, *probe*). To finish the fix, make `CourseGraphBuilder.trace_explanation_path` delegate to it.

### 9. Builder hygiene (fixed for SkillGraphBuilder; CourseGraphBuilder not touched)
`SkillGraphBuilder(strict=False, clock=None)`:
* idempotent: the edge key is the edge type, so a relation is stored once; re-adding updates properties (last write wins, but an empty value never erases a real one). Different edge types between the same nodes coexist.
* `add_node` on an existing node with another type keeps the type, records a warning in `builder.warnings` and logs (strict: `ValueError`); same-type calls merge properties; a placeholder is upgraded.
* `add_edge` to unknown nodes still creates an untyped node (some callers and a test rely on it) but marks it `placeholder=True`, warns and records it; `strict=True` raises `ValueError` and adds nothing. **Default left lenient on purpose**, `tests/evals/test_context_graph_explanation.py::test_structural_report_hand_graphs` asserts the placeholder behaviour.
* the builder never reads the wall clock: `last_seen` is the explicit argument, else `clock()`, else absent.
* `add_edge`'s three parameters are now positional-only so edges can carry a `source` property.
Not fixed: `CourseGraphBuilder` still doubles edges on re-ingest (`graph.ingest_idempotence` 0.5) and stamps `last_seen` with the wall clock (`graph.determinism_across_clock_ticks` 0.0).

### 10. normalize_text collapses C, C++, C#, F#, .NET (fixed via a new key function)
`skillllm_v2.schema.graph_key(name)`: keeps a trailing `+`/`#` run (c++, c#, f#), a leading `.` (.net) and a `.` before an alphabetic segment (node.js, asp.net); otherwise identical to `normalize_text` (checked on names without such symbols). `normalize_text` is unchanged, so alias/taxonomy/extraction matching (139 call sites) behaves exactly as before. `pipeline/inference.py` uses `graph_key` for graph node ids in the graph stage and when attaching courses. Not applied to `CourseGraphBuilder` ids (off limits), which still collide.

### 11. SkillGraphBuilder edges carry almost no provenance (fixed)
HAS_SKILL: `source`, `confidence`, `evidence_text`, `last_seen` (when known); REQUIRES/PREFERS: `importance`, `requirement_strength`, `source_sentence`; TEACHES: `confidence`, `depth`, `learning_outcome_evidence`, `source`. Populated from the new optional arguments of `add_user_skill` / `add_job_requirement` / `add_course_skill`; unknown values are `""` / `None` / `"unknown"`, never invented. The pipeline's graph stage now passes the user skill's evidence and source.

### 12. `generate_recommendation_explanation` (fixed)
`evidence` now holds the best profile evidence (`evidence_per_skill`, default 1) for matched, missing and gap skills only (matching by surface form, canonical name or id), not one line per profile skill. New keys: `missing_skill_evidence` (per missing skill: lines, or the note "no evidence in profile") and `matched_without_evidence`. `limitations` are computed and checkable: missing skills without profile evidence, matched skills with no evidence text or best evidence confidence < 0.5, missing skills no supplied course covers, no job match, no course recommendations; empty list when none applies. The two fixed boilerplate strings are gone.

## Existing eval-suite tests that assert the old (buggy) behaviour
`tests/evals/` is read-only for this task, so these were not edited and now fail by design:
* `test_context_graph_explanation.py::test_graph_run_smoke` asserts `graph.skill_builder.provenance_complete == 0.0` (bug 11 fixed).
* `test_context_graph_explanation.py::test_match_audit_clean_mutations_and_completeness` asserts `evidence_relevance < 1.0` (bug 12 fixed).
Also failing, but not from this work (another change to `match/job_matcher.py` / `gap_detector.py` in the working tree): `tests/evals/test_match_gap_e2e.py::{test_match_property_checks_flag_case_sensitivity_and_pass_when_fixed, test_gap_property_checks_flag_broken_engine}`.

## Tests added
`tests/skillllm_v2/test_context_bugfixes.py` (24), `test_graph_bugfixes.py` (47), `test_match_explanation_bugfixes.py` (12).
