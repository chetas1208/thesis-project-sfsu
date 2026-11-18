# L06 — Taxonomy linking against public ESCO gold

## 1. Audit of the legacy "Linking Accuracy" (~0.890)

**Verdict:** NOT AN ACCURACY: the reported 'Linking Accuracy' (0.890374 Full CAG/Graph, 0.889834 safe wrapper) is coverage = share of predicted spans that received any canonical id. No gold link was compared. It is additionally partly circular: the 711-row alias table it links into was built from SkillSpan train/dev gold surfaces (601 rows) and 76 high-confidence Mistral predictions on the SAME test split, and its BM25 fallback links almost any string that shares a token with an entry.

- Computed as: pipeline/smoke_runner.py: linked_count / total over PREDICTED spans (share with any non-null canonical id).
- Alias table: 711 rows, sources {'manual_curated': 34, 'skillspan_gold_train_dev': 601, 'mistral_high_confidence': 76}.
- `linker_train/dev.jsonl`: 2 toy rows each; reranker: placeholder (train_taxonomy_reranker writes a README only); build_linker_dataset writes an empty file.
- Consequence: the 0.890 must not be cited as linking accuracy anywhere (integrity flag raised in Progress.md).

## 2. Gold and protocol

- Gold: Decorte et al. 2022 TECH+HOUSE (SkillSpan test spans -> ESCO v1.1.0); HF `TechWolf/skill-extraction-{tech,house}`, CC-BY-4.0. Decorte *validation* (420 mentions) is used for every threshold; *test* (1700 mentions: 1062 with an ESCO label, 638 NIL = LABEL NOT PRESENT / UNDERSPECIFIED) is reporting-only.
- All Decorte sentences come from the SkillSpan **test** split; 1698/1700 test mentions aligned to a repo SkillSpan test id (`test-<i>`), cluster = SkillSpan posting (`source:idx`).
- Gold label -> URI resolution in ESCO v1.1.0: {'label_resolution_pref_exact': 1241, 'label_resolution_pref_norm': 1} -> unresolved 0 mentions (excluded from the linkable set); unresolved labels: {}.
- Mention = (subset, sentence, span); multi-label spans count a hit if top-k contains any gold concept.
- Linking is context-free on GOLD spans, so extraction errors are excluded (upper bound for the pipeline).
- seen = L02 buckets S1+S2 (surface, or its normalisation, is a SkillSpan TRAIN gold span), unseen = L02 S4 (same `generalization.Bucketer`); HARD / SOFT = SkillSpan tag of the aligned gold span. variant_surface = surface differs from every gold preferred label; alias_listed = surface is an ESCO alternative label of a gold concept (dictionary-style lookups trivially succeed here; it is the alias-resolution accuracy, not generalisation).
- `cascade_esco@tau_val`: the validation-tuned threshold is 1.0 (only an exact alias/normalised hit survives); it is effectively a lookup that abstains otherwise.

## 3. Results (test; 95% posting-cluster bootstrap CIs)

| system | top1_acc_linkable | top1_acc_linkable_ci | recall@3 | recall@3_ci | recall@5 | recall@5_ci | mrr@10 | mrr@10_ci | nil_aware_acc | abstain_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | 0.000 |
| always_nil | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.375 | 1.000 |
| exact | 0.161 | [0.130, 0.191] | 0.164 | [0.133, 0.195] | 0.164 | [0.133, 0.195] | 0.162 | [0.132, 0.193] | 0.465 | 0.878 |
| shipped_bridge | 0.157 | [0.126, 0.190] | 0.162 | [0.130, 0.197] | 0.162 | [0.130, 0.197] | 0.159 | [0.128, 0.193] | 0.446 | 0.826 |
| bm25@never | 0.238 | [0.211, 0.265] | 0.377 | [0.334, 0.419] | 0.450 | [0.406, 0.493] | 0.331 | [0.299, 0.363] | 0.218 | 0.092 |
| bm25@tau_val | 0.238 | [0.211, 0.265] | 0.377 | [0.334, 0.419] | 0.450 | [0.406, 0.493] | 0.331 | [0.299, 0.363] | 0.380 | 0.528 |
| dense_minilm@never | 0.488 | [0.451, 0.525] | 0.721 | [0.686, 0.755] | 0.779 | [0.748, 0.809] | 0.619 | [0.588, 0.650] | 0.305 | 0.000 |
| dense_minilm@tau_val | 0.488 | [0.451, 0.525] | 0.721 | [0.686, 0.755] | 0.779 | [0.748, 0.809] | 0.619 | [0.588, 0.650] | 0.532 | 0.615 |
| cascade_esco@never | 0.324 | [0.286, 0.360] | 0.440 | [0.399, 0.481] | 0.506 | [0.460, 0.550] | 0.400 | [0.362, 0.438] | 0.272 | 0.089 |
| cascade_esco@tau_val | 0.324 | [0.286, 0.360] | 0.440 | [0.399, 0.481] | 0.506 | [0.460, 0.550] | 0.400 | [0.362, 0.438] | 0.461 | 0.878 |

NIL detection (positive class = NIL: gold has no ESCO label; predicted NIL = abstain; posting-cluster CIs). Systems that never abstain (`random`, `*@never`) score NIL recall 0 by construction except when they have no candidate. `always_nil` shows the floor: NIL prevalence in test makes predict-everything-NIL already reach the F1 printed on its row, so `exact` / `cascade_esco@tau_val` (NIL F1 ~0.58) add little NIL skill beyond abstaining:

| system | nil_precision | nil_precision_ci | nil_recall | nil_recall_ci | nil_f1 | nil_f1_ci |
|---|---|---|---|---|---|---|
| random | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| always_nil | 0.375 | [0.342, 0.408] | 1.000 | [1.000, 1.000] | 0.546 | [0.510, 0.580] |
| exact | 0.416 | [0.379, 0.450] | 0.972 | [0.959, 0.983] | 0.582 | [0.546, 0.615] |
| shipped_bridge | 0.422 | [0.382, 0.458] | 0.928 | [0.906, 0.948] | 0.580 | [0.542, 0.613] |
| bm25@never | 0.756 | [0.654, 0.836] | 0.185 | [0.139, 0.235] | 0.297 | [0.231, 0.362] |
| bm25@tau_val | 0.489 | [0.441, 0.534] | 0.688 | [0.636, 0.737] | 0.572 | [0.526, 0.615] |
| dense_minilm@never | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] |
| dense_minilm@tau_val | 0.510 | [0.462, 0.556] | 0.835 | [0.803, 0.867] | 0.633 | [0.591, 0.673] |
| cascade_esco@never | 0.776 | [0.679, 0.856] | 0.185 | [0.139, 0.235] | 0.299 | [0.232, 0.363] |
| cascade_esco@tau_val | 0.416 | [0.379, 0.450] | 0.972 | [0.959, 0.983] | 0.582 | [0.546, 0.615] |

| system | nil_f1[tech] | nil_f1[house] | nil_f1[HARD] | nil_f1[SOFT] | nil_f1[seen] | nil_f1[unseen] |
|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| always_nil | 0.571 | 0.514 | 0.623 | 0.469 | 0.487 | 0.578 |
| exact | 0.619 | 0.539 | 0.714 | 0.470 | 0.576 | 0.587 |
| shipped_bridge | 0.619 | 0.534 | 0.707 | 0.467 | 0.579 | 0.582 |
| bm25@never | 0.324 | 0.262 | 0.431 | 0.095 | 0.435 | 0.227 |
| bm25@tau_val | 0.633 | 0.490 | 0.674 | 0.445 | 0.595 | 0.559 |
| dense_minilm@never | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dense_minilm@tau_val | 0.681 | 0.578 | 0.787 | 0.499 | 0.665 | 0.622 |
| cascade_esco@never | 0.324 | 0.265 | 0.432 | 0.096 | 0.440 | 0.227 |
| cascade_esco@tau_val | 0.619 | 0.539 | 0.714 | 0.470 | 0.576 | 0.587 |

Strata (top-1 accuracy on linkable mentions; seen = L02 S1+S2, unseen = L02 S4):

| system | top1[tech] | top1[house] | top1[HARD] | top1[SOFT] | top1[seen] | top1[unseen] | top1[HARD_seen] | top1[HARD_unseen] | top1[SOFT_seen] | top1[SOFT_unseen] |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| always_nil | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| exact | 0.208 | 0.110 | 0.349 | 0.029 | 0.372 | 0.028 | 0.493 | 0.093 | 0.096 | 0.008 |
| shipped_bridge | 0.208 | 0.102 | 0.333 | 0.034 | 0.367 | 0.026 | 0.479 | 0.073 | 0.112 | 0.012 |
| bm25@never | 0.266 | 0.208 | 0.353 | 0.154 | 0.298 | 0.200 | 0.368 | 0.333 | 0.136 | 0.156 |
| bm25@tau_val | 0.266 | 0.208 | 0.353 | 0.154 | 0.298 | 0.200 | 0.368 | 0.333 | 0.136 | 0.156 |
| dense_minilm@never | 0.533 | 0.439 | 0.658 | 0.368 | 0.672 | 0.368 | 0.761 | 0.473 | 0.472 | 0.336 |
| dense_minilm@tau_val | 0.533 | 0.439 | 0.658 | 0.368 | 0.672 | 0.368 | 0.761 | 0.473 | 0.472 | 0.336 |
| cascade_esco@never | 0.366 | 0.278 | 0.537 | 0.172 | 0.507 | 0.203 | 0.636 | 0.353 | 0.216 | 0.154 |
| cascade_esco@tau_val | 0.366 | 0.278 | 0.537 | 0.172 | 0.507 | 0.203 | 0.636 | 0.353 | 0.216 | 0.154 |

L02 buckets (S1 exact seen, S2 normalised seen, S3 canonical seen / surface unseen, S4 fully unseen; same `Bucketer` as L02):

| system | top1[S1] | n[S1] | top1[S2] | n[S2] | top1[S3] | n[S3] | top1[S4] | n[S4] |
|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 321 | 0.000 | 85 | 0.000 | 10 | 0.000 | 646 |
| always_nil | 0.000 | 321 | 0.000 | 85 | 0.000 | 10 | 0.000 | 646 |
| exact | 0.386 | 321 | 0.318 | 85 | 0.200 | 10 | 0.028 | 646 |
| shipped_bridge | 0.380 | 321 | 0.318 | 85 | 0.100 | 10 | 0.026 | 646 |
| bm25@never | 0.293 | 321 | 0.318 | 85 | 0.300 | 10 | 0.200 | 646 |
| bm25@tau_val | 0.293 | 321 | 0.318 | 85 | 0.300 | 10 | 0.200 | 646 |
| dense_minilm@never | 0.676 | 321 | 0.659 | 85 | 0.700 | 10 | 0.368 | 646 |
| dense_minilm@tau_val | 0.676 | 321 | 0.659 | 85 | 0.700 | 10 | 0.368 | 646 |
| cascade_esco@never | 0.517 | 321 | 0.471 | 85 | 0.700 | 10 | 0.203 | 646 |
| cascade_esco@tau_val | 0.517 | 321 | 0.471 | 85 | 0.700 | 10 | 0.203 | 646 |

Alias-resolution accuracy (top-1 when the surface is an ESCO *alternative* label of the gold concept = `alias_listed`; `pref_label_surface` = surface equals the gold preferred label; `variant_surface` = differs from every gold preferred label):

| system | top1[pref_label_surface] | n[pref_label_surface] | top1[alias_listed] | n[alias_listed] | top1[variant_surface] | n[variant_surface] |
|---|---|---|---|---|---|---|
| random | 0.000 | 123 | 0.000 | 51 | 0.000 | 939 |
| always_nil | 0.000 | 123 | 0.000 | 51 | 0.000 | 939 |
| exact | 1.000 | 123 | 0.941 | 51 | 0.051 | 939 |
| shipped_bridge | 0.886 | 123 | 0.824 | 51 | 0.062 | 939 |
| bm25@never | 0.341 | 123 | 0.843 | 51 | 0.225 | 939 |
| bm25@tau_val | 0.341 | 123 | 0.843 | 51 | 0.225 | 939 |
| dense_minilm@never | 0.935 | 123 | 0.980 | 51 | 0.429 | 939 |
| dense_minilm@tau_val | 0.935 | 123 | 0.980 | 51 | 0.429 | 939 |
| cascade_esco@never | 0.943 | 123 | 0.941 | 51 | 0.243 | 939 |
| cascade_esco@tau_val | 0.943 | 123 | 0.941 | 51 | 0.243 | 939 |

Thresholds (validation only): bm25: tau=16.13692683932551, val acc 0.460 (never-abstain 0.267); dense_minilm: tau=0.7647480964660645, val acc 0.581 (never-abstain 0.312); cascade_esco: tau=1.0, val acc 0.540 (never-abstain 0.329)

Paired comparisons vs `cascade_esco@tau_val` (Holm within metric family):

| comparison | metric | delta | CI | p_holm |
|---|---|---|---|---|
| cascade_esco@tau_val − random | top1_acc_linkable | 0.3239 | [0.286, 0.360] | 0.0090 |
| cascade_esco@tau_val − random | nil_aware_accuracy | 0.4612 | [0.427, 0.493] | 0.0090 |
| cascade_esco@tau_val − always_nil | top1_acc_linkable | 0.3239 | [0.286, 0.360] | 0.0090 |
| cascade_esco@tau_val − always_nil | nil_aware_accuracy | 0.0859 | [0.065, 0.108] | 0.0090 |
| cascade_esco@tau_val − exact | top1_acc_linkable | 0.1629 | [0.139, 0.187] | 0.0090 |
| cascade_esco@tau_val − exact | nil_aware_accuracy | -0.0041 | [-0.008, -0.001] | 0.0100 |
| cascade_esco@tau_val − shipped_bridge | top1_acc_linkable | 0.1667 | [0.142, 0.191] | 0.0090 |
| cascade_esco@tau_val − shipped_bridge | nil_aware_accuracy | 0.0147 | [0.005, 0.024] | 0.0100 |
| cascade_esco@tau_val − bm25@never | top1_acc_linkable | 0.0857 | [0.061, 0.110] | 0.0090 |
| cascade_esco@tau_val − bm25@never | nil_aware_accuracy | 0.2429 | [0.206, 0.279] | 0.0090 |
| cascade_esco@tau_val − bm25@tau_val | top1_acc_linkable | 0.0857 | [0.061, 0.110] | 0.0090 |
| cascade_esco@tau_val − bm25@tau_val | nil_aware_accuracy | 0.0812 | [0.056, 0.107] | 0.0090 |
| cascade_esco@tau_val − dense_minilm@never | top1_acc_linkable | -0.1638 | [-0.196, -0.133] | 0.0090 |
| cascade_esco@tau_val − dense_minilm@never | nil_aware_accuracy | 0.1565 | [0.110, 0.200] | 0.0090 |
| cascade_esco@tau_val − dense_minilm@tau_val | top1_acc_linkable | -0.1638 | [-0.196, -0.133] | 0.0090 |
| cascade_esco@tau_val − dense_minilm@tau_val | nil_aware_accuracy | -0.0712 | [-0.097, -0.045] | 0.0090 |
| cascade_esco@tau_val − cascade_esco@never | top1_acc_linkable | 0.0000 | [0.000, 0.000] | 1.0000 |
| cascade_esco@tau_val − cascade_esco@never | nil_aware_accuracy | 0.1894 | [0.151, 0.227] | 0.0090 |

## 4. The shipped linker

- Bridge repo canonical entry -> ESCO (exact normalised labels, no gold): {'entries': 696, 'stage_counts': {'unmapped': 588, 'name_pref': 59, 'alias_pref': 1, 'name_alt': 48}, 'mapped_share': 0.15517241379310343, 'multi_uri_entries': 2}.
- On gold surfaces: {'repo_linked': 1464, 'top1_unbridgeable': 1234, 'repo_unlinked': 245}.
- Replayed cascade rank-1 equals `TaxonomyLinker.link` on 150/150 sampled surfaces.

## 5. Circularity audit of the legacy number and label-free diagnostics (NOT accuracy)

**Is the ~0.890 circular? Yes, in the sense that matters:** there are no gold links at all; the score is a function of the linker and the dictionary it links into, so it cannot detect a wrong link.

- Reproduced exactly from `artifacts/skillllm_v2/full_cag_graph/linked_skills.jsonl`: 0.890374 over 2244 predicted spans (= linked / total). Same for the other stored files: mistral_gliner_taxonomy 0.890271, safe_wrapper 0.889834.
- Method mix of the linked spans: alias 46.3%, bm25 52.1%, fuzzy 1.6%; the BM25 fallback accepts any string sharing a token with an entry, so it is not a correctness signal.
- 601 of 711 alias rows come from SkillSpan train/dev gold spans; 76% of linked predicted spans carry the taxonomy tag SKILLSPAN_GOLD.
- Test contamination: 67/76 test-mined alias rows are test gold surfaces (alias rows `mistral_high_confidence` mined from anchor predictions on this very test split); 41.4% of test gold mentions have a surface in the table ({'manual_curated': 111, 'skillspan_gold_train_dev': 674, 'mistral_high_confidence': 152}).
- No discrimination: coverage is 89.5% on predicted spans that exactly match a gold span (n=1576) vs 87.9% on predicted spans that do not (n=668); 73.6% of the latter's links are BM25 fallbacks.
- Null control (strings with no specific ESCO skill: random non-gold n-grams, Decorte UNDERSPECIFIED mentions): the shipped linker links 64.6% of 861 random non-gold 1-3-grams of the test sentences (method mix {'alias': 19, 'bm25': 521, 'fuzzy': 16}); Decorte UNDERSPECIFIED mentions: n=111, coverage 82.0%.
- All of the above are DIAGNOSTICS on predicted spans; none is a linking accuracy. Legitimate numbers are those in section 3 (Decorte gold).

Cascade-on-ESCO coverage of the predicted spans (diagnostic only):

{'pred_spans': 2233, 'cascade_esco_coverage_at_tau': 0.13255709807433946, 'cascade_esco_nil_rate_at_tau': 0.8674429019256605, 'method_mix': {'alias': 296, 'none': 221, 'bm25': 1648, 'fuzzy': 68}, 'shipped_bridge_coverage': 0.17733990147783252}

## 6. Human annotation sample (for A6 / L11)

`annotation_sample.csv` (blinded: candidate order shuffled per item, no scores, no system pick) and the separate `annotation_answer_key.jsonl`. {'n_items': 300, 'pool_spans': 2233, 'n_strata': 16, 'control_items': 205, 'cells': {'HARD_SKILL|bm25_high|True|False': 36, 'HARD_SKILL|bm25_high|True|True': 33, 'HARD_SKILL|exact_alias|False|False': 13, 'HARD_SKILL|exact_alias|False|True': 28, 'HARD_SKILL|fuzzy|True|False': 12, 'HARD_SKILL|fuzzy|True|True': 9, 'HARD_SKILL|none|True|False': 18, 'HARD_SKILL|none|True|True': 19, 'SOFT_SKILL|bm25_high|True|False': 46, 'SOFT_SKILL|bm25_high|True|True': 23, 'SOFT_SKILL|exact_alias|False|False': 9, 'SOFT_SKILL|exact_alias|False|True': 10, 'SOFT_SKILL|fuzzy|True|False': 11, 'SOFT_SKILL|fuzzy|True|True': 10, 'SOFT_SKILL|none|True|False': 13, 'SOFT_SKILL|none|True|True': 10}, 'pool_strata_sizes': {'HARD_SKILL|bm25_high|True|False': 427, 'HARD_SKILL|bm25_high|True|True': 339, 'HARD_SKILL|exact_alias|False|False': 33, 'HARD_SKILL|exact_alias|False|True': 239, 'HARD_SKILL|fuzzy|True|False': 23, 'HARD_SKILL|fuzzy|True|True': 12, 'HARD_SKILL|none|True|False': 80, 'HARD_SKILL|none|True|True': 95, 'SOFT_SKILL|bm25_high|True|False': 735, 'SOFT_SKILL|bm25_high|True|True': 147, 'SOFT_SKILL|exact_alias|False|False': 9, 'SOFT_SKILL|exact_alias|False|True': 15, 'SOFT_SKILL|fuzzy|True|False': 18, 'SOFT_SKILL|fuzzy|True|True': 15, 'SOFT_SKILL|none|True|False': 32, 'SOFT_SKILL|none|True|True': 14}}

## 7. Caveats

- Decorte et al. report no inter-annotator agreement; gold is a single annotation pass.
- 65 SkillSpan test postings -> cluster bootstrap over few clusters; intervals are wide by design.
- The system has no trained taxonomy reranker (placeholder), so no learned linker is evaluated.
- `exact` and `shipped_bridge` abstain intrinsically; their top-k lists are short, so recall@k ~ top-1.
- Decorte validation/test are both drawn from SkillSpan test; the repo alias table contains 76 aliases mined from anchor predictions on that same split (affects `shipped_bridge` only; no ESCO ids leak).
