# Taxonomy Linking Error Analysis

Alias-table-backed `TaxonomyLinker` loaded 696 canonical entries from `data/skillllm_v2/taxonomy/skill_aliases.jsonl` (grouped by `canonical_skill_id`, taxonomy priority ESCO > ONET > LIGHTCAST > CUSTOM > SKILLSPAN_GOLD), replacing the previous 4-entry hardcoded fallback.

Linking is attach-only: re-linking never changes `surface_form`/`start`/`end`/`label`/`source`, so it cannot affect extraction Exact/Token/Soft/Hard Skill F1 -- only the separate `linking_accuracy` (coverage) metric.

## Mistral + GLiNER + Taxonomy (`mistral_gliner_taxonomy`)

- Total predicted spans: 2469
- Linking coverage (old 4-entry fallback): 0.069259
- Linking coverage (new 696-entry alias table): 0.882139
- Correctly-extracted (TP) spans: 1542
- TP linking coverage (new table): 0.896239

### Link method breakdown (new table)

| Method | Count |
|---|---|
| bm25 | 1139 |
| alias | 999 |
| none | 291 |
| fuzzy | 40 |

### Per-label coverage (new table)

| Label | Total | Linked | Coverage |
|---|---|---|---|
| HARD_SKILL | 1313 | 1158 | 0.881950 |
| SOFT_SKILL | 1156 | 1020 | 0.882353 |

### Most common unlinked TP surface forms (candidates for alias-table extension)

| Surface form | Count |
|---|---|
| message | 1 |
| wealthtech | 1 |
| subscriptions | 1 |
| resource groups | 1 |
| sonarqube | 1 |
| infer# | 1 |
| iis ) | 1 |
| yaml | 1 |
| appservices | 1 |
| servicebus | 1 |
| keyvault | 1 |
| k8s | 1 |
| containers | 1 |
| postgis | 1 |
| opencv | 1 |
| numpy | 1 |
| shapely | 1 |
| estimate timelines | 1 |
| deliver initiatives | 1 |
| fixed income instruments | 1 |

## Full CAG/Graph (`full_cag_graph`)

- Total predicted spans: 2601
- Linking coverage (old 4-entry fallback): 0.062284
- Linking coverage (new 696-entry alias table): 0.838908
- Correctly-extracted (TP) spans: 1367
- TP linking coverage (new table): 0.874909

### Link method breakdown (new table)

| Method | Count |
|---|---|
| bm25 | 1124 |
| alias | 1014 |
| none | 419 |
| fuzzy | 44 |

### Per-label coverage (new table)

| Label | Total | Linked | Coverage |
|---|---|---|---|
| HARD_SKILL | 1314 | 1156 | 0.879756 |
| SOFT_SKILL | 1287 | 1026 | 0.797203 |

### Most common unlinked TP surface forms (candidates for alias-table extension)

| Surface form | Count |
|---|---|
| plm | 2 |
| message | 1 |
| wealthtech | 1 |
| subscriptions | 1 |
| resource groups | 1 |
| sonarqube | 1 |
| infer# | 1 |
| iis ) | 1 |
| yaml | 1 |
| appservices | 1 |
| servicebus | 1 |
| keyvault | 1 |
| k8s | 1 |
| containers | 1 |
| postgis | 1 |
| opencv | 1 |
| numpy | 1 |
| shapely | 1 |
| estimate timelines | 1 |
| communicate | 1 |

## Notes

- `linking_error_cases.jsonl` lists every correctly-extracted (TP) span across all stages above that the new alias-table linker still fails to attach a `canonical_skill_id` to (capped at 200 per stage). These are the highest-value additions for `MANUAL_ALIAS_PAIRS` in `skillllm_v2.normalize.skill_alias_builder` or for an external ESCO/O*NET/Lightcast export.
