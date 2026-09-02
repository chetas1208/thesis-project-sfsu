# SkillLLM_v2 Codebase Audit

Generated: `2026-07-13T21:37:00+00:00`

## Repository State

- Branch: `feature/skillllm-v2-research-refresh-20260709`
- Recent commit available: `False`
- Git status:
```
## No commits yet on feature/skillllm-v2-research-refresh-20260709
?? README.md
?? Snakefile
?? artifacts/
?? configs/
?? data/
?? docs/
?? inputs/
?? logs/
?? models/
?? reports/
?? requirements.txt
?? scripts/
?? src/
?? tests/
```

## Modules and Stages

- Python modules under `src/skillllm_v2`: `87`
- Scripts: `38`
- Config files: `8`
- Stage status files: `31`
- Production-ready success stages: `23`
- Smoke-only stages: `8`

## Real Metrics Inspected

| System | Exact F1 | Token F1 | Partial F1 | HARD F1 | SOFT F1 | Artifact |
|---|---:|---:|---:|---:|---:|---|
| BERT baseline | 0.059150 | 0.274922 | 0.176406 | 0.129799 | 0.018796 | `artifacts/skillllm_v2/bert_baseline/metrics.json` |
| GLiNER | 0.075833 | 0.114409 | 0.135365 | 0.054559 | 0.096571 | `artifacts/skillllm_v2/gliner_baseline/metrics.json` |
| Raw Mistral LoRA | 0.670069 | 0.787886 | 0.813845 | 0.726450 | 0.604046 | `artifacts/skillllm_v2/mistral_lora/metrics.json` |
| Old Boundary Refiner | 0.670069 | 0.787886 | 0.813845 | 0.726450 | 0.604046 | `artifacts/skillllm_v2/mistral_boundary/metrics.json` |
| Old Mistral + GLiNER + Taxonomy | 0.670062 | 0.787575 | 0.814438 | 0.728020 | 0.602305 | `artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json` |
| Old Full CAG/Graph | 0.671102 | 0.787575 | 0.815702 | 0.728618 | 0.603755 | `artifacts/skillllm_v2/full_cag_graph/metrics.json` |
| SkillLLM_v2 Safe Wrapper | 0.671102 | 0.787886 | 0.813845 | 0.726450 | 0.603755 | `artifacts/skillllm_v2/safe_wrapper/metrics.json` |

## Wrapper Diagnosis

| Wrapper | Candidate Exact Delta | Selected Exact Delta | Candidate Soft Delta | Selected Soft Delta |
|---|---:|---:|---:|---:|
| Old Boundary Refiner | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Old Mistral + GLiNER + Taxonomy | -0.000007 | -0.000007 | -0.001741 | -0.001741 |
| Old Full CAG/Graph | 0.001033 | 0.001033 | -0.000291 | -0.000291 |
| SkillLLM_v2 Safe Wrapper | -0.670069 | -0.670069 | -0.604046 | -0.604046 |

## Answers

1. Modules exist under `src/skillllm_v2` for baselines, extraction, verification, normalization, graph, matching, training, evaluation, and pipeline orchestration.
2. Pipeline stages are represented by JSON status markers under `artifacts/skillllm_v2/status`.
3. Stages with success markers and full metrics are production-ready for reporting, subject to artifact provenance.
4. Stages ending in `_smoke` are smoke-only gates.
5. Staleness is determined by provenance checks against Raw Mistral prediction timestamps.
6. Full-test metrics are only claimed when they come from persisted `metrics.json` artifacts.
7. Smoke/limited metrics must remain labeled by their own status/config artifacts.
8. Old boundary and full wrapper variants hurt extraction because exact and soft-skill F1 drop below Raw Mistral.
9. Wrapper modules help when they preserve anchor spans and attach normalization/provenance metadata.
10. Current best thesis claim: SkillLLM_v2 should use the fine-tuned Mistral LoRA as the extraction anchor; wrapper components are valuable for normalization, provenance, CAG context, and graph explanations only when gated so they cannot degrade extraction.
