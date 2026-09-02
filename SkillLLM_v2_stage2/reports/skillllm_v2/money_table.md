# SkillLLM_v2 — Results Summary

SkillSpan evaluation metrics from verified artifact JSON files.

---

## At a glance

Baselines vs the Raw Mistral LoRA anchor.

| System | Exact Span F1 | Token F1 | Partial Span F1 | HARD_SKILL F1 | Soft Skill F1 |
| --- | --- | --- | --- | --- | --- |
| BERT baseline | 0.059150 | 0.274922 | 0.176406 | 0.129799 | 0.018796 |
| GLiNER | 0.075833 | 0.114409 | 0.135365 | 0.054559 | 0.096571 |
| Raw Mistral LoRA (anchor) | 0.670069 | 0.787886 | 0.813845 | 0.726450 | 0.604046 |

---

## Baselines & anchor

| System | Exact Span F1 | Token F1 | Partial Span F1 | HARD_SKILL F1 | Soft Skill F1 | Gate Decision |
| --- | --- | --- | --- | --- | --- | --- |
| BERT baseline | 0.059150 | 0.274922 | 0.176406 | 0.129799 | 0.018796 | NA |
| GLiNER | 0.075833 | 0.114409 | 0.135365 | 0.054559 | 0.096571 | NA |
| Raw Mistral LoRA (anchor) | 0.670069 | 0.787886 | 0.813845 | 0.726450 | 0.604046 | anchor |

---

## Legacy wrapper stages (pre-gate eval)

| System | Exact Span F1 | Token F1 | Partial Span F1 | HARD_SKILL F1 | Soft Skill F1 | Linking Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Old Boundary Refiner | 0.670069 | 0.787886 | 0.813845 | 0.726450 | 0.604046 | NA |
| Old Mistral + GLiNER + Taxonomy | 0.670062 | 0.787575 | 0.814438 | 0.728020 | 0.602305 | 0.890271 |
| Old Full CAG/Graph | 0.671102 | 0.787575 | 0.815702 | 0.728618 | 0.603755 | 0.890374 |

---

## Safe wrapper variants (gated)

| System | Exact Span F1 | Token F1 | Partial Span F1 | HARD_SKILL F1 | Soft Skill F1 | Linking Accuracy | Gate Decision | Candidate Exact Span F1 | Regression vs Raw Mistral |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mistral + Boundary Refiner (safe wrapper) | 0.671111 | 0.787886 | 0.815111 | 0.727048 | 0.605505 | NA | accept | 0.671111 | 0.001042 |
| Mistral + GLiNER + Taxonomy (safe wrapper) | 0.670062 | 0.787575 | 0.814438 | 0.728020 | 0.602305 | 0.890271 | accept | 0.670062 | -0.000007 |
| SkillLLM_v2 Full CAG/Graph (safe wrapper) | 0.671102 | 0.787575 | 0.815702 | 0.728618 | 0.603755 | 0.890374 | accept | 0.671102 | 0.001033 |

---

## Downstream metrics & status

| System | Job Match nDCG@10 | Course Rec P@5 | Regression vs Raw Mistral | Status |
| --- | --- | --- | --- | --- |
| BERT baseline | NA | NA | -0.610919 | present |
| GLiNER | NA | NA | -0.594236 | present |
| Raw Mistral LoRA (anchor) | NA | NA | 0.000000 | present |
| Old Boundary Refiner | NA | NA | 0.000000 | present |
| Old Mistral + GLiNER + Taxonomy | NA | NA | -0.000007 | present |
| Old Full CAG/Graph | NA | NA | 0.001033 | present |
| Mistral + Boundary Refiner (safe wrapper) | NA | NA | 0.001042 | selected_candidate |
| Mistral + GLiNER + Taxonomy (safe wrapper) | NA | NA | -0.000007 | selected_candidate |
| SkillLLM_v2 Full CAG/Graph (safe wrapper) | NA | NA | 0.001033 | selected_candidate |

---

## Artifact paths

- **BERT baseline** — `artifacts/skillllm_v2/bert_baseline/metrics.json`
- **GLiNER** — `artifacts/skillllm_v2/gliner_baseline/metrics.json`
- **Raw Mistral LoRA (anchor)** — `artifacts/skillllm_v2/mistral_lora/metrics.json`
- **Old Boundary Refiner** — `artifacts/skillllm_v2/mistral_boundary/metrics.json`
- **Old Mistral + GLiNER + Taxonomy** — `artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json`
- **Old Full CAG/Graph** — `artifacts/skillllm_v2/full_cag_graph/metrics.json`
- **Mistral + Boundary Refiner (safe wrapper)** — `artifacts/skillllm_v2/safe_wrapper/boundary_refiner/metrics.json`
- **Mistral + GLiNER + Taxonomy (safe wrapper)** — `artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/metrics.json`
- **SkillLLM_v2 Full CAG/Graph (safe wrapper)** — `artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json`

---

**Reading this table:** Raw Mistral LoRA (anchor) is the strongest span-extraction system on its own and is the non-regression anchor for every "(safe wrapper)" row (see configs/skillllm_v2_wrapper_guardrails.yaml and skillllm_v2.eval.non_regression_gate). For "(safe wrapper)" rows, the F1 columns report the gate's *selected* extraction: the wrapper's own output if `Gate Decision = accept`, otherwise the unmodified Raw Mistral anchor predictions (`Gate Decision = fallback_to_anchor`). "Candidate Exact Span F1" is the wrapper variant's own pre-gate exact-span F1, kept for transparency even on fallback. "Linking Accuracy" is the wrapper's taxonomy-linking coverage, produced for downstream job/course recommendation regardless of the gate decision -- this is the basis for the claim that SkillLLM_v2's CAG/Graph layer improves product utility (normalization, explainability, recommendation) even on rows where it does not improve raw span-extraction F1 over Raw Mistral.
