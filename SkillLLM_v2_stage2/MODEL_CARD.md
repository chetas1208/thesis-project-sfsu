---
# TEMPLATE: fill the <placeholders> and review every claim before publishing an adapter. Nothing has been published.
library_name: peft
base_model: <public Hub id of the base model, e.g. mistralai/Mistral-Small-3.1-24B-Instruct-2503>
license: <decide after the licence review below>
language: en
tags: [skill-extraction, information-extraction, lora, peft, resumes, job-postings]
datasets: [SkillSpan]
---

# <skillllm-anchor | skillllm-loop-v1 | skillllm-loop-v2>: skill span extraction adapter

A LoRA adapter for Mistral-Small-3.1-24B-Instruct that extracts skill mentions (exact character spans, labelled `HARD_SKILL` or `SOFT_SKILL`) from resumes and job postings. It is used through the `skillllm` Python library (`docs/LIBRARY.md`), alone or in a three-adapter majority-vote ensemble.

| Adapter | Training | LoRA | Local path (research repo) |
|---|---|---|---|
| anchor | 80 epochs, 2 GPUs, 180.1 GPU-h | r=16, alpha=32 | `artifacts/skillllm_v2/mistral_lora/model` |
| loop v1 | 4-bit QLoRA loop, 1.59 GPU-h | r=8, alpha=16 | `artifacts/skillllm_v2/qlora_loop/best` |
| loop v2 | 4-bit QLoRA loop, 3.17 GPU-h | r=16, alpha=32 | `artifacts/skillllm_v2/qlora_loop_v2/best` |

All target the language-model attention and MLP projections of the base. The base model is **not** included; obtain it under its own licence.

## Intended use

- Extracting explicit skill mentions from English resume and job-posting text for research, search, analytics, or as a first pass before human review.
- Feeding skill sets into profile-versus-job gap analysis (`SkillLLM.analyze`), where results should be treated as suggestions.

## Out of scope

- Automated hiring, screening, ranking or rejection of people. The system was not evaluated for fairness across demographic groups, languages, or resume formats, and must not be the sole basis of any decision about a person.
- Inferring skills that are not stated, assessing skill level, or verifying that a candidate has a skill.
- Non-English text, very long documents without chunking, or text outside the resume/job-posting domain (not evaluated).

## How to use

```python
from skillllm_v2 import SkillLLM
llm = SkillLLM.from_pretrained("<base model dir or Hub id>", adapters={"anchor": "<adapter dir or Hub id>"}, allow_hub=True)
result = llm.extract("...")
```

Needs `pip install "skillllm[model]"` and a CUDA GPU with about 16 GB free for 4-bit inference.

## Training data

SkillSpan (Zhang et al., NAACL 2022): English job-posting sentences annotated with skill spans (`train`/`dev` splits used for training, `test` held out). The local copy used for the SFT data is `data/source/skillspan_hf`; its dataset card declares **CC BY 4.0**. Attribution is required, and the licence terms of the underlying job-posting text and of any redistribution of derived adapters should be confirmed with the dataset authors before release. Training samples: 4,800 SFT examples for the anchor.

## Evaluation (held-out SkillSpan test, 974 documents with at least one gold span; pooled exact-span match)

| System | Precision | Recall | Exact-span F1 | Token F1 | Partial F1 |
|---|---|---|---|---|---|
| anchor | 0.6735 | 0.6667 | 0.6701 | 0.7879 | 0.8138 |
| loop v1 | 0.6864 | 0.6291 | 0.6565 | 0.7846 | 0.8040 |
| loop v2 | 0.6720 | 0.6512 | 0.6614 | 0.7957 | 0.8170 |
| majority vote (>=2 of 3) | 0.7220 | 0.6605 | 0.6899 | 0.7962 | 0.8213 |

Ensemble minus anchor, exact-span F1 +0.0198 (paired bootstrap 95% CI [+0.0041, +0.0356]); not significant after Holm correction over 5 metrics (p about 0.08). Single adapters are statistically indistinguishable from the anchor. **One training seed per adapter and one split.** Full tables: `reports/skillllm_v2/ensemble_eval.md`, `qlora_loop_v2_significance_vs_anchor.md`.

Agreement among the three adapters is a usable confidence signal: spans predicted by 1, 2 or 3 adapters are correct 29.0%, 54.5% and 80.2% of the time (estimated on the same test split; re-fit with `fit_vote_calibration`). The adapters' own per-span confidence is a constant (0.86) and carries no information.

## Limitations

- About one third of predicted spans are wrong at the exact-span level; many errors are boundary disagreements (partial-span F1 is about 0.81 versus exact 0.67).
- Soft skills are weaker than hard skills (F1 0.60 versus 0.73 for the anchor).
- Trained on a single English dataset with one annotation convention; domain shift (other occupations, informal resumes, other languages) is untested.
- Adapters share a base model and data, so ensemble errors are correlated.
- The model can output strings that are not in the text; the library locates them and drops spans it cannot place.
- Ensembling costs 3x generation per document.
- Taxonomy linking is an alias lookup, not a validated entity linker (no linking-accuracy evaluation exists).
- Real-model loading through the packaged library path (`SkillLLM.from_pretrained`) has been unit-tested with fakes only; the numbers above come from the research evaluation scripts, which use the same prompt, greedy decoding and parsing.

## Safety and privacy

- Resumes contain personal data. Run the model locally, do not log inputs, and follow applicable data-protection law. The library does no network calls (except if you opt into Hub downloads).
- Treat extracted skills as claims made in the text, not verified facts. Do not use them to infer protected attributes.
- The model follows instructions embedded in the text only to the extent that it returns JSON; do not feed its output to a downstream agent without validation (untrusted-input handling still applies).

## Licences (review before publishing)

- Adapter weights: `<choose; at minimum compatible with the base model licence and the training-data terms>`.
- Base model Mistral-Small-3.1-24B-Instruct-2503: check the licence on its model card (stated by its publisher as Apache-2.0 at the time of writing; verify).
- SkillSpan: CC BY 4.0 (attribution: Zhang, Jensen, Plank, "SkillSpan: Hard and Soft Skill Extraction from English Job Postings", NAACL 2022).
- `skillllm` library: MIT, Copyright (c) 2026 Chetas Parekh.

## Citation

`<thesis / repository citation to be added by the author>`
