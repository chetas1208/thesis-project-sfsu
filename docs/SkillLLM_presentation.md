# SkillLLM_v2 — 5-Minute Recording Script (No Slides)

Record yourself talking over your screen. No slide deck — just these files open and flip between them as noted.

---

## Before you hit record

**Open exactly these 3 files (split screen or quick Alt-Tab):**

1. `SkillLLM_v2_stage2/reports/skillllm_v2/money_table.md` — leave this visible most of the time
2. `SkillLLM_v2/results/skillspan_baselines/summary_report.json` — have it ready in a tab
3. `SkillLLM_v2_stage2/reports/skillllm_v2/demo/job_matches.json` — optional, last 30 seconds

Optional fourth tab if asked later: `docs/thesis/main.tex` (abstract only)

**Do not open:** `extraction_ablation.md` (smoke test, misleading numbers)

---

## Full script (~500 words · ~5 minutes at normal pace)

### [0:00–0:45] Intro — show `money_table.md` (top of file)

> Hi, this is a five-minute update on SkillLLM_v2, my CSC 898 thesis project at SFSU.
>
> The problem is straightforward: job postings and resumes mention skills as free text — things like "Python," "communication," "machine learning" — but applications like job matching and course recommendations need structured skills with exact spans, hard versus soft labels, and links to a skill taxonomy.
>
> We evaluate on SkillSpan, the standard benchmark for hard and soft skill extraction from English job postings. The question I'm answering is: can a locally run large language model do this well, and can we build a full career-intelligence pipeline on top without breaking extraction quality?

---

### [0:45–1:35] What we built — still on `money_table.md`, scroll slowly

> The project has two phases.
>
> Phase A, in SkillLLM_v2, was a controlled benchmark of five open-weight instruction-tuned models — same data, same QLoRA training setup, same evaluation code on two RTX 3090 GPUs. Mistral Small 3.1 24B won at zero point six six five exact span F1. Gemma 26B failed to run on our hardware, so it's documented as infeasible, not ranked.
>
> Phase B, in SkillLLM_v2_stage2, takes that winner and builds the full system: an eighty-epoch Mistral LoRA extractor as the anchor, plus GLiNER and dictionary matching for recall, taxonomy linking, a NetworkX graph over skills and jobs and courses, and modules for job matching, gap detection, and course recommendations. Everything passes through a non-regression gate so the wrapper can't silently degrade the anchor's span predictions.

---

### [1:35–2:45] Results — point at rows in `money_table.md`

> Here are the numbers that matter. I'm reading from our primary results table.
>
> Under the same Stage-2 evaluation protocol, BERT baseline gets zero point zero five nine exact span F1. Zero-shot GLiNER gets zero point zero seven six. The raw Mistral LoRA anchor gets **zero point six seven zero**.
>
> That's roughly eleven times better than the classical baselines — not a marginal improvement.
>
> The full context-aware graph wrapper — that's the row labeled "SkillLLM_v2 Full CAG/Graph safe wrapper" — hits zero point six seven one exact span F1, so plus zero point zero zero one over the anchor, and **eighty-nine percent taxonomy linking coverage**.
>
> That plus-zero-point-zero-zero-one is intentional. The wrapper's job isn't to massively boost F1. It's to add normalization, linking, and downstream reasoning while the safe gate guarantees we never ship extraction worse than the anchor. All three wrapper variants passed the non-regression gate with an "accept" decision.

---

### [2:45–3:30] Architecture — scroll `money_table.md` or switch to repo tree briefly

> Architecturally, text goes in, Mistral produces JSON skill spans, GLiNER and the dictionary add candidates, those get merged and verified, spans get linked to ESCO-style taxonomy aliases, and the graph connects users, skills, jobs, and courses. Cached context holds stable profile and taxonomy summaries; retrieval pulls dynamic job text at inference time.
>
> The pipeline has thirty-one stages, all succeeded in our last full run. End-to-end demo outputs exist — extracted profile skills, job matches, skill gaps, and course recommendations on synthetic demo inputs.

---

### [3:30–4:15] Honest gaps — switch to `demo/job_matches.json` briefly, then back to `money_table.md`

> What's done: the multi-model benchmark, eighty-epoch Mistral training — about ninety hours on two GPUs — full pipeline evaluation, and a thesis first draft with a sixty-nine-paper bibliography.
>
> What's not quantified yet: job match nDCG and course recommendation precision show NA in our eval table because we don't have labeled downstream evaluation sets. The matching and recommendation modules run on demo data — you can see job match output here — but I am not claiming validated recommendation metrics in the thesis. That's an explicit scope boundary.

---

### [4:15–5:00] Close — `money_table.md`, anchor row visible

> Bottom line: SkillLLM_v2 extracts skills at **zero point six seven exact span F1** using a locally fine-tuned Mistral model, far ahead of BERT and GLiNER, with a guarded wrapper that adds **eighty-nine percent skill linking** and a working demo pipeline. Thesis draft is in progress; remaining work is format check, PDF build, and optionally a small labeled set for downstream eval if time allows.
>
> Thanks — that's where the project stands right now.

---

## Timing summary

| Section | Time | On screen |
|---------|------|-----------|
| Intro | 0:00–0:45 | `money_table.md` |
| What we built | 0:45–1:35 | `money_table.md` |
| **Results** | 1:35–2:45 | `money_table.md` (point at rows) |
| Architecture + pipeline | 2:45–3:30 | `money_table.md` or repo tree |
| Gaps + demo | 3:30–4:15 | `job_matches.json` → `money_table.md` |
| Close | 4:15–5:00 | `money_table.md` (anchor row) |

---

## Three numbers to memorize (no reading needed)

- **0.670** — Mistral anchor exact span F1  
- **0.059 / 0.076** — BERT / GLiNER (baselines)  
- **89%** — taxonomy linking coverage  

---

## If you run short (~4 min)

Add 30 seconds on soft vs hard skills: hard F1 is **0.726**, soft F1 is **0.604** — soft skills remain harder, consistent with prior SkillSpan literature.

## If you run long (~6 min)

Cut the architecture paragraph (2:45–3:30) to two sentences: "Mistral extracts, wrapper links and graphs, gate protects quality."

---

## Recording checklist

- [ ] Close email/notifications  
- [ ] Zoom/screen recorder set to capture **the file tabs**, not a blank desktop  
- [ ] Font size large enough to read F1 numbers on `money_table.md`  
- [ ] Read results section once out loud before recording  
- [ ] One take is fine — this is a status update, not a performance  

---

*All numbers from `SkillLLM_v2_stage2/reports/skillllm_v2/money_table.md` and `SkillLLM_v2/results/skillspan_baselines/summary_report.json`.*
