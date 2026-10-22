# Pre-training feasibility memo: domain-adaptive and task-adaptive pretraining for SkillLLM_v2

Draft for the thesis committee. Status as of 2026-10-22. This is a **feasibility memo, not a result**: no pretraining was run, and no model was loaded. The measurements below are CPU-only counts made for this memo with the tokenizer saved beside the anchor adapter (`artifacts/skillllm_v2/mistral_lora/model/tokenizer.json`), scripts kept outside the repository. Posting text in the local job-description file is private and licensed material; only counts are reported here and no text was copied.

Evidence conventions: numbers from repository files are cited by path; **derived** = arithmetic shown on read numbers; **from memory, verify** = my recollection of a published figure that I could not check, because the `Research Papers` folder (`/home/923873155/Sem 4 Research/Research Papers`) is **empty** (verified with `ls -la`, 2026-10-22) and `Resume+Job Data/` is also empty. The only references available are bibliography entries (`docs/thesis/references.bib`), which carry titles and authors but not the papers' content.

## 1. Question and short answer

Question: would continued pretraining on unlabeled in-domain text (domain-adaptive pretraining, DAPT) or on the task's own unlabeled text (task-adaptive pretraining, TAPT; Gururangan et al., ACL 2020, bibliography key `gururangan2020dontstop`) improve the skill extractor, and is it feasible with the resources of this project (CPU for analysis; two 24 GB-class RTX 3090 GPUs shared with other jobs; a stated total cap of 8 GPU-h across the new GPU experiments in `docs/GPU_EXPERIMENTS_LOG.md`, which excludes the earlier loop runs)?

Short answer: **not worth doing for the 24B model.** Three independent reasons, each quantified below: (1) the in-domain unlabeled text that exists locally is only 3.2 to 4.9 million tokens of job postings and none of the deployment domain (resumes); (2) DAPT of a 24B model on that corpus costs 4 to 20 GPU-h per pass-count/seed before the required re-fine-tuning and evaluation, which exhausts the 8 GPU-h cap; (3) even a genuine +1 point F1 gain could not be detected on the 974-document test set, whose minimum detectable effect is about 3 points (`reports/skillllm_v2/qlora_loop_summary.md`), so the experiment cannot produce thesis-quality evidence at this budget. The realistic alternatives and their ranking are in section 7.

## 2. Background (what DAPT and TAPT are, and what I can and cannot cite)

Gururangan et al. (2020) continue masked-language-model pretraining of a general encoder (RoBERTa) on a large unlabeled domain corpus (DAPT) or on the unlabeled text of the downstream task's own training set (TAPT), then fine-tune, and report gains on in-domain classification tasks. Specific figures I recall, **from memory, verify against the paper**: DAPT corpora of the order of 2 to 8 billion tokens (for example roughly 7.6 B for biomedical papers, 8.1 B for computer-science papers, 6.7 B for news, 2.1 B for reviews), about 12.5 K steps at batch 2,048; TAPT on the task text for about 100 epochs; gains typically one to a few F1 points, larger when the domain is far from the original pretraining data, with TAPT cheap and DAPT expensive. Two relevant differences from this project: the paper's models were encoders of about 125 M parameters trained with full-parameter updates, whereas this project's extractor is a 24B decoder with 4-bit weights and LoRA adapters that is already instruction-tuned. I have no evidence in this repository about whether job-posting text helps such a model.

## 3. How much unlabeled in-domain text actually exists

All counts are for the Mistral tokenizer with truncation disabled. (An initial run with the tokenizer's built-in 512-token truncation capped every document at 512 tokens and gave a wrong total of 0.96 M; I caught and corrected this, and mention it because the saved `tokenizer.json` ships with truncation on.)

**Table P1. Text inventory.**

| Source | Records | Words | Mistral tokens | Notes |
|---|---:|---:|---:|---|
| SkillSpan train text (the SFT input text) | 4,800 rows | 93,453 | 124,441 | the TAPT-scale corpus; 3,952 unique texts |
| SkillSpan dev + test text | 3,174 + 3,569 rows | 82,843 | 110,555 | using test text, even unlabeled, would be transductive |
| SkillSpan all splits | 11,543 rows | 176,296 | 234,996 | labelled corpus total |
| Local job-posting pool `/home/923873155/Sem 4 Research/SkillLLM_v2/Extract_jobDesc/data/job_descriptions.jsonl` (field `job_description_text`) | 1,887 postings | 2,697,939 | **4,907,501** | 19.6 M characters; 112 postings are exact duplicates after whitespace/case normalisation; median 1,457 tokens, 90th percentile 3,772, one outlier of 143,180 |
| Same pool after the project's own deduplication (`.../data/processed/jobs_deduped.jsonl`, field `raw_text`) | 1,246 postings | 1,721,960 | **3,191,685** | `.../reports/dedup_report.json`: 1,887 to 1,246 (536 exact, 3 normalised, 102 near duplicates at similarity 0.92) |
| Same pool, sentence-segmented (`.../data/processed/job_sentences.jsonl`) | 56,767 sentences | 1,221,216 | 1,663,273 | about 29 tokens per sentence |
| Resumes | **0** | 0 | 0 | `Resume+Job Data/` is empty; no resume text exists anywhere I looked |
| `inputs/` and `data/` of this repository | synthetic course catalogue (171 courses, 17,448 words), synthetic evaluation cases, SkillSpan-derived files | | | `inputs/` is 380 KB and synthetic; `data/` (30 MB) re-encodes SkillSpan and contains no new text |

Provenance and quality of the pool (from `Extract_jobDesc/README.md`): postings scraped from SimplifyJobs GitHub repositories (new-graduate and internship tables), so the pool is tech-heavy, early-career and US/international; raw scrape, "can contain private or licensed material". Existing weak labels: `data/silver/skillspan_style_silver.jsonl` (56,767 sentences, 6,217 with at least one span = 11%; 9,878 hard and 2,176 soft spans per `reports/dataset_stats.json`) produced by `src/weak_label/weak_label.py`, whose first lines are a hand-written keyword dictionary; its precision is unknown, `data/gold/*.jsonl` are all empty (0 lines; `gold_validation_report.json` totals 0), and `data/gold_review/review_sheet.csv` has 250 rows of which 60 have non-empty `corrected_spans` and `notes` (I counted; who filled them is not recorded).

Contamination check (counts only). I lowercased and tokenised the pool and the SkillSpan test texts into 8-word shingles: 536 of the 1,887 postings share at least one 8-gram with the test texts, but the shared 8-grams number only 93 distinct strings out of 22,085 test 8-grams, which is the signature of boilerplate (equal-opportunity and benefits statements), not verbatim reuse of test sentences. This is not a formal decontamination (no fuzzy matching, 8-word window) but it gives no sign of test leakage through the pool. Whether the pool overlaps the base model's own pretraining data is unknowable.

**Table P2. Scale against typical DAPT corpora** (remembered sizes, **from memory, verify**).

| Comparison | Ratio (derived) |
|---|---:|
| Pool (raw 4.9 M tokens) versus the smallest remembered DAPT corpus (about 2.1 B) | about 430x smaller |
| Pool (deduplicated 3.2 M) versus the largest remembered (about 8.1 B) | about 2,500x smaller |
| Pool versus the labelled SkillSpan corpus (235 K tokens) | 21x larger (raw), 14x larger (deduplicated) |
| SkillSpan train text (124 K) versus pool (4.9 M) | pool is 39x larger |

Conclusion for section 3: what exists is **TAPT-scale, not DAPT-scale**. It is 21x the labelled corpus, which makes it useful as a source of silver labels (section 7), but it is two to three orders of magnitude below the corpora in the DAPT literature, and it covers job postings only.

## 4. Cost estimate for continued pretraining of the 24B (4-bit, one GPU)

Throughput basis: the anchor training ran 1.184 samples per second at 351.9 tokens per sample on 2 GPUs (`artifacts/skillllm_v2/mistral_lora/model/training_summary.json`; Table T2 of `training.md`), which is about 417 tokens/s, or **208 tokens/s per GPU**, a figure that includes the per-epoch evaluation time (an estimated 15.6 h of 90 h, not recorded). Excluding that estimate gives about 252 tokens/s per GPU. Both are derived and assume the same 4-bit QLoRA kernels, gradient checkpointing and similar sequence lengths; the true figure for packed 1,024-token text could differ by a factor of about two in either direction, which I cannot measure here.

**Table P3. GPU-hours for continued pretraining on one GPU (derived).**

| Corpus | Tokens | Per epoch at 252 / 208 tok/s | 3 epochs |
|---|---:|---:|---:|
| SkillSpan train text only (TAPT-style) | 124,441 | 0.14 / 0.17 GPU-h | 0.4 to 0.5 GPU-h |
| Deduplicated pool | 3,191,685 | 3.5 / 4.3 GPU-h | 10.6 to 12.8 GPU-h |
| Raw pool | 4,907,501 | 5.4 / 6.5 GPU-h | 16.2 to 19.6 GPU-h |

A DAPT run is only the first of three costs. A usable experiment also needs (a) re-fine-tuning on SkillSpan (3.2 GPU-h for a loop-v2 run; `training.md` Table T7), (b) a 974-document test evaluation (about 1.2 GPU-h; same table), and (c) at least three seeds per arm to estimate seed variance (the two identical anchor runs already differ by 0.005 exact-F1). One arm therefore costs roughly (3.5 GPU-h for one epoch of the deduplicated pool, up to 19.6 GPU-h for three epochs of the raw pool) + 3.2 + 1.2 = 8 to 24 GPU-h for a single seed (derived), against a stated cap of 8 GPU-h for the whole project and a loop-v2 total of 4.4 GPU-h with evaluation. Implementation issues that add risk but that I did not cost: LoRA-based continued pretraining on a 4-bit base must either be merged into a bf16 base (about 48 GB, more than a 24 GB card) or be stacked with the SFT adapter; the code has no causal-LM-on-raw-text trainer (the existing trainers format chat examples), and the loop trainer would need a new data path; adapter-merge changes numerics versus the evaluated 4-bit setting.

## 5. Expected benefit

- **Evidence in this repository: none.** No experiment compares a domain-adapted and a plain base.
- **Weak prior.** The literature's DAPT gains are for encoders of modest size fine-tuned fully on classification tasks and are largest for domains far from the original corpus (from memory, verify). Job postings are plain English business text, and a 24B instruction model very likely saw similar text; the anchor already reaches 0.670 exact-F1 and 0.814 partial-span F1, so the headroom is mostly boundary and annotation-convention disagreement (`reports/skillllm_v2/eval_suite/REPORT.md`: 325 boundary shifts versus 317 misses, 294 spurious and 113 label confusions at the anchor), which domain vocabulary exposure does not obviously address.
- **A TAPT-like term is already inside the anchor.** Its loss covers all tokens, including the 124,441 input-text tokens, for 80 epochs (`training.md` section 2.1). The loop adapters mask the text; their slightly lower scores cannot be attributed to that, because nothing is ablated, but a fresh TAPT stage would partly duplicate what the anchor already did.
- **My judgement (not a measured quantity):** a plausible gain is within about 1 point or zero; I have no basis for a larger figure.
- **Measurability.** At 974 test documents the minimum detectable paired effect is about 3 points for comparisons between independently trained adapters (the wrapper variants have MDEs of only 0.23 to 0.34 points, but only because they alter 9 to 19 documents of the same predictions). A 1-point effect would need roughly nine times as many test documents (MDE scales as 1/sqrt(n); derived, back-of-envelope) or many seeds. The experiment could not confirm or refute the expected gain.
- **Domain mismatch.** The deployment input is resumes and profiles; the pool is postings. Postings-only pretraining does not touch the largest unevaluated shift (`INFERENCE_AND_EVALS.md` section 4, item 12: resume text is not evaluated).

## 6. Risks

1. Forgetting or damage to instruction following and strict-JSON output (the system relies on parseable JSON; an unparsable generation aborts an evaluation loop, per the audit).
2. Numerics: merging into a lower-precision base changes behaviour relative to every evaluated number.
3. Licensing and terms of use: the pool is scraped from employer career pages; the README warns it may contain licensed material; training on it and releasing an adapter needs review (`MODEL_CARD.md` already defers licence review).
4. Contamination: no sign beyond boilerplate (section 3), but the check is shallow.
5. Budget: the experiment would crowd out the replicate runs that the existing results need more (section 7, `BEST_PLAN.md`).
6. Selection effects: any DAPT arm tuned against the same test split would repeat the dev-to-test shrinkage already seen in the loop (4 to 5 points; `training.md` section 4.3).

## 7. Ranked recommendation

**Table P4. Options ranked by expected value per cost.** GPU-hour figures are derived from Tables P3 and `training.md` T7; "value" is my qualitative judgement.

| Rank | Option | Estimated cost | Why / caveat |
|---:|---|---|---|
| 1 | **Do not run DAPT or CPT of the 24B.** | 0 | Sections 3 to 6. Revisit only if a large resume corpus (order of 10^8 tokens or more) and a larger test set with at least 5,000 gold-bearing documents become available. |
| 2 | **Prompt compression** (running as GPU Experiment 1 / loop v3, `prompt.version: short`; about 77 fixed tokens by my count, 41 by the log's, against about 300; training tokens per epoch 599,308 against 1,688,908 by my count, derived) | cap 3.2 GPU-h plus about 1.2 GPU-h evaluation; **pending** | Cuts train and prefill tokens by roughly 2.8x to 3.1x; the best possible claim is "no detectable loss"; adapters need the matching prompt at inference. The log itself predicts the latency criterion will probably fail because generation is decode-bound. |
| 3 | **Out-of-distribution evaluation set** (annotate 200 to 300 resume and posting sentences, finishing the existing 250-row `review_sheet.csv` of 2026 postings plus new resume text) | human time, about 0 GPU-h of training, one inference pass | Highest information per cost: the deployment domain is unmeasured. Gives a domain-shift number for the anchor and the ensemble. |
| 4 | **Self-training / pseudo-label distillation from ensemble agreement** on the pool (being piloted as GPU Experiment 2 / loop v4: 600 postings, up to 2,400 chunks after removing 55 postings that overlap SkillSpan, silver spans = at least 2 of 3 adapters, warm start from v2; **pending**, `training.md` Table T13). My own variant would use spans on which all three agree (precision 0.80, recall 0.51 on SkillSpan test; `training.md` Table T11), add them to SkillSpan and retrain a loop adapter | pseudo-labelling cost (derived, one-document-at-a-time at 3.6 to 4.4 s per adapter): 2,000 sentences 6.0 to 7.4 GPU-h (the planned 2,400 chunks would be 7.2 to 8.8 GPU-h unbatched, i.e. the whole 8 GPU-h cap, so batched generation, which the log specifies but whose speed is unmeasured, is essential); 5,000 sentences 15.1 to 18.4 GPU-h; all 56,767 sentences 171 to 209 GPU-h, i.e. as much as the anchor. Batched generation is not measured here and is the necessary optimisation. Then 3.2 GPU-h per training run | Silver labels are 20% wrong at 3-of-3 agreement and miss half of true spans, so unlabeled-span positions become false negatives; use positive-only or agreement-masked loss. Correlated members confirm each other's errors. Same MDE problem (about 3 points). Worth a small pilot only after batching and after rank 3. |
| 5 | **TAPT on SkillSpan train text only** | 0.4 to 0.5 GPU-h for 3 epochs (Table P3) | Cheap but largely redundant with the anchor's all-token loss; expected effect about zero; also needs a new trainer path. Low priority. |
| 6 | **Small-encoder adaptation for the linker and span verifier** (a 100 M-class encoder, continued pretraining on the 3.2 to 4.9 M-token pool) | order of minutes to an hour of GPU (6ND estimate 3.2e15 FLOPs per epoch, derived; not measured) | This is the one place where the pool is in the right size regime. But the blocker is labels, not pretraining: there are no gold canonical links, so linking accuracy cannot be measured, and the dense linker is disabled (`sentence_transformers` missing, `reports/skillllm_v2/gpu_component_availability.md`); on real merged candidates the trained verifier has AUROC 0.579 against 0.835 on its synthetic dev set (`reports/skillllm_v2/eval_suite/REPORT.md`). Do after gold links exist. |

Decision rule proposed for the thesis: report DAPT as "considered and rejected on feasibility and measurability grounds, with the numbers in Tables P1 to P3", and spend the freed GPU budget on replicate seeds (`BEST_PLAN.md` step list), which the existing claims need more than any pretraining stage.

## 8. Items I could not verify or that disagree

- DAPT corpus sizes, step counts and gain sizes in sections 2 and 3 are **from memory, verify**; the paper is not in the repository (the folder is empty).
- The 208 and 252 tokens/s per GPU figures are derived from the anchor run with approximate token counts; real CPT throughput is unmeasured.
- The 8 GPU-h cap is defined in `docs/GPU_EXPERIMENTS_LOG.md` as a total across the new experiments (the earlier v2 loop and its test evaluation are excluded); the log bills generation wall-seconds, so evaluation passes count against it.
- Whether the keyword-dictionary silver labels are accurate is unknown; no gold exists.
- The pool's relationship to the base model's pretraining data is unknown.
