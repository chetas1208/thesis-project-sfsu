# Evaluation card — SkillLLM_v2 research evaluation (proposed name: SkillGraphLM)

Generated 2026-11-18T08:28:42+00:00 by `python -m skillllm_v2.evals.research.repro_card` (re-runnable). Machine-readable twin: `eval_card.json` (complete lists; this page abbreviates long lists).

## 1. Code version

- Repository `/home/923873155/Sem 4 Research`, branch `main`, commit `54aab906d548d871ad855bdbb09b7eea18ddf0d5` (2026-11-13T21:12:37+00:00).
- Working tree dirty: **True** (9 entries under SkillLLM_v2_stage2). The evaluation code is uncommitted, so the commit alone does not identify it. The file hashes below and in `intermediate_outputs` do.

## 2. Datasets

SkillSpan (HF export, CC-BY-4.0, Zhang et al. 2022):

| file | bytes | sha256 |
|---|---|---|
| data/source/skillspan_hf/.gitattributes | 2307 | f4e703ea6e44bbebe53aceed2a89c11e40b88b7ae130c480c89860ab805ffc8f |
| data/source/skillspan_hf/README.md | 1291 | 5f30719a8689118ddc6933df468d363efeae54a220b80e90661e24f9815dbd68 |
| data/source/skillspan_hf/dev.json | 1036099 | 3732937c6869fe191c749b4efe7ffd597ce84eea1192f09b7fe3bd1eb459e2b4 |
| data/source/skillspan_hf/test.json | 1116219 | 8438eda34939a5877b984ea5daf9e6ff39b2d5c7e3d73cca0499d3de3784d24a |
| data/source/skillspan_hf/train.json | 2207996 | 5e884c3426087d14fc437d328eb72132082edbfe1e776292ae0e6e9b36c1e1f6 |

Derived SkillLLM_v2 data: 11 files; course benchmark inputs: 4 files; eval inputs: 72 files (hashes in JSON).

| file | sha256 |
|---|---|
| data/skillllm_v2/bio/dev.jsonl | b8ca34d0a0af5c14 |
| data/skillllm_v2/bio/test.jsonl | 0acaf5ca232bd542 |
| data/skillllm_v2/bio/train.jsonl | 6586c2feebd171bc |
| data/skillllm_v2/linker_dev.jsonl | c056845cf25fbd44 |
| data/skillllm_v2/linker_train.jsonl | c056845cf25fbd44 |
| data/skillllm_v2/mistral_sft/dev.jsonl | 698a3d33f9fd6eee |
| data/skillllm_v2/mistral_sft/test.jsonl | e4280d26987f92ff |
| data/skillllm_v2/mistral_sft/train.jsonl | 54dbe6156c7a5cf2 |
| data/skillllm_v2/taxonomy/skill_aliases.jsonl | afb3055760d0ad15 |
| data/skillllm_v2/verifier_dev.jsonl | 392da61f3f1c4693 |
| data/skillllm_v2/verifier_train.jsonl | ef452ee6e2ec40db |
| inputs/courses/benchmark_scenarios.jsonl | 99d436e2f8e274f4 |
| inputs/courses/course_catalog.jsonl | f2f6b48d0aac54d3 |
| inputs/courses/course_catalog_large.jsonl | 1b42b903e75f53fa |
| inputs/courses/relevance_labels.jsonl | ce64a36c7e7f4e98 |

External datasets (`data/external/MANIFEST.jsonl`, sha256 `6b5fe2f8f7895c9fa8bb59f3aebb43e13e5906dd409e451a047b057e33a324b3`):

| name | version_or_commit | license | downloaded_by | n_files |
|---|---|---|---|---|
| cpr_recover | 95a7fa7311abc6fb0b8ef49d74d393f56b1bc05f | CC BY-NC-SA 4.0 | A5 | 5 |
| al_cpl | 11397eaa3ea01c5e06fefcfc1c02a84f21164081 | CC BY-NC-SA 4.0 | A5 | 9 |
| lecturebank | b237565ded5b112295fd5bc39035f53b0dd4286b | unspecified (no license file; public research release, citation requested) | A5 | 6 |
| green_hf | a3500da00e98b04ba7af65a230586f6432755bb7 | CC BY 4.0 (HF card) | A3 | 5 |
| sayfullina_hf | 624b4f8ff58be7de4c5df1565b998cd8c8fd92f7 | unknown (HF card says 'unknown'; research use only, not redistributed) | A3 | 5 |
| skillskape_hf | d3a3c6e49a08157068b6b258acf4659269ab76cd | unspecified on HF card (research release) | A3 | 5 |
| skillspan_paper | ACL Anthology 2022.naacl-main.366 | CC BY 4.0 (ACL Anthology) | A1 | 1 |
| esco_skills_v1.1.0 | ESCO v1.1.0 (version used by the Decorte gold) | ESCO: European Commission, reuse under Commission Decision 2011/833/EU (free reuse incl. commercial, with attribution) | A4 | 4 |
| esco_skills_v1.2.1 | ESCO v1.2.1 (latest available in the API on download day) | ESCO: European Commission, reuse under Commission Decision 2011/833/EU (free reuse incl. commercial, with attribution) | A4 | 4 |
| decorte_skill_extraction_tech | 5f0cb05df03cd1cd5048ff1d64639bf9bd4cb36f | CC-BY-4.0 | A4 | 3 |
| decorte_skill_extraction_house | dcb313cad615a8f5148ef5addc01ef090fc9f295 | CC-BY-4.0 | A4 | 3 |
| decorte_skill_extraction_techwolf | ed83e50ff6bc7e44008e98a82f6143ddaf59a255 | CC-BY-4.0 | A4 | 2 |

## 3. Models and adapters

- Base model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (local `/home/923873155/Sem 4 Research/SkillLLM_v2_stage2/models/mistral-small-3.1-24b-instruct-2503` -> `/usr/data/models/skillllm-v2/models/mistral-small-3.1-24b-instruct-2503`); revision: not recorded locally (plain snapshot directory, no HF cache metadata).
- Base config/tokenizer hashes: SYSTEM_PROMPT.txt=33a5139b633a, chat_template.json=d4b1a286509c, config.json=ce3ec410cac7, generation_config.json=8e8cb26b98a0, model.safetensors.index.json=664a049408e8, params.json=23febf6d98a7, preprocessor_config.json=0f1a312ed75c, processor_config.json=4ceef0d0cbf0, special_tokens_map.json=c98562eff7be, tekken.json=c604f35d1035, tokenizer.json=b76085f99233, tokenizer_config.json=85d2cc5ea2cd
- Base weights: 11 safetensors files, not hashed in this run (pass --hash-base-weights; ~96 GB).

| adapter | weights_sha256 | config_sha256 | checkpoints |
|---|---|---|---|
| artifacts/skillllm_v2/mistral_lora/model | 36ff344f684844a6 | 96818dcb87a1daec | checkpoint-5925, checkpoint-6000 |
| artifacts/skillllm_v2/qlora_loop/best | eba864e87e91e858 | 560c41c048e9ac9f | - |
| artifacts/skillllm_v2/qlora_loop_smoke/best | 15279d317900d09d | 560c41c048e9ac9f | - |
| artifacts/skillllm_v2/qlora_loop_v2/best | b196b4acca45d6c3 | 96818dcb87a1daec | - |
| artifacts/skillllm_v2/qlora_loop_v3_shortprompt/best | 3c56c1dcb8f63cad | 96818dcb87a1daec | - |
| artifacts/skillllm_v2/qlora_loop_v4_distill/best | 1e75b5a532530d64 | 96818dcb87a1daec | - |

Hugging Face cache revisions used by the evaluation (models downloaded by agents are listed with their snapshot):

| repo | snapshots | cache |
|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 1110a243fdf4 | /home/923873155/.cache/huggingface/hub |
| microsoft/deberta-v3-base | 8ccc9b6f3619 | /usr/data/models/skillllm-v2-stage2/hf_cache_a2/hub |
| urchade/gliner_medium-v2.1 | 40ec419335d0 | /usr/data/models/skillllm-v2-stage2/hf_cache_a2/hub |
| bert-base-cased | cd5ef92a9fb2 | /usr/data/models/skillllm-v2-stage2/hf_cache_a3/hub |
| jjzha/jobbert-base-cased | eee1c8da6200 | /usr/data/models/skillllm-v2-stage2/hf_cache_a3/hub |
| microsoft/deberta-v3-base | 8ccc9b6f3619 | /usr/data/models/skillllm-v2-stage2/hf_cache_a3/hub |
| urchade/gliner_medium-v2.1 | 40ec419335d0 | /usr/data/models/skillllm-v2-stage2/hf_cache_a3/hub |

## 4. Predictions

| file | rows | sha256 | mtime_utc |
|---|---|---|---|
| artifacts/skillllm_v2/bert_baseline/predictions.jsonl | 3569 | 8f7145b6096a7971 | 2026-07-09T23:38:37+00:00 |
| artifacts/skillllm_v2/evals_gpu/anchor_dev_logprobs/predictions.jsonl | 914 | 0c6bed963b0da331 | 2026-11-17T22:47:08+00:00 |
| artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/predictions.jsonl | 974 | b4c75aa87bc9ce70 | 2026-11-17T22:53:28+00:00 |
| artifacts/skillllm_v2/evals_gpu/anchor_test_nospan/predictions.jsonl | 2595 | 45e383df48097461 | 2026-11-17T23:24:15+00:00 |
| artifacts/skillllm_v2/evals_gpu/anchor_test_nospan/shards/shard0of2/predictions.jsonl | 1298 | 15f7192a32661031 | 2026-11-17T23:24:14+00:00 |
| artifacts/skillllm_v2/evals_gpu/anchor_test_nospan/shards/shard1of2/predictions.jsonl | 1297 | 1ee2507c5a65e0fc | 2026-11-17T23:18:30+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/predictions.jsonl | 3569 | 0c71b22ceb8a9d65 | 2026-11-18T06:53:55+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/runs/bert-base-cased_s42/predictions.jsonl | 3569 | b8a6b261815e5db6 | 2026-11-17T23:28:00+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/runs/bert-base-cased_s43/predictions.jsonl | 3569 | 75f9061f1363d465 | 2026-11-17T23:37:26+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/runs/bert-base-cased_s44/predictions.jsonl | 3569 | 6e0dcb0dd40f42a6 | 2026-11-17T23:46:42+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/runs/jjzha__jobbert-base-cased_s42/predictions.jsonl | 3569 | 0c71b22ceb8a9d65 | 2026-11-17T23:33:32+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/runs/jjzha__jobbert-base-cased_s43/predictions.jsonl | 3569 | 1693f04045f0ae2a | 2026-11-17T23:42:48+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/runs/jjzha__jobbert-base-cased_s44/predictions.jsonl | 3569 | 915542e7737d74d9 | 2026-11-17T23:49:20+00:00 |
| artifacts/skillllm_v2/evals_gpu/bert_fair/smoke/runs/jjzha__jobbert-base-cased_s42/predictions.jsonl | 200 | d5ff44a50c7369f6 | 2026-11-17T21:03:13+00:00 |
| artifacts/skillllm_v2/evals_gpu/robustness_anchor/predictions.jsonl | 1373 | dbdd5e09cbbbab13 | 2026-11-18T01:16:03+00:00 |
| artifacts/skillllm_v2/evals_gpu/robustness_anchor/shards/shard0of2/predictions.jsonl | 687 | d5b604b17996deab | 2026-11-18T01:16:01+00:00 |
| artifacts/skillllm_v2/evals_gpu/robustness_anchor/shards/shard1of2/predictions.jsonl | 686 | 4a2bb8054b6579c2 | 2026-11-18T01:12:20+00:00 |
| artifacts/skillllm_v2/evals_gpu/robustness_bert_fair/predictions.jsonl | 1373 | 8ed842ee010f6207 | 2026-11-18T07:03:42+00:00 |
| artifacts/skillllm_v2/evals_gpu/robustness_gliner_zeroshot/predictions.jsonl | 1373 | c5c47be40d75d8df | 2026-11-17T21:49:06+00:00 |
| artifacts/skillllm_v2/evals_gpu/robustness_safe_wrapper/predictions.jsonl | 1373 | 53d4a248976ac9d2 | 2026-11-18T07:21:47+00:00 |
| artifacts/skillllm_v2/evals_gpu/robustness_token_lookup/predictions.jsonl | 1373 | 033fee9beb1fcc60 | 2026-11-17T21:25:44+00:00 |
| artifacts/skillllm_v2/evals_gpu/smoke/anchor_test_logprobs/predictions.jsonl | 4 | a3d61685c8432305 | 2026-11-17T20:58:14+00:00 |
| artifacts/skillllm_v2/evals_gpu/smoke/rss_bert_fair/predictions.jsonl | 100 | d1ecd487a222424e | 2026-11-18T07:38:29+00:00 |
| artifacts/skillllm_v2/evals_gpu/smoke/rss_gliner_zeroshot/predictions.jsonl | 100 | c478dd0e8c77d5d0 | 2026-11-18T07:37:55+00:00 |
| artifacts/skillllm_v2/evals_gpu/smoke/rss_safe_wrapper/predictions.jsonl | 100 | 51a8c2e44d963564 | 2026-11-18T07:39:35+00:00 |
| artifacts/skillllm_v2/evals_gpu/smoke/rss_token_lookup/predictions.jsonl | 100 | b8d20ba1978edc3c | 2026-11-18T07:36:48+00:00 |
| artifacts/skillllm_v2/evals_gpu/test974_bert_fair/predictions.jsonl | 974 | 93eb4c52ef44df2e | 2026-11-18T07:31:34+00:00 |
| artifacts/skillllm_v2/evals_gpu/test974_gliner_zeroshot/predictions.jsonl | 974 | 363b6f6ac6606489 | 2026-11-17T21:38:23+00:00 |
| artifacts/skillllm_v2/evals_gpu/test974_safe_wrapper/predictions.jsonl | 974 | 20c34995e73d8e62 | 2026-11-17T21:52:54+00:00 |
| artifacts/skillllm_v2/evals_gpu/test974_token_lookup/predictions.jsonl | 974 | 448df7d0bc890315 | 2026-11-17T21:20:17+00:00 |
| artifacts/skillllm_v2/evals_gpu/xdata_anchor/predictions.jsonl | 600 | fca6712893b882bc | 2026-11-18T01:46:06+00:00 |
| artifacts/skillllm_v2/evals_gpu/xdata_anchor/shards/shard0of2/predictions.jsonl | 300 | 075d409dcfabd085 | 2026-11-18T01:46:05+00:00 |
| artifacts/skillllm_v2/evals_gpu/xdata_anchor/shards/shard1of2/predictions.jsonl | 300 | 41f81ae9f7da4471 | 2026-11-18T01:40:52+00:00 |
| artifacts/skillllm_v2/evals_gpu/xdata_bert_fair/predictions.jsonl | 600 | c31b93bb8ec3f54b | 2026-11-18T07:07:02+00:00 |
| artifacts/skillllm_v2/evals_gpu/xdata_gliner_zeroshot/predictions.jsonl | 600 | 9b9525236daa2b8a | 2026-11-17T21:55:21+00:00 |
| artifacts/skillllm_v2/evals_gpu/xdata_safe_wrapper/predictions.jsonl | 600 | f50dba2e7211a598 | 2026-11-18T07:27:58+00:00 |
| artifacts/skillllm_v2/evals_gpu/xdata_token_lookup/predictions.jsonl | 600 | 70dd107407b146c7 | 2026-11-17T21:49:08+00:00 |
| artifacts/skillllm_v2/full_cag_graph/predictions.jsonl | 974 | 6e123ffd118928ab | 2026-07-13T21:33:47+00:00 |
| artifacts/skillllm_v2/gliner_baseline/predictions.jsonl | 3569 | 743403d3fc8dd36d | 2026-07-09T23:38:20+00:00 |
| artifacts/skillllm_v2/mistral_boundary/predictions.jsonl | 974 | cce3137d16931366 | 2026-07-13T19:08:06+00:00 |
| artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl | 974 | 4308decd3a57ec5e | 2026-07-13T20:20:40+00:00 |
| artifacts/skillllm_v2/mistral_lora/predictions.jsonl | 974 | ac5917d0267670b0 | 2026-07-13T19:08:02+00:00 |
| artifacts/skillllm_v2/qlora_loop_eval/predictions.jsonl | 974 | b3c3589033644702 | 2026-10-21T17:00:47+00:00 |
| artifacts/skillllm_v2/qlora_loop_v2_eval/predictions.jsonl | 974 | e4d305ddda358a6c | 2026-10-22T03:36:45+00:00 |
| artifacts/skillllm_v2/qlora_loop_v3_eval/predictions.jsonl | 974 | 6e17040c89e48a7b | 2026-10-22T12:00:23+00:00 |
| artifacts/skillllm_v2/qlora_loop_v4_eval/predictions.jsonl | 974 | 1027a8955e314e3a | 2026-10-22T17:58:10+00:00 |
| artifacts/skillllm_v2/safe_wrapper/boundary_refiner/predictions.jsonl | 974 | 39c34a24645f65d8 | 2026-10-21T08:45:47+00:00 |
| artifacts/skillllm_v2/safe_wrapper/full_cag_graph/predictions.jsonl | 974 | 945e748c0ba0bcce | 2026-10-21T08:45:47+00:00 |
| artifacts/skillllm_v2/safe_wrapper/mistral_gliner_taxonomy/predictions.jsonl | 974 | 1ffef118b4661578 | 2026-10-21T08:45:47+00:00 |
| artifacts/skillllm_v2/safe_wrapper/predictions.jsonl | 974 | 945e748c0ba0bcce | 2026-10-21T08:45:47+00:00 |
| artifacts/skillllm_v2/safe_wrapper_gpu_smoke/boundary_refiner/predictions.jsonl | 20 | 693044d217cb87f2 | 2026-07-09T23:11:10+00:00 |
| artifacts/skillllm_v2/safe_wrapper_gpu_smoke/full_cag_graph/predictions.jsonl | 20 | 879e2aece1037011 | 2026-07-09T23:11:10+00:00 |
| artifacts/skillllm_v2/safe_wrapper_gpu_smoke/mistral_gliner_taxonomy/predictions.jsonl | 20 | 79f28b8db8ddcb95 | 2026-07-09T23:11:10+00:00 |
| artifacts/skillllm_v2/safe_wrapper_gpu_smoke/predictions.jsonl | 20 | 879e2aece1037011 | 2026-07-09T23:11:10+00:00 |

## 5. Taxonomy, graph, retrieval index

- Taxonomy `data/skillllm_v2/taxonomy/skill_aliases.jsonl` sha256 `afb3055760d0ad15a6d95bd0454728d1aa1e716b6e3cdfc3a028c8d3a712e7e6`: 711 alias rows, 696 concepts, alias sources {'manual_curated': 34, 'mistral_high_confidence': 76, 'skillspan_gold_train_dev': 601}.
- External taxonomies: [{'dir': 'data/external/esco', 'n_files': 1380}].
- Graph: The course/skill graph is rebuilt in memory from the catalogue + taxonomy on every run (no persisted production snapshot); its identity is the hash of these inputs plus the code commit. Inputs: `inputs/courses/course_catalog_large.jsonl`=1b42b903e75f, `inputs/courses/course_catalog.jsonl`=f2f6b48d0aac, `data/skillllm_v2/taxonomy/skill_aliases.jsonl`=afb3055760d0, `artifacts/skillllm_v2/smoke/graph.json`=3e1a235d55e0
- Retrieval index: BM25 (k1=1.5, b=0.75) and the dense index (sentence-transformers/all-MiniLM-L6-v2, CPU) are rebuilt from the catalogue text at run time; no index file is persisted. Identity = catalogue hash + model revision in the HF cache section.

## 6. Prompts

| name | file | sha256 | default |
|---|---|---|---|
| extraction/original/system | reports/skillllm_v2/evals/repro/prompts/extraction_original_system.txt | b8988b207b3c24a6 | True |
| extraction/original/user_template | reports/skillllm_v2/evals/repro/prompts/extraction_original_user_template.txt | b2458f41c56802ef | True |
| extraction/short/system | reports/skillllm_v2/evals/repro/prompts/extraction_short_system.txt | 1b389698c05bffe1 | False |
| extraction/short/user_template | reports/skillllm_v2/evals/repro/prompts/extraction_short_user_template.txt | d52307c69140b28c | False |
| base_model_chat_template.json | reports/skillllm_v2/evals/repro/prompts/base_model_chat_template.json | d4b1a286509cd7a4 |  |
| base_model_default_SYSTEM_PROMPT.txt | reports/skillllm_v2/evals/repro/prompts/base_model_default_SYSTEM_PROMPT.txt | 33a5139b633aed86 |  |
| anchor_adapter_chat_template.jinja | reports/skillllm_v2/evals/repro/prompts/anchor_adapter_chat_template.jinja | 5f42291cf1b71a8e |  |

The shipped anchor and loop-v1/v2 adapters were trained and evaluated with the `original` prompt; `short` applies only to the loop-v3 adapter trained with it.

## 7. Seeds

- Layer seeds: {'L01_extraction': 42, 'L02_generalization': 42, 'L03_robustness': 42, 'L04_calibration': 42, 'L05_wrapper': 42, 'L06_linking': 42, 'L07_compaction': 42, 'L08_graph_retrieval': 42, 'L09_course_rec': 42, 'L10_significance': 42, 'L11_human_eval': 42, 'L12_systems': 42}
- Training config seeds: {'artifacts/skillllm_v2/mistral_lora/model/mistral_lora_train_config.yaml': 42}
- Bootstrap: seed 42, 2,000 resamples (10,000 headline); recommender benchmark seed 20260101, split salt `skillllm-recsys-loop-v1`.

## 8. Software environment

- **cpu_miniconda** `/opt/miniconda/bin/python` Python 3.13.12; pip freeze `reports/skillllm_v2/evals/repro/env_cpu_miniconda.txt` (sha256 adb2d7254e1a). Key packages: numpy 2.5.2; scipy 1.17.1; scikit-learn 1.8.0; torch 2.5.1+cu121; transformers 5.14.1; peft 0.19.1; accelerate 1.13.0; sentence-transformers 5.4.1; networkx 3.6.1; matplotlib 3.10.9; pytest 9.0.3
- **gpu_venv** `/home/923873155/Sem 4 Research/SkillLLM_v2/.venv/bin/python` Python 3.12.3; pip freeze `reports/skillllm_v2/evals/repro/env_gpu_venv.txt` (sha256 05ed077eeaf9). Key packages: numpy 1.26.3; scipy 1.16.3; scikit-learn 1.8.0; torch 2.6.0; transformers 5.8.0.dev0; peft 0.11.1; bitsandbytes 0.49.2; accelerate 1.9.0; gliner 0.2.8; networkx 3.6.1; matplotlib 3.10.9
- torch / CUDA / cuDNN / cuda_available (GPU venv; availability as seen by the card process, which the CPU launcher runs without GPUs): `2.6.0+cu124 12.4 91900 False`
- nvidia-smi: `0, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 00000000:1A:00.0
1, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 00000000:68:00.0`
- nvcc: `Build cuda_12.2.r12.2/compiler.32965470_0`

## 9. Hardware

- Host `srva`, Linux-6.8.0-147-generic-x86_64-with-glibc2.39
- CPU: CPU(s):                                  20 | Model name:                              Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz | Thread(s) per core:                      2 | Socket(s):                               1 | CPU(s) scaling MHz:                      89%
- MemTotal:       131571780 kB
- GPUs: 0, NVIDIA GeForce RTX 3090, 24576 MiB | 1, NVIDIA GeForce RTX 3090, 24576 MiB
- /usr/data: /dev/mapper/data--vg-data--lv  2.7T  594G  2.2T  22% /usr/data

## 10. Exact commands per layer (from results.json meta.argv)

| layer | command | utc | git | status |
|---|---|---|---|---|
| L01_extraction | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.extraction_full | 2026-11-18T06:54:54+00:00 | 54aab906d548 dirty | present |
| L02_generalization | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.generalization | 2026-11-18T06:58:38+00:00 | 54aab906d548 dirty | present |
| L03_robustness | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.robustness | 2026-11-18T07:35:14+00:00 | 54aab906d548 dirty | present |
| L04_calibration | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.calibration_selective | 2026-11-18T07:08:45+00:00 | 54aab906d548 dirty | present |
| L05_wrapper | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.wrapper_contribution | 2026-11-17T21:26:54+00:00 | 54aab906d548 dirty | present |
| L06_linking | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.linking_eval --seed 42 --n-boot 2000 | 2026-11-18T08:23:32+00:00 | 54aab906d548 dirty | present |
| L07_compaction | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.compaction --seed 42 --n-boot 2000 --n-profiles 120 | 2026-11-17T21:16:49+00:00 | 54aab906d548 dirty | present |
| L08_graph_retrieval | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.graph_retrieval --seed 42 --n-boot 2000 | 2026-11-18T07:08:07+00:00 | 54aab906d548 dirty | present |
| L09_course_rec | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.course_rec --seed 42 --n-boot 2000 | 2026-11-18T07:14:29+00:00 | 54aab906d548 dirty | present |
| L10_significance | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.significance | 2026-11-18T08:28:22+00:00 | 54aab906d548 dirty | present |
| L11_human_eval | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.human_eval_samples --seed 42 --n-boot 2000 | 2026-11-18T08:25:23+00:00 | 54aab906d548 dirty | present |
| L12_systems | PYTHONPATH=src /opt/miniconda/bin/python -m skillllm_v2.evals.research.systems | 2026-11-18T07:42:00+00:00 | 54aab906d548 dirty | present |
| (all) | bash scripts/evals/run_research_evals.sh [--smoke] | - | - | runner |

Heavy jobs run through `/home/923873155/server-guardrails/bin/guardrails run cpu|--ram 16 gpu0|gpu1 -- <cmd>`; GPU jobs (anchor re-inference, fair BERT-family baselines, robustness) are listed with their logs in `Progress.md`.

## 11. Intermediate outputs

462 files under `reports/skillllm_v2/evals/`, `inputs/evals/research/` and `artifacts/skillllm_v2/evals_gpu/`, each with sha256 in `eval_card.json` (`intermediate_outputs.files`). GPU jobs with DONE.json: ['artifacts/skillllm_v2/evals_gpu/anchor_dev_logprobs/DONE.json', 'artifacts/skillllm_v2/evals_gpu/anchor_test_logprobs/DONE.json', 'artifacts/skillllm_v2/evals_gpu/anchor_test_nospan/DONE.json', 'artifacts/skillllm_v2/evals_gpu/bert_fair/DONE.json', 'artifacts/skillllm_v2/evals_gpu/robustness_anchor/DONE.json', 'artifacts/skillllm_v2/evals_gpu/robustness_bert_fair/DONE.json', 'artifacts/skillllm_v2/evals_gpu/robustness_gliner_zeroshot/DONE.json', 'artifacts/skillllm_v2/evals_gpu/robustness_safe_wrapper/DONE.json', 'artifacts/skillllm_v2/evals_gpu/robustness_token_lookup/DONE.json', 'artifacts/skillllm_v2/evals_gpu/test974_bert_fair/DONE.json', 'artifacts/skillllm_v2/evals_gpu/test974_gliner_zeroshot/DONE.json', 'artifacts/skillllm_v2/evals_gpu/test974_safe_wrapper/DONE.json', 'artifacts/skillllm_v2/evals_gpu/test974_token_lookup/DONE.json', 'artifacts/skillllm_v2/evals_gpu/xdata_anchor/DONE.json', 'artifacts/skillllm_v2/evals_gpu/xdata_bert_fair/DONE.json', 'artifacts/skillllm_v2/evals_gpu/xdata_gliner_zeroshot/DONE.json', 'artifacts/skillllm_v2/evals_gpu/xdata_safe_wrapper/DONE.json', 'artifacts/skillllm_v2/evals_gpu/xdata_token_lookup/DONE.json'].

## 12. Known reproducibility limits

- Stored anchor offsets were produced by an older `schema.find_offsets` (first plain-substring match); the current code matches word boundaries first, so re-running the anchor cannot reproduce stored offsets byte-for-byte (Progress.md integrity-flag, A3). Model-output identity and offset effects are reported separately by L01/L12.
- One training run per adapter; two same-seed 80-epoch runs differed by 0.005 exact F1 (run-1 adapter overwritten).
- Base-model revision is not recorded in the local copy; the config/tokenizer hashes above identify it.
- The course catalogue, scenarios and recommender labels are synthetic and project-authored.

