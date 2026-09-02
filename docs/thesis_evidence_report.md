# Thesis Research Evidence Report

This file is **not** part of the CSC 898 thesis PDF. It maps major claims to repository evidence for verification.

Repository snapshot: `SkillLLM_v2` commit `0627abf5170ccbe5fe01e5ab1238dc5ac340b880` (branch `skill-llm-v2`); Stage-2 audited from `SkillLLM_v2_stage2/` file artifacts (no git commits at run manifest time).

| Research claim | Repository evidence | File / path | Confidence |
|---|---|---|---|
| SkillSpan train/dev/test = 4800/3174/3569 | summary_report.json splits | `SkillLLM_v2/results/skillspan_baselines/summary_report.json` | High |
| Legacy Mistral exact span F1 = 0.665 | metrics in summary report | `SkillLLM_v2/results/skillspan_baselines/mistral-small-3.1-24b-instruct-2503/metrics.json` | High |
| Legacy Qwen exact span F1 = 0.649 | summary report | same directory tree | High |
| Legacy Gemma E4B exact span F1 = 0.636 | summary report | same | High |
| Legacy Llama exact span F1 = 0.632 | summary report | same | High |
| Gemma 26B benchmark failed | status failed + errors.jsonl | `SkillLLM_v2/results/skillspan_baselines/gemma-4-26b-a4b-it/errors.jsonl` | High |
| Stage-2 Mistral 80-epoch exact span F1 = 0.670 | metrics.json | `SkillLLM_v2_stage2/artifacts/skillllm_v2/mistral_lora/metrics.json` | High |
| Stage-2 BERT exact span F1 = 0.059 | metrics.json | `SkillLLM_v2_stage2/artifacts/skillllm_v2/bert_baseline/metrics.json` | High |
| Stage-2 GLiNER zero-shot exact span F1 = 0.076 | metrics.json + training_mode note | `SkillLLM_v2_stage2/artifacts/skillllm_v2/gliner_baseline/metrics.json` | High |
| Full CAG/graph wrapper exact span F1 = 0.671 | safe_wrapper metrics | `SkillLLM_v2_stage2/artifacts/skillllm_v2/safe_wrapper/full_cag_graph/metrics.json` | High |
| Linking coverage = 0.890 | gate metrics JSON | same file | High |
| Job match nDCG@10 unavailable | matching_eval.csv | `SkillLLM_v2_stage2/reports/skillllm_v2/matching_eval.csv` | High |
| Course rec P@5 unavailable | course_recommendation_eval.csv | `SkillLLM_v2_stage2/reports/skillllm_v2/course_recommendation_eval.csv` | High |
| Mistral training 80 epochs, ~324218s | training_summary.json | `SkillLLM_v2_stage2/artifacts/skillllm_v2/mistral_lora/model/training_summary.json` | High |
| 2× RTX 3090 hardware | run_manifest.json | `SkillLLM_v2_stage2/reports/skillllm_v2/research_refresh/run_manifest.json` | High |
| Generative LoRA SFT (not token fusion) | train_sft_v2.py + mistral_extractor.py | `SkillLLM_v2/src/train/train_sft_v2.py`, Stage-2 extract module | High |
| Smoke ablation F1 ~0.05–0.20 | ablation_raw.json | `SkillLLM_v2_stage2/reports/skillllm_v2/ablation_raw.json` | High (smoke only) |
| Extract_jobDesc BERT entity F1 0.462 gold | test_metrics.json | `SkillLLM_v2/Extract_jobDesc/outputs/gold_bert_base_cased/test_metrics.json` | High |
| Earlier resume benchmark nnose span F1 0.535 | comparison.md | `Experiments/skill_extraction_benchmark/outputs/reports/comparison.md` | Medium (different metrics) |
| SFSU requires dept-approved template | official web page | grad.sfsu.edu thesis guidelines | High |

## Conflicts documented

| Topic | Values | Resolution in thesis |
|---|---|---|
| Mistral exact F1 | 0.665 (3 ep) vs 0.670 (80 ep) | Report both with training protocol labels |
| extraction_ablation.md vs money_table | 0.05 vs 0.67 | Smoke vs full test; primary claims use money_table |
| wrapper_error_analysis Δ signs vs gate accept | small negative soft F1 delta | Use gate JSON and money_table as authoritative for reporting |

## Garbage-text scan target

After editing thesis sources, run:

```bash
grep -RniE 'TODO|FIXME|TBD|XXX|lorem|placeholder|dummy|insert here|citation needed' docs/thesis/ \
  | grep -v 'thesisauthor\|thesisadvisor\|thesiscommittee\|thesisstudentid\|SUBMISSION'
```

Administrative `\newcommand` placeholders in `main.tex` are intentional.
