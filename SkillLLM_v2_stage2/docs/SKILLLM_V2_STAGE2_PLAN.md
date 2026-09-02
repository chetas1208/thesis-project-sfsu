# SkillLLM_v2 Stage-2 Plan

## Objective

Convert SkillLLM_v2 from a multi-model benchmarking repo into a thesis-facing Context-Aware Graph-Augmented Skill Intelligence System.

## Active Loop

1. User profile text is processed by the skill extraction wrapper.
2. Extracted spans are normalized to canonical taxonomy skills.
3. User skills are compared with job requirements.
4. Missing, weak, outdated, and uncertain skills are detected.
5. Courses are recommended to close gaps.
6. Completed courses or newly detected skills update the user profile.

## Model Policy

Mistral Small 3.1 24B LoRA is the only active main LLM extractor. GLiNER is retained only as a lightweight recall booster. Dictionary and taxonomy matching provide traceable deterministic candidates. Qwen, Gemma, and Llama are legacy benchmark evidence only.

## Validation Gates

- `bash scripts/setup_skillllm_v2_stage2.sh`
- `bash scripts/run_skillllm_v2_smoke.sh 5`
- `PYTHONPATH=src pytest tests/`
- `bash scripts/run_skillllm_v2_extraction_ablation.sh 5`
- Review `docs/master_merge_checklist.md`
