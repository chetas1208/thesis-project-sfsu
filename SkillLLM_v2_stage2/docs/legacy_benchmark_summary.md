# Legacy SkillSpan Benchmark Summary

## Final Model Selection

Mistral Small 3.1 24B LoRA is the selected Stage-2 base extractor because it produced the strongest completed SkillSpan benchmark result while remaining feasible on the available 2x24GB GPU environment.

| Model | Exact Span F1 | Token F1 | Partial Span F1 | Outcome |
| --- | ---: | ---: | ---: | --- |
| mistral-small-3.1-24b | 0.665 | 0.730 | 0.798 | Selected |
| qwen3-14b | 0.649 | 0.714 | 0.789 | Legacy evidence only |
| gemma-4-e4b-it | 0.636 | 0.705 | 0.781 | Legacy evidence only |
| llama-3.1-8b-instruct | 0.632 | 0.704 | 0.764 | Legacy evidence only |

Gemma 4 26B A4B was dropped because repeated benchmark attempts failed under the available hardware. The failures were consistent with model loading and memory infeasibility on 2x24GB GPUs, including quantized dispatch/offload and CUDA out-of-memory errors.

## Stage-2 Money Table Scope

Qwen, Gemma, and Llama are no longer active Stage-2 training paths. Their role is limited to model-selection evidence. The active thesis money table instead benchmarks systems that test the SkillLLM_v2 contribution directly:

- BERT baseline
- GLiNER
- Raw Mistral LoRA
- Mistral + Boundary Refiner
- Mistral + GLiNER + Taxonomy
- SkillLLM_v2 Full CAG/Graph

## Historical Artifact Locations

The old raw benchmark artifacts existed in the original `SkillLLM_v2` repo before Stage-2 cleanup:

- `results/skillspan_baselines/`
- `results/skillspan_baselines_smoke_limit_5/`
- `logs/skillspan_benchmark*.log`
- `logs/skillspan_failed_gemma_retry*.log`
- `models/`
- `.venv/`
- `.cache/`
- `Extract_jobDesc/data/`
- `Extract_jobDesc/logs/`
- `Extract_jobDesc/outputs/`
- `Extract_jobDesc/venv/`

Those large or legacy experiment artifacts are intentionally not active in this Stage-2 repo. The original repo was not modified.
