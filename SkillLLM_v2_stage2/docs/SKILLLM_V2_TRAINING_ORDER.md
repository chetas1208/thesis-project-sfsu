# SkillLLM_v2 Training Order

Full training must not start until its smoke gate passes.

1. Build SkillSpan BIO data.
2. Smoke and train/evaluate the BERT baseline.
3. Smoke and run/fine-tune GLiNER when supported.
4. Build Mistral SFT data.
5. Smoke and train or verify the Mistral LoRA adapter.
6. Evaluate Raw Mistral LoRA.
7. Evaluate Mistral + Boundary Refiner.
8. Evaluate Mistral + GLiNER + Taxonomy.
9. Build/train/evaluate span verifier.
10. Build/train or skip taxonomy reranker based on label availability.
11. Evaluate Full CAG/Graph.
12. Generate ablations and the money table.
