# GPU Component Availability

- Config: `configs/skillllm_v2_training_80ep_2gpu.yaml`
- CUDA available: `True`
- Visible GPUs: `2`

| Component | GPU Required | GPU Capable | Training Mode | Missing |
| --- | --- | --- | --- | --- |
| bert_baseline | False | True | current implementation is CPU-fast token baseline; GPU BERT Trainer is not the active implementation | none |
| gliner | False | True | zero-shot/fallback unless GLiNER fine-tuning support is added | none |
| mistral_lora | True | True | 2-GPU torchrun QLoRA/LoRA | none |
| span_verifier | False | False | CPU scikit-learn TF-IDF/logistic verifier by design | none |
| taxonomy_reranker | False | False | retrieval-only unless supervised labels and embedding/reranker deps are provided | sentence_transformers |
| graph_match_course | False | False | CPU NetworkX/scoring logic by design | none |

## Warnings

- BERT baseline currently uses the deterministic CPU-fast token classifier, not a GPU Hugging Face Trainer.
