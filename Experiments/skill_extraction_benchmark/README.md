# 5-Model Skill Extraction Benchmark

A fully reproducible, research-grade pipeline to benchmark 5 skill extraction models on the `jjzha/skillspan` dataset.

## Models Compared
1. **NNOSE** (kNN-augmented token classification)
2. **GLiNER** (Zero-shot/few-shot spans)
3. **Skill-LLM** (QLoRA instruction tuning on Llama 3 8B)
4. **ConTeXT** (Sentence-transformer dense retrieval)
5. **JobBERT** (Standard token classification baseline)
6. **Gemma 4** (QLoRA instruction tuning producing exact JSON arrays, mapping string generation text back to token matching)

## Generative Baselines
### Gemma 4
Gemma 4 is included as a strong, open-weight generative baseline (`google/gemma-4-e4b-it`). Unlike token classification, it generates skills strictly in JSON format via fine-tuning. We implement robust text-to-token span recovery mapping the generated exact string matches back to their original token alignment in the source sentence when evaluating.

## Features
- **Canonical data pipeline**: Unifies data into a single JSONL schema.
- **Unified evaluation**: Span-level P/R/F1, text-match F1, document-level set F1, Jaccard similarity.
- **Efficiency tracking**: Measures inference latency and peak GPU memory per model.
- **End-to-End Orchestration**: A single script handles everything from download to the final markdown comparison table.

## Setup
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Environment Variables
cp .env.example .env
# Edit .env and supply your HF_TOKEN (needed for Llama 3)
```

## Running
To run the entire pipeline end-to-end:
```bash
bash scripts/40_run_all.sh
```

## Outputs
- **Models**: Saved in `outputs/models/`
- **Predictions**: Sentence-level in `outputs/predictions/` and document-level in `outputs/predictions/document_level/`
- **Metrics**: Detailed metrics in `outputs/metrics/`
- **Report**: Final markdown and CSV tables in `outputs/reports/`
