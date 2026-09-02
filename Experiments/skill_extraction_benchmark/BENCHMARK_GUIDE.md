# Benchmarking & Testing Guide

This guide explains how to properly test, benchmark, and debug the 6-model Skill Extraction pipeline locally on your RTX 3090s.

## 1. Environment Verification

Before running the full pipeline, ensure the virtual environment is correctly set up and models are cached.

```bash
cd "/home/923873155/Sem 4 Research/Experiments/skill_extraction_benchmark"

# Activate the environment
source venv/bin/activate

# Verify your huggingface token is loaded
cat .env
```

## 2. Running the Full End-to-End Benchmark

If you want to run the full pipeline in a single go (producing the final comparison tables automatically), simply run:

```bash
bash scripts/40_run_all.sh
```

This script will sequentially:
1. Download `jjzha/skillspan` and create canonical `train/dev/test` splits.
2. Export the data into 6 model-specific prompt formats.
3. Train all 6 models one by one.
4. Run inference locally using all 6 trained checkpoints.
5. Aggregate the metrics and output the unified `comparison.md` and `comparison.csv`.

## 3. Testing Individual Models (Step-by-Step)

If you prefer to benchmark or debug a single model (e.g. Gemma 4), you can run its pipeline phase-by-phase.

### Phase A: Data Prep
Run the shared canonical parser, then the model-specific exporter:
```bash
python scripts/00_download_skillspan.py
python scripts/01_prepare_canonical.py
python scripts/07_export_gemma4.py   # export Gemma 4 dataset
```

### Phase B: Training
Model configurations (batch size, epochs, QLoRA) are controlled by `configs/gemma4.yaml`.
```bash
python scripts/15_train_gemma4.py
```
> **Performance Note**: For local testing on your 3090s, if training is taking too long for testing purposes, you can edit `configs/gemma4.yaml` and set `num_train_epochs: 1`.

### Phase C: Inference
```bash
python scripts/25_predict_gemma4.py
```
This will output `outputs/predictions/gemma4_test.jsonl` (sentence-level) and `outputs/predictions/document_level/gemma4_doc.jsonl` (doc-level).

### Phase D: Evaluation
You can evaluate whatever models happen to be locally complete at the moment:
```bash
python scripts/30_evaluate_span_metrics.py
python scripts/31_evaluate_document_metrics.py
python scripts/32_build_comparison_table.py
```

## 4. Debugging & Common Issues

- **Out of Memory (OOM) Errors**: If you encounter CUDA OOM errors during `Skill-LLM` or `Gemma 4` training, open their config files (`configs/skill_llm.yaml` / `configs/gemma4.yaml`), reduce `per_device_train_batch_size: 1`, and ensure `gradient_checkpointing: true` and `load_in_4bit: true` are active.
- **Hugging Face Unauthorised**: If Llama 3 or Gemma 4 fails to download, ensure that `HF_TOKEN` in the `.env` file corresponds to an account that has explicitly accepted the license agreement on the Hugging Face website.
- **Wiping Slate Clean**: To restart the entire benchmark cleanly from scratch, simply delete the generated folders:
  ```bash
  rm -rf data/raw data/exports outputs/models outputs/predictions outputs/metrics outputs/reports
  ```

## 5. Interpreting the Output Results

Once execution completes, your final benchmark tables will be located here:
- `outputs/reports/comparison.md` (Formatted markdown table)
- `outputs/reports/comparison.csv` (Raw spreadsheet data)

**Key Metrics Tracked:**
1. **Span F1**: Sequence-level token exact match (perfectly bounded boundaries).
2. **Text F1**: Text-level match (did it extract the right string representation without enforcing strict token boundaries?).
3. **Doc Set F1 / Jaccard**: Document-level skill tracking useful for candidate-level ATS resume tracking.
4. **Time/Sent (s)**: Wall-clock latency for parsing a sentence during inference.
5. **Peak Mem (MB)**: Maximum GPU footprint monitored via CUDA allocations.
