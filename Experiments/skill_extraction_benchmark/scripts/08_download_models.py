#!/usr/bin/env python3
"""Pre-download all required Hugging Face models to the local cache."""

import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_models")

def main():
    # Load environment variables (especially HF_TOKEN)
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set. Gated models like Llama 3 or Gemma may fail to download.")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
    from sentence_transformers import SentenceTransformer
    from gliner import GLiNER

    # 1. NNOSE Backbone
    logger.info("Downloading NNOSE backbone: jjzha/jobbert-base-cased")
    AutoTokenizer.from_pretrained("jjzha/jobbert-base-cased")
    AutoModelForTokenClassification.from_pretrained("jjzha/jobbert-base-cased", ignore_mismatched_sizes=True)

    # 2. GLiNER
    logger.info("Downloading GLiNER: urchade/gliner_medium-v2.1")
    GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    # 3. Skill-LLM
    logger.info("Downloading Skill-LLM bases: meta-llama/Meta-Llama-3-8B-Instruct")
    try:
        AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        # Just loading config/pointers is often enough to cache, but let's download the weights
        # We don't load into memory fully, we just ensure it's in HF cache. We can use snapshot_download.
        from huggingface_hub import snapshot_download
        snapshot_download("meta-llama/Meta-Llama-3-8B-Instruct", ignore_patterns=["*.pth", "*.pt"]) # keep safetensors
    except Exception as e:
        logger.error(f"Failed to download Llama 3: {e}")

    # 4. ConTeXT
    logger.info("Downloading ConTeXT: TechWolf/ConTeXT-Skill-Extraction-base")
    SentenceTransformer("TechWolf/ConTeXT-Skill-Extraction-base", trust_remote_code=True)

    # 5. JobBERT
    logger.info("Downloading JobBERT: jjzha/jobbert_skill_extraction")
    AutoTokenizer.from_pretrained("jjzha/jobbert_skill_extraction")
    AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_skill_extraction")

    # 6. Gemma 4
    gemma_model = "google/gemma-4-e4b-it"
    gemma_fallback = "google/gemma-4-e2b-it"
    logger.info(f"Downloading Gemma 4 base: {gemma_model}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(gemma_model, ignore_patterns=["*.pth", "*.pt"])
    except Exception as e:
        logger.warning(f"Failed to download primary Gemma 4 ({e}). Trying fallback: {gemma_fallback}")
        try:
            snapshot_download(gemma_fallback, ignore_patterns=["*.pth", "*.pt"])
        except Exception as e2:
            logger.error(f"Failed to download fallback Gemma 4: {e2}")

    logger.info("All models downloaded to HuggingFace cache successfully (or handled gracefully).")

if __name__ == "__main__":
    main()
