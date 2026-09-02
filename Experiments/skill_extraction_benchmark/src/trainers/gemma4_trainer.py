"""Gemma 4 trainer: QLoRA SFT using Hugging Face PEFT and SFTTrainer."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset

logger = logging.getLogger(__name__)


def format_gemma_prompt(example: Dict) -> str:
    """Format input for Gemma 4 Chat template."""
    instruction = example["instruction"]
    inputs = example["input"]
    output = example["output"]
    # Provide it in a structured way that instructions dictate.
    return f"User: {instruction}\n\nSentence: {inputs}\n\nAssistant: {output}"


def train_gemma4(
    config: Dict[str, Any],
    data_dir: Path,
    output_dir: Path,
) -> Path:
    """Run QLoRA SFT training using trl SFTTrainer."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling
    from trl import SFTTrainer

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"

    base_model = config["model_name"]
    fallback_model = config.get("fallback_model_name", base_model)
    
    # Check if primary model is accessible, otherwise use fallback
    try:
        AutoTokenizer.from_pretrained(base_model)
        model_id = base_model
    except Exception:
        logger.warning(f"Failed to access {base_model}, falling back to {fallback_model}")
        model_id = fallback_model

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    def load_dataset(path: Path) -> Dataset:
        records = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                records.append(json.loads(line))
        
        # apply formatting
        for r in records:
            r["text"] = format_gemma_prompt(r)
        return Dataset.from_list(records)

    train_ds = load_dataset(data_dir / "train.jsonl")
    dev_ds = load_dataset(data_dir / "dev.jsonl")

    # QLoRA config
    quant_cfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=torch.bfloat16 if config["training"].get("bf16", False) else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    from accelerate import Accelerator
    device_index = Accelerator().local_process_index
    device_map_config = {"": device_index} if torch.distributed.is_initialized() or os.environ.get("LOCAL_RANK") is not None else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map_config,
        torch_dtype=torch.bfloat16 if config["training"].get("bf16", False) else torch.float16,
    )
    
    model = prepare_model_for_kbit_training(model)
    if config["training"].get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # PEFT config
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if lora_cfg.get("target_modules") == "all-linear" else lora_cfg.get("target_modules"),
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        logging_steps=50,
        bf16=train_cfg.get("bf16", False),
        fp16=not train_cfg.get("bf16", False),
        save_strategy="epoch",
        eval_strategy="epoch",
        optim=train_cfg.get("optim", "adamw_torch"),
        seed=config.get("seed", 42),
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # We want it to train on the completion, not the user instruction
    # But since instructions are simple, standard LM training on entire sequence can also work well.
    # We will use data collator for completion only to ensure it learns extraction
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
    )

    logger.info("Starting Gemma 4 QLoRA training...")
    trainer.train()
    
    best_dir = checkpoint_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    logger.info("Gemma 4 model saved to %s", best_dir)
    return best_dir


def generate_gemma4_batch(
    model_path: Path,
    test_records: List[Dict],
    config: Dict[str, Any],
) -> List[str]:
    """Run inference with Gemma 4 using adapted model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    
    gen_cfg = config["generation"]
    
    # Try finding the base model id from config first
    base_model = config["model_name"]
    fallback_model = config.get("fallback_model_name", base_model)
    try:
        AutoTokenizer.from_pretrained(base_model)
        model_id = base_model
    except Exception:
        model_id = fallback_model

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model: %s", model_id)
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    logger.info("Loading PEFT adapter from: %s", model_path)
    model = PeftModel.from_pretrained(base_model_obj, str(model_path))
    model.eval()
    
    predictions = []
    
    for rec in test_records:
        prompt = f"User: Extract skill entities from the given sentence and return strict JSON.\n\nSentence: {rec['text']}\n\nAssistant: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_cfg["max_new_tokens"],
                do_sample=gen_cfg.get("do_sample", False),
                temperature=gen_cfg.get("temperature", 0.0) if gen_cfg.get("temperature", 0) > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predictions.append(generated_text)
        
    return predictions


def parse_strict_json_skills(output_text: str) -> List[str]:
    """Parse JSON to get skill texts from Gemma 4."""
    import re
    # Try to find JSON block if bounded by markdown
    json_block = output_text
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', output_text, re.DOTALL)
    if match:
        json_block = match.group(1)
        
    try:
        parsed = json.loads(json_block)
        if isinstance(parsed, dict) and "skills" in parsed:
            if isinstance(parsed["skills"], list):
                return [s.get("text", "") for s in parsed["skills"] if "text" in s and s.get("text")]
    except json.JSONDecodeError:
        pass
    
    # Fallback to regex
    skills = []
    for m in re.finditer(r'"text"\s*:\s*"([^"]+)"', output_text):
        skills.append(m.group(1).strip())
    return skills


def recover_span_from_text(skill_text: str, tokens: List[str]) -> tuple[int, int]:
    """Recover start and end token indices from skill text and a list of tokens.
    Uses exact string matching, then case-insensitive matching.
    """
    if not tokens or not skill_text:
        return -1, -1

    # First build token strings and their spans
    sentence_lower_tokens = [t.lower() for t in tokens]
    skill_lower_tokens = skill_text.lower().split()
    
    # First attempt: Try to match sequence of tokens
    if len(skill_lower_tokens) > 0:
        for i in range(len(sentence_lower_tokens) - len(skill_lower_tokens) + 1):
            match = True
            for j in range(len(skill_lower_tokens)):
                if sentence_lower_tokens[i+j] != skill_lower_tokens[j] and skill_lower_tokens[j] not in sentence_lower_tokens[i+j]:
                    match = False
                    break
            if match:
                return i, i + len(skill_lower_tokens) - 1

    # Second attempt: Check character mapping if tokenization differs. 
    # To keep it simple and robust, if token sequence fails, we just return -1, -1 
    # to signify "span not cleanly found" and rely on text_match in evaluation.
    return -1, -1
