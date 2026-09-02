from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from skillllm_v2.config import load_yaml
from skillllm_v2.extract.mistral_extractor import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from skillllm_v2.io_utils import read_jsonl, write_json


def train(config: dict[str, Any]) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        import torch
        import transformers
        from transformers import AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("Mistral LoRA training requires torch, transformers, datasets, peft, and bitsandbytes.") from exc

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_path = config["model_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(AutoTokenizer, model_path)
    _ensure_chat_template(tokenizer, model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "attn_implementation": config.get("attn_implementation", "sdpa"),
    }
    if bool(config.get("use_4bit", True)):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        model_kwargs["device_map"] = {"": local_rank} if world_size > 1 else "auto"

    model = _load_model(transformers, model_path, model_kwargs)
    if bool(config.get("use_4bit", True)):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    train_rows = read_jsonl(config["train_file"])
    dev_rows = read_jsonl(config["dev_file"])
    train_dataset = Dataset.from_list([{"text": _format_training_text(tokenizer, row)} for row in train_rows])
    dev_dataset = Dataset.from_list([{"text": _format_training_text(tokenizer, row)} for row in dev_rows])

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=int(config.get("max_seq_length", 2048)))

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    dev_dataset = dev_dataset.map(tokenize, batched=True, remove_columns=["text"])

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(config.get("lora_r", 64)),
        lora_alpha=int(config.get("lora_alpha", 128)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        target_modules=config.get("target_modules", r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"),
    )
    model = get_peft_model(model, lora)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("batch_size", 1)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
        learning_rate=float(config.get("learning_rate", 0.00012)),
        num_train_epochs=float(config.get("num_train_epochs", config.get("epochs", 3))),
        lr_scheduler_type="cosine",
        warmup_ratio=float(config.get("warmup_ratio", 0.05)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        logging_steps=int(config.get("logging_steps", 5)),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=int(config.get("save_total_limit", 2)),
        remove_unused_columns=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if bool(config.get("use_4bit", True)) else "adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False if world_size > 1 else None,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "training_mode": "trained_lora",
        "model_path": model_path,
        "output_dir": str(output_dir),
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "train_metrics": dict(train_result.metrics),
        "eval_metrics": eval_metrics,
        "world_size": world_size,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def _ensure_chat_template(tokenizer: Any, model_path: str) -> None:
    if getattr(tokenizer, "chat_template", None) is not None:
        return
    template_file = Path(model_path) / "chat_template.json"
    if template_file.exists():
        data = json.loads(template_file.read_text(encoding="utf-8"))
        if "chat_template" in data:
            tokenizer.chat_template = data["chat_template"]
            return
    # Fallback: standard Mistral v3 instruct format
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}[SYSTEM_PROMPT]{{ message['content'] }}[/SYSTEM_PROMPT]{% endif %}"
        "{% if message['role'] == 'user' %}[INST]{{ message['content'] }}[/INST]{% endif %}"
        "{% if message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    )


def _load_tokenizer(auto_tokenizer: Any, model_path: str) -> Any:
    attempts = [
        {"trust_remote_code": True, "fix_mistral_regex": True},
        {"trust_remote_code": True},
        {"trust_remote_code": True, "use_fast": False},
    ]
    errors = []
    for kwargs in attempts:
        try:
            return auto_tokenizer.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Failed to load tokenizer for {model_path}: {' | '.join(errors)}")


def _load_model(transformers: Any, model_path: str, model_kwargs: dict[str, Any]) -> Any:
    class_names = ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]
    errors = []
    for class_name in class_names:
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            continue
        try:
            return model_class.from_pretrained(model_path, **model_kwargs)
        except Exception as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Failed to load model for {model_path}: {' | '.join(errors)}")


def _format_training_text(tokenizer: Any, row: dict[str, Any]) -> str:
    text = row["input"]
    skills = []
    for skill in row.get("output", {}).get("skills", []):
        skills.append(
            {
                "text": skill.get("surface_form") or skill.get("text"),
                "label": skill.get("label", "HARD_SKILL"),
            }
        )
    target = json.dumps({"skills": skills}, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        {"role": "assistant", "content": target},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return rendered + (tokenizer.eos_token or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_yaml(args.config))


if __name__ == "__main__":
    main()
