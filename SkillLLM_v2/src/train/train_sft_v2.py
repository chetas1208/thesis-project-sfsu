from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.runtime_utils import (
    apply_overrides,
    ensure_dir,
    experiment_name_from_config,
    format_training_text,
    load_model_and_tokenizer,
    load_yaml_config,
    read_jsonl,
    save_yaml_config,
    seed_everything,
    serialize_skills,
    write_json,
)


def build_dataset(records, tokenizer, config):
    dataset_rows = []
    prompt_template = config.get("prompt_template_file")
    fewshot_template = config.get("fewshot_prompt_file") or None
    include_offsets = bool(config.get("training_target_include_offsets", True))

    for record in records:
        dataset_rows.append(
            {
                "text": format_training_text(
                    tokenizer=tokenizer,
                    text=record["text"],
                    target_json=serialize_skills(record["skills"], include_offsets=include_offsets),
                    prompt_template_file=prompt_template,
                    fewshot_template_file=fewshot_template,
                    config=config,
                ),
            }
        )
    return dataset_rows


def train_sft(config: dict[str, object]) -> dict[str, object]:
    seed_everything(int(config.get("seed", 13)))

    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        import torch
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover - setup validation
        raise RuntimeError("Training dependencies are missing. Run scripts/setup_env.sh first.") from exc

    train_records = read_jsonl(config["train_file"])
    val_records = read_jsonl(config["val_file"])

    tokenizer, model = load_model_and_tokenizer(config, for_training=True)
    if (config.get("use_4bit", False) or config.get("use_8bit", False)) and not config.get(
        "skip_prepare_model_for_kbit_training", False
    ):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("use_gradient_checkpointing", False)),
        )

    train_dataset = Dataset.from_list(build_dataset(train_records, tokenizer, config))
    val_dataset = Dataset.from_list(build_dataset(val_records, tokenizer, config))

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(config["max_seq_length"]),
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_batch, batched=True, remove_columns=["text"])

    output_dir = ensure_dir(config["output_dir"])
    save_yaml_config(output_dir / "resolved_config.yaml", config)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    report_to = "wandb" if (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_PROJECT")) else "none"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config.get("eval_batch_size", config["batch_size"])),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["epochs"]),
        lr_scheduler_type="cosine",
        warmup_ratio=float(config.get("warmup_ratio", 0.05)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if (config.get("use_4bit", False) or config.get("use_8bit", False)) else "adamw_torch",
        report_to=report_to,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        seed=int(config.get("seed", 13)),
    )

    target_modules = config["target_modules"]
    if not isinstance(target_modules, str):
        target_modules = list(target_modules)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    trainer_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "experiment_name": experiment_name_from_config(config),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_metrics": trainer_metrics,
        "eval_metrics": eval_metrics,
    }
    write_json(Path(output_dir) / "training_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the v2 exact-span extraction pipeline.")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--override", action="append", default=[], help="Config override in key=value format.")
    args = parser.parse_args()

    config = apply_overrides(load_yaml_config(args.config), args.override)
    train_sft(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
