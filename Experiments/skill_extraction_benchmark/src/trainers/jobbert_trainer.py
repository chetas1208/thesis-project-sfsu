"""JobBERT trainer: standard HuggingFace token-classification fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.tokenizer_utils import align_tokens_to_subwords, subword_predictions_to_word_tags
from src.trainers.nnose_trainer import LABEL2ID, LABEL_LIST, ID2LABEL, NERDataset, _compute_metrics

logger = logging.getLogger(__name__)


def train_jobbert(
    train_records: Sequence[Dict],
    dev_records: Sequence[Dict],
    config: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
) -> Path:
    """Fine-tune JobBERT for token classification."""
    base_model = config["base_model"]
    train_cfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds = NERDataset(train_records, tokenizer, max_length=train_cfg["max_length"])
    dev_ds = NERDataset(dev_records, tokenizer, max_length=train_cfg["max_length"])
    collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    checkpoint_dir = output_dir / "checkpoint"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=train_cfg.get("save_total_limit", 2),
        fp16=True,
        seed=seed,
        logging_steps=50,
        report_to="none",
    )

    callbacks = []
    patience = train_cfg.get("patience", 3)
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    best_dir = checkpoint_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    logger.info("JobBERT model saved to %s", best_dir)
    return best_dir


def predict_jobbert(
    model_path: Path,
    test_records: Sequence[Dict],
    max_length: int = 256,
) -> List[Dict]:
    """Run JobBERT token-classification inference."""
    import torch
    from src.data.canonical_schema import bio_to_spans

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results: List[Dict] = []

    with torch.no_grad():
        for rec in test_records:
            aligned = align_tokens_to_subwords(
                tokenizer, rec["tokens"], rec.get("ner_tags", ["O"] * len(rec["tokens"])),
                max_length=max_length, label2id=LABEL2ID,
            )
            input_ids = torch.tensor([aligned["input_ids"]], device=device)
            attention_mask = torch.tensor([aligned["attention_mask"]], device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

            tags = subword_predictions_to_word_tags(
                aligned["word_ids"], preds,
                id2label=ID2LABEL, num_words=len(rec["tokens"]),
            )

            spans = bio_to_spans(rec["tokens"], tags, label="SKILL")

            results.append({
                "id": rec.get("id", rec.get("sentence_id", "")),
                "doc_id": rec.get("doc_id", ""),
                "tokens": rec["tokens"],
                "predicted_tags": tags,
                "predicted_spans": [
                    {"text": s.text, "start_token": s.start_token, "end_token": s.end_token, "score": 1.0}
                    for s in spans
                ],
                "predicted_skill_texts": [s.text for s in spans],
            })

    return results
