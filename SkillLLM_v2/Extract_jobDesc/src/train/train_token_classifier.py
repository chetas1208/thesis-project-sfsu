import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABELS = ["O", "B-hard_skill", "I-hard_skill", "B-soft_skill", "I-soft_skill"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_record(record: Dict[str, object]) -> Dict[str, object]:
    tokens = list(record.get("tokens", []))
    ner_tags = list(record.get("ner_tags", []))
    if len(tokens) != len(ner_tags):
        raise ValueError(f"Token/tag length mismatch for {record.get('id')}")

    return {
        "id": str(record.get("id", "")),
        "tokens": [str(token) for token in tokens],
        "labels": [LABEL_TO_ID[str(tag)] for tag in ner_tags],
    }


def load_dataset(path: str) -> Dataset:
    records = [normalize_record(record) for record in read_jsonl(path)]
    return Dataset.from_list(records)


def extract_entities(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    entities: List[Tuple[str, int, int]] = []
    active_label: Optional[str] = None
    start_index: Optional[int] = None

    for index, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            prefix = "O"
            label = None
        else:
            prefix, label = tag.split("-", 1)

        if active_label is not None:
            should_close = prefix != "I" or label != active_label
            if should_close:
                entities.append((active_label, int(start_index), index))
                active_label = None
                start_index = None

        if prefix == "B":
            active_label = label
            start_index = index
        elif prefix == "I" and active_label is None:
            active_label = label
            start_index = index

    return entities


def compute_entity_metrics(true_sequences: List[List[str]], pred_sequences: List[List[str]]) -> Dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold_tags, pred_tags in zip(true_sequences, pred_sequences):
        gold_entities = set(extract_entities(gold_tags))
        pred_entities = set(extract_entities(pred_tags))
        true_positive += len(gold_entities & pred_entities)
        false_positive += len(pred_entities - gold_entities)
        false_negative += len(gold_entities - pred_entities)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@dataclass
class TokenizedDatasets:
    train: Dataset
    eval: Dataset
    test: Optional[Dataset]


def tokenize_and_align(tokenizer, dataset: Dataset, max_length: int) -> Dataset:
    def transform(batch: Dict[str, List[object]]) -> Dict[str, List[object]]:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        aligned_labels: List[List[int]] = []
        for batch_index, labels in enumerate(batch["labels"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            label_ids: List[int] = []
            previous_word = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word:
                    label_ids.append(labels[word_id])
                else:
                    tag = ID_TO_LABEL[int(labels[word_id])]
                    if tag.startswith("B-"):
                        tag = "I-" + tag[2:]
                    label_ids.append(LABEL_TO_ID.get(tag, labels[word_id]))
                previous_word = word_id
            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    return dataset.map(transform, batched=True, remove_columns=dataset.column_names)


def prepare_datasets(train_file: str, eval_file: str, test_file: Optional[str], tokenizer, max_length: int) -> TokenizedDatasets:
    train_dataset = tokenize_and_align(tokenizer, load_dataset(train_file), max_length)
    eval_dataset = tokenize_and_align(tokenizer, load_dataset(eval_file), max_length)
    test_dataset = tokenize_and_align(tokenizer, load_dataset(test_file), max_length) if test_file else None
    return TokenizedDatasets(train=train_dataset, eval=eval_dataset, test=test_dataset)


def metrics_from_predictions(eval_prediction) -> Dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    true_sequences: List[List[str]] = []
    pred_sequences: List[List[str]] = []
    token_correct = 0
    token_total = 0

    for predicted_ids, label_ids in zip(predictions, labels):
        true_tags: List[str] = []
        pred_tags: List[str] = []
        for pred_id, label_id in zip(predicted_ids, label_ids):
            if int(label_id) == -100:
                continue
            true_tag = ID_TO_LABEL[int(label_id)]
            pred_tag = ID_TO_LABEL[int(pred_id)]
            true_tags.append(true_tag)
            pred_tags.append(pred_tag)
            token_total += 1
            if true_tag == pred_tag:
                token_correct += 1
        true_sequences.append(true_tags)
        pred_sequences.append(pred_tags)

    entity_metrics = compute_entity_metrics(true_sequences, pred_sequences)
    token_accuracy = token_correct / token_total if token_total else 0.0
    return {
        "token_accuracy": token_accuracy,
        "entity_precision": entity_metrics["precision"],
        "entity_recall": entity_metrics["recall"],
        "entity_f1": entity_metrics["f1"],
    }


def save_test_metrics(trainer: Trainer, test_dataset: Optional[Dataset], output_dir: str) -> None:
    if test_dataset is None:
        return
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    with open(os.path.join(output_dir, "test_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a token-classification model for skill span extraction.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--test-file")
    parser.add_argument("--model-name", default="bert-base-cased")
    parser.add_argument("--output-dir", default="outputs/token_classifier")
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = prepare_datasets(
        args.train_file,
        args.eval_file,
        args.test_file,
        tokenizer,
        args.max_length,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": datasets.train,
        "eval_dataset": datasets.eval,
        "data_collator": DataCollatorForTokenClassification(tokenizer=tokenizer),
        "compute_metrics": metrics_from_predictions,
    }
    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_metrics = trainer.evaluate(metric_key_prefix="eval")
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, indent=2)

    save_test_metrics(trainer, datasets.test, args.output_dir)


if __name__ == "__main__":
    main()
