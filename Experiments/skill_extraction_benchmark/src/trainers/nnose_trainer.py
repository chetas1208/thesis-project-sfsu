"""NNOSE trainer: token-classification fine-tuning + FAISS kNN datastore.

Re-implements the NNOSE approach (Zhang et al., EACL 2024) using modern
PyTorch / HuggingFace APIs so everything runs from a single environment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.tokenizer_utils import align_tokens_to_subwords

logger = logging.getLogger(__name__)

LABEL_LIST = ["O", "B-SKILL", "I-SKILL"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


# ── Dataset ──────────────────────────────────────────────────────────

class NERDataset(Dataset):
    """Token classification dataset for NNOSE / JobBERT."""

    def __init__(self, records: Sequence[Dict], tokenizer, max_length: int = 256):
        self.items: List[Dict] = []
        for rec in records:
            aligned = align_tokens_to_subwords(
                tokenizer, rec["tokens"], rec["ner_tags"],
                max_length=max_length, label2id=LABEL2ID,
            )
            self.items.append({
                "input_ids": aligned["input_ids"],
                "attention_mask": aligned["attention_mask"],
                "labels": aligned["labels"],
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


# ── Metrics ──────────────────────────────────────────────────────────

def _compute_metrics(eval_pred):
    from seqeval.metrics import f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)

    true_seqs: List[List[str]] = []
    pred_seqs: List[List[str]] = []

    for pred_row, label_row in zip(preds, labels):
        true_seq: List[str] = []
        pred_seq: List[str] = []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            true_seq.append(ID2LABEL.get(l, "O"))
            pred_seq.append(ID2LABEL.get(p, "O"))
        true_seqs.append(true_seq)
        pred_seqs.append(pred_seq)

    return {
        "precision": precision_score(true_seqs, pred_seqs),
        "recall": recall_score(true_seqs, pred_seqs),
        "f1": f1_score(true_seqs, pred_seqs),
    }


# ── Training ─────────────────────────────────────────────────────────

def train_encoder(
    train_records: Sequence[Dict],
    dev_records: Sequence[Dict],
    config: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
) -> Path:
    """Fine-tune a token-classification encoder and save the best checkpoint."""
    backbone = config["backbone"]
    train_cfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        backbone,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds = NERDataset(train_records, tokenizer, max_length=train_cfg["max_length"])
    dev_ds = NERDataset(dev_records, tokenizer, max_length=train_cfg["max_length"])
    collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    checkpoint_dir = output_dir / "encoder"
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
        save_total_limit=2,
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
    trainer.save_model(str(checkpoint_dir / "best"))
    tokenizer.save_pretrained(str(checkpoint_dir / "best"))

    logger.info("Encoder saved to %s", checkpoint_dir / "best")
    return checkpoint_dir / "best"


# ── Datastore ────────────────────────────────────────────────────────

def build_datastore(
    model_path: Path,
    train_records: Sequence[Dict],
    config: Dict[str, Any],
    output_dir: Path,
    seed: int = 42,
) -> Path:
    """Extract token embeddings from training data and build a FAISS index."""
    import faiss

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = config["training"]["max_length"]
    ds_cfg = config["datastore"]

    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for rec in train_records:
            aligned = align_tokens_to_subwords(
                tokenizer, rec["tokens"], rec["ner_tags"],
                max_length=max_length, label2id=LABEL2ID,
            )
            input_ids = torch.tensor([aligned["input_ids"]], device=device)
            attention_mask = torch.tensor([aligned["attention_mask"]], device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Use last hidden state
            hidden = outputs.hidden_states[-1][0].cpu().numpy()  # (seq_len, hidden_dim)
            labels_arr = aligned["labels"]

            for i, lab in enumerate(labels_arr):
                if lab != -100:
                    all_embeddings.append(hidden[i])
                    all_labels.append(lab)

    embeddings = np.stack(all_embeddings, axis=0).astype(np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    # Optional: random subsample
    sample_pct = ds_cfg.get("sample_percentage", 100)
    if sample_pct < 100:
        rng = np.random.RandomState(seed)
        n_keep = max(1, int(len(embeddings) * sample_pct / 100))
        indices = rng.choice(len(embeddings), n_keep, replace=False)
        embeddings = embeddings[indices]
        labels_array = labels_array[indices]

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine after L2-norm)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    datastore_dir = output_dir / "datastore"
    datastore_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(datastore_dir / "index.faiss"))
    np.save(str(datastore_dir / "labels.npy"), labels_array)

    logger.info("Datastore: %d vectors, dim=%d → %s", len(embeddings), dim, datastore_dir)
    return datastore_dir


def knn_augmented_predict(
    model_path: Path,
    datastore_dir: Path,
    test_records: Sequence[Dict],
    config: Dict[str, Any],
) -> List[Dict]:
    """Run inference with kNN-augmented logits (the NNOSE approach).

    For each token:
        p_final = (1 - λ) * p_model + λ * p_knn
    where p_knn is a softmax over the k nearest datastore labels.
    """
    import faiss

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    index = faiss.read_index(str(datastore_dir / "index.faiss"))
    ds_labels = np.load(str(datastore_dir / "labels.npy"))

    ds_cfg = config["datastore"]
    k = ds_cfg["k"]
    lam = ds_cfg["lambda_value"]
    temperature = ds_cfg["temperature"]
    max_length = config["training"]["max_length"]
    num_labels = len(LABEL_LIST)

    results: List[Dict] = []

    with torch.no_grad():
        for rec in test_records:
            aligned = align_tokens_to_subwords(
                tokenizer, rec["tokens"], rec["ner_tags"],
                max_length=max_length, label2id=LABEL2ID,
            )
            input_ids = torch.tensor([aligned["input_ids"]], device=device)
            attention_mask = torch.tensor([aligned["attention_mask"]], device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits[0].cpu().numpy()  # (seq_len, num_labels)
            hidden = outputs.hidden_states[-1][0].cpu().numpy()

            # Model probabilities
            from scipy.special import softmax
            p_model = softmax(logits, axis=-1)

            # kNN probabilities
            word_ids = aligned["word_ids"]
            labels_out = aligned["labels"]

            p_combined = np.copy(p_model)
            for i in range(len(labels_out)):
                if labels_out[i] == -100:
                    continue
                query = hidden[i:i+1].astype(np.float32).copy()
                faiss.normalize_L2(query)
                distances, indices = index.search(query, k)

                # Build kNN distribution
                knn_dist = np.zeros(num_labels, dtype=np.float32)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0:
                        continue
                    knn_dist[ds_labels[idx]] += np.exp(dist / temperature)

                knn_sum = knn_dist.sum()
                if knn_sum > 0:
                    p_knn = knn_dist / knn_sum
                    p_combined[i] = (1 - lam) * p_model[i] + lam * p_knn

            # Map back to word-level predictions
            from src.data.tokenizer_utils import subword_predictions_to_word_tags
            word_preds = np.argmax(p_combined, axis=-1)
            tags = subword_predictions_to_word_tags(
                word_ids, word_preds.tolist(),
                id2label=ID2LABEL, num_words=len(rec["tokens"]),
            )

            from src.data.canonical_schema import bio_to_spans
            spans = bio_to_spans(rec["tokens"], tags, label="SKILL")

            results.append({
                "id": rec["id"],
                "doc_id": rec["doc_id"],
                "tokens": rec["tokens"],
                "predicted_tags": tags,
                "predicted_spans": [
                    {"text": s.text, "start_token": s.start_token, "end_token": s.end_token, "score": 1.0}
                    for s in spans
                ],
                "predicted_skill_texts": [s.text for s in spans],
            })

    return results
