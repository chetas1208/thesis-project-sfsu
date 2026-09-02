"""ConTeXT trainer: sentence-transformer fine-tuning for skill retrieval."""

from __future__ import annotations

import json
import logging
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def load_pairs(path: Path) -> List[Dict]:
    """Load sentence–skill pairs from a JSONL file."""
    records: List[Dict] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append({
            "sentence": str(row["sentence"]),
            "skill": str(row["skill"]),
            "label": int(row["label"]),
            "sentence_id": str(row.get("sentence_id", "")),
            "doc_id": str(row.get("doc_id", "")),
        })
    return records


def train_context(
    config: Dict[str, Any],
    data_dir: Path,
    output_dir: Path,
) -> Path:
    """Fine-tune a ConTeXT sentence-transformer on sentence–skill pairs."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"

    base_model = config["base_model"]
    train_cfg = config["training"]

    model = SentenceTransformer(base_model, trust_remote_code=True)
    train_rows = load_pairs(data_dir / "train.jsonl")

    examples = [
        InputExample(texts=[row["sentence"], row["skill"]], label=float(row["label"]))
        for row in train_rows
    ]
    if not examples:
        raise ValueError("No training examples for ConTeXT")

    train_loader = DataLoader(examples, batch_size=int(train_cfg["train_batch_size"]), shuffle=True)
    train_loss = losses.CosineSimilarityLoss(model)

    epochs = int(train_cfg["num_train_epochs"])
    steps_per_epoch = max(1, ceil(len(examples) / int(train_cfg["train_batch_size"])))
    warmup_steps = max(1, int(steps_per_epoch * epochs * float(train_cfg.get("warmup_ratio", 0.1))))

    logger.info("ConTeXT training: %d examples, %d epochs, lr=%s", len(examples), epochs, train_cfg["learning_rate"])

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(train_cfg["learning_rate"])},
        output_path=str(checkpoint_dir),
        show_progress_bar=True,
    )

    logger.info("ConTeXT model saved to %s", checkpoint_dir)
    return checkpoint_dir


def sweep_threshold(
    model_path: Path,
    dev_data_path: Path,
    encode_batch_size: int = 64,
) -> tuple[float, Dict[str, float]]:
    """Sweep cosine similarity thresholds on the dev set to find optimal."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import precision_recall_fscore_support

    model = SentenceTransformer(str(model_path), trust_remote_code=True)
    dev_rows = load_pairs(dev_data_path)

    # Encode all unique texts
    sentences = sorted({r["sentence"] for r in dev_rows})
    skills = sorted({r["skill"] for r in dev_rows})
    all_texts = sentences + skills

    embeddings = model.encode(all_texts, batch_size=encode_batch_size, normalize_embeddings=True, show_progress_bar=False)
    text_to_vec = {text: embeddings[i] for i, text in enumerate(all_texts)}

    # Compute scores
    scores = []
    labels = []
    for row in dev_rows:
        s_vec = text_to_vec[row["sentence"]]
        k_vec = text_to_vec[row["skill"]]
        scores.append(float(np.dot(s_vec, k_vec)))
        labels.append(row["label"])

    # Sweep
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}

    unique_scores = sorted(set(scores))
    candidates = [unique_scores[0] - 0.01] + unique_scores + [unique_scores[-1] + 0.01]

    for threshold in candidates:
        preds = [1 if s >= threshold else 0 for s in scores]
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {"precision": float(p), "recall": float(r), "f1": float(f1)}

    logger.info("Best dev threshold: %.4f (F1=%.4f)", best_threshold, best_f1)
    return best_threshold, best_metrics


def predict_context(
    model_path: Path,
    candidate_skills: List[str],
    test_records: Sequence[Dict],
    threshold: float = 0.5,
    top_k: int = 5,
    encode_batch_size: int = 64,
) -> List[Dict]:
    """Run ConTeXT inference: rank candidate skills per sentence."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), trust_remote_code=True)

    # Encode candidate skills
    skill_vecs = model.encode(candidate_skills, batch_size=encode_batch_size, normalize_embeddings=True, show_progress_bar=False)

    results: List[Dict] = []
    for rec in test_records:
        text = rec.get("text", " ".join(rec.get("tokens", [])))
        sent_vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]

        # Compute similarities
        sims = skill_vecs @ sent_vec
        ranked_indices = np.argsort(-sims)

        predicted_skills: List[str] = []
        predicted_spans: List[Dict] = []
        for idx in ranked_indices[:top_k]:
            score = float(sims[idx])
            if score < threshold:
                break
            skill_text = candidate_skills[idx]
            predicted_skills.append(skill_text)
            predicted_spans.append({
                "text": skill_text,
                "start_token": -1,
                "end_token": -1,
                "score": round(score, 4),
            })

        results.append({
            "id": rec.get("sentence_id", rec.get("id", "")),
            "doc_id": rec.get("doc_id", ""),
            "tokens": rec.get("tokens", []),
            "predicted_spans": predicted_spans,
            "predicted_skill_texts": predicted_skills,
        })

    return results
