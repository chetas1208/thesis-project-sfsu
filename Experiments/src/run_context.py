import argparse
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from common import (
    ensure_benchmark_env,
    ensure_dir,
    load_config,
    load_env_file,
    read_json_records,
    resolve_path,
    write_json,
    write_jsonl,
    write_run_metadata,
)


def _load_pairs(path: Path) -> List[Dict]:
    rows = read_json_records(path)
    normalized = []
    for row in rows:
        sentence = str(row["sentence"])
        skill = str(row["skill"])
        label = int(row["label"])
        normalized.append({"sentence": sentence, "skill": skill, "label": label})
    return normalized


def _binary_metrics(gold: Sequence[int], predicted: Sequence[int]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predicted,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(gold, predicted) if gold else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _candidate_thresholds(scores: Sequence[float]) -> List[float]:
    if not scores:
        return [0.5]
    unique_scores = sorted({float(score) for score in scores})
    thresholds = [unique_scores[0] - 1e-6]
    thresholds.extend(unique_scores)
    thresholds.append(unique_scores[-1] + 1e-6)
    return thresholds


def _choose_best_threshold(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "accuracy": 0.0}
    for threshold in _candidate_thresholds(scores):
        predictions = [1 if score >= threshold else 0 for score in scores]
        metrics = _binary_metrics(labels, predictions)
        if metrics["f1"] > best_metrics["f1"] or (
            metrics["f1"] == best_metrics["f1"] and threshold < best_threshold
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _encode_texts(model, texts: Iterable[str], batch_size: int):
    texts = list(texts)
    if not texts:
        return {}
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {text: embeddings[index] for index, text in enumerate(texts)}


def _score_pairs(model, rows: Sequence[Dict], batch_size: int) -> List[float]:
    import numpy as np

    sentences = sorted({row["sentence"] for row in rows})
    skills = sorted({row["skill"] for row in rows})
    sentence_vectors = _encode_texts(model, sentences, batch_size)
    skill_vectors = _encode_texts(model, skills, batch_size)
    scores = []
    for row in rows:
        sentence_vector = sentence_vectors[row["sentence"]]
        skill_vector = skill_vectors[row["skill"]]
        scores.append(float(np.dot(sentence_vector, skill_vector)))
    return scores


def _group_positive_skills(rows: Sequence[Dict]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for row in rows:
        if int(row["label"]) != 1:
            continue
        grouped.setdefault(row["sentence"], [])
        if row["skill"] not in grouped[row["sentence"]]:
            grouped[row["sentence"]].append(row["skill"])
    return grouped


def _retrieval_metrics(model, candidate_pool: Sequence[str], test_rows: Sequence[Dict], batch_size: int) -> tuple[Dict, List[Dict]]:
    import numpy as np

    if not candidate_pool:
        metrics = {
            "candidate_pool_size": 0,
            "evaluated_sentences": 0,
            "gold_skill_mentions": 0,
            "oov_gold_skill_mentions": 0,
            "oov_gold_skill_rate": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
        }
        return metrics, []

    sentence_to_gold = _group_positive_skills(test_rows)
    candidate_pool = list(candidate_pool)
    candidate_vectors = _encode_texts(model, candidate_pool, batch_size)
    sentence_vectors = _encode_texts(model, sentence_to_gold.keys(), batch_size)

    total_gold = 0
    total_oov = 0
    recall_sums = {1: 0.0, 3: 0.0, 5: 0.0}
    evaluated_sentences = 0
    details = []

    candidate_matrix = np.stack([candidate_vectors[skill] for skill in candidate_pool], axis=0)
    for sentence, gold_skills in sentence_to_gold.items():
        gold_in_vocab = [skill for skill in gold_skills if skill in candidate_vectors]
        total_gold += len(gold_skills)
        total_oov += len(gold_skills) - len(gold_in_vocab)
        sentence_vector = sentence_vectors[sentence]
        similarities = candidate_matrix @ sentence_vector
        ranked_indices = np.argsort(-similarities)
        ranked_skills = [candidate_pool[index] for index in ranked_indices.tolist()]

        sentence_detail = {
            "sentence": sentence,
            "gold_skills": gold_skills,
            "gold_in_vocab": gold_in_vocab,
            "top_5_skills": ranked_skills[:5],
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
        }

        if gold_in_vocab:
            evaluated_sentences += 1
            for k in (1, 3, 5):
                top_k = set(ranked_skills[:k])
                recall_value = len(top_k.intersection(gold_in_vocab)) / len(gold_in_vocab)
                recall_sums[k] += recall_value
                sentence_detail[f"recall_at_{k}"] = recall_value

        details.append(sentence_detail)

    metrics = {
        "candidate_pool_size": len(candidate_pool),
        "evaluated_sentences": evaluated_sentences,
        "gold_skill_mentions": total_gold,
        "oov_gold_skill_mentions": total_oov,
        "oov_gold_skill_rate": (total_oov / total_gold) if total_gold else 0.0,
        "recall_at_1": (recall_sums[1] / evaluated_sentences) if evaluated_sentences else 0.0,
        "recall_at_3": (recall_sums[3] / evaluated_sentences) if evaluated_sentences else 0.0,
        "recall_at_5": (recall_sums[5] / evaluated_sentences) if evaluated_sentences else 0.0,
    }
    return metrics, details


def _train_model(model, train_rows: Sequence[Dict], defaults: Dict, output_dir: Path) -> Dict:
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader

    training_examples = [
        InputExample(texts=[row["sentence"], row["skill"]], label=float(row["label"]))
        for row in train_rows
    ]
    if not training_examples:
        raise ValueError("Context training requires at least one training pair.")

    train_loader = DataLoader(
        training_examples,
        batch_size=int(defaults["train_batch_size"]),
        shuffle=True,
    )
    train_loss = losses.CosineSimilarityLoss(model)
    total_steps = max(1, ceil(len(training_examples) / int(defaults["train_batch_size"])))
    warmup_steps = max(1, int(total_steps * int(defaults["num_train_epochs"]) * float(defaults["warmup_ratio"])))

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=int(defaults["num_train_epochs"]),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(defaults["learning_rate"])},
        output_path=str(output_dir),
        show_progress_bar=False,
    )
    return {
        "train_pairs": len(training_examples),
        "warmup_steps": warmup_steps,
        "num_train_epochs": int(defaults["num_train_epochs"]),
        "learning_rate": float(defaults["learning_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the ConTeXT retrieval baseline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    model_cfg = config["models"]["context"]
    defaults = config["training_defaults"]["context"]
    input_dir = resolve_path(model_cfg["input_dir"])
    results_dir = ensure_dir(resolve_path(model_cfg["results_dir"]))
    checkpoint_dir = ensure_dir(results_dir / "checkpoints")

    train_file = input_dir / "train.json"
    dev_file = input_dir / "dev.json"
    test_file = input_dir / "test.json"

    if args.dry_run:
        print(f"train_file={train_file}")
        print(f"dev_file={dev_file}")
        print(f"test_file={test_file}")
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"base_model={model_cfg['base_model']}")
        print(f"mode={args.mode}")
        return

    train_rows = _load_pairs(train_file)
    dev_rows = _load_pairs(dev_file)
    test_rows = _load_pairs(test_file)

    from sentence_transformers import SentenceTransformer

    if args.mode in {"train", "full"}:
        model = SentenceTransformer(model_cfg["base_model"], trust_remote_code=True)
        train_metadata = _train_model(model, train_rows, defaults, checkpoint_dir)
    else:
        if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
            raise FileNotFoundError(
                f"No trained ConTeXT checkpoint found at {checkpoint_dir}. Run --mode train or --mode full first."
            )
        train_metadata = {}

    inference_model_path = checkpoint_dir if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) else model_cfg["base_model"]
    model = SentenceTransformer(str(inference_model_path), trust_remote_code=True)

    dev_scores = _score_pairs(model, dev_rows, int(defaults["encode_batch_size"]))
    dev_labels = [int(row["label"]) for row in dev_rows]
    threshold, dev_metrics = _choose_best_threshold(dev_scores, dev_labels)

    test_scores = _score_pairs(model, test_rows, int(defaults["encode_batch_size"]))
    test_labels = [int(row["label"]) for row in test_rows]
    predicted_labels = [1 if score >= threshold else 0 for score in test_scores]
    test_metrics = _binary_metrics(test_labels, predicted_labels)

    candidate_pool = sorted({row["skill"] for row in train_rows + dev_rows if int(row["label"]) == 1})
    retrieval_metrics, retrieval_details = _retrieval_metrics(
        model,
        candidate_pool,
        test_rows,
        int(defaults["encode_batch_size"]),
    )

    pair_scores = []
    for row, score, predicted in zip(test_rows, test_scores, predicted_labels):
        pair_scores.append(
            {
                "sentence": row["sentence"],
                "skill": row["skill"],
                "gold_label": int(row["label"]),
                "cosine_score": float(score),
                "predicted_label": int(predicted),
            }
        )

    write_jsonl(results_dir / "pair_scores.jsonl", pair_scores)
    write_json(results_dir / "metrics.json", {**test_metrics, "threshold": threshold, "dev_metrics": dev_metrics})
    write_json(results_dir / "retrieval_metrics.json", retrieval_metrics)
    write_jsonl(results_dir / "retrieval_details.jsonl", retrieval_details)
    write_run_metadata(
        results_dir / "run_metadata.json",
        {
            "model": "context",
            "task_type": "retrieval",
            "base_model": model_cfg["base_model"],
            "checkpoint": str(inference_model_path),
            "threshold": threshold,
            "candidate_pool_size": len(candidate_pool),
            **train_metadata,
        },
    )
    print(f"Pair scores written to {results_dir / 'pair_scores.jsonl'}")


if __name__ == "__main__":
    main()
