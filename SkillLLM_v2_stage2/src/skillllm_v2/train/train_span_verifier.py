from __future__ import annotations

import argparse
import json
from pathlib import Path


def train_span_verifier(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = _dataset_stats(data_dir)
    examples = _load_examples(data_dir / "verifier_train.jsonl")
    artifact = {
        "model_type": "rule_based_span_verifier",
        "min_confidence": 0.35,
        "stats": stats,
        "note": "Fallback verifier artifact. scikit-learn was unavailable or the training data had one class.",
    }
    if len({label for _, label in examples}) == 2:
        try:
            import joblib
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            model = Pipeline(
                [
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                    ("classifier", LogisticRegression(max_iter=500, class_weight="balanced")),
                ]
            )
            texts, labels = zip(*examples)
            model.fit(texts, labels)
            model_dir = output_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_dir / "verifier_model.joblib")
            artifact.update(
                {
                    "model_type": "tfidf_logistic_regression_span_verifier",
                    "model_path": "model/verifier_model.joblib",
                    "note": "Lightweight classifier trained from SkillSpan positives and generated negatives.",
                }
            )
        except Exception as exc:  # pragma: no cover - exercised only when optional dependency fails.
            artifact["training_warning"] = str(exc)
    (output_dir / "verifier_config.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (output_dir / "training_args.json").write_text(json.dumps({"classifier": "tfidf_logistic_regression", "full_llm_training": False}, indent=2), encoding="utf-8")


def _dataset_stats(data_dir: Path) -> dict[str, dict[str, int]]:
    stats = {}
    for split in ["train", "dev"]:
        path = data_dir / f"verifier_{split}.jsonl"
        positives = negatives = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                positives += int(bool(row.get("is_valid_skill")))
                negatives += int(not row.get("is_valid_skill"))
        stats[split] = {"positives": positives, "negatives": negatives}
    return stats


def _load_examples(path: Path) -> list[tuple[str, int]]:
    examples: list[tuple[str, int]] = []
    if not path.exists():
        return examples
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        candidate = row.get("candidate", {})
        text = f"{candidate.get('surface_form', '')} [SEP] {row.get('text', '')}"
        examples.append((text, int(bool(row.get("is_valid_skill")))))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/skillllm_v2")
    parser.add_argument("--output-dir", default="artifacts/skillllm_v2/span_verifier")
    args = parser.parse_args()
    train_span_verifier(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
