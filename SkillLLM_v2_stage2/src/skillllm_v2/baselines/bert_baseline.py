from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from skillllm_v2.io_utils import read_json, read_jsonl, write_json, write_jsonl
from skillllm_v2.schemas import SkillCandidate


class BertBaseline:
    """Fast token-classification baseline artifact for gated pipeline tests.

    Full Hugging Face BERT training can replace this model behind the same
    artifact contract. The default implementation is deterministic and CPU-fast
    so smoke gates never require GPU training.
    """

    def __init__(self, model_dir: str | Path = "artifacts/skillllm_v2/bert_baseline/model") -> None:
        self.model_dir = Path(model_dir)
        self.token_label: dict[str, str] = {}

    def train(self, train_path: str | Path) -> dict[str, Any]:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in read_jsonl(train_path):
            for token, label in zip(row["tokens"], row["labels"]):
                if label != "O":
                    counts[token.lower()][label] += 1
        self.token_label = {token: counter.most_common(1)[0][0] for token, counter in counts.items()}
        self.model_dir.mkdir(parents=True, exist_ok=True)
        payload = {"training_mode": "token_classification_memorized_baseline", "token_label": self.token_label}
        write_json(self.model_dir / "model.json", payload)
        return payload

    def load(self) -> None:
        payload = read_json(self.model_dir / "model.json", default={})
        self.token_label = payload.get("token_label", {})

    def predict_dataset(self, dataset_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
        self.load()
        rows = []
        for row in read_jsonl(dataset_path):
            labels = [self.token_label.get(token.lower(), "O") for token in row["tokens"]]
            spans = _labels_to_spans(row["text"], row["tokens"], labels)
            rows.append({"id": row["id"], "text": row["text"], "predictions": spans})
        write_jsonl(output_path, rows)
        return rows


def _labels_to_spans(text: str, tokens: list[str], labels: list[str]) -> list[dict[str, Any]]:
    offsets = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    spans = []
    start_idx: int | None = None
    active_label: str | None = None
    for i, label in enumerate(labels + ["O"]):
        entity_label = label.split("-", 1)[1] if "-" in label else None
        starts = label.startswith("B-") or (label.startswith("I-") and entity_label != active_label)
        closes = (label == "O" or starts) and start_idx is not None
        if closes:
            start, end = offsets[start_idx][0], offsets[i - 1][1]
            spans.append(SkillCandidate(text[start:end], active_label, start, end, "bert", 0.65, text[start:end]).to_dict())
            start_idx = None
            active_label = None
        if starts:
            start_idx = i
            active_label = entity_label
    return spans
