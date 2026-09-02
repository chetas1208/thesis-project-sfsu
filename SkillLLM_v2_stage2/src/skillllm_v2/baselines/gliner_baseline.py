from __future__ import annotations

from pathlib import Path
from typing import Any

from skillllm_v2.extract.gliner_extractor import GLiNERExtractor
from skillllm_v2.io_utils import write_json, write_jsonl
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split


class GLiNERBaseline:
    def __init__(self, output_dir: str | Path = "artifacts/skillllm_v2/gliner_baseline", model_name: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.extractor = GLiNERExtractor(model_name=model_name, allow_fallback=True)

    def train_or_prepare(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        note = "GLiNER fine-tuning skipped; zero-shot/few-shot baseline used"
        try:
            import gliner  # noqa: F401

            training_mode = "zero_shot"
        except ImportError:
            training_mode = "skipped_dependency_missing"
            note = "GLiNER dependency missing; deterministic fallback candidate extractor used"
        payload = {"training_mode": training_mode, "note": note}
        write_json(self.output_dir / "model" / "config.json", payload)
        return payload

    def predict(self, dataset_root: str | Path, split: str, output_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
        rows = []
        for doc in load_skillspan_split(Path(dataset_root), split, limit=limit):
            rows.append({"id": doc["id"], "text": doc["text"], "predictions": self.extractor.extract(doc["text"])})
        write_jsonl(output_path, rows)
        return rows
