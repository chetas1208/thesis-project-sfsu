from __future__ import annotations

from pathlib import Path


def train_taxonomy_reranker(output_dir: str = "artifacts/skillllm_v2/taxonomy_reranker") -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("Placeholder for optional taxonomy reranker training.\n", encoding="utf-8")
