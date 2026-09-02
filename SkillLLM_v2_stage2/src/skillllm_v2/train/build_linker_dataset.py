from __future__ import annotations

from pathlib import Path


def build_linker_dataset(output_path: str = "data/skillllm_v2/linker_dev.jsonl") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("", encoding="utf-8")
