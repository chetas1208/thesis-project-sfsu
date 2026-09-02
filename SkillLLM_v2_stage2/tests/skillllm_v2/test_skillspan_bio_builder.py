from pathlib import Path

from skillllm_v2.train.build_skillspan_bio import build_bio_dataset


def test_build_skillspan_bio_smoke(tmp_path: Path) -> None:
    paths = build_bio_dataset("data/source/skillspan_hf", tmp_path, limit=3)
    assert Path(paths["train"]).exists()
