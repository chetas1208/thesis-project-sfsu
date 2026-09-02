from pathlib import Path

from skillllm_v2.train.build_mistral_sft_data import build_mistral_sft_data


def test_mistral_sft_builder(tmp_path: Path) -> None:
    paths = build_mistral_sft_data("data/source/skillspan_hf", tmp_path, limit=2)
    assert Path(paths["train"]).exists()
