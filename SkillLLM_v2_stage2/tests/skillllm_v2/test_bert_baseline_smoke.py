from pathlib import Path

from skillllm_v2.baselines import BertBaseline
from skillllm_v2.train.build_skillspan_bio import build_bio_dataset


def test_bert_baseline_smoke(tmp_path: Path) -> None:
    bio = tmp_path / "bio"
    build_bio_dataset("data/source/skillspan_hf", bio, limit=10)
    model = BertBaseline(tmp_path / "model")
    model.train(bio / "train.jsonl")
    rows = model.predict_dataset(bio / "test.jsonl", tmp_path / "predictions.jsonl")
    assert rows
