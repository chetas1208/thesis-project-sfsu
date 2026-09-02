from __future__ import annotations

import argparse
from pathlib import Path

from skillllm_v2.baselines.bert_baseline import BertBaseline


def train_bert_baseline(bio_dir: str = "data/skillllm_v2/bio", output_dir: str = "artifacts/skillllm_v2/bert_baseline/model") -> dict:
    return BertBaseline(output_dir).train(Path(bio_dir) / "train.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bio-dir", default="data/skillllm_v2/bio")
    parser.add_argument("--output-dir", default="artifacts/skillllm_v2/bert_baseline/model")
    args = parser.parse_args()
    train_bert_baseline(args.bio_dir, args.output_dir)


if __name__ == "__main__":
    main()
