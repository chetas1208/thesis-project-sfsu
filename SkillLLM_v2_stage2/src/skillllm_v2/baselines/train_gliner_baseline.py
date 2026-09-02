from __future__ import annotations

import argparse

from skillllm_v2.baselines.gliner_baseline import GLiNERBaseline


def train_or_run_gliner_baseline(output_dir: str = "artifacts/skillllm_v2/gliner_baseline") -> dict:
    return GLiNERBaseline(output_dir).train_or_prepare()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/skillllm_v2/gliner_baseline")
    args = parser.parse_args()
    train_or_run_gliner_baseline(args.output_dir)


if __name__ == "__main__":
    main()
