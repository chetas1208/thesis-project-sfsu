from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from skillllm_v2.io_utils import write_jsonl
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

INSTRUCTION = "Extract all HARD_SKILL and SOFT_SKILL spans from the text. Return strict JSON with surface_form, label, start, and end."
# Rows carry no prompt (the prompt template is applied at train/inference time, see extract.mistral_extractor.PROMPT_VERSIONS);
# for a non-default prompt_version the row is tagged so a dataset can be traced to the prompt it is meant for.


def build_mistral_sft_data(
    dataset_root: str | Path, output_dir: str | Path, limit: int | None = None, prompt_version: str = "original"
) -> dict[str, str]:
    from skillllm_v2.extract.mistral_extractor import PROMPT_VERSIONS

    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt_version {prompt_version!r}; choose one of {sorted(PROMPT_VERSIONS)}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ["train", "dev", "test"]:
        rows = []
        for doc in load_skillspan_split(Path(dataset_root), split, limit=limit):
            row = {"id": doc["id"], "instruction": INSTRUCTION, "input": doc["text"], "output": {"skills": doc["gold_spans"]}}
            if prompt_version != "original":
                row["prompt_version"] = prompt_version
            rows.append(row)
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/source/skillspan_hf")
    parser.add_argument("--output-dir", default="data/skillllm_v2/mistral_sft")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt-version", default="original")
    args = parser.parse_args()
    build_mistral_sft_data(args.dataset_path, args.output_dir, args.limit, args.prompt_version)


if __name__ == "__main__":
    main()
