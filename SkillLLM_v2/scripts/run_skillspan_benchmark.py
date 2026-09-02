#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.chdir(ROOT_DIR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

LOCAL_PYTHON_HEADERS = ROOT_DIR / ".cache" / "apt_python_headers" / "extracted" / "usr" / "include"
if LOCAL_PYTHON_HEADERS.exists():
    include_dirs = [
        str(LOCAL_PYTHON_HEADERS),
        str(LOCAL_PYTHON_HEADERS / "python3.12"),
        str(LOCAL_PYTHON_HEADERS / "x86_64-linux-gnu" / "python3.12"),
    ]
    existing_include_path = os.environ.get("C_INCLUDE_PATH")
    if existing_include_path:
        include_dirs.append(existing_include_path)
    os.environ["C_INCLUDE_PATH"] = ":".join(include_dirs)

from src.benchmark.pipeline import run_benchmark
from src.runtime_utils import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SkillSpan multi-model benchmark.")
    parser.add_argument("--config", required=True, help="Benchmark config yaml.")
    parser.add_argument("--dataset-path", required=True, help="Dataset root or candidate path.")
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    parser.add_argument("--model", default=None, help="Optional single model name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test record limit per split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing model outputs.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    results = run_benchmark(
        benchmark_config=config,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        model_name=args.model,
    )
    print(f"Benchmark outputs: {results['output_dir']}")
    if args.model and results["models"].get(args.model, {}).get("status") == "failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
