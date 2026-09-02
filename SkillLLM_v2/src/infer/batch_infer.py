from __future__ import annotations

import argparse
from pathlib import Path

from src.infer.export_predictions import export_predictions_file
from src.infer.generate_predictions import generate_raw_predictions
from src.runtime_utils import apply_overrides, ensure_dir, load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run batch inference over val/test splits.")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Dataset splits to run.")
    parser.add_argument("--adapter-path", default=None, help="Optional adapter directory.")
    parser.add_argument("--override", action="append", default=[], help="Config override in key=value format.")
    args = parser.parse_args()

    config = apply_overrides(load_yaml_config(args.config), args.override)
    prediction_dir = ensure_dir(Path(config["output_dir"]) / "predictions")
    adapter_path = args.adapter_path or config["output_dir"]

    for split in args.splits:
        input_key = f"{split}_file"
        if input_key not in config:
            raise KeyError(f"Missing '{input_key}' in config.")
        raw_file = prediction_dir / f"raw_{split}.jsonl"
        final_file = prediction_dir / f"{split}_predictions.jsonl"
        generate_raw_predictions(config, config[input_key], str(raw_file), adapter_path=adapter_path)
        export_predictions_file(config, str(raw_file), str(final_file))
        print(f"Wrote {final_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
