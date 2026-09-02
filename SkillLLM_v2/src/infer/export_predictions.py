from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.formatting.response_parser import parse_model_response
from src.postprocess.span_recover import SpanRecoverer
from src.runtime_utils import load_yaml_config, read_jsonl, write_jsonl


def export_predictions_file(
    config: dict[str, Any],
    raw_file: str,
    output_file: str,
) -> str:
    raw_records = read_jsonl(raw_file)
    recoverer = SpanRecoverer(
        threshold=float(config.get("recovery_threshold", 0.92)),
        debug_dir=str(Path(config["output_dir"]) / "prediction_debug"),
    )

    exported_records = []
    for record in raw_records:
        parse_result = parse_model_response(record["raw_output"], source_text=None)
        predicted_skills = []
        unresolved = []
        for skill in parse_result.skills:
            if "start" in skill and "end" in skill:
                predicted_skills.append(
                    {
                        "text": skill["text"],
                        "label": skill["label"],
                        "start": skill["start"],
                        "end": skill["end"],
                        "recovery_type": "exact",
                    }
                )
            else:
                unresolved.append(skill)

        if config.get("use_span_recovery", False) and unresolved:
            recovered, trace = recoverer.recover_example(record["id"], record["text"], unresolved)
            predicted_skills.extend(recovered)
        else:
            trace = {"id": record["id"], "rejected": unresolved}
            for skill in unresolved:
                parse_result.rejected_items.append({"reason": "missing_offsets_without_recovery", "item": skill})

        exported_records.append(
            {
                "id": record["id"],
                "text": record["text"],
                "predicted_skills": sorted(predicted_skills, key=lambda item: (item["start"], item["end"], item["label"])),
                "raw_output": record["raw_output"],
                "repair_actions": parse_result.repair_actions,
                "rejected_items": parse_result.rejected_items,
                "generation_seconds": record.get("generation_seconds"),
                "gpu_memory_mb": record.get("gpu_memory_mb"),
                "trace_rejections": trace.get("rejected", []),
            }
        )

    write_jsonl(output_file, exported_records)
    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse and normalize raw generation outputs.")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--raw-file", required=True, help="Raw generation jsonl file.")
    parser.add_argument("--output-file", required=True, help="Normalized prediction jsonl file.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    export_predictions_file(config, args.raw_file, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
