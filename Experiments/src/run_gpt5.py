import argparse
import json
import os
from pathlib import Path

from common import (
    ensure_benchmark_env,
    ensure_dir,
    load_config,
    load_env_file,
    read_json_records,
    resolve_path,
    write_jsonl,
    write_run_metadata,
)
from model_adapters.gpt5 import build_prompt, normalize_gpt5_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GPT-5 prompt baseline for skill extraction.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    model_cfg = config["models"]["gpt5"]
    defaults = config["training_defaults"]["gpt5"]
    prompt_template = resolve_path(config["paths"]["prompt_template"]).read_text(encoding="utf-8")
    results_dir = ensure_dir(resolve_path(model_cfg["results_dir"]))
    raw_dir = ensure_dir(results_dir / "raw")
    raw_output = raw_dir / f"{args.split}_responses.jsonl"
    predictions_output = results_dir / "predictions.jsonl"
    canonical_path = resolve_path(config["paths"]["canonical_dir"]) / f"{args.split}.jsonl"
    records = read_json_records(canonical_path) if canonical_path.exists() else []
    if args.limit is not None:
        records = records[: args.limit]

    if args.dry_run:
        print(f"Would query {model_cfg['base_model']} for {len(records)} records from {canonical_path}")
        return

    if not records:
        raise FileNotFoundError(f"No canonical records found for split '{args.split}' at {canonical_path}")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required to run the GPT-5 baseline.")

    from openai import OpenAI  # pylint: disable=import-outside-toplevel

    client = OpenAI(api_key=api_key)
    reasoning_effort = args.reasoning_effort or defaults["reasoning_effort"]
    raw_rows = []
    unified_rows = []
    response_schema = {
        "type": "object",
        "properties": {
            "sentence": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "start_char": {"type": "integer"},
                        "end_char": {"type": "integer"},
                        "canonical_name": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                    },
                    "required": ["text", "start_char", "end_char", "canonical_name", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["sentence", "skills"],
        "additionalProperties": False,
    }

    for record in records:
        prompt = build_prompt(prompt_template, record)
        response = client.responses.create(
            model=model_cfg["base_model"],
            reasoning={"effort": reasoning_effort},
            input=[
                {"role": "system", "content": "You extract technical skill spans from technical resume text."},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "skill_extraction",
                    "schema": response_schema,
                    "strict": True,
                }
            },
        )
        payload = json.loads(response.output_text)
        raw_rows.append({"id": record["id"], "response": payload})
        unified_rows.append({"id": record["id"], "predicted_spans": normalize_gpt5_response(record["text"], payload)})

    write_jsonl(raw_output, raw_rows)
    write_jsonl(predictions_output, unified_rows)
    write_run_metadata(
        results_dir / "run_metadata.json",
        {
            "model": "gpt5",
            "base_model": model_cfg["base_model"],
            "split": args.split,
            "reasoning_effort": reasoning_effort,
            "raw_output_file": str(raw_output),
        },
    )
    print(f"Unified predictions written to {predictions_output}")


if __name__ == "__main__":
    main()
