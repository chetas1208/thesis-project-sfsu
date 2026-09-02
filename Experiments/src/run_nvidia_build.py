import argparse
import json
import os
import time
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
from model_adapters.gpt5 import build_prompt
from model_adapters.nvidia_build import normalize_nvidia_build_response


SYSTEM_PROMPT = """You extract technical skill spans from technical resume text.
Return valid JSON only.
Schema:
{
  "skills": ["exact skill text from the input", "..."]
}
Rules:
- Extract only skill mentions that appear verbatim in the sentence.
- Return the exact surface strings only, not offsets.
- If there are no skills, return an empty skills array.
- Do not include explanation, reasoning, markdown, or extra keys."""


def _read_prompt_records(input_dir: Path, split: str) -> list[dict]:
    prompt_file = input_dir / f"{split}_prompts.jsonl"
    rows = read_json_records(prompt_file)
    normalized = []
    for row in rows:
        normalized.append(
            {
                "id": row["id"],
                "resume_id": row.get("resume_id"),
                "section_name": row.get("section_name"),
                "text": row["text"],
                "prompt": row["prompt"],
            }
        )
    return normalized


def _build_nvidia_user_prompt(record: dict) -> str:
    parts = []
    if record.get("resume_id"):
        parts.append(f"Resume ID: {record['resume_id']}")
    if record.get("section_name"):
        parts.append(f"Section: {record['section_name']}")
    parts.append("Input sentence:")
    parts.append(record["text"])
    return "\n".join(parts)


def _reasoning_kwargs(model_id: str, defaults: dict) -> dict:
    if "gpt-oss" not in model_id:
        return {}
    return {"reasoning_effort": defaults["reasoning_effort"]}


def _max_tokens_for_model(model_id: str, defaults: dict) -> int:
    if "gemma" in model_id:
        return max(defaults["max_tokens"], 900)
    return defaults["max_tokens"]


def _read_jsonl_tolerant(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSONL line {line_number} in {path}")
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _append_jsonl(path: Path, record: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
        handle.flush()


def _sanitize_jsonl(path: Path) -> list[dict]:
    rows = _read_jsonl_tolerant(path)
    if path.exists():
        write_jsonl(path, rows)
    return rows


def _request_with_retries(client, *, model: str, messages: list[dict], defaults: dict, max_retries: int) -> tuple[object, int]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=defaults["temperature"],
                top_p=defaults["top_p"],
                max_tokens=_max_tokens_for_model(model, defaults),
                stream=False,
                timeout=180.0,
                **_reasoning_kwargs(model, defaults),
            )
            return response, attempt
        except Exception as exc:  # pragma: no cover - network/provider variance
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = min(30, 2 ** (attempt - 1))
            print(f"Request failed on attempt {attempt}/{max_retries}: {exc}. Retrying in {sleep_seconds}s.")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"NVIDIA Build request failed after {max_retries} attempts: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVIDIA Build prompt baselines for skill extraction.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    if args.model_key not in config["models"]:
        raise KeyError(f"Unknown model key: {args.model_key}")

    model_cfg = config["models"][args.model_key]
    defaults = config["training_defaults"]["nvidia_build"]
    input_dir = resolve_path(model_cfg["input_dir"])
    prompt_records = _read_prompt_records(input_dir, args.split)
    if args.limit is not None:
        prompt_records = prompt_records[: args.limit]

    results_dir = ensure_dir(resolve_path(model_cfg["results_dir"]))
    raw_dir = ensure_dir(results_dir / "raw")
    raw_output = raw_dir / f"{args.split}_responses.jsonl"
    predictions_output = results_dir / "predictions.jsonl"

    if args.dry_run:
        print(f"Would query NVIDIA Build model {model_cfg['base_model']} for {len(prompt_records)} records")
        return

    api_key = os.environ.get("NVIDIA_NIM_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_NIM_API_KEY is required to run NVIDIA Build baselines.")

    from openai import OpenAI  # pylint: disable=import-outside-toplevel

    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )

    if args.restart:
        existing_raw_rows = []
        existing_prediction_rows = []
        if raw_output.exists():
            raw_output.unlink()
        if predictions_output.exists():
            predictions_output.unlink()
    else:
        existing_raw_rows = _sanitize_jsonl(raw_output)
        existing_prediction_rows = _sanitize_jsonl(predictions_output)

    completed_ids = {row["id"] for row in existing_prediction_rows if row.get("id")}
    pending_records = [record for record in prompt_records if record["id"] not in completed_ids]
    print(
        f"Resuming {args.model_key} on split={args.split}: "
        f"{len(completed_ids)} completed, {len(pending_records)} pending, total={len(prompt_records)}"
    )

    for index, record in enumerate(pending_records, start=1):
        try:
            response, attempt_count = _request_with_retries(
                client,
                model=model_cfg["base_model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_nvidia_user_prompt(record)},
                ],
                defaults=defaults,
                max_retries=args.max_retries,
            )
            message = response.choices[0].message
            content = message.content or ""
            reasoning_text = getattr(message, "reasoning", None)
            raw_row = {
                "id": record["id"],
                "response_text": content,
                "reasoning_text": reasoning_text,
                "model": model_cfg["base_model"],
                "attempts": attempt_count,
            }
            try:
                predicted_spans = normalize_nvidia_build_response(record["text"], content)
            except Exception as exc:  # pragma: no cover - model output variance
                raw_row["parse_error"] = str(exc)
                predicted_spans = []
        except Exception as exc:  # pragma: no cover - provider instability
            raw_row = {
                "id": record["id"],
                "response_text": "",
                "reasoning_text": "",
                "model": model_cfg["base_model"],
                "request_error": str(exc),
            }
            predicted_spans = []

        _append_jsonl(raw_output, raw_row)
        _append_jsonl(predictions_output, {"id": record["id"], "predicted_spans": predicted_spans})
        if index % 10 == 0 or index == len(pending_records):
            finished = len(completed_ids) + index
            print(f"{args.model_key}: completed {finished}/{len(prompt_records)}")

    write_run_metadata(
        results_dir / "run_metadata.json",
        {
            "model": args.model_key,
            "provider": "nvidia_build",
            "base_model": model_cfg["base_model"],
            "split": args.split,
            "raw_output_file": str(raw_output),
            "prediction_file": str(predictions_output),
            "completed_predictions": len(_read_jsonl_tolerant(predictions_output)),
        },
    )
    print(f"Unified predictions written to {predictions_output}")


if __name__ == "__main__":
    main()
