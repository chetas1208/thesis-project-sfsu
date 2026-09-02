from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

from src.benchmark.alignment import align_predicted_skills
from src.benchmark.dataset import derive_smoke_output_dir, prepare_benchmark_dataset
from src.benchmark.evaluate import evaluate_predictions
from src.benchmark.parsing import parse_benchmark_output
from src.benchmark.reporting import LEADERBOARD_COLUMNS, write_leaderboard, write_summary_report
from src.infer.generate_predictions import generate_raw_predictions
from src.runtime_utils import ensure_dir, load_yaml_config, read_json, read_jsonl, write_jsonl
from src.train.train_sft_v2 import train_sft


ROOT_DIR = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "You are SkillsLLM, an information extraction model specialized in identifying skill mentions "
    "from resumes, job postings, and professional text. Your job is to extract only explicit skill "
    "spans from the input text. Do not infer skills that are not directly stated. Return valid JSON only."
)

USER_PROMPT_TEMPLATE = """Extract all skill spans from the text below.

Rules:
- Extract exact surface spans from the input text.
- Do not paraphrase.
- Do not normalize names unless a separate normalized field is requested.
- Do not invent missing skills.
- Include technical skills, tools, programming languages, frameworks, libraries, platforms, methodologies, domain skills, and soft skills only when explicitly mentioned.
- Ignore job titles, company names, degrees, dates, locations, and generic responsibilities unless they are explicitly skill mentions.
- If no skills are present, return an empty list.
- Return valid JSON only.

Text:
{{ text }}

Output JSON schema:
{
  "skills": [
    {
      "text": "exact skill span from input",
      "label": "HARD_SKILL or SOFT_SKILL or SKILL"
    }
  ]
}

Important:
- The "text" value must exactly match a substring from the input.
- Use "HARD_SKILL" for technical/domain/tool skills.
- Use "SOFT_SKILL" for interpersonal or behavioral skills.
- Use "SKILL" only if the dataset does not distinguish hard and soft skills.
"""

MODEL_CONFIG_MAP = {
    "gemma-4-26b-a4b-it": ROOT_DIR / "configs" / "gemma4_26b_a4b_v2.yaml",
    "gemma-4-e4b-it": ROOT_DIR / "configs" / "gemma4_e4b_v2.yaml",
    "llama-3.1-8b-instruct": ROOT_DIR / "configs" / "baseline_llama31_8b.yaml",
    "mistral-small-3.1-24b-instruct-2503": ROOT_DIR / "configs" / "mistral_small_3_1_v2.yaml",
    "qwen3-14b": ROOT_DIR / "configs" / "qwen3_14b_v2.yaml",
}


def _parse_version(version: str) -> tuple[int, ...]:
    parts = []
    for item in version.split("."):
        value = ""
        for char in item:
            if char.isdigit():
                value += char
            else:
                break
        if value:
            parts.append(int(value))
    return tuple(parts)


def validate_environment() -> list[str]:
    errors: list[str] = []
    try:
        import tokenizers
        import transformers
    except ImportError as exc:
        return [f"Missing runtime dependency: {exc}"]

    if _parse_version(transformers.__version__) < (4, 51, 0):
        errors.append(
            f"transformers {transformers.__version__} is too old; expected >= 4.51.0 for the cached benchmark models."
        )
    if _parse_version(tokenizers.__version__) < (0, 21, 0):
        errors.append(f"tokenizers {tokenizers.__version__} is too old; expected >= 0.21.0.")
    return errors


def _runtime_versions() -> dict[str, str]:
    versions = {}
    for module_name in ("torch", "transformers", "tokenizers", "bitsandbytes", "peft", "accelerate"):
        try:
            module = __import__(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[module_name] = f"unavailable: {exc}"
        else:
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _resolve_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((ROOT_DIR / path).resolve())


def resolve_model_config(
    model_name: str,
    benchmark_config: dict[str, Any],
    benchmark_model_config: dict[str, Any],
    prepared_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    if model_name not in MODEL_CONFIG_MAP:
        raise KeyError(f"Unsupported benchmark model: {model_name}")

    base_config = load_yaml_config(MODEL_CONFIG_MAP[model_name])
    benchmark_section = benchmark_config["benchmark"]
    base_config.update(
        {
            "experiment_name": model_name,
            "base_model_name": model_name,
            "local_model_path": _resolve_path(base_config["local_model_path"]),
            "train_file": str(prepared_dir / "train.jsonl"),
            "val_file": str(prepared_dir / "dev.jsonl"),
            "test_file": str(prepared_dir / "test.jsonl"),
            "output_dir": str(model_dir / "training"),
            "prompt_template_file": None,
            "fewshot_prompt_file": None,
            "use_fewshot_during_training": False,
            "use_fewshot_for_inference": False,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": USER_PROMPT_TEMPLATE,
            "training_target_include_offsets": False,
            "batch_size": int(benchmark_section.get("batch_size", base_config.get("batch_size", 1))),
            "eval_batch_size": int(benchmark_section.get("batch_size", base_config.get("eval_batch_size", 1))),
            "temperature": float(benchmark_model_config.get("temperature", 0.0)),
            "top_p": float(benchmark_model_config.get("top_p", 1.0)),
            "max_new_tokens": int(benchmark_model_config.get("max_tokens", 1024)),
        }
    )
    if (
        model_name == "llama-3.1-8b-instruct"
        or model_name.startswith("gemma-4")
        or model_name.startswith("mistral-small-3.1")
    ):
        base_config["use_4bit"] = True
    if model_name.startswith("gemma-4") or model_name.startswith("mistral-small-3.1"):
        base_config["device_map"] = "balanced"
        if model_name == "gemma-4-26b-a4b-it":
            base_config["use_4bit"] = True
            base_config["use_8bit"] = False
            base_config["max_memory"] = {0: "14GiB", 1: "14GiB", "cpu": "160GiB"}
            base_config["quantization_cpu_offload"] = True
            base_config["offload_folder"] = str(model_dir / "offload")
            base_config["preferred_model_classes"] = ["AutoModelForImageTextToText", "AutoModelForCausalLM"]
            base_config["skip_prepare_model_for_kbit_training"] = False
        else:
            base_config["max_memory"] = {0: "8GiB", 1: "22GiB", "cpu": "96GiB"}
        base_config["target_modules"] = (
            r".*language_model.*("
            r"q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
            r")$"
        )
    if model_name.startswith("gemma-4"):
        base_config["max_seq_length"] = min(int(base_config.get("max_seq_length", 2048)), 1024)
        base_config["preferred_model_classes"] = ["AutoModelForImageTextToText", "AutoModelForCausalLM"]
    if model_name == "gemma-4-26b-a4b-it":
        base_config["max_seq_length"] = 256
    if model_name == "gemma-4-e4b-it":
        base_config["cuda_visible_devices"] = "1"
        base_config["device_map"] = {"": 0}
        base_config["max_memory"] = {0: "22GiB", "cpu": "96GiB"}
        base_config["max_seq_length"] = 512
    return base_config


def normalize_predictions(
    raw_file: str | Path,
    output_file: str | Path,
    errors_file: str | Path,
) -> dict[str, Any]:
    raw_records = read_jsonl(raw_file)
    normalized_rows = []
    error_rows = []

    parse_failures = 0
    unaligned_predictions = 0
    total_predictions = 0

    for record in raw_records:
        parse_result = parse_benchmark_output(record.get("raw_output", ""))
        if parse_result.parse_failed:
            parse_failures += 1
            error_rows.append(
                {
                    "type": "parse_failure",
                    "id": record["id"],
                    "error": parse_result.parse_error,
                    "raw_output": record.get("raw_output", ""),
                }
            )

        for rejected in parse_result.rejected_items:
            error_rows.append(
                {
                    "type": "rejected_item",
                    "id": record["id"],
                    "error": rejected.get("reason"),
                    "item": rejected.get("item"),
                    "raw_output": record.get("raw_output", ""),
                }
            )

        aligned, record_unaligned = align_predicted_skills(record["text"], parse_result.skills)
        total_predictions += len(aligned)
        unaligned_predictions += record_unaligned

        normalized_rows.append(
            {
                "id": record["id"],
                "text": record["text"],
                "predicted_skills": aligned,
                "parse_failed": parse_result.parse_failed,
                "repair_actions": parse_result.repair_actions,
                "rejected_items": parse_result.rejected_items,
                "generation_seconds": record.get("generation_seconds"),
                "gpu_memory_mb": record.get("gpu_memory_mb"),
            }
        )

    write_jsonl(output_file, normalized_rows)
    write_jsonl(errors_file, error_rows)
    return {
        "parse_failures": parse_failures,
        "total_records": len(raw_records),
        "unaligned_predictions": unaligned_predictions,
        "total_predictions": total_predictions,
    }


def _model_artifacts(model_dir: Path) -> dict[str, Path]:
    training_dir = model_dir / "training"
    return {
        "training_dir": training_dir,
        "training_summary": training_dir / "training_summary.json",
        "raw_outputs": model_dir / "raw_outputs.jsonl",
        "predictions": model_dir / "predictions.jsonl",
        "metrics": model_dir / "metrics.json",
        "errors": model_dir / "errors.jsonl",
    }


def _last_model_failure(errors_path: Path) -> str | None:
    if not errors_path.exists():
        return None
    try:
        rows = read_jsonl(errors_path)
    except Exception:
        return None

    failure_reason = None
    for row in rows:
        if row.get("type") == "model_failure" and row.get("error"):
            failure_reason = str(row["error"])
    return failure_reason


def load_existing_model_result(output_dir: str | Path, model_name: str) -> dict[str, Any]:
    artifacts = _model_artifacts(Path(output_dir) / model_name)
    result: dict[str, Any] = {
        "status": "pending",
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }

    if artifacts["metrics"].exists():
        result["status"] = "completed"
        result["metrics"] = read_json(artifacts["metrics"])
    elif artifacts["predictions"].exists() or artifacts["raw_outputs"].exists() or _training_complete(artifacts["training_dir"]):
        result["status"] = "incomplete"

    failure_reason = _last_model_failure(artifacts["errors"])
    if failure_reason and "metrics" not in result:
        result["status"] = "failed"
        result["failure_reason"] = failure_reason
    return result


def _training_complete(training_dir: Path) -> bool:
    return (training_dir / "adapter_config.json").exists() or (training_dir / "adapter_model.safetensors").exists()


def should_skip_model(artifacts: dict[str, Path], overwrite: bool) -> bool:
    return artifacts["predictions"].exists() and artifacts["metrics"].exists() and not overwrite


def _apply_runtime_environment(config: dict[str, Any]) -> dict[str, str | None]:
    updates = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if config.get("cuda_visible_devices"):
        updates["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])

    previous = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        os.environ[key] = value
    return previous


def _restore_runtime_environment(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run_single_model(
    benchmark_config: dict[str, Any],
    benchmark_model_config: dict[str, Any],
    prepared_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    model_name = benchmark_model_config["name"]
    model_dir = ensure_dir(output_dir / model_name)
    artifacts = _model_artifacts(model_dir)
    result: dict[str, Any] = {
        "status": "pending",
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }

    if should_skip_model(artifacts, overwrite=overwrite):
        result["status"] = "skipped"
        return result

    start_time = time.perf_counter()
    config = resolve_model_config(model_name, benchmark_config, benchmark_model_config, prepared_dir, model_dir)
    previous_environment = _apply_runtime_environment(config)

    try:
        if overwrite or not _training_complete(artifacts["training_dir"]):
            train_sft(config)

        if overwrite or not artifacts["raw_outputs"].exists():
            generate_raw_predictions(
                config,
                input_file=str(prepared_dir / f"{benchmark_config['benchmark']['split']}.jsonl"),
                output_file=str(artifacts["raw_outputs"]),
                adapter_path=str(artifacts["training_dir"]),
            )

        if overwrite or not artifacts["predictions"].exists():
            normalize_predictions(
                raw_file=artifacts["raw_outputs"],
                output_file=artifacts["predictions"],
                errors_file=artifacts["errors"],
            )

        total_runtime = time.perf_counter() - start_time
        metrics = evaluate_predictions(
            gold_file=prepared_dir / f"{benchmark_config['benchmark']['split']}.jsonl",
            prediction_file=artifacts["predictions"],
            output_file=artifacts["metrics"],
            total_runtime_seconds=total_runtime,
        )
        result["metrics"] = metrics
        result["status"] = "completed"
        return result
    except Exception as exc:
        failure_record = {
            "type": "model_failure",
            "model_name": model_name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime": _runtime_versions(),
        }
        result["status"] = "failed"
        result["failure_reason"] = str(exc)
        if not artifacts["errors"].exists():
            write_jsonl(artifacts["errors"], [failure_record])
        else:
            rows = read_jsonl(artifacts["errors"])
            rows.append(failure_record)
            write_jsonl(artifacts["errors"], rows)
        return result
    finally:
        _restore_runtime_environment(previous_environment)


def run_benchmark(
    benchmark_config: dict[str, Any],
    dataset_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    limit: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    errors = validate_environment()
    if errors:
        raise RuntimeError("; ".join(errors))

    resolved_output_dir = derive_smoke_output_dir(output_dir, limit)
    ensure_dir(resolved_output_dir)

    benchmark_section = benchmark_config["benchmark"]
    dataset_metadata = prepare_benchmark_dataset(
        dataset_path=dataset_path,
        output_dir=resolved_output_dir,
        split=str(benchmark_section["split"]),
        seed=int(benchmark_section["seed"]),
        limit=limit,
    )
    prepared_dir = Path(dataset_metadata["prepared_dir"])

    if model_name is not None:
        selected_models = [model for model in benchmark_config["models"] if model["name"] == model_name]
    else:
        selected_models = list(benchmark_config["models"])

    if not selected_models:
        raise ValueError(f"No benchmark model found for selection: {model_name}")

    model_results: dict[str, dict[str, Any]] = {}
    for model in selected_models:
        model_results[model["name"]] = run_single_model(
            benchmark_config=benchmark_config,
            benchmark_model_config=model,
            prepared_dir=prepared_dir,
            output_dir=Path(resolved_output_dir),
            overwrite=overwrite,
        )

    full_results = {
        model["name"]: load_existing_model_result(resolved_output_dir, model["name"])
        for model in benchmark_config["models"]
    }
    for name, result in model_results.items():
        existing = full_results.get(name, {})
        merged = {**existing, **result}
        if "metrics" not in merged and existing.get("metrics"):
            merged["metrics"] = existing["metrics"]
        if merged.get("metrics") or merged.get("status") in {"completed", "skipped"}:
            merged.pop("failure_reason", None)
        full_results[name] = merged

    for model in benchmark_config["models"]:
        metrics_path = Path(resolved_output_dir) / model["name"] / "metrics.json"
        if metrics_path.exists():
            full_results.setdefault(model["name"], {})["metrics"] = read_json(metrics_path)
            if full_results[model["name"]].get("status") == "pending":
                full_results[model["name"]]["status"] = "completed"

    write_leaderboard(resolved_output_dir, full_results)
    write_summary_report(resolved_output_dir, benchmark_config, dataset_metadata, full_results)
    return {
        "output_dir": str(resolved_output_dir),
        "dataset": dataset_metadata,
        "models": full_results,
        "leaderboard_columns": LEADERBOARD_COLUMNS,
    }
