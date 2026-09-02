from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.runtime_utils import (
    apply_chat_template,
    build_messages,
    load_model_and_tokenizer,
    load_yaml_config,
    maybe_get_max_memory_mb,
    read_jsonl,
    write_jsonl,
)


def _model_device(model) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - defensive
        return "cpu"


def generate_raw_predictions(
    config: dict[str, Any],
    input_file: str,
    output_file: str,
    adapter_path: str | None = None,
) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - setup validation
        raise RuntimeError("torch must be installed before running inference.") from exc

    tokenizer, model = load_model_and_tokenizer(config, for_training=False, adapter_path=adapter_path)
    examples = read_jsonl(input_file)
    output_rows = []
    batch_size = int(config.get("eval_batch_size", config.get("batch_size", 1)))
    device = _model_device(model)

    for batch_start in range(0, len(examples), batch_size):
        batch_examples = examples[batch_start : batch_start + batch_size]
        prompts = []
        for example in batch_examples:
            messages = build_messages(
                text=example["text"],
                prompt_template_file=config.get("prompt_template_file"),
                fewshot_template_file=config.get("fewshot_prompt_file") or None,
                use_fewshot=bool(config.get("use_fewshot_for_inference", True)),
                system_prompt=config.get("system_prompt"),
                user_prompt_template=config.get("user_prompt_template"),
            )
            prompts.append(
                apply_chat_template(
                    tokenizer=tokenizer,
                    messages=messages,
                    add_generation_prompt=True,
                    non_thinking_mode=bool(config.get("non_thinking_mode", False)),
                )
            )

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(config["max_seq_length"]),
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        start_time = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(config.get("max_new_tokens", 512)),
                do_sample=bool(float(config.get("temperature", 0.0)) > 0.0 or float(config.get("top_p", 1.0)) < 1.0),
                temperature=float(config.get("temperature", 0.0)),
                top_p=float(config.get("top_p", 1.0)),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_duration = time.perf_counter() - start_time

        generated_tokens = generated[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        gpu_memory_mb = maybe_get_max_memory_mb()

        for example, prompt, raw_output in zip(batch_examples, prompts, decoded):
            output_rows.append(
                {
                    "id": example["id"],
                    "text": example["text"],
                    "prompt": prompt,
                    "raw_output": raw_output.strip(),
                    "generation_seconds": batch_duration / max(len(batch_examples), 1),
                    "gpu_memory_mb": gpu_memory_mb,
                }
            )

    write_jsonl(output_file, output_rows)
    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate raw model outputs for span extraction.")
    parser.add_argument("--config", required=True, help="YAML config file.")
    parser.add_argument("--input-file", required=True, help="Input jsonl file.")
    parser.add_argument("--output-file", required=True, help="Raw generation output jsonl file.")
    parser.add_argument("--adapter-path", default=None, help="Optional adapter directory.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    generate_raw_predictions(config, args.input_file, args.output_file, adapter_path=args.adapter_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
