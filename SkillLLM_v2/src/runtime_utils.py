from __future__ import annotations

import inspect
import gc
import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    import orjson
except ImportError:  # pragma: no cover - optional dependency during bootstrapping
    orjson = None


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["config_path"] = str(Path(path).resolve())
    return config


def save_yaml_config(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - optional in early setup stages
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - optional in early setup stages
        pass


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "rb") as handle:
        payload = handle.read()
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload.decode("utf-8"))


def write_json(path: str | os.PathLike[str], data: Any, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    if orjson is not None:
        option = 0
        if indent:
            option |= orjson.OPT_INDENT_2
        payload = orjson.dumps(data, option=option)
        with open(path, "wb") as handle:
            handle.write(payload)
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            if orjson is not None:
                records.append(orjson.loads(raw_line))
            else:
                records.append(json.loads(raw_line.decode("utf-8")))
    return records


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "wb") as handle:
        for row in rows:
            if orjson is not None:
                payload = orjson.dumps(row)
            else:
                payload = json.dumps(row, ensure_ascii=False).encode("utf-8")
            handle.write(payload + b"\n")


def load_text(path: str | os.PathLike[str]) -> str:
    return Path(path).read_text(encoding="utf-8")


def serialize_skills(skills: list[dict[str, Any]], include_offsets: bool = True) -> str:
    ordered_skills = sorted(
        skills,
        key=lambda item: (item.get("start", 10**9), item.get("end", 10**9), item.get("label", ""), item.get("text", "")),
    )
    serialized = []
    for item in ordered_skills:
        row = {
            "text": item.get("text"),
            "label": item.get("label"),
        }
        if include_offsets:
            row["start"] = item.get("start")
            row["end"] = item.get("end")
        serialized.append(row)
    return json.dumps({"skills": serialized}, ensure_ascii=False, separators=(",", ":"))


def render_prompt(template_path: str | os.PathLike[str], **context: Any) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:  # pragma: no cover - setup validation
        raise RuntimeError("jinja2 is required to render prompt templates.") from exc

    template = Template(load_text(template_path))
    return template.render(**context)


def render_prompt_template(template_text: str, **context: Any) -> str:
    try:
        from jinja2 import Template
    except ImportError as exc:  # pragma: no cover - setup validation
        raise RuntimeError("jinja2 is required to render prompt templates.") from exc

    template = Template(template_text)
    return template.render(**context)


def build_messages(
    text: str,
    prompt_template_file: str | None,
    fewshot_template_file: str | None = None,
    use_fewshot: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> list[dict[str, str]]:
    if user_prompt_template:
        prompt = render_prompt_template(user_prompt_template, text=text)
    else:
        if prompt_template_file is None:
            raise ValueError("prompt_template_file is required when user_prompt_template is not provided.")
        template_path = fewshot_template_file if use_fewshot and fewshot_template_file else prompt_template_file
        prompt = render_prompt(template_path, text=text)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool, non_thinking_mode: bool) -> str:
    signature = inspect.signature(tokenizer.apply_chat_template)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if "enable_thinking" in signature.parameters:
        kwargs["enable_thinking"] = not non_thinking_mode
    return tokenizer.apply_chat_template(messages, **kwargs)


def format_training_text(
    tokenizer: Any,
    text: str,
    target_json: str,
    prompt_template_file: str | None,
    fewshot_template_file: str | None,
    config: dict[str, Any],
) -> str:
    messages = build_messages(
        text=text,
        prompt_template_file=prompt_template_file,
        fewshot_template_file=fewshot_template_file,
        use_fewshot=bool(config.get("use_fewshot_during_training", False)),
        system_prompt=config.get("system_prompt"),
        user_prompt_template=config.get("user_prompt_template"),
    )
    messages.append({"role": "assistant", "content": target_json})
    rendered = apply_chat_template(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=False,
        non_thinking_mode=bool(config.get("non_thinking_mode", False)),
    )
    eos_token = getattr(tokenizer, "eos_token", None) or ""
    return rendered + eos_token


def parse_override_value(raw_value: str) -> Any:
    return yaml.safe_load(raw_value)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value format.")
        key_path, raw_value = override.split("=", 1)
        value = parse_override_value(raw_value)
        cursor = updated
        keys = key_path.split(".")
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return updated


def experiment_name_from_config(config: dict[str, Any]) -> str:
    if config.get("experiment_name"):
        return str(config["experiment_name"])
    output_dir = Path(config["output_dir"])
    if output_dir.name:
        return output_dir.name
    return Path(config["local_model_path"]).name


def resolve_compute_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def detect_attn_implementation(config: dict[str, Any]) -> str | None:
    if not config.get("use_flash_attention_if_available", False):
        return "sdpa"
    return "flash_attention_2" if flash_attention_available() else "sdpa"


def build_bitsandbytes_config(config: dict[str, Any], dtype: Any):
    from transformers import BitsAndBytesConfig

    if config.get("use_8bit", False):
        kwargs: dict[str, Any] = {
            "load_in_8bit": True,
        }
    else:
        kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": dtype,
        }
    if config.get("quantization_cpu_offload", False):
        # Some large cached local models cannot fit fully on the available GPUs
        # even after quantization. This allows accelerate/transformers to keep
        # the overflow modules on CPU instead of failing during dispatch.
        kwargs["llm_int8_enable_fp32_cpu_offload"] = True
    return BitsAndBytesConfig(**kwargs)


def patch_bitsandbytes_param_constructors() -> None:
    try:
        from bitsandbytes.nn import modules as bnb_modules
        from bitsandbytes.autograd._functions import MatmulLtState
    except Exception:  # pragma: no cover - optional dependency guard
        return

    for class_name in ("Int8Params", "Params4bit"):
        param_class = getattr(bnb_modules, class_name, None)
        if param_class is None or getattr(param_class, "_skillllm_accepts_hf_init_flag", False):
            continue

        original_new = param_class.__new__

        def patched_new(cls, *args, _original_new=original_new, **kwargs):
            kwargs.pop("_is_hf_initialized", None)
            return _original_new(cls, *args, **kwargs)

        param_class.__new__ = staticmethod(patched_new)
        param_class._skillllm_accepts_hf_init_flag = True

    if not hasattr(MatmulLtState, "memory_efficient_backward"):
        MatmulLtState.memory_efficient_backward = False


def load_model_and_tokenizer(
    config: dict[str, Any],
    for_training: bool,
    adapter_path: str | None = None,
):
    try:
        import torch
        from peft import PeftModel
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - setup validation
        raise RuntimeError("transformers, peft, and torch must be installed before loading models.") from exc

    local_model_path = config["local_model_path"]
    local_model_dir = Path(config["local_model_path"])
    tokenizer = None
    tokenizer_errors: list[str] = []
    tokenizer_attempts = (
        {"use_fast": True},
        {"use_fast": True, "extra_special_tokens": {}},
        {"use_fast": False},
        {"use_fast": False, "extra_special_tokens": {}},
    )
    for attempt in tokenizer_attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                **attempt,
            )
            break
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            tokenizer_errors.append(f"{attempt}: {type(exc).__name__}: {exc}")
    if tokenizer is None:
        try:
            import tokenizers
            from transformers import __version__ as transformers_version
        except Exception:  # pragma: no cover - defensive
            tokenizers = None
            transformers_version = "unknown"
        tokenizers_version = getattr(tokenizers, "__version__", "unknown")
        joined = " | ".join(tokenizer_errors)
        raise RuntimeError(
            "Failed to load tokenizer for "
            f"{local_model_path}. transformers={transformers_version}, tokenizers={tokenizers_version}. "
            f"Attempts: {joined}"
        )
    if not getattr(tokenizer, "chat_template", None):
        chat_template_json = local_model_dir / "chat_template.json"
        chat_template_jinja = local_model_dir / "chat_template.jinja"
        if chat_template_json.exists():
            payload = read_json(chat_template_json)
            template = payload["chat_template"] if isinstance(payload, dict) else None
            if template:
                tokenizer.chat_template = template
        elif chat_template_jinja.exists():
            tokenizer.chat_template = chat_template_jinja.read_text(encoding="utf-8")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right" if for_training else "left"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    dtype = resolve_compute_dtype()
    model_kwargs["torch_dtype"] = dtype

    attn_implementation = detect_attn_implementation(config)
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if config.get("max_memory"):
        model_kwargs["max_memory"] = config["max_memory"]
    if config.get("offload_folder"):
        offload_folder = ensure_dir(config["offload_folder"])
        model_kwargs["offload_folder"] = str(offload_folder)
        model_kwargs["offload_state_dict"] = True

    use_4bit = bool(config.get("use_4bit", False))
    use_8bit = bool(config.get("use_8bit", False))
    if use_4bit or use_8bit:
        if not torch.cuda.is_available():
            raise RuntimeError("Quantized loading requires CUDA.")
        patch_bitsandbytes_param_constructors()
        model_kwargs["quantization_config"] = build_bitsandbytes_config(config, dtype)
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        default_device_map = {"": local_rank} if for_training and world_size > 1 else "auto"
        model_kwargs["device_map"] = config.get("device_map", default_device_map)
    elif not for_training and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = None
    model_errors: list[str] = []
    available_model_classes = {"AutoModelForCausalLM": AutoModelForCausalLM}
    for optional_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        optional_class = getattr(transformers, optional_name, None)
        if optional_class is not None:
            available_model_classes[optional_name] = optional_class

    preferred_model_classes = config.get("preferred_model_classes") or []
    ordered_class_names = list(preferred_model_classes) + [
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ]
    model_classes = []
    seen_class_names = set()
    for class_name in ordered_class_names:
        if class_name in seen_class_names or class_name not in available_model_classes:
            continue
        seen_class_names.add(class_name)
        model_classes.append(available_model_classes[class_name])

    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(local_model_path, **model_kwargs)
            break
        except Exception as exc:  # pragma: no cover - model/version dependent
            model_errors.append(f"{model_class.__name__}: {type(exc).__name__}: {exc}")
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if model is None:
        raise RuntimeError(f"Failed to load model for {local_model_path}. Attempts: {' | '.join(model_errors)}")

    if adapter_path:
        adapter_dir = Path(adapter_path)
        if (adapter_dir / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=for_training)

    if for_training and config.get("use_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if not for_training:
        model.eval()

    return tokenizer, model


def maybe_detect_gpu_memory() -> list[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                values.append(int(line))
            except ValueError:
                continue
    return values


def maybe_get_max_memory_mb() -> int | None:
    memory_values = maybe_detect_gpu_memory()
    if not memory_values:
        return None
    return max(memory_values)


def strip_markdown_fences(text: str) -> str:
    fenced_block = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_block:
        return fenced_block.group(1).strip()
    return text.strip()
