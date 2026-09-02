from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml

from skillllm_v2.io_utils import write_json, write_text


def run_gpu_preflight(config_path: str, require_two_gpus: bool = True) -> dict[str, Any]:
    config = _load_config(config_path)
    report: dict[str, Any] = {
        "config": config_path,
        "cuda": _cuda_report(),
        "components": {},
        "blocking_errors": [],
        "warnings": [],
    }

    cuda = report["cuda"]
    if not cuda["torch_cuda_available"]:
        report["blocking_errors"].append("CUDA is not available to PyTorch.")
    if require_two_gpus and cuda["torch_cuda_device_count"] < 2:
        report["blocking_errors"].append(f"Expected at least 2 visible GPUs; PyTorch sees {cuda['torch_cuda_device_count']}.")

    report["components"]["bert_baseline"] = _component(
        gpu_required=False,
        gpu_capable=_has("torch") and _has("transformers") and cuda["torch_cuda_available"],
        training_mode="current implementation is CPU-fast token baseline; GPU BERT Trainer is not the active implementation",
        missing=_missing(["torch", "transformers"]),
    )
    if not report["components"]["bert_baseline"]["gpu_training_implemented"]:
        report["warnings"].append("BERT baseline currently uses the deterministic CPU-fast token classifier, not a GPU Hugging Face Trainer.")

    gliner_missing = _missing(["gliner", "torch"])
    report["components"]["gliner"] = _component(
        gpu_required=False,
        gpu_capable=not gliner_missing and cuda["torch_cuda_available"],
        training_mode="zero-shot/fallback unless GLiNER fine-tuning support is added",
        missing=gliner_missing,
    )
    if gliner_missing:
        report["warnings"].append("GLiNER package is unavailable; GLiNER baseline will use fallback/skip mode.")

    mistral_missing = _missing(["torch", "transformers", "peft", "datasets", "bitsandbytes"])
    model_path = config.get("model", {}).get("model_path", "models/mistral-small-3.1-24b-instruct-2503")
    sft_train = "data/skillllm_v2/mistral_sft/train.jsonl"
    sft_dev = "data/skillllm_v2/mistral_sft/dev.jsonl"
    mistral_path_missing = [path for path in [model_path, sft_train, sft_dev] if not Path(path).exists()]
    has_mistral_loader = _transformers_has_mistral_loader()
    report["components"]["mistral_lora"] = {
        "gpu_required": True,
        "gpu_capable": not mistral_missing and not mistral_path_missing and cuda["torch_cuda_device_count"] >= 2 and has_mistral_loader,
        "gpu_training_implemented": True,
        "training_mode": "2-GPU torchrun QLoRA/LoRA",
        "missing_dependencies": mistral_missing,
        "missing_paths": mistral_path_missing,
        "has_mistral3_loader_class": has_mistral_loader,
    }
    if mistral_missing:
        report["blocking_errors"].append(f"Mistral LoRA training dependencies missing: {mistral_missing}")
    if mistral_path_missing:
        report["blocking_errors"].append(f"Mistral LoRA required paths missing: {mistral_path_missing}")
    if not has_mistral_loader:
        report["blocking_errors"].append("Installed transformers lacks a loader class suitable for Mistral3Config.")

    report["components"]["span_verifier"] = _component(
        gpu_required=False,
        gpu_capable=False,
        training_mode="CPU scikit-learn TF-IDF/logistic verifier by design",
        missing=_missing(["sklearn"]),
    )
    report["components"]["taxonomy_reranker"] = _component(
        gpu_required=False,
        gpu_capable=_has("sentence_transformers") and cuda["torch_cuda_available"],
        training_mode="retrieval-only unless supervised labels and embedding/reranker deps are provided",
        missing=[] if _has("sentence_transformers") else ["sentence_transformers"],
    )
    report["components"]["graph_match_course"] = _component(
        gpu_required=False,
        gpu_capable=False,
        training_mode="CPU NetworkX/scoring logic by design",
        missing=_missing(["networkx"]),
    )

    write_json("reports/skillllm_v2/gpu_component_availability.json", report)
    write_text("reports/skillllm_v2/gpu_component_availability.md", _markdown(report))
    if report["blocking_errors"]:
        raise SystemExit("\n".join(report["blocking_errors"]))
    return report


def _cuda_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_importable": False, "torch_cuda_available": False, "torch_cuda_device_count": 0, "devices": []}
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            devices.append({"index": index, "name": torch.cuda.get_device_name(index)})
    return {
        "torch_importable": True,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": devices,
    }


def _component(gpu_required: bool, gpu_capable: bool, training_mode: str, missing: list[str]) -> dict[str, Any]:
    return {
        "gpu_required": gpu_required,
        "gpu_capable": gpu_capable,
        "gpu_training_implemented": False,
        "training_mode": training_mode,
        "missing_dependencies": missing,
    }


def _has(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _missing(module_names: list[str]) -> list[str]:
    return [name for name in module_names if not _has(name)]


def _transformers_has_mistral_loader() -> bool:
    try:
        import transformers
    except ImportError:
        return False
    return any(
        getattr(transformers, class_name, None) is not None
        for class_name in ["AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"]
    )


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GPU Component Availability",
        "",
        f"- Config: `{report['config']}`",
        f"- CUDA available: `{report['cuda']['torch_cuda_available']}`",
        f"- Visible GPUs: `{report['cuda']['torch_cuda_device_count']}`",
        "",
        "| Component | GPU Required | GPU Capable | Training Mode | Missing |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, component in report["components"].items():
        missing = ", ".join(component.get("missing_dependencies", []) + component.get("missing_paths", [])) or "none"
        lines.append(
            f"| {name} | {component.get('gpu_required')} | {component.get('gpu_capable')} | "
            f"{component.get('training_mode')} | {missing} |"
        )
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    if report["blocking_errors"]:
        lines.extend(["", "## Blocking Errors", ""])
        lines.extend(f"- {error}" for error in report["blocking_errors"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-two-gpus", action="store_true")
    args = parser.parse_args()
    run_gpu_preflight(args.config, args.require_two_gpus)


if __name__ == "__main__":
    main()
