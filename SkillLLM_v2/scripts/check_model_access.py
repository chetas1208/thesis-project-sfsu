from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    local_dir: str


MODEL_SPECS = (
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct"),
    ModelSpec("Qwen/Qwen3-14B", "qwen3-14b"),
    ModelSpec("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "mistral-small-3.1-24b-instruct-2503"),
    ModelSpec("google/gemma-4-E4B-it", "gemma-4-e4b-it"),
    ModelSpec("google/gemma-4-26B-A4B-it", "gemma-4-26b-a4b-it"),
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)
CONFIG_FILES = ("config.json", "generation_config.json")
MODEL_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "model.safetensors.index.json", "pytorch_model.bin.index.json")


def has_any_file(directory: Path, names: Iterable[str]) -> bool:
    return any((directory / name).exists() for name in names)


def has_any_pattern(directory: Path, patterns: Iterable[str]) -> bool:
    return any(directory.glob(pattern) for pattern in patterns)


def describe_remote_access(repo_id: str) -> str:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
    except ImportError:
        return "Install huggingface_hub to diagnose remote access."

    api = HfApi()
    try:
        api.model_info(repo_id)
    except GatedRepoError:
        return "Remote access is gated. Accept the license and run huggingface-cli login."
    except RepositoryNotFoundError:
        return "Repository was not found with the current token."
    except HfHubHTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code == 401:
            return "Remote access is unauthorized. Refresh your Hugging Face token."
        return f"Remote access failed with HTTP {status_code}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return f"Remote access check failed: {exc}"
    return "Remote repository is visible. The local download is still incomplete."


def validate_model(root: Path, spec: ModelSpec) -> list[str]:
    local_dir = root / spec.local_dir
    errors: list[str] = []

    if not local_dir.exists():
        errors.append(f"Missing local directory: {local_dir}")
        errors.append(describe_remote_access(spec.repo_id))
        return errors

    if not (local_dir / "tokenizer_config.json").exists() and not has_any_file(local_dir, TOKENIZER_FILES):
        errors.append(f"Tokenizer files missing in {local_dir}")

    if not has_any_file(local_dir, CONFIG_FILES):
        errors.append(f"Model config missing in {local_dir}")

    if not has_any_pattern(local_dir, MODEL_PATTERNS):
        errors.append(f"Model weights missing in {local_dir}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local model downloads and access.")
    parser.add_argument(
        "--models-root",
        default="models",
        help="Directory containing downloaded Hugging Face model folders.",
    )
    args = parser.parse_args()

    models_root = Path(args.models_root).resolve()
    failures = {}

    for spec in MODEL_SPECS:
        errors = validate_model(models_root, spec)
        if errors:
            failures[spec.repo_id] = errors
        else:
            print(f"[OK] {spec.repo_id} -> {models_root / spec.local_dir}")

    if failures:
        print("\nModel access check failed.\n", file=sys.stderr)
        for repo_id, errors in failures.items():
            print(repo_id, file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        return 1

    print("\nAll requested models are present and appear complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
