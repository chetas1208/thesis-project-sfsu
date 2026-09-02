import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from common import ensure_benchmark_env, ensure_dir, load_config, load_env_file, resolve_path


DEFAULT_DATASETS = [
    "rayyankauchali0/resume-dataset",
    "snehaanbhawal/resume-dataset",
]


def slug_to_dirname(slug: str) -> str:
    owner, dataset = slug.split("/", 1)
    return f"{owner}_{dataset}".replace("-", "_")


def configure_kaggle_credentials() -> bool:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        return False

    kaggle_dir = ensure_dir(Path.home() / ".kaggle")
    kaggle_json_path = kaggle_dir / "kaggle.json"
    payload = {"username": username, "key": key}
    kaggle_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(kaggle_json_path, 0o600)
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(kaggle_dir))
    return True


def copy_tree_contents(source_dir: Path, destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for path in sorted(source_dir.rglob("*")):
        relative = path.relative_to(source_dir)
        target = destination_dir / relative
        if path.is_dir():
            ensure_dir(target)
            continue
        ensure_dir(target.parent)
        shutil.copy2(path, target)


def download_with_kaggle_api(slug: str, destination_dir: Path, force: bool) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    ensure_dir(destination_dir)
    api.dataset_download_files(slug, path=str(destination_dir), unzip=True, quiet=False, force=force)


def download_with_kagglehub(slug: str, destination_dir: Path, force: bool) -> None:
    import inspect

    import kagglehub

    download_kwargs = {}
    signature = inspect.signature(kagglehub.dataset_download)
    if "force_download" in signature.parameters:
        download_kwargs["force_download"] = force
    downloaded_path = Path(kagglehub.dataset_download(slug, **download_kwargs))
    copy_tree_contents(downloaded_path, destination_dir)


def available_methods(requested_method: str, credentials_available: bool) -> Iterable[str]:
    if requested_method == "auto":
        if credentials_available:
            return ("kaggle_api", "kagglehub")
        return ("kagglehub",)
    return (requested_method,)


def download_dataset(
    slug: str,
    output_root: Path,
    requested_method: str,
    force: bool,
    dry_run: bool,
    credentials_available: bool,
) -> dict:
    destination_dir = output_root / slug_to_dirname(slug)
    summary = {
        "dataset": slug,
        "destination": str(destination_dir),
        "method": None,
        "status": "pending",
        "error": None,
    }

    if dry_run:
        summary["status"] = "dry_run"
        return summary

    errors: List[str] = []
    for method in available_methods(requested_method, credentials_available):
        try:
            if method == "kaggle_api":
                download_with_kaggle_api(slug, destination_dir, force)
            elif method == "kagglehub":
                download_with_kagglehub(slug, destination_dir, force)
            else:
                raise ValueError(f"Unsupported download method: {method}")
            summary["method"] = method
            summary["status"] = "downloaded"
            return summary
        except BaseException as exc:  # pragma: no cover - network/auth dependent
            if isinstance(exc, KeyboardInterrupt):
                raise
            errors.append(f"{method}: {exc}")

    summary["status"] = "failed"
    summary["error"] = " | ".join(errors) if errors else "unknown error"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle resume datasets into data/raw/source.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--output-dir", default="data/raw/source")
    parser.add_argument("--method", choices=["auto", "kaggle_api", "kagglehub"], default="auto")
    parser.add_argument("--force", action="store_true", help="Force a re-download when supported.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)
    credentials_available = configure_kaggle_credentials()

    output_root = ensure_dir(resolve_path(args.output_dir))
    results = [
        download_dataset(
            slug,
            output_root,
            args.method,
            args.force,
            args.dry_run,
            credentials_available=credentials_available,
        )
        for slug in args.datasets
    ]

    for result in results:
        status = result["status"]
        line = f"{result['dataset']}: {status} -> {result['destination']}"
        if result["method"]:
            line += f" via {result['method']}"
        if result["error"]:
            line += f" ({result['error']})"
        print(line)

    failures = [result for result in results if result["status"] == "failed"]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
