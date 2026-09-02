from __future__ import annotations

from pathlib import Path


class GateError(RuntimeError):
    pass


def require_paths(paths: list[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise GateError(f"Missing required paths: {missing}")


def require_no_legacy_active_paths() -> None:
    banned_roots = [Path("configs"), Path("scripts"), Path("src/skillllm_v2")]
    banned_terms = ["qwen", "gemma", "llama", "retry", "skillspan_benchmark"]
    offenders: list[str] = []
    for root in banned_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and any(term in path.name.lower() for term in banned_terms):
                offenders.append(str(path))
    if offenders:
        raise GateError(f"Active legacy model experiment files remain: {offenders}")


def require_success(tracker: object, stages: list[str]) -> None:
    missing = [stage for stage in stages if not tracker.has_success(stage)]
    if missing:
        raise GateError(f"Required stage success markers are missing: {missing}")
