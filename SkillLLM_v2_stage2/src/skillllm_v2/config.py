from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read SkillLLM_v2 YAML configs.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return data


def repo_root_from(file_path: str | Path) -> Path:
    path = Path(file_path).resolve()
    for parent in [path, *path.parents]:
        if (parent / "src" / "skillllm_v2").exists():
            return parent
    return Path.cwd()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` (dicts merge, everything else is replaced)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``extends: <path>`` (or list of paths) parent chain.

    Parent paths are resolved relative to the repository root first, then
    relative to the child file. The child's keys win over the parent's.
    ``extends`` is removed from the returned mapping. ``load_yaml`` is unchanged.
    """
    file_path = Path(path).resolve()
    if file_path in _seen:
        raise ValueError(f"Circular 'extends' chain at {file_path}")
    data = load_yaml(file_path)
    parents = data.pop("extends", None)
    if not parents:
        return data
    if isinstance(parents, (str, Path)):
        parents = [parents]
    root = repo_root_from(file_path)
    merged: dict[str, Any] = {}
    for parent in parents:
        candidates = [root / parent, file_path.parent / parent]
        parent_path = next((c for c in candidates if c.exists()), None)
        if parent_path is None:
            raise FileNotFoundError(f"extends target not found for {file_path}: {parent}")
        merged = deep_merge(merged, load_config(parent_path, _seen + (file_path,)))
    return deep_merge(merged, data)
