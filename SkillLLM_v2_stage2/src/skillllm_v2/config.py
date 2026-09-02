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
