from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import time
from typing import Any

from skillllm_v2.io_utils import write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint_artifact(path: str | Path) -> dict[str, Any]:
    """Return mtime + size + optional content hash for a single artifact path."""
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    stat = p.stat()
    result: dict[str, Any] = {
        "exists": True,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
    }
    # Content-hash small files (< 10 MB) to detect silent in-place rewrites.
    if stat.st_size < 10 * 1024 * 1024:
        try:
            result["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            pass
    return result


class ArtifactTracker:
    def __init__(self, artifacts_dir: str | Path = "artifacts/skillllm_v2") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.status_dir = self.artifacts_dir / "status"
        self.logs_dir = self.artifacts_dir / "logs"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def success_path(self, stage_name: str) -> Path:
        return self.status_dir / f"{stage_name}.success.json"

    def failure_path(self, stage_name: str) -> Path:
        return self.status_dir / f"{stage_name}.failed.json"

    def has_success(self, stage_name: str) -> bool:
        return self.success_path(stage_name).exists()

    def invalidate_downstream(self, start_stage: str) -> list[str]:
        """Delete success markers for start_stage and all stages that follow it.

        Returns the list of stages whose markers were removed.
        """
        from skillllm_v2.pipeline.stage_registry import STAGE_ORDER

        try:
            idx = STAGE_ORDER.index(start_stage)
        except ValueError:
            raise KeyError(f"Unknown stage: {start_stage}")

        removed: list[str] = []
        for stage in STAGE_ORDER[idx:]:
            p = self.success_path(stage)
            if p.exists():
                p.unlink()
                removed.append(stage)
        return removed

    def write_status(
        self,
        stage_name: str,
        system_name: str,
        status: str,
        started_at: str,
        started_time: float,
        config: str,
        inputs: list[str],
        outputs: list[str],
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        finished_at = utc_now()
        output_fingerprints = {p: fingerprint_artifact(p) for p in outputs}
        payload = {
            "stage": stage_name,
            "system_name": system_name,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": time.perf_counter() - started_time,
            "config": config,
            "inputs": inputs,
            "outputs": outputs,
            "output_fingerprints": output_fingerprints,
            "metrics": metrics or {},
            "error": error,
        }
        path = self.success_path(stage_name) if status == "success" else self.failure_path(stage_name)
        stale = self.failure_path(stage_name) if status == "success" else self.success_path(stage_name)
        if stale.exists():
            stale.unlink()
        write_json(path, payload)
