"""Shared plumbing for the layered research evaluation: result records, provenance hashes and the layer writer.

Every layer writes ``reports/skillllm_v2/evals/<layer>/results.json`` with the schema below so the leaderboard, the
significance layer and the evaluation card can consume any layer without knowing its internals.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
GIT_ROOT = REPO_ROOT.parent  # SkillLLM_v2_stage2 is tracked by the outer repository; its nested .git is empty
EVALS_ROOT = REPO_ROOT / "reports" / "skillllm_v2" / "evals"
SCHEMA_VERSION = "1.0"

STATUSES = ("measured", "not_available", "pending_human", "pending_gpu", "failed")
DIRECTIONS = ("higher", "lower", "none")
REQUIRED = ("layer", "system", "metric", "value", "n_units", "unit", "split", "dataset", "direction", "status")


def record(layer: str, system: str, metric: str, value: Optional[float], *, n_units: int, unit: str, split: str,
           dataset: str, direction: str, status: str = "measured", ci: Optional[Sequence[float]] = None,
           ci_method: str = "", provenance: Sequence[str | Path] = (), caveats: Sequence[str] = (),
           reason: str = "", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One measured (or explicitly unavailable) quantity. ``value`` must be None unless status == 'measured'."""
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r} not in {DIRECTIONS}")
    if status != "measured" and value is not None:
        raise ValueError("only measured records may carry a value")
    if status != "measured" and not reason:
        raise ValueError("unavailable records need a reason")
    if status == "measured" and value is None:
        raise ValueError("measured records need a value")
    return {
        "layer": layer, "system": system, "metric": metric,
        "value": None if value is None else float(value),
        "ci_low": None if not ci else float(ci[0]), "ci_high": None if not ci else float(ci[1]), "ci_method": ci_method,
        "n_units": int(n_units), "unit": unit, "split": split, "dataset": dataset, "direction": direction,
        "status": status, "reason": reason, "provenance": [provenance_entry(p) for p in provenance],
        "caveats": list(caveats), "details": details or {},
    }


@lru_cache(maxsize=None)
def _sha256(path: str, size: int, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    return _sha256(str(p.resolve()), st.st_size, st.st_mtime)


def sha256_tree(path: str | Path, pattern: str = "*") -> str:
    """Hash of a directory: sorted (relative path, file hash) pairs. Large weight dirs are hashed fully, so call once."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(q for q in root.rglob(pattern) if q.is_file()):
        h.update(f"{f.relative_to(root)}\0{sha256_file(f)}\n".encode())
    return h.hexdigest()


def rel(path: str | Path) -> str:
    p = Path(os.path.abspath(path))  # abspath, not resolve: artifacts/ is a symlink and paths must stay repo-relative
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def provenance_entry(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return {"path": rel(p), "sha256": None, "bytes": None, "missing": True}
    if p.is_dir():
        return {"path": rel(p), "sha256": sha256_tree(p), "bytes": None, "dir": True}
    return {"path": rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}


def git_info() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(GIT_ROOT), *args], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001 - git missing is recorded, not fatal
            return ""
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", str(REPO_ROOT)))}


def run_metadata(seed: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(), "seed": seed, "argv": sys.argv, "python": sys.version.split()[0],
        "platform": platform.platform(), "host": platform.node(), "pid": os.getpid(), **(extra or {}),
    }


def validate(records: Iterable[Dict[str, Any]]) -> List[str]:
    problems = []
    for i, r in enumerate(records):
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            problems.append(f"record {i}: missing {missing}")
        elif r["status"] == "measured" and r["value"] is None:
            problems.append(f"record {i} ({r['system']}/{r['metric']}): measured without value")
    return problems


def write_layer(layer: str, records: List[Dict[str, Any]], report_md: str, *, seed: int, smoke: bool = False,
                out_root: Path = EVALS_ROOT, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    """Write <out_root>/<layer>/{results.json, REPORT.md}; smoke runs go to <layer>/smoke/ and are never headline numbers."""
    problems = validate(records)
    if problems:
        raise ValueError("invalid records:\n" + "\n".join(problems[:20]))
    out = out_root / layer / ("smoke" if smoke else "")
    out.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(seed, {"layer": layer, "smoke": smoke, **(extra_meta or {})})
    (out / "results.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2, default=str) + "\n")
    banner = "> SMOKE RUN - tiny subset, not a result.\n\n" if smoke else ""
    (out / "REPORT.md").write_text(banner + report_md.rstrip() + "\n")
    return out


def load_layer(layer: str, out_root: Path = EVALS_ROOT) -> Optional[Dict[str, Any]]:
    p = out_root / layer / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def md_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str], fmt: str = "{:.4f}") -> str:
    def cell(v: Any) -> str:
        if v is None:
            return "-"
        return fmt.format(v) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
