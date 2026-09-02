"""diagnose_stale_artifacts.py

Reads every stage success marker, compares timestamps to detect downstream
artifacts that pre-date their upstream dependencies, and prints a stale-pipeline
diagnosis.  Exits 0 when nothing is stale, 1 when stale stages are found.

Usage:
    python -m skillllm_v2.pipeline.diagnose_stale_artifacts
    python -m skillllm_v2.pipeline.diagnose_stale_artifacts --artifacts-dir artifacts/skillllm_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from skillllm_v2.pipeline.stage_registry import STAGE_ORDER


def _load_success(artifacts_dir: Path, stage: str) -> dict[str, Any] | None:
    p = artifacts_dir / "status" / f"{stage}.success.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ts(record: dict[str, Any] | None) -> str | None:
    if record is None:
        return None
    return record.get("finished_at") or record.get("started_at")


def diagnose(artifacts_dir: str | Path = "artifacts/skillllm_v2") -> list[dict[str, Any]]:
    """Return list of stale stage dicts. Empty list = pipeline is clean."""
    adir = Path(artifacts_dir)
    records = {s: _load_success(adir, s) for s in STAGE_ORDER}

    stale: list[dict[str, Any]] = []
    # Build ordered list; a stage is stale if it ran BEFORE any of its upstream
    # stages that also have a success marker.
    for idx, stage in enumerate(STAGE_ORDER):
        rec = records[stage]
        if rec is None:
            continue
        stage_ts = _ts(rec)
        if stage_ts is None:
            continue
        for upstream in STAGE_ORDER[:idx]:
            up_rec = records[upstream]
            if up_rec is None:
                continue
            up_ts = _ts(up_rec)
            if up_ts and up_ts > stage_ts:
                stale.append({
                    "stage": stage,
                    "stage_finished_at": stage_ts,
                    "upstream": upstream,
                    "upstream_finished_at": up_ts,
                    "delta_seconds": _delta(up_ts, stage_ts),
                })
                break  # one upstream is enough to flag this stage

    return stale


def _delta(newer: str, older: str) -> float | None:
    try:
        from datetime import datetime, timezone
        def _parse(s: str) -> datetime:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (_parse(newer) - _parse(older)).total_seconds()
    except Exception:
        return None


def artifact_summary(artifacts_dir: str | Path = "artifacts/skillllm_v2") -> None:
    """Print a table of every stage: timestamp, stale flag, key metric."""
    adir = Path(artifacts_dir)
    print(f"\n{'Stage':<42} {'Finished':>24}  {'Stale':>5}  Key metric")
    print("-" * 95)
    stale_set = {s["stage"] for s in diagnose(adir)}
    for stage in STAGE_ORDER:
        rec = _load_success(adir, stage)
        if rec is None:
            print(f"  {stage:<40} {'MISSING':>24}  {'?':>5}")
            continue
        ts = _ts(rec) or "?"
        flag = "YES" if stage in stale_set else "ok"
        metrics = rec.get("metrics") or {}
        kv = next(
            (f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)),
            next((f"{k}={v}" for k, v in metrics.items() if v is not None), ""),
        )
        print(f"  {stage:<40} {ts:>24}  {flag:>5}  {kv}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose stale artifacts in the SkillLLM_v2 pipeline")
    parser.add_argument("--artifacts-dir", default="artifacts/skillllm_v2")
    parser.add_argument("--summary", action="store_true", default=True)
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args()

    if args.summary and not args.as_json:
        artifact_summary(args.artifacts_dir)

    stale = diagnose(args.artifacts_dir)

    if args.as_json:
        print(json.dumps(stale, indent=2))
        return 1 if stale else 0

    if stale:
        print(f"STALE STAGES ({len(stale)}):")
        for item in stale:
            print(
                f"  {item['stage']} (finished {item['stage_finished_at']}) is stale — "
                f"upstream {item['upstream']} re-ran at {item['upstream_finished_at']} "
                f"(+{item['delta_seconds']:.0f}s after)"
            )
        print()
        print("To fix: run scripts/invalidate_stale_downstream_from_mistral.sh")
        print("Then:   run scripts/resume_downstream_from_80epoch_mistral.sh")
        return 1

    print("Pipeline is clean — no stale artifacts found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
