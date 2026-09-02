"""prediction_provenance_check.py

Verifies that downstream prediction files are newer than the upstream
Mistral predictions they depend on, and that the Mistral predictions
were produced by the 80-epoch trained adapter (not the smoke fallback).

Exits 0 when provenance is clean, 1 when stale or suspect.

Usage:
    python -m skillllm_v2.eval.prediction_provenance_check
    python -m skillllm_v2.eval.prediction_provenance_check --fix-hint
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


MISTRAL_PREDICTIONS = "artifacts/skillllm_v2/mistral_lora/predictions.jsonl"
MISTRAL_METRICS     = "artifacts/skillllm_v2/mistral_lora/metrics.json"
MISTRAL_MODEL_DIR   = "artifacts/skillllm_v2/mistral_lora/model"

DOWNSTREAM_FILES = [
    "artifacts/skillllm_v2/mistral_boundary/predictions.jsonl",
    "artifacts/skillllm_v2/mistral_boundary/metrics.json",
    "artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl",
    "artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json",
    "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl",
    "artifacts/skillllm_v2/full_cag_graph/metrics.json",
    "reports/skillllm_v2/money_table.csv",
    "reports/skillllm_v2/money_table.md",
]


def _mtime(path: str) -> datetime | None:
    p = Path(path)
    if not p.exists():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)


def _load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def check_provenance() -> list[str]:
    issues: list[str] = []

    upstream_mtime = _mtime(MISTRAL_PREDICTIONS)
    if upstream_mtime is None:
        issues.append(f"MISSING upstream: {MISTRAL_PREDICTIONS}")
        return issues

    print(f"Upstream Mistral predictions mtime: {upstream_mtime.isoformat()}")

    # Verify upstream is the 80-epoch trained model (not fallback).
    metrics = _load_json(MISTRAL_METRICS)
    if metrics:
        notes = str(metrics.get("notes", ""))
        if "fallback used: True" in notes or "fallback_used: True" in notes:
            issues.append(
                "Mistral metrics note fallback was used — "
                "predictions may be from a smoke/random baseline, not the 80-epoch model."
            )
        f1_val = metrics.get("exact_span_f1") or metrics.get("micro_f1") or 0.0
        if isinstance(f1_val, float) and f1_val < 0.40:
            issues.append(
                f"Mistral F1={f1_val:.4f} is suspiciously low for a trained model. "
                "Verify adapter loaded correctly."
            )
    else:
        issues.append(f"MISSING Mistral metrics: {MISTRAL_METRICS}")

    # Check model adapter exists.
    model_dir = Path(MISTRAL_MODEL_DIR)
    adapter_files = list(model_dir.glob("adapter_model*")) + list(model_dir.glob("adapter_config*"))
    if not adapter_files:
        issues.append(
            f"No adapter_model/adapter_config files in {MISTRAL_MODEL_DIR}. "
            "The 80-epoch LoRA adapter may be missing."
        )

    # Check downstream files are NEWER than upstream Mistral predictions.
    for ds_path in DOWNSTREAM_FILES:
        ds_mtime = _mtime(ds_path)
        if ds_mtime is None:
            issues.append(f"MISSING downstream: {ds_path}")
            continue
        if ds_mtime < upstream_mtime:
            delta = (upstream_mtime - ds_mtime).total_seconds()
            issues.append(
                f"STALE: {ds_path}\n"
                f"  downstream mtime : {ds_mtime.isoformat()}\n"
                f"  upstream  mtime  : {upstream_mtime.isoformat()}\n"
                f"  upstream is {delta:.0f}s NEWER — downstream was built before Mistral re-eval"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check prediction provenance against 80-epoch Mistral adapter")
    parser.add_argument("--fix-hint", action="store_true", help="Print remediation commands when issues are found")
    args = parser.parse_args()

    print("Prediction provenance check")
    print("=" * 60)

    issues = check_provenance()
    print()

    if issues:
        print(f"FAILED ({len(issues)} issue(s)):")
        for issue in issues:
            print(f"  - {issue}")
        if args.fix_hint:
            print()
            print("Fix:")
            print("  bash scripts/invalidate_stale_downstream_from_mistral.sh")
            print("  nohup bash scripts/nohup_resume_downstream_from_80epoch_mistral.sh \\")
            print("    > logs/resume_downstream_bootstrap.log 2>&1 &")
        return 1

    print("Provenance clean — all downstream files are newer than 80-epoch Mistral predictions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
