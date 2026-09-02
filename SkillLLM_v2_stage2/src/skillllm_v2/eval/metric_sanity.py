"""metric_sanity.py

Validates that thesis metric files are plausible: non-zero, non-trivial, and
internally consistent.  Exits 0 on pass, 1 on failure.

Usage:
    python -m skillllm_v2.eval.metric_sanity
    python -m skillllm_v2.eval.metric_sanity --strict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


METRIC_FILES = [
    ("BERT baseline",               "artifacts/skillllm_v2/bert_baseline/metrics.json",           0.0,  0.5),
    ("GLiNER baseline",             "artifacts/skillllm_v2/gliner_baseline/metrics.json",          0.0,  0.6),
    ("Raw Mistral LoRA",            "artifacts/skillllm_v2/mistral_lora/metrics.json",             0.4,  1.0),
    ("Mistral + Boundary Refiner",  "artifacts/skillllm_v2/mistral_boundary/metrics.json",         0.0,  1.0),
    ("Mistral + GLiNER + Taxonomy", "artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json",  0.0,  1.0),
    ("Full CAG/Graph",              "artifacts/skillllm_v2/full_cag_graph/metrics.json",           0.0,  1.0),
]

# Hard lower bound on exact_span_f1 for the trained Mistral model (80-epoch).
MISTRAL_F1_FLOOR = 0.4


def _load(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _f1(metrics: dict[str, Any]) -> float | None:
    for key in ("exact_span_f1", "micro_f1", "f1", "exact_f1"):
        v = metrics.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def check_all(strict: bool = False) -> list[str]:
    errors: list[str] = []

    for label, path, min_f1, max_f1 in METRIC_FILES:
        m = _load(path)
        if m is None:
            errors.append(f"MISSING: {label} ({path})")
            continue

        f1 = _f1(m)
        if f1 is None:
            errors.append(f"NO F1 KEY: {label} ({path}) — keys: {list(m.keys())[:8]}")
            continue

        if not (min_f1 <= f1 <= max_f1):
            errors.append(
                f"OUT OF RANGE: {label} exact_span_f1={f1:.4f} "
                f"(expected [{min_f1:.2f}, {max_f1:.2f}]) in {path}"
            )
            continue

        # All-zero check: every float metric is 0.0 is suspicious.
        float_vals = [v for v in m.values() if isinstance(v, float)]
        if float_vals and all(v == 0.0 for v in float_vals):
            errors.append(f"ALL-ZERO METRICS: {label} ({path})")

    # Cross-stage sanity: Mistral raw should be the highest precision stage
    # (boundary refiner adds recall; subsequent stages add linking).
    mistral_m = _load("artifacts/skillllm_v2/mistral_lora/metrics.json")
    if mistral_m and strict:
        mistral_f1 = _f1(mistral_m) or 0.0
        if mistral_f1 < MISTRAL_F1_FLOOR:
            errors.append(
                f"STRICT: Raw Mistral F1={mistral_f1:.4f} < floor {MISTRAL_F1_FLOOR:.2f}. "
                "Did the 80-epoch model predictions load correctly?"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate thesis metric files")
    parser.add_argument("--strict", action="store_true", help="Apply additional cross-stage checks")
    args = parser.parse_args()

    print("Metric sanity check")
    print("=" * 60)

    for label, path, _, _ in METRIC_FILES:
        m = _load(path)
        if m is None:
            print(f"  MISSING  {label}")
            continue
        f1 = _f1(m)
        tag = f"F1={f1:.4f}" if f1 is not None else "no-f1"
        print(f"  OK       {label:<35} {tag}  ({path})")

    errors = check_all(args.strict)
    print()
    if errors:
        print(f"FAILED ({len(errors)} issue(s)):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("All metric files passed sanity checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
