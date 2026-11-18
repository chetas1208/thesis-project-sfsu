"""Tests for L12 systems helpers (latency block, Pareto plot, corner cases) and L03 report plumbing; synthetic fixtures."""
from __future__ import annotations

import numpy as np

from skillllm_v2.evals.research import systems as S


def test_latency_block_percentiles_and_throughput():
    rows = [{"latency_seconds": float(i), "predictions": [{}] * (i % 3), "n_gen_tokens": 10 * i, "n_prompt_tokens": 50}
            for i in range(1, 101)]
    b = S.latency_block(rows)
    assert b["per_doc"]["n"] == 100
    assert abs(b["per_doc"]["p50"] - 50.5) < 1e-9
    assert b["per_doc"]["p95"] > b["per_doc"]["p50"]
    assert abs(b["throughput_docs_per_hour"] - 3600 / 50.5) < 1e-6
    assert b["gen_tokens_per_second"]["p50"] == 10.0
    total_spans = sum(len(r["predictions"]) for r in rows)
    assert abs(b["seconds_per_predicted_span_aggregate"] - sum(range(1, 101)) / total_spans) < 1e-9


def test_latency_block_empty():
    b = S.latency_block([])
    assert b["per_doc"]["n"] == 0 and b["throughput_docs_per_hour"] is None


def test_plot_pareto_writes_png_and_skips_missing(tmp_path):
    pts = [{"system": "a", "exact_f1": 0.6, "exact_f1_ci": (0.55, 0.65), "latency_p95_s": 10.0, "memory_gb": 20.0,
            "memory_kind": "GPU VRAM (reserved)"},
           {"system": "b", "exact_f1": 0.3, "exact_f1_ci": (0.25, 0.35), "latency_p95_s": 0.1, "memory_gb": 1.0,
            "memory_kind": "CPU RSS (process)"},
           {"system": "c", "exact_f1": 0.1, "exact_f1_ci": (0.05, 0.15), "latency_p95_s": None, "memory_gb": None,
            "memory_kind": None}]
    paths = S.plot_pareto(pts, tmp_path)
    assert {p.name for p in paths} == {"pareto_f1_vs_latency.png", "pareto_f1_vs_memory.png"}
    assert all(p.stat().st_size > 1000 for p in paths)
    assert S.plot_pareto([], tmp_path) == []


def test_boot_pct_ci_brackets_percentile():
    a = np.random.default_rng(0).exponential(2.0, 400)
    lo, hi = S.boot_pct_ci(a, 95, 300, 42)
    assert lo < np.percentile(a, 95) < hi
    assert S.boot_pct_ci([1.0], 95, 10, 1) == (None, None)
