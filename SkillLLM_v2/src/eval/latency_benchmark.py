from __future__ import annotations

import argparse
import statistics

from src.runtime_utils import maybe_get_max_memory_mb, read_jsonl, write_json


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize latency and throughput metrics.")
    parser.add_argument("--raw-file", required=True, help="Raw generation jsonl file.")
    parser.add_argument("--output-file", required=True, help="Metric output json file.")
    parser.add_argument("--experiment-name", required=True, help="Experiment name.")
    args = parser.parse_args()

    raw_records = read_jsonl(args.raw_file)
    timings = [float(record["generation_seconds"]) for record in raw_records if record.get("generation_seconds") is not None]
    total_seconds = sum(timings)
    throughput = len(timings) / total_seconds if total_seconds else 0.0
    gpu_memory_values = [record.get("gpu_memory_mb") for record in raw_records if record.get("gpu_memory_mb") is not None]

    metrics = {
        "experiment_name": args.experiment_name,
        "metric_type": "latency",
        "samples": len(timings),
        "average_latency_seconds": statistics.mean(timings) if timings else 0.0,
        "median_latency_seconds": statistics.median(timings) if timings else 0.0,
        "p95_latency_seconds": percentile(timings, 0.95),
        "throughput_samples_per_second": throughput,
        "gpu_memory_mb": max(gpu_memory_values) if gpu_memory_values else maybe_get_max_memory_mb(),
    }
    write_json(args.output_file, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
