"""GPU memory tracking utilities."""

from __future__ import annotations


def get_peak_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated in MB. Returns 0.0 if no GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:
        pass
    return 0.0


def reset_gpu_memory_stats() -> None:
    """Reset GPU memory tracking stats."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
