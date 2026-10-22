"""Shared contract for the evaluation suite: result type, context, and seeded bootstrap helpers (stdlib + random only)."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class EvalContext:
    """What an eval may rely on. Evals must work offline on CPU and never need a GPU or model weights."""

    root: Path = Path(".")
    seed: int = 42
    n_boot: int = 2000
    smoke: bool = False  # tiny subsets, for tests
    extra: Dict[str, Any] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)


@dataclass
class EvalResult:
    """One measured quantity. `ci` is a 95% interval when the metric is a mean over independent units."""

    eval_name: str  # e.g. "linking.recall_at_5"
    module: str  # extract | verify | calibrate | link | context | graph | explain | match | gap | e2e
    metric: str
    value: Optional[float]
    n: int  # number of independent evaluation units
    dataset: str  # name + split; say "synthetic" when it is
    ci: Optional[Tuple[float, float]] = None
    details: Dict[str, Any] = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)
    status: str = "ok"  # ok | skipped | failed
    reason: str = ""  # why skipped/failed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ci"] = list(self.ci) if self.ci else None
        return d


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05,
                 stat: Callable[[Sequence[float]], float] = lambda v: sum(v) / len(v)) -> Tuple[float, float]:
    """Percentile bootstrap CI of `stat` over per-unit values."""
    vals = list(values)
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(stat([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return (stats[int(alpha / 2 * n_boot)], stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Paired bootstrap of mean(a - b): delta, 95% CI and two-sided p (share of resamples crossing zero, doubled)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be equal-length and non-empty")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    delta = sum(diffs) / n
    lo, hi = means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]
    below = sum(1 for m in means if m <= 0) / n_boot
    p = min(1.0, 2 * min(below, 1 - below))
    return {"delta": delta, "ci_low": lo, "ci_high": hi, "p": p}
