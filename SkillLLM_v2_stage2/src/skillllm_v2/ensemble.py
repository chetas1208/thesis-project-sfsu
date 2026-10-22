"""Span-voting ensemble layer (pure Python, no heavy dependencies).

A span is a mapping with at least ``start``, ``end`` and (optionally) ``label``.
Given the predictions of several systems on the same text, a span
``(start, end, label)`` is kept when at least ``min_votes`` distinct systems
predict it exactly. Each system gets at most one vote per span.

This is a faithful port of ``vote()`` in ``scripts/ensemble_eval.py`` (the
reference implementation used for the pre-declared held-out evaluation):
:func:`vote_documents` returns identical output, including order, for the same
inputs; ``tests/test_ensemble.py`` proves this on the saved predictions.
The only additions are opt-in: provenance annotation, position ordering and
skipping spans that carry no offsets.

Majority voting over ``n`` systems uses ``min_votes = n // 2 + 1``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

Span = Mapping[str, Any]
Order = Literal["reference", "position"]

__all__ = [
    "majority_threshold",
    "resolve_min_votes",
    "vote_spans",
    "vote_documents",
    "VoteCalibration",
    "default_vote_calibration",
    "fit_vote_calibration",
    "DEFAULT_CALIBRATION_NOTE",
]


def majority_threshold(n_systems: int) -> int:
    """Strict majority: 2 of 3, 2 of 2, 3 of 4, 3 of 5 ..."""
    if n_systems < 1:
        raise ValueError("need at least one system")
    return n_systems // 2 + 1


def resolve_min_votes(vote: str | int | None, n_systems: int) -> int:
    """Translate ``'majority' | 'unanimous' | 'union' | int`` to a vote threshold."""
    if isinstance(vote, bool):
        raise ValueError("vote must be 'majority', 'unanimous', 'union' or an int, not a bool")
    if isinstance(vote, int):
        min_votes = vote
    elif vote == "majority":
        min_votes = majority_threshold(n_systems)
    elif vote == "unanimous":
        min_votes = n_systems
    elif vote == "union":
        min_votes = 1
    else:
        raise ValueError(f"Unknown vote rule {vote!r}; expected 'majority', 'unanimous', 'union' or an int")
    if not 1 <= min_votes <= n_systems:
        raise ValueError(f"min_votes must be between 1 and the number of systems ({n_systems}); got {min_votes}")
    return min_votes


def vote_spans(
    per_system_predictions: Sequence[Sequence[Span]],
    min_votes: int,
    *,
    system_names: Sequence[str] | None = None,
    annotate: bool = False,
    order: Order = "reference",
    skip_unlocated: bool = False,
) -> list[dict[str, Any]]:
    """Vote over ONE document.

    Args:
        per_system_predictions: ``[system][span]``; spans are dicts with ``start``, ``end``, optional ``label``
            and optional ``confidence``.
        min_votes: minimum number of distinct systems that must predict the exact ``(start, end, label)``.
        system_names: names used in the provenance annotation (default ``system0, system1, ...``).
        annotate: add ``votes`` (n_votes, n_systems, voted_by) to each kept span. When False the output is
            byte-identical to the reference implementation.
        order: ``'reference'`` sorts by ``str((start, end, label))`` exactly like the reference script;
            ``'position'`` sorts by ``(start, end, label)``.
        skip_unlocated: ignore spans whose ``start``/``end`` is ``None`` (they cannot match across systems).
            The reference implementation does not skip them.

    Returns:
        Kept spans. Each is a copy of the first voting system's span, with ``confidence`` replaced by the mean
        of the voting systems' confidences (rounded to 4 decimals, only if any system gave one) and
        ``source`` set to ``"ensemble:<votes>of<systems>"``.
    """
    n_systems = len(per_system_predictions)
    if n_systems == 0:
        raise ValueError("per_system_predictions must contain at least one system")
    if not 1 <= min_votes <= n_systems:
        raise ValueError(f"min_votes must be between 1 and {n_systems}; got {min_votes}")
    names = list(system_names) if system_names is not None else [f"system{i}" for i in range(n_systems)]
    if len(names) != n_systems:
        raise ValueError(f"system_names has {len(names)} entries for {n_systems} systems")

    votes: dict[tuple[Any, ...], list[tuple[int, Span]]] = defaultdict(list)
    for sys_index, spans in enumerate(per_system_predictions):
        seen: set[tuple[Any, ...]] = set()
        for sp in spans:
            if skip_unlocated and (sp.get("start") is None or sp.get("end") is None):
                continue
            key = (sp["start"], sp["end"], sp.get("label"))
            if key in seen:  # one vote per system per span
                continue
            seen.add(key)
            votes[key].append((sys_index, sp))

    items = list(votes.items())
    if order == "reference":
        items.sort(key=lambda kv: str(kv[0]))
    elif order == "position":
        items.sort(key=lambda kv: (kv[0][0] if kv[0][0] is not None else -1, kv[0][1] if kv[0][1] is not None else -1, str(kv[0][2])))
    else:
        raise ValueError(f"Unknown order {order!r}")

    kept: list[dict[str, Any]] = []
    for _, entries in items:
        if len(entries) < min_votes:
            continue
        sp = dict(entries[0][1])
        confs = [s.get("confidence") for _, s in entries if s.get("confidence") is not None]
        if confs:
            sp["confidence"] = round(sum(confs) / len(confs), 4)
        sp["source"] = f"ensemble:{len(entries)}of{n_systems}"
        if annotate:
            sp["votes"] = {
                "n_votes": len(entries),
                "n_systems": n_systems,
                "min_votes": min_votes,
                "voted_by": [names[i] for i, _ in entries],
            }
        kept.append(sp)
    return kept


def vote_documents(
    per_system_docs: Sequence[Sequence[Sequence[Span]]],
    min_votes: int,
    **kwargs: Any,
) -> list[list[dict[str, Any]]]:
    """Vote over many documents: ``per_system_docs[system][doc][span]`` -> ``[doc][span]``.

    Equivalent to ``vote()`` in ``scripts/ensemble_eval.py`` (same output for the same input).
    """
    if not per_system_docs:
        raise ValueError("per_system_docs must contain at least one system")
    n_docs = len(per_system_docs[0])
    for i, docs in enumerate(per_system_docs):
        if len(docs) != n_docs:
            raise ValueError(f"system {i} has {len(docs)} documents, expected {n_docs}")
    return [
        vote_spans([sys_docs[d] for sys_docs in per_system_docs], min_votes, **kwargs)
        for d in range(n_docs)
    ]


# --------------------------------------------------------------------------- vote -> precision calibration
DEFAULT_CALIBRATION_NOTE = (
    "Estimated on the SkillSpan test split (974 docs) with the anchor + QLoRA-loop v1 + v2 adapters; "
    "re-fit on your own held-out data with fit_vote_calibration()."
)


@dataclass
class VoteCalibration:
    """Maps "how many of the ``n_systems`` systems predicted this exact span" to an estimated precision.

    ``precision[v]`` is the probability that a span predicted by exactly ``v`` systems is correct (exact span
    match). ``counts[v] = {"correct": c, "total": n}`` are the counts it was estimated from (empty for hand-entered
    tables). Use it as the span ``confidence`` instead of the constant per-extractor prior.
    """

    n_systems: int
    precision: dict[int, float]
    counts: dict[int, dict[str, int]] = field(default_factory=dict)
    smoothing: float = 0.0
    source: str = "custom"
    note: str = ""

    def confidence(self, votes: int) -> float | None:
        """Estimated precision for a span with ``votes`` agreeing systems (None if the table has no entry)."""
        return self.precision.get(int(votes))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["precision"] = {str(k): v for k, v in self.precision.items()}
        d["counts"] = {str(k): v for k, v in self.counts.items()}
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VoteCalibration":
        return cls(
            n_systems=int(data["n_systems"]),
            precision={int(k): float(v) for k, v in data["precision"].items()},
            counts={int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in (data.get("counts") or {}).items()},
            smoothing=float(data.get("smoothing", 0.0)),
            source=data.get("source", "custom"),
            note=data.get("note", ""),
        )


def default_vote_calibration(n_systems: int) -> VoteCalibration | None:
    """The shipped table (only defined for 3 systems: anchor, QLoRA-loop v1, v2); None for any other count."""
    if n_systems != 3:
        return None
    return VoteCalibration(
        n_systems=3,
        precision={1: 0.2901, 2: 0.5450, 3: 0.8018},
        counts={1: {"correct": 273, "total": 941}, 2: {"correct": 351, "total": 644}, 3: {"correct": 1145, "total": 1428}},
        smoothing=0.0,
        source="skillspan_test_974_docs (reports/skillllm_v2/ensemble_vote_confidence.md)",
        note=DEFAULT_CALIBRATION_NOTE,
    )


def fit_vote_calibration(
    gold_spans_per_doc: Sequence[Sequence[Span]],
    per_system_predictions: Sequence[Sequence[Sequence[Span]]],
    smoothing: float = 1.0,
    source: str = "fitted",
) -> VoteCalibration:
    """Fit the vote -> precision table on your own held-out data.

    Args:
        gold_spans_per_doc: ``[doc][span]`` gold spans (dicts with ``start``, ``end``, ``label``).
        per_system_predictions: ``[system][doc][span]`` predictions of every system on the same documents
            (same layout as :func:`vote_documents`).
        smoothing: Laplace pseudo-count ``a``; precision = ``(correct + a) / (total + 2a)``. 0 gives the raw
            proportion (and 0.5 is returned for a vote count that never occurred only if ``smoothing > 0``).

    A span counts once per system; "correct" means its exact ``(start, end, label)`` is in the gold spans.
    """
    if not per_system_predictions:
        raise ValueError("per_system_predictions must contain at least one system")
    n_systems = len(per_system_predictions)
    n_docs = len(gold_spans_per_doc)
    for i, docs in enumerate(per_system_predictions):
        if len(docs) != n_docs:
            raise ValueError(f"system {i} has {len(docs)} documents but gold has {n_docs}")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    correct = {v: 0 for v in range(1, n_systems + 1)}
    total = {v: 0 for v in range(1, n_systems + 1)}
    for d in range(n_docs):
        gold = {(g["start"], g["end"], g.get("label")) for g in gold_spans_per_doc[d]}
        counts: dict[tuple[Any, ...], int] = defaultdict(int)
        for sys_docs in per_system_predictions:
            seen: set[tuple[Any, ...]] = set()
            for sp in sys_docs[d]:
                key = (sp["start"], sp["end"], sp.get("label"))
                if key in seen:
                    continue
                seen.add(key)
                counts[key] += 1
        for key, v in counts.items():
            total[v] += 1
            correct[v] += key in gold
    precision: dict[int, float] = {}
    for v in total:
        denom = total[v] + 2 * smoothing
        if denom > 0:
            precision[v] = round((correct[v] + smoothing) / denom, 4)
    return VoteCalibration(
        n_systems=n_systems,
        precision=precision,
        counts={v: {"correct": correct[v], "total": total[v]} for v in total},
        smoothing=float(smoothing),
        source=source,
        note=f"Fitted on {n_docs} documents with Laplace smoothing {smoothing}.",
    )
