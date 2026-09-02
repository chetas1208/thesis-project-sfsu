from __future__ import annotations


def mean_match_score(rows: list[dict]) -> float:
    return sum(float(row.get("match_score", 0.0)) for row in rows) / max(1, len(rows))
