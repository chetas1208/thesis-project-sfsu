from __future__ import annotations


def coverage_at_k(recommendations: list[dict], target_skills: set[str], k: int = 5) -> float:
    covered = set()
    for row in recommendations[:k]:
        covered.update(row.get("covers_skills", []))
    return len(covered & target_skills) / max(1, len(target_skills))
