from __future__ import annotations

import math
from typing import Any

from skillllm_v2.schema import normalize_text


def coverage_at_k(recommendations: list[dict[str, Any]], target_skills: set[str], k: int = 5) -> float:
    """Fraction of target skills covered by the top-k recommendations."""
    if not target_skills:
        return 1.0
    covered = set()
    for row in recommendations[:k]:
        skills = row.get("covered_missing_skills") or row.get("covers_skills") or row.get("teaches_skills") or []
        for s in skills:
            if isinstance(s, dict):
                covered.add(normalize_text(s.get("canonical_skill_name") or s.get("skill_name") or ""))
            else:
                covered.add(normalize_text(str(s)))
    normalized_targets = {normalize_text(s) for s in target_skills}
    return len(covered & normalized_targets) / max(1, len(normalized_targets))


def weighted_gap_coverage_at_k(
    recommendations: list[dict[str, Any]],
    gap_skills: list[dict[str, Any]],
    k: int = 5,
) -> float:
    """Weighted Skill-Gap Coverage@K measuring how much missing-skill mass is covered.
    
    Each gap skill has an importance/priority weight. This metric computes:
        (sum of weights of covered gaps) / (total weight of all gaps)
    """
    if not gap_skills:
        return 1.0

    total_weight = 0.0
    skill_weights: dict[str, float] = {}
    for item in gap_skills:
        name = normalize_text(item.get("canonical_skill_name") or item.get("skill_name") or "")
        weight = float(item.get("priority_score", item.get("importance", item.get("gap_weight", 1.0))))
        skill_weights[name] = max(skill_weights.get(name, 0.0), weight)

    total_weight = sum(skill_weights.values())
    if total_weight <= 0.0:
        return 1.0

    covered_skills: set[str] = set()
    for rec in recommendations[:k]:
        taught = rec.get("covered_missing_skills") or rec.get("covers_skills") or rec.get("teaches_skills") or []
        for s in taught:
            if isinstance(s, dict):
                covered_skills.add(normalize_text(s.get("canonical_skill_name") or s.get("skill_name") or ""))
            else:
                covered_skills.add(normalize_text(str(s)))

    covered_weight = sum(w for name, w in skill_weights.items() if name in covered_skills)
    return round(covered_weight / total_weight, 6)


def prerequisite_violation_rate(
    recommendations: list[dict[str, Any]],
    user_known_skills: set[str],
    k: int = 5,
    allow_chained_learning: bool = True,
) -> float:
    """Fraction of recommended courses in top-k that violate prerequisite constraints.
    
    If allow_chained_learning is True, a prerequisite satisfied by a course
    recommended earlier in the sequence is considered satisfied.
    """
    top_k_recs = recommendations[:k]
    if not top_k_recs:
        return 0.0

    known = {normalize_text(s) for s in user_known_skills}
    violations = 0

    for rec in top_k_recs:
        prereqs = rec.get("prerequisites") or []
        has_violation = False
        for p in prereqs:
            p_name = normalize_text(p.get("skill_name") if isinstance(p, dict) else str(p))
            if p_name and p_name not in known:
                has_violation = True
                break
        if has_violation:
            violations += 1

        if allow_chained_learning:
            taught = rec.get("covers_skills") or rec.get("teaches_skills") or []
            for s in taught:
                name = normalize_text(s.get("canonical_skill_name") or s.get("skill_name") if isinstance(s, dict) else str(s))
                if name:
                    known.add(name)

    return round(violations / len(top_k_recs), 6)


def redundancy_at_k(
    recommendations: list[dict[str, Any]],
    gap_skills: list[dict[str, Any]],
    k: int = 5,
) -> float:
    """Measures redundant overlap of taught skills across distinct courses in top-k.
    
    A lower score indicates a diverse curriculum where each course targets distinct
    missing gaps rather than repeatedly teaching the same already-closed skills.
    """
    top_k_recs = recommendations[:k]
    if len(top_k_recs) <= 1:
        return 0.0

    gap_names = {
        normalize_text(item.get("canonical_skill_name") or item.get("skill_name") or "")
        for item in gap_skills
    }

    course_gap_skills: list[set[str]] = []
    for rec in top_k_recs:
        taught = rec.get("covered_missing_skills") or rec.get("covers_skills") or rec.get("teaches_skills") or []
        skills = set()
        for s in taught:
            name = normalize_text(s.get("canonical_skill_name") or s.get("skill_name") if isinstance(s, dict) else str(s))
            if name in gap_names:
                skills.add(name)
        course_gap_skills.append(skills)

    pair_count = 0
    total_redundant = 0
    for i in range(len(course_gap_skills)):
        for j in range(i + 1, len(course_gap_skills)):
            overlap = course_gap_skills[i] & course_gap_skills[j]
            total_redundant += len(overlap)
            pair_count += 1

    if pair_count == 0 or not gap_names:
        return 0.0

    avg_overlap_per_pair = total_redundant / pair_count
    return round(avg_overlap_per_pair / max(1, len(gap_names)), 6)


def explanation_completeness_rate(recommendations: list[dict[str, Any]], k: int = 5) -> float:
    """Fraction of recommendations having complete traceable graph explanations.
    
    A recommendation is complete if it has:
    1. Identified user missing skills it resolves.
    2. Explicit multi-hop graph explanation path.
    3. Verifiable learning outcome evidence.
    4. Prerequisite compatibility reasoning.
    """
    top_k_recs = recommendations[:k]
    if not top_k_recs:
        return 0.0

    complete = 0
    for rec in top_k_recs:
        has_skills = bool(rec.get("covered_missing_skills") or rec.get("covers_skills") or rec.get("teaches_skills"))
        has_explanation = bool(rec.get("explanation_path") or (isinstance(rec.get("explanation"), list) and rec.get("explanation")))
        has_evidence = bool(rec.get("evidence"))
        has_prereq_status = "prerequisites" in rec or "prerequisite_status" in rec or "prerequisites_satisfied" in rec
        if has_skills and has_explanation and has_evidence and has_prereq_status:
            complete += 1

    return round(complete / len(top_k_recs), 6)


def precision_at_k(recommendations: list[dict[str, Any]], relevant_course_ids: set[str], k: int = 5) -> float:
    """Precision@K against gold or proxy relevant course identifiers."""
    top_k_recs = recommendations[:k]
    if not top_k_recs:
        return 0.0
    hits = sum(1 for rec in top_k_recs if rec.get("course_id") in relevant_course_ids)
    return round(hits / len(top_k_recs), 6)


def ndcg_at_k(recommendations: list[dict[str, Any]], course_relevance_scores: dict[str, float], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain (nDCG@K)."""
    top_k_recs = recommendations[:k]
    if not top_k_recs:
        return 0.0

    dcg = 0.0
    for rank, rec in enumerate(top_k_recs, start=1):
        rel = float(course_relevance_scores.get(rec.get("course_id", ""), 0.0))
        dcg += (2.0**rel - 1.0) / math.log2(rank + 1)

    # Ideal DCG
    ideal_scores = sorted(course_relevance_scores.values(), reverse=True)[:k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal_scores, start=1))

    if idcg <= 0.0:
        return 0.0
    return round(dcg / idcg, 6)
