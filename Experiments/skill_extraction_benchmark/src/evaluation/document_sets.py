"""Document-level set evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.data.normalizers import normalize_skill_text


def document_set_metrics(doc_predictions: Sequence[Dict]) -> Dict[str, Any]:
    """Compute document-level set P/R/F1 and Jaccard similarity.

    Each document has:
        predicted_skill_texts: list[str]
        gold_skill_texts: list[str]  (if available)
    """
    tp = fp = fn = 0
    jaccard_scores: List[float] = []
    total_gold_skills = 0
    total_pred_skills = 0
    docs_evaluated = 0

    for doc in doc_predictions:
        gold_raw = doc.get("gold_skill_texts", [])
        pred_raw = doc.get("predicted_skill_texts", [])

        if not gold_raw:
            continue

        gold_set = {normalize_skill_text(s) for s in gold_raw if normalize_skill_text(s)}
        pred_set = {normalize_skill_text(s) for s in pred_raw if normalize_skill_text(s)}

        if not gold_set:
            continue

        docs_evaluated += 1
        total_gold_skills += len(gold_set)
        total_pred_skills += len(pred_set)

        intersection = gold_set & pred_set
        tp += len(intersection)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        union = gold_set | pred_set
        jaccard = len(intersection) / len(union) if union else 0.0
        jaccard_scores.append(jaccard)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
    avg_gold = total_gold_skills / docs_evaluated if docs_evaluated else 0.0
    avg_pred = total_pred_skills / docs_evaluated if docs_evaluated else 0.0

    return {
        "doc_precision": round(precision, 6),
        "doc_recall": round(recall, 6),
        "doc_f1": round(f1, 6),
        "doc_jaccard": round(avg_jaccard, 6),
        "docs_evaluated": docs_evaluated,
        "avg_gold_skills_per_doc": round(avg_gold, 2),
        "avg_pred_skills_per_doc": round(avg_pred, 2),
        "doc_tp": tp,
        "doc_fp": fp,
        "doc_fn": fn,
    }
