"""Span-level evaluation metrics for skill extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.data.normalizers import normalize_skill_text


def exact_span_match_metrics(
    gold_records: Sequence[Dict],
    pred_records: Sequence[Dict],
) -> Dict[str, Any]:
    """Compute exact span-match P/R/F1 across all records.

    Matching is on (start_token, end_token) tuples within each sentence.
    """
    tp = fp = fn = 0

    gold_by_id = {str(r.get("sentence_id", r.get("id", ""))): r for r in gold_records}
    pred_by_id = {str(r.get("id", "")): r for r in pred_records}

    for rec_id, gold in gold_by_id.items():
        gold_set = set()
        for span in gold.get("spans", []):
            if isinstance(span, dict):
                gold_set.add((span.get("start_token", -1), span.get("end_token", -1)))

        pred = pred_by_id.get(rec_id, {})
        pred_set = set()
        for span in pred.get("predicted_spans", []):
            if isinstance(span, dict) and span.get("start_token", -1) >= 0:
                pred_set.add((span["start_token"], span["end_token"]))

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "exact_span_tp": tp,
        "exact_span_fp": fp,
        "exact_span_fn": fn,
        "exact_span_precision": round(precision, 6),
        "exact_span_recall": round(recall, 6),
        "exact_span_f1": round(f1, 6),
    }


def text_match_metrics(
    gold_records: Sequence[Dict],
    pred_records: Sequence[Dict],
) -> Dict[str, Any]:
    """Compute text-match P/R/F1 — matches on normalized skill text.

    This is the primary metric for models that don't produce token indices
    (e.g., GLiNER, Skill-LLM, ConTeXT).
    """
    tp = fp = fn = 0

    gold_by_id = {str(r.get("sentence_id", r.get("id", ""))): r for r in gold_records}
    pred_by_id = {str(r.get("id", "")): r for r in pred_records}

    for rec_id, gold in gold_by_id.items():
        gold_set = set()
        for span in gold.get("spans", []):
            if isinstance(span, dict):
                gold_set.add(normalize_skill_text(span.get("text", "")))
        gold_set.discard("")

        pred = pred_by_id.get(rec_id, {})
        pred_set = set()
        for text in pred.get("predicted_skill_texts", []):
            normalized = normalize_skill_text(text)
            if normalized:
                pred_set.add(normalized)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "text_match_tp": tp,
        "text_match_fp": fp,
        "text_match_fn": fn,
        "text_match_precision": round(precision, 6),
        "text_match_recall": round(recall, 6),
        "text_match_f1": round(f1, 6),
    }


def token_level_metrics(
    gold_records: Sequence[Dict],
    pred_records: Sequence[Dict],
) -> Dict[str, Any]:
    """Compute token-level P/R/F1 using seqeval."""
    from seqeval.metrics import f1_score, precision_score, recall_score

    gold_by_id = {str(r.get("sentence_id", r.get("id", ""))): r for r in gold_records}
    pred_by_id = {str(r.get("id", "")): r for r in pred_records}

    all_gold_tags: List[List[str]] = []
    all_pred_tags: List[List[str]] = []

    for rec_id, gold in gold_by_id.items():
        gold_tags = gold.get("bio_tags", [])
        if not gold_tags:
            continue

        pred = pred_by_id.get(rec_id, {})
        pred_tags = pred.get("predicted_tags", ["O"] * len(gold_tags))

        # Ensure same length
        if len(pred_tags) < len(gold_tags):
            pred_tags = pred_tags + ["O"] * (len(gold_tags) - len(pred_tags))
        elif len(pred_tags) > len(gold_tags):
            pred_tags = pred_tags[:len(gold_tags)]

        all_gold_tags.append(gold_tags)
        all_pred_tags.append(pred_tags)

    if not all_gold_tags:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    return {
        "token_precision": round(precision_score(all_gold_tags, all_pred_tags), 6),
        "token_recall": round(recall_score(all_gold_tags, all_pred_tags), 6),
        "token_f1": round(f1_score(all_gold_tags, all_pred_tags), 6),
    }
