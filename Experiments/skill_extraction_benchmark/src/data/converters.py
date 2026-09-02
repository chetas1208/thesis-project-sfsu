"""Converters from canonical format to each model-specific export format."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

from src.data.canonical_schema import CanonicalRecord, SpanAnnotation


# ── NNOSE / JobBERT (token classification) ───────────────────────────


def canonical_to_token_classification(record: CanonicalRecord) -> Dict[str, Any]:
    """Convert one canonical record to NNOSE/JobBERT token-classification format."""
    return {
        "id": record.sentence_id,
        "doc_id": record.doc_id,
        "tokens": record.tokens,
        "ner_tags": record.bio_tags,
    }


# ── GLiNER ───────────────────────────────────────────────────────────


def canonical_to_gliner(record: CanonicalRecord) -> Dict[str, Any]:
    """Convert one canonical record to GLiNER training format.

    GLiNER expects:
        tokenized_text: list of tokens
        ner: list of [start_idx, end_idx, label] (inclusive indices)
        ner_labels: list of unique label strings
    """
    ner_entries = []
    for span in record.spans:
        ner_entries.append([span.start_token, span.end_token, "skill"])

    return {
        "id": record.sentence_id,
        "doc_id": record.doc_id,
        "tokenized_text": record.tokens,
        "ner": ner_entries,
        "ner_labels": ["skill"],
    }


# ── Skill-LLM (instruction-tuning) ──────────────────────────────────

_SKILL_LLM_INSTRUCTION = "Extract skill entities from the given sentence."


def _build_context_window(tokens: Sequence[str], start: int, end: int, window: int = 5) -> str:
    """Extract a context window around a span for Skill-LLM output."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(tokens), end + 1 + window)
    return " ".join(tokens[ctx_start:ctx_end])


def canonical_to_skill_llm(record: CanonicalRecord) -> Dict[str, Any]:
    """Convert one canonical record to Skill-LLM instruction-tuning format."""
    skills_output: List[Dict[str, str]] = []
    for span in record.spans:
        skills_output.append({
            "skill_span": span.text,
            "context": _build_context_window(record.tokens, span.start_token, span.end_token),
        })

    output_json = json.dumps({"SKILL": skills_output}, ensure_ascii=False)

    return {
        "instruction": _SKILL_LLM_INSTRUCTION,
        "input": record.text,
        "output": output_json,
        "id": record.sentence_id,
        "doc_id": record.doc_id,
    }


def build_skill_llm_dataset_info(data_dir: Path) -> Dict[str, Any]:
    """Create dataset_info.json for LLaMA-Factory compatibility."""
    return {
        "skillspan_train": {
            "file_name": str(data_dir / "train.json"),
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "skillspan_dev": {
            "file_name": str(data_dir / "dev.json"),
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "skillspan_test": {
            "file_name": str(data_dir / "test.json"),
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
    }


# ── ConTeXT (sentence–skill similarity pairs) ────────────────────────


def _collect_all_skill_texts(records: Sequence[CanonicalRecord]) -> List[str]:
    """Collect all unique skill surface forms from a set of records."""
    skills: Set[str] = set()
    for rec in records:
        for span in rec.spans:
            skills.add(span.text)
    return sorted(skills)


def canonical_to_context_pairs(
    record: CanonicalRecord,
    all_skill_texts: Sequence[str],
    negatives_per_positive: int = 5,
    rng: random.Random | None = None,
) -> List[Dict[str, Any]]:
    """Create positive and hard-negative sentence–skill pairs for one record.

    Positive pairs: one per gold skill span text in this sentence.
    Negative pairs: random skills from the inventory that are NOT gold spans.
    """
    if rng is None:
        rng = random.Random(42)

    gold_texts = {span.text for span in record.spans}
    pairs: List[Dict[str, Any]] = []

    # Positive pairs
    for skill_text in sorted(gold_texts):
        pairs.append({
            "sentence_id": record.sentence_id,
            "doc_id": record.doc_id,
            "sentence": record.text,
            "skill": skill_text,
            "label": 1,
        })

    # Negative pairs
    if gold_texts:
        negative_pool = [s for s in all_skill_texts if s not in gold_texts]
        num_negatives = min(len(negative_pool), len(gold_texts) * negatives_per_positive)
        if negative_pool and num_negatives > 0:
            chosen_negatives = rng.sample(negative_pool, num_negatives)
            for neg_skill in chosen_negatives:
                pairs.append({
                    "sentence_id": record.sentence_id,
                    "doc_id": record.doc_id,
                    "sentence": record.text,
                    "skill": neg_skill,
                    "label": 0,
                })

    return pairs
