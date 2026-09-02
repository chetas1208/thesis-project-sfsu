"""Subword ↔ word alignment utilities for HuggingFace tokenizers."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def align_tokens_to_subwords(
    tokenizer,
    tokens: Sequence[str],
    bio_tags: Sequence[str],
    max_length: int = 256,
    label2id: Optional[Dict[str, int]] = None,
) -> Dict:
    """Tokenise a pre-tokenised sentence and align BIO tags to subword pieces.

    Returns a dict suitable for HuggingFace token-classification models:
        input_ids, attention_mask, labels  (all as lists of int)
    """
    if label2id is None:
        label2id = {"O": 0, "B-SKILL": 1, "I-SKILL": 2}

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=False,
    )

    word_ids = encoding.word_ids()
    labels: List[int] = []
    previous_word_id: Optional[int] = None

    for word_id in word_ids:
        if word_id is None:
            # Special tokens ([CLS], [SEP], [PAD])
            labels.append(-100)
        elif word_id != previous_word_id:
            # First subword of a new word → use original tag
            labels.append(label2id.get(bio_tags[word_id], 0))
        else:
            # Continuation subword → propagate I- or ignore
            orig_tag = bio_tags[word_id]
            if orig_tag.startswith("B-"):
                labels.append(label2id.get(orig_tag.replace("B-", "I-"), 0))
            else:
                labels.append(label2id.get(orig_tag, 0))
        previous_word_id = word_id

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "word_ids": word_ids,
    }


def subword_predictions_to_word_tags(
    word_ids: Sequence[Optional[int]],
    logits_or_preds: Sequence[int],
    id2label: Optional[Dict[int, str]] = None,
    num_words: Optional[int] = None,
) -> List[str]:
    """Map subword-level predictions back to word-level BIO tags.

    Uses first-subword voting: each word gets the tag predicted for
    its first subword piece.
    """
    if id2label is None:
        id2label = {0: "O", 1: "B-SKILL", 2: "I-SKILL"}

    word_tags: Dict[int, str] = {}
    for sw_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_tags:
            pred_id = logits_or_preds[sw_idx]
            word_tags[word_id] = id2label.get(pred_id, "O")

    n = num_words or (max(word_tags.keys()) + 1 if word_tags else 0)
    return [word_tags.get(i, "O") for i in range(n)]
