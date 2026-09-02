from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_skillspan_split(dataset_root: Path, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    root = _resolve_dataset_root(dataset_root)
    path = root / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing SkillSpan split: {path}")
    docs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if limit is not None and i >= limit:
            break
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row["tokens"]
        text, offsets = _tokens_to_text(tokens)
        gold_spans = []
        gold_spans.extend(_spans_from_tags(text, offsets, row.get("tags_knowledge", []), "HARD_SKILL"))
        gold_spans.extend(_spans_from_tags(text, offsets, row.get("tags_skill", []), "SOFT_SKILL"))
        docs.append(
            {
                "id": f"{split}-{i}",
                "document_id": row.get("idx", i),
                "source": row.get("source", "skillspan"),
                "tokens": tokens,
                "text": text,
                "gold_spans": sorted(gold_spans, key=lambda item: (item["start"], item["end"], item["label"])),
                "gold_token_tags": {"tags_knowledge": row.get("tags_knowledge", []), "tags_skill": row.get("tags_skill", [])},
            }
        )
    return docs


def build_verifier_dataset(dataset_root: Path, output_root: Path, limit: int | None = None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev"]:
        docs = load_skillspan_split(dataset_root, split, limit=limit)
        out_path = output_root / f"verifier_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for doc in docs:
                for span in doc["gold_spans"]:
                    handle.write(json.dumps(_verifier_row(doc["text"], span, True, span["label"], span["surface_form"], "gold", {"reason": "gold_span"})) + "\n")
                    expanded = _expanded_negative(doc["text"], span)
                    if expanded:
                        handle.write(json.dumps(_verifier_row(doc["text"], expanded, False, None, span["surface_form"], "boundary_noise", {"reason": "boundary_expanded"})) + "\n")
                for negative in _generic_negatives(doc["text"]):
                    handle.write(json.dumps(_verifier_row(doc["text"], negative, False, None, None, "synthetic_negative", {"reason": "generic_phrase"})) + "\n")


def _resolve_dataset_root(path: Path) -> Path:
    if (path / "train.json").exists():
        return path
    candidates = [
        Path("data/source/skillspan_hf"),
        path.parent / "source" / "skillspan_hf",
    ]
    for candidate in candidates:
        if (candidate / "train.json").exists():
            return candidate
    return path


def _tokens_to_text(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text = ""
    offsets = []
    for token in tokens:
        if text:
            text += " "
        start = len(text)
        text += token
        offsets.append((start, len(text)))
    return text, offsets


def _spans_from_tags(text: str, offsets: list[tuple[int, int]], tags: list[str], label: str) -> list[dict[str, Any]]:
    spans = []
    start_idx: int | None = None
    for i, tag in enumerate(tags + ["O"]):
        is_entity = i < len(tags) and tag != "O"
        starts_new = is_entity and (tag.startswith("B") or start_idx is None)
        closes = (not is_entity or starts_new) and start_idx is not None
        if closes:
            end_idx = i - 1
            start, end = offsets[start_idx][0], offsets[end_idx][1]
            spans.append({"surface_form": text[start:end], "label": label, "start": start, "end": end})
            start_idx = None
        if starts_new:
            start_idx = i
    return spans


def _expanded_negative(text: str, span: dict[str, Any]) -> dict[str, Any] | None:
    start = max(0, span["start"] - 5)
    end = min(len(text), span["end"] + 5)
    if start == span["start"] and end == span["end"]:
        return None
    return {**span, "surface_form": text[start:end], "start": start, "end": end}


def _generic_negatives(text: str) -> list[dict[str, Any]]:
    output = []
    for phrase in ["experience", "knowledge", "responsibilities", "team"]:
        start = text.lower().find(phrase)
        if start >= 0:
            output.append({"surface_form": text[start : start + len(phrase)], "label": "HARD_SKILL", "start": start, "end": start + len(phrase)})
    return output


def _verifier_row(
    text: str,
    candidate: dict[str, Any],
    gold_is_skill: bool,
    gold_label: str | None,
    corrected_span: str | None,
    source: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "text": text,
        "candidate_span": candidate["surface_form"],
        "start": candidate["start"],
        "end": candidate["end"],
        "candidate_label": candidate["label"],
        "gold_is_skill": gold_is_skill,
        "gold_label": gold_label,
        "corrected_span": corrected_span,
        "source": source,
        "metadata": metadata,
        "candidate": {
            "surface_form": candidate["surface_form"],
            "label": candidate["label"],
            "start": candidate["start"],
            "end": candidate["end"],
            "source": "dictionary",
            "confidence": 0.9 if gold_is_skill else 0.6,
            "evidence": candidate["surface_form"],
            "canonical_skill_id": None,
            "canonical_skill_name": None,
            "metadata": metadata,
        },
        "is_valid_skill": gold_is_skill,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", default="data/skillllm_v2")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build_verifier_dataset(Path(args.dataset_path), Path(args.output_dir), args.limit)


if __name__ == "__main__":
    main()
