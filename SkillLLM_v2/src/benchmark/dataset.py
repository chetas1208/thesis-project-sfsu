from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.runtime_utils import ensure_dir, read_jsonl, write_json, write_jsonl


SPLIT_ALIASES = {
    "train": ("train",),
    "dev": ("dev", "validation", "val"),
    "test": ("test",),
}

LABEL_MAP = {
    "SKILL": "SOFT_SKILL",
    "KNOWLEDGE": "HARD_SKILL",
}


@dataclass(frozen=True)
class DatasetDiscovery:
    dataset_type: str
    root: Path
    split_files: dict[str, Path]
    used_fallback_split: bool


def _candidate_roots(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append((Path.cwd() / path).resolve())

    repo_root = Path(__file__).resolve().parents[2]
    known = [
        repo_root / "data" / "source" / "skillspan_hf",
        repo_root / "data" / "skillspan",
    ]
    for candidate in known:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _find_split_files(root: Path) -> dict[str, Path]:
    split_files: dict[str, Path] = {}
    if not root.exists():
        return split_files

    for split_name, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            for extension in (".jsonl", ".json", ".csv", ".conll", ".bio", ".txt"):
                candidate = root / f"{alias}{extension}"
                if candidate.exists():
                    split_files[split_name] = candidate
                    break
            if split_name in split_files:
                break
    return split_files


def _detect_file_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix in {".conll", ".bio", ".txt"}:
        return "bio_conll"

    with path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        second_line = handle.readline().strip()
    if first_line.startswith("{") and second_line.startswith("{"):
        return "jsonl"
    if first_line.startswith("["):
        return "json"
    if "," in first_line and not first_line.startswith("{"):
        return "csv"
    return "json"


def discover_dataset(dataset_path: str | Path) -> DatasetDiscovery:
    for root in _candidate_roots(dataset_path):
        split_files = _find_split_files(root)
        if not split_files:
            continue
        dataset_type = _detect_file_format(next(iter(split_files.values())))
        return DatasetDiscovery(
            dataset_type=dataset_type,
            root=root,
            split_files=split_files,
            used_fallback_split=False,
        )
    raise FileNotFoundError(f"Could not discover SkillSpan dataset from {dataset_path!s}.")


def token_offsets(tokens: list[str]) -> list[dict[str, int]]:
    offsets: list[dict[str, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        start = cursor
        end = start + len(token)
        offsets.append({"start": start, "end": end})
        cursor = end + (1 if index < len(tokens) - 1 else 0)
    return offsets


def _spans_from_bio(tokens: list[str], tags: list[str], label: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    offsets = token_offsets(tokens)
    text = " ".join(tokens)
    active_start: int | None = None

    def flush(end_index: int) -> None:
        nonlocal active_start
        if active_start is None:
            return
        start = offsets[active_start]["start"]
        end = offsets[end_index]["end"]
        spans.append(
            {
                "text": text[start:end],
                "label": label,
                "start": start,
                "end": end,
                "token_start": active_start,
                "token_end": end_index,
            }
        )
        active_start = None

    for index, tag in enumerate(tags):
        if tag == "B":
            flush(index - 1)
            active_start = index
        elif tag == "I":
            if active_start is None:
                active_start = index
        else:
            flush(index - 1)
    flush(len(tags) - 1)
    return spans


def _token_labels(tokens: list[str], skills: list[dict[str, Any]]) -> list[str]:
    labels = ["O"] * len(tokens)
    for span in skills:
        prefix = span["label"]
        for token_index in range(int(span["token_start"]), int(span["token_end"]) + 1):
            labels[token_index] = f"{'B' if token_index == span['token_start'] else 'I'}-{prefix}"
    return labels


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    file_type = _detect_file_format(path)
    if file_type == "jsonl":
        return read_jsonl(path)
    if file_type == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Expected a JSON array in {path}.")
    if file_type == "csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported dataset file format for {path}.")


def _canonicalize_skillspan_rows(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    canonical = []
    for row_index, row in enumerate(rows):
        tokens = list(row["tokens"])
        text = " ".join(tokens)
        hard_spans = _spans_from_bio(tokens, list(row["tags_knowledge"]), LABEL_MAP["KNOWLEDGE"])
        soft_spans = _spans_from_bio(tokens, list(row["tags_skill"]), LABEL_MAP["SKILL"])
        skills = sorted(hard_spans + soft_spans, key=lambda item: (item["start"], item["end"], item["label"], item["text"]))
        offsets = token_offsets(tokens)
        canonical.append(
            {
                "id": f"{split_name}_{row.get('idx', row_index)}_{row_index}",
                "document_id": row.get("idx", row_index),
                "source": row.get("source"),
                "text": text,
                "tokens": tokens,
                "token_offsets": offsets,
                "token_labels": _token_labels(tokens, skills),
                "skills": [
                    {
                        "text": item["text"],
                        "label": item["label"],
                        "start": item["start"],
                        "end": item["end"],
                    }
                    for item in skills
                ],
            }
        )
    return canonical


def _fallback_split(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    train_end = int(len(indices) * 0.8)
    dev_end = train_end + int(len(indices) * 0.1)

    split_indices = {
        "train": set(indices[:train_end]),
        "dev": set(indices[train_end:dev_end]),
        "test": set(indices[dev_end:]),
    }
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for record_index, record in enumerate(records):
        for split_name, selected in split_indices.items():
            if record_index in selected:
                splits[split_name].append(record)
                break
    return splits


def _load_bio_conll(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tokens: list[str] = []
    skill_tags: list[str] = []
    knowledge_tags: list[str] = []
    doc_id = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if tokens:
                rows.append(
                    {
                        "idx": doc_id,
                        "tokens": tokens,
                        "tags_skill": skill_tags,
                        "tags_knowledge": knowledge_tags,
                        "source": None,
                    }
                )
                tokens, skill_tags, knowledge_tags = [], [], []
                doc_id += 1
            continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Unsupported BIO/CoNLL line in {path}: {raw_line!r}")
        token = parts[0]
        skill_tag = parts[1].replace("B-", "").replace("I-", "")
        knowledge_tag = parts[2].replace("B-", "").replace("I-", "")
        tokens.append(token)
        skill_tags.append(skill_tag if skill_tag in {"B", "I"} else "O")
        knowledge_tags.append(knowledge_tag if knowledge_tag in {"B", "I"} else "O")

    if tokens:
        rows.append(
            {
                "idx": doc_id,
                "tokens": tokens,
                "tags_skill": skill_tags,
                "tags_knowledge": knowledge_tags,
                "source": None,
            }
        )
    return rows


def _load_split_records(path: Path) -> list[dict[str, Any]]:
    file_type = _detect_file_format(path)
    if file_type == "bio_conll":
        return _load_bio_conll(path)
    return _load_json_records(path)


def derive_smoke_output_dir(output_dir: str | Path, limit: int | None) -> Path:
    root = Path(output_dir)
    if limit is None:
        return root
    return root.parent / f"{root.name}_smoke_limit_{limit}"


def prepare_benchmark_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    split: str,
    seed: int,
    limit: int | None = None,
) -> dict[str, Any]:
    discovery = discover_dataset(dataset_path)
    prepared_dir = ensure_dir(Path(output_dir) / "_prepared_data")

    if all(name in discovery.split_files for name in ("train", "dev", "test")):
        source_splits = {
            split_name: _load_split_records(file_path)
            for split_name, file_path in discovery.split_files.items()
        }
        used_fallback = False
    else:
        merged_records: list[dict[str, Any]] = []
        for file_path in discovery.split_files.values():
            merged_records.extend(_load_split_records(file_path))
        source_splits = _fallback_split(merged_records, seed)
        used_fallback = True

    canonical_splits: dict[str, list[dict[str, Any]]] = {}
    for split_name, rows in source_splits.items():
        if limit is not None:
            rows = rows[:limit]
        canonical_splits[split_name] = _canonicalize_skillspan_rows(rows, split_name=split_name)
        write_jsonl(prepared_dir / f"{split_name}.jsonl", canonical_splits[split_name])

    metadata = {
        "dataset_type": discovery.dataset_type,
        "dataset_root": str(discovery.root),
        "requested_dataset_path": str(dataset_path),
        "used_fallback_split": used_fallback,
        "split": split,
        "seed": seed,
        "limit": limit,
        "prepared_dir": str(prepared_dir),
        "splits": {
            split_name: {
                "records": len(records),
                "path": str(prepared_dir / f"{split_name}.jsonl"),
            }
            for split_name, records in canonical_splits.items()
        },
    }
    write_json(prepared_dir / "metadata.json", metadata)
    return metadata
