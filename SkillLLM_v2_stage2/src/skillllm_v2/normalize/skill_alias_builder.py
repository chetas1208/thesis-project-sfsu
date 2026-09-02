"""skill_alias_builder.py

Builds a local skill alias table used by the taxonomy linker, dictionary
extractor, and boundary refiner/selector for "exact alias" evidence.

Sources (in priority order -- earlier sources win on alias conflicts):
  1. Manually curated technical/soft-skill aliases (``CUSTOM``).
  2. External taxonomy exports if present under ``data/skillllm_v2/taxonomy/``
     (``esco_skills.jsonl|csv``, ``onet_skills.jsonl|csv``,
     ``lightcast_skills.jsonl|csv``). Skipped silently if absent.
  3. Gold SkillSpan train+dev skill spans (``SKILLSPAN_GOLD``).
  4. High-confidence Raw Mistral LoRA test predictions (``CUSTOM``,
     source=``mistral_high_confidence``).

Output: ``data/skillllm_v2/taxonomy/skill_aliases.jsonl``, one record per
(canonical, alias) pair:

    {"canonical_skill_id": "...", "canonical_skill_name": "...",
     "alias": "...", "normalized_alias": "...",
     "taxonomy": "ESCO"|"ONET"|"LIGHTCAST"|"CUSTOM"|"SKILLSPAN_GOLD",
     "label_hint": "HARD_SKILL"|"SOFT_SKILL"|null, "source": "..."}

Usage:
    python -m skillllm_v2.normalize.skill_alias_builder
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import re
from typing import Any

from skillllm_v2.io_utils import read_json, read_jsonl, write_jsonl
from skillllm_v2.schema import normalize_text
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

DEFAULT_DATASET_ROOT = "data/source/skillspan_hf"
DEFAULT_MISTRAL_PREDICTIONS = "artifacts/skillllm_v2/mistral_lora/predictions.jsonl"
DEFAULT_OUTPUT = "data/skillllm_v2/taxonomy/skill_aliases.jsonl"
DEFAULT_TAXONOMY_DIR = "data/skillllm_v2/taxonomy"

# (canonical_skill_name, alias, label_hint)
MANUAL_ALIAS_PAIRS: list[tuple[str, str, str]] = [
    ("Amazon Web Services", "AWS", "HARD_SKILL"),
    ("Amazon Web Services", "Amazon Web Services", "HARD_SKILL"),
    ("continuous integration/continuous deployment", "CI/CD", "HARD_SKILL"),
    ("continuous integration/continuous deployment", "continuous integration", "HARD_SKILL"),
    ("continuous integration/continuous deployment", "continuous deployment", "HARD_SKILL"),
    ("continuous integration/continuous deployment", "continuous integration/continuous deployment", "HARD_SKILL"),
    ("JavaScript", "JS", "HARD_SKILL"),
    ("JavaScript", "JavaScript", "HARD_SKILL"),
    ("machine learning", "ML", "HARD_SKILL"),
    ("machine learning", "machine learning", "HARD_SKILL"),
    ("natural language processing", "NLP", "HARD_SKILL"),
    ("natural language processing", "natural language processing", "HARD_SKILL"),
    ("deep learning", "DL", "HARD_SKILL"),
    ("deep learning", "deep learning", "HARD_SKILL"),
    ("structured query language", "SQL", "HARD_SKILL"),
    ("structured query language", "structured query language", "HARD_SKILL"),
    ("PyTorch Lightning", "PyTorch Lightning", "HARD_SKILL"),
    ("React", "React", "HARD_SKILL"),
    ("React", "React.js", "HARD_SKILL"),
    ("React", "ReactJS", "HARD_SKILL"),
    ("Node.js", "Node.js", "HARD_SKILL"),
    ("Node.js", "NodeJS", "HARD_SKILL"),
    ("Python", "Python", "HARD_SKILL"),
    ("Docker", "Docker", "HARD_SKILL"),
    ("Kubernetes", "Kubernetes", "HARD_SKILL"),
    ("PyTorch", "PyTorch", "HARD_SKILL"),
    ("communication", "communication skills", "SOFT_SKILL"),
    ("communication", "communication", "SOFT_SKILL"),
    ("leadership", "leadership skills", "SOFT_SKILL"),
    ("leadership", "leadership", "SOFT_SKILL"),
    ("problem solving", "problem-solving skills", "SOFT_SKILL"),
    ("problem solving", "problem solving", "SOFT_SKILL"),
    ("teamwork", "teamwork", "SOFT_SKILL"),
    ("collaboration", "collaboration", "SOFT_SKILL"),
]

EXTERNAL_TAXONOMIES = [
    ("ESCO", ["esco_skills.jsonl", "esco_skills.csv", "esco_skills.json"]),
    ("ONET", ["onet_skills.jsonl", "onet_skills.csv", "onet_skills.json"]),
    ("LIGHTCAST", ["lightcast_skills.jsonl", "lightcast_skills.csv", "lightcast_skills.json"]),
]


def _slug(name: str) -> str:
    norm = normalize_text(name)
    slug = re.sub(r"\s+", "-", norm)
    return slug or "skill"


def _record(canonical_name: str, alias: str, taxonomy: str, label_hint: str | None, source: str) -> dict[str, Any]:
    return {
        "canonical_skill_id": _slug(canonical_name),
        "canonical_skill_name": canonical_name,
        "alias": alias,
        "normalized_alias": normalize_text(alias),
        "taxonomy": taxonomy,
        "label_hint": label_hint,
        "source": source,
    }


def _manual_records() -> list[dict[str, Any]]:
    return [_record(name, alias, "CUSTOM", label_hint, "manual_curated") for name, alias, label_hint in MANUAL_ALIAS_PAIRS]


def _external_records(taxonomy_dir: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    base = Path(taxonomy_dir)
    if not base.exists():
        return records
    for taxonomy_name, filenames in EXTERNAL_TAXONOMIES:
        for filename in filenames:
            path = base / filename
            if not path.exists():
                continue
            rows = _load_external_rows(path)
            for row in rows:
                name = row.get("preferredLabel") or row.get("canonical_skill_name") or row.get("name") or row.get("skill")
                if not name:
                    continue
                aliases = row.get("altLabels") or row.get("aliases") or row.get("alias") or []
                if isinstance(aliases, str):
                    aliases = [part.strip() for part in re.split(r"[|;,]", aliases) if part.strip()]
                label_hint = row.get("label_hint")
                records.append(_record(name, name, taxonomy_name, label_hint, f"{taxonomy_name.lower()}_export"))
                for alias in aliases:
                    records.append(_record(name, alias, taxonomy_name, label_hint, f"{taxonomy_name.lower()}_export"))
            break  # only load the first matching file for this taxonomy
    return records


def _load_external_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(str(path))
    if path.suffix.lower() == ".json":
        data = read_json(str(path), default=[])
        return data if isinstance(data, list) else data.get("skills", [])
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def _gold_skillspan_records(dataset_root: str, splits: tuple[str, ...] = ("train", "dev"), min_count: int = 2) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    surface_counts: dict[tuple[str, str], Counter[str]] = {}
    for split in splits:
        try:
            docs = load_skillspan_split(Path(dataset_root), split, limit=None)
        except FileNotFoundError:
            continue
        for doc in docs:
            for span in doc["gold_spans"]:
                surface = doc["text"][span["start"] : span["end"]]
                norm = normalize_text(surface)
                if not norm:
                    continue
                key = (norm, span["label"])
                counts[key] += 1
                surface_counts.setdefault(key, Counter())[surface] += 1

    records: list[dict[str, Any]] = []
    for (norm, label), count in counts.items():
        if count < min_count:
            continue
        canonical_surface = surface_counts[(norm, label)].most_common(1)[0][0]
        records.append(_record(canonical_surface, canonical_surface, "SKILLSPAN_GOLD", label, "skillspan_gold_train_dev"))
    return records


def _mistral_high_confidence_records(predictions_path: str, min_confidence: float = 0.8, min_count: int = 2) -> list[dict[str, Any]]:
    if not Path(predictions_path).exists():
        return []
    counts: Counter[tuple[str, str]] = Counter()
    surface_counts: dict[tuple[str, str], Counter[str]] = {}
    for row in read_jsonl(predictions_path):
        for pred in row.get("predictions", []):
            if pred.get("source") != "mistral":
                continue
            if float(pred.get("confidence", 0.0)) < min_confidence:
                continue
            surface = pred["surface_form"]
            norm = normalize_text(surface)
            if not norm:
                continue
            key = (norm, pred["label"])
            counts[key] += 1
            surface_counts.setdefault(key, Counter())[surface] += 1

    records: list[dict[str, Any]] = []
    for (norm, label), count in counts.items():
        if count < min_count:
            continue
        canonical_surface = surface_counts[(norm, label)].most_common(1)[0][0]
        records.append(_record(canonical_surface, canonical_surface, "CUSTOM", label, "mistral_high_confidence"))
    return records


def load_alias_records(paths: list[str] | None = None) -> list[dict[str, Any]]:
    """Load alias records produced by :func:`build_alias_table`.

    ``paths`` defaults to ``[DEFAULT_OUTPUT]``. Missing files are skipped.
    """
    records: list[dict[str, Any]] = []
    for path in paths or [DEFAULT_OUTPUT]:
        if Path(path).exists():
            records.extend(read_jsonl(path))
    return records


def build_alias_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map ``normalized_alias`` -> first matching alias record (priority order preserved)."""
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        index.setdefault(record["normalized_alias"], record)
    return index


def build_alias_table(
    dataset_root: str = DEFAULT_DATASET_ROOT,
    mistral_predictions: str = DEFAULT_MISTRAL_PREDICTIONS,
    taxonomy_dir: str = DEFAULT_TAXONOMY_DIR,
    output_path: str = DEFAULT_OUTPUT,
) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in _manual_records() + _external_records(taxonomy_dir) + _gold_skillspan_records(dataset_root) + _mistral_high_confidence_records(mistral_predictions):
        dedup_key = record["normalized_alias"]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        all_records.append(record)

    write_jsonl(output_path, all_records)
    return all_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SkillLLM_v2 skill alias table")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mistral-predictions", default=DEFAULT_MISTRAL_PREDICTIONS)
    parser.add_argument("--taxonomy-dir", default=DEFAULT_TAXONOMY_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = build_alias_table(args.dataset_root, args.mistral_predictions, args.taxonomy_dir, args.output)
    by_taxonomy: Counter[str] = Counter(r["taxonomy"] for r in records)
    by_canonical = len({r["canonical_skill_id"] for r in records})
    print(f"Wrote {len(records)} alias records ({by_canonical} canonical skills) to {args.output}")
    for taxonomy, count in sorted(by_taxonomy.items()):
        print(f"  {taxonomy}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
