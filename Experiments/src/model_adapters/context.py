from pathlib import Path
from typing import Dict, List

from common import ensure_dir, write_json


DEFAULT_NEGATIVE_SKILLS = [
    "Kubernetes",
    "Docker",
    "TensorFlow",
    "Tableau",
    "Terraform",
]


def _sentence_skill_names(record: Dict) -> List[str]:
    names = []
    for item in record.get("normalized_skills", []):
        canonical = item.get("canonical_name") or item.get("text")
        if canonical and canonical not in names:
            names.append(canonical)
    for span in record.get("spans", []):
        if span["text"] not in names:
            names.append(span["text"])
    return names


def export_context_dataset(splits: Dict[str, List[Dict]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    skill_pool = sorted(
        {
            skill_name
            for records in splits.values()
            for record in records
            for skill_name in _sentence_skill_names(record)
        }
    )

    for split_name, records in splits.items():
        pairs = []
        for record in records:
            positives = _sentence_skill_names(record)
            seen_pairs = set()
            for skill_name in positives:
                key = (record["text"], skill_name, 1)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append({"sentence": record["text"], "skill": skill_name, "label": 1})

            negative_candidates = [skill for skill in skill_pool if skill not in positives]
            if not negative_candidates:
                negative_candidates = [skill for skill in DEFAULT_NEGATIVE_SKILLS if skill not in positives]
            for skill_name in negative_candidates[: max(1, min(3, len(positives) or 1))]:
                key = (record["text"], skill_name, 0)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pairs.append({"sentence": record["text"], "skill": skill_name, "label": 0})
        write_json(output_dir / f"{split_name}.json", pairs)
