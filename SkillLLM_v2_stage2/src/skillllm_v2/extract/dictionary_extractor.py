from __future__ import annotations

import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any

from skillllm_v2.schema import SkillCandidate, normalize_text


class DictionaryExtractor:
    def __init__(self, alias_paths: list[str] | None = None, fuzzy_threshold: float = 0.92) -> None:
        self.alias_paths = [Path(path) for path in alias_paths or []]
        self.fuzzy_threshold = fuzzy_threshold
        self.aliases = self._load_aliases()
        if not self.aliases:
            self.aliases = _default_aliases()

    def extract(self, text: str) -> list[dict[str, Any]]:
        candidates: list[SkillCandidate] = []
        seen: set[tuple[int, int, str]] = set()
        for alias, item in sorted(self.aliases.items(), key=lambda row: len(row[0]), reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                start, end = match.span()
                key = (start, end, item["label"])
                if key in seen:
                    continue
                seen.add(key)
                surface = text[start:end]
                candidates.append(
                    SkillCandidate(
                        surface,
                        item["label"],
                        start,
                        end,
                        "dictionary",
                        0.82,
                        surface,
                        item.get("canonical_skill_id"),
                        item.get("canonical_skill_name"),
                        {"match_method": "exact_alias", "alias": alias},
                    )
                )
        candidates.extend(self._fuzzy_candidates(text, seen))
        return [candidate.to_dict() for candidate in sorted(candidates, key=lambda item: (item.start or 0, item.end or 0))]

    def _fuzzy_candidates(self, text: str, seen: set[tuple[int, int, str]]) -> list[SkillCandidate]:
        tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z][A-Za-z0-9+#.-]*", text)]
        output: list[SkillCandidate] = []
        for alias, item in self.aliases.items():
            alias_len = len(alias.split())
            if alias_len < 2:
                continue
            for width in {alias_len, alias_len + 1}:
                for i in range(0, max(0, len(tokens) - width + 1)):
                    start, end = tokens[i][1], tokens[i + width - 1][2]
                    phrase = text[start:end]
                    score = SequenceMatcher(None, normalize_text(alias), normalize_text(phrase)).ratio()
                    if score < self.fuzzy_threshold:
                        continue
                    key = (start, end, item["label"])
                    if key in seen:
                        continue
                    seen.add(key)
                    output.append(
                        SkillCandidate(
                            phrase,
                            item["label"],
                            start,
                            end,
                            "dictionary",
                            min(0.72, score),
                            phrase,
                            item.get("canonical_skill_id"),
                            item.get("canonical_skill_name"),
                            {"match_method": "fuzzy_alias", "alias": alias, "score": score},
                        )
                    )
        return output

    def _load_aliases(self) -> dict[str, dict[str, Any]]:
        aliases: dict[str, dict[str, Any]] = {}
        for path in self.alias_paths:
            if not path.exists():
                continue
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("skills", [])
            elif path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            else:
                rows = [{"alias": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                canonical_name = row.get("canonical_skill_name") or row.get("name") or row.get("skill") or row.get("alias")
                label = row.get("label") or "HARD_SKILL"
                if label not in {"HARD_SKILL", "SOFT_SKILL"}:
                    label = "HARD_SKILL"
                for alias in _aliases_from_row(row):
                    aliases[alias] = {
                        "label": label,
                        "canonical_skill_id": row.get("canonical_skill_id") or row.get("id"),
                        "canonical_skill_name": canonical_name,
                    }
        return aliases


def _aliases_from_row(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ["alias", "aliases", "canonical_skill_name", "name", "skill"]:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif isinstance(value, str):
            values.extend(part.strip() for part in re.split(r"[|;,]", value) if part.strip())
    return list(dict.fromkeys(values))


def _default_aliases() -> dict[str, dict[str, Any]]:
    hard = [
        "Python",
        "SQL",
        "machine learning",
        "AWS",
        "Amazon Web Services",
        "Docker",
        "Kubernetes",
        "PyTorch",
        "Excel",
        "Business to Business",
        "Information Technology",
        "Web Technology",
        "Consulting",
        "IT Consulting",
        "javascript",
        "reactjs",
        "java",
    ]
    soft = ["communication", "leadership", "collaboration", "teamwork", "problem solving"]
    aliases: dict[str, dict[str, Any]] = {}
    for value in hard:
        aliases[value] = {"label": "HARD_SKILL", "canonical_skill_id": None, "canonical_skill_name": value}
    for value in soft:
        aliases[value] = {"label": "SOFT_SKILL", "canonical_skill_id": None, "canonical_skill_name": value}
    aliases["AWS"]["canonical_skill_name"] = "Amazon Web Services"
    return aliases
