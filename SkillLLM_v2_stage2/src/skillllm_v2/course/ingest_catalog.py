from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence


@dataclass(slots=True)
class CourseRecord:
    course_id: str
    title: str
    description: str = ""
    learning_outcomes: list[str] = field(default_factory=list)
    teaches_skills: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    level: str = "intermediate"  # "beginner" | "intermediate" | "advanced"
    duration_hours: float = 20.0
    provider: str = "Generic"
    rating: float = 4.5
    cost_usd: float = 0.0
    url: str | None = None
    canonical_skills: list[str] = field(default_factory=list)
    canonical_prerequisites: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CourseCatalog:
    """Manages loading, validation, and storage of course catalogues."""

    def __init__(self, courses: Sequence[CourseRecord | dict[str, Any]] | None = None) -> None:
        self._courses: dict[str, CourseRecord] = {}
        if courses:
            self.add_courses(courses)

    def add_course(self, item: CourseRecord | dict[str, Any]) -> CourseRecord:
        if isinstance(item, CourseRecord):
            record = item
        else:
            record = self._dict_to_record(item)
        self._courses[record.course_id] = record
        return record

    def add_courses(self, items: Sequence[CourseRecord | dict[str, Any]]) -> None:
        for item in items:
            self.add_course(item)

    def get_course(self, course_id: str) -> CourseRecord | None:
        return self._courses.get(course_id)

    def list_courses(self) -> list[CourseRecord]:
        return list(self._courses.values())

    def __len__(self) -> int:
        return len(self._courses)

    def __iter__(self):
        return iter(self._courses.values())

    @classmethod
    def load_from_file(cls, path: str | Path) -> CourseCatalog:
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Course catalog file not found: {file_path}")

        records: list[CourseRecord] = []
        suffix = file_path.suffix.lower()

        if suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        payload = json.loads(line)
                        records.append(cls._dict_to_record(payload))
        elif suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("courses", [])
                for item in items:
                    records.append(cls._dict_to_record(item))
        elif suffix == ".csv":
            with file_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(cls._csv_row_to_record(row))
        else:
            raise ValueError(f"Unsupported course catalog file format: {suffix}")

        return cls(records)

    def save_to_jsonl(self, path: str | Path) -> None:
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for record in self._courses.values():
                f.write(json.dumps(record.to_dict()) + "\n")

    @staticmethod
    def _split_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(item).strip() for item in val if str(item).strip()]
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return []
            if val.startswith("[") and val.endswith("]"):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
            parts = [p.strip() for p in val.replace(";", "|").replace(",", "|").split("|")]
            return [p for p in parts if p]
        return []

    @classmethod
    def _dict_to_record(cls, d: dict[str, Any]) -> CourseRecord:
        course_id = str(d.get("course_id") or d.get("id") or "").strip()
        if not course_id:
            raise ValueError(f"Course entry missing 'course_id': {d}")
        title = str(d.get("title") or d.get("course_name") or d.get("name") or course_id).strip()
        description = str(d.get("description") or "").strip()

        learning_outcomes = cls._split_list(d.get("learning_outcomes") or d.get("outcomes") or [])
        teaches_skills = cls._split_list(d.get("teaches_skills") or d.get("skills") or d.get("taught_skills") or [])
        prerequisites = cls._split_list(d.get("prerequisites") or d.get("prerequisite_skills") or [])

        level = str(d.get("level") or "intermediate").strip().lower()
        if level not in {"beginner", "intermediate", "advanced"}:
            level = "intermediate"

        duration_hours = float(d.get("duration_hours") or d.get("duration") or 20.0)
        provider = str(d.get("provider") or "Generic").strip()
        rating = float(d.get("rating") or 4.5)
        cost_usd = float(d.get("cost_usd") or d.get("cost") or 0.0)
        url = d.get("url")
        canonical_skills = cls._split_list(d.get("canonical_skills") or [])
        canonical_prerequisites = cls._split_list(d.get("canonical_prerequisites") or [])
        metadata = dict(d.get("metadata") or {})

        return CourseRecord(
            course_id=course_id,
            title=title,
            description=description,
            learning_outcomes=learning_outcomes,
            teaches_skills=teaches_skills,
            prerequisites=prerequisites,
            level=level,
            duration_hours=duration_hours,
            provider=provider,
            rating=rating,
            cost_usd=cost_usd,
            url=url,
            canonical_skills=canonical_skills,
            canonical_prerequisites=canonical_prerequisites,
            metadata=metadata,
        )

    @classmethod
    def _csv_row_to_record(cls, row: dict[str, Any]) -> CourseRecord:
        return cls._dict_to_record(dict(row))
