import json
import re
from pathlib import Path

STAGE2_DIR = Path(__file__).resolve().parent.parent
PARSED_DIR = STAGE2_DIR / "inputs" / "syllabi_parsed"
OUT_FILE = STAGE2_DIR / "inputs" / "courses" / "sfsu_catalog.jsonl"

def add_unique(target, items):
    """Append items to target, skipping case-insensitive duplicates.Such as the Case of Multiple instances of CSC 220"""
    seen = {t.lower() for t in target}
    for item in items:
        if item and item.lower() not in seen:
            target.append(item)
            seen.add(item.lower())



def course_difficulty_determinant(course):
    number = int(re.search(r"\d+",course).group())
    if number<300:
        return "beginner"
    if number<600:
        return "intermediate"
    return "advanced"


def main():
    # 1. SET UP: one entry per course, keyed by course_id so syllabi of the same course merge
    courses = {}

    # 2. LOOP: read every parsed syllabus and merge it into its course
    for path in sorted(PARSED_DIR.glob("*.json")):
        extracted = json.loads(path.read_text(encoding="utf-8"))["extracted"]  # 2a LOAD the LLM's answers
        course_id = extracted["course_id"]                                      # 2b IDENTIFY the course

        # 2c CREATE: first syllabus of this course -> basics now, lists filled in 2d
        if course_id not in courses:
            courses[course_id] = {
                "course_id": course_id,
                "title": extracted.get("course_title") or course_id,
                "learning_outcomes": [],
                "teaches_skills": [],
                "topics": [],
                "prerequisites": [],  # recommender expects skills; ours are courses, kept in metadata
                "level": course_difficulty_determinant(course_id),
                "provider": "SFSU",
                "metadata": {"prereq_courses": [], "source_files": []},
            }

        # 2d MERGE: add this syllabus's data, skipping repeats
        entry = courses[course_id]
        add_unique(entry["learning_outcomes"], extracted.get("learning_outcomes") or [])
        add_unique(entry["topics"], extracted.get("topics") or [])
        add_unique(entry["teaches_skills"], [s["skill"] for s in extracted.get("skills") or []])
        # prerequisites are saved as groups ({"any_of": [...]}): list every course mentioned
        add_unique(entry["metadata"]["prereq_courses"],
                   [c for p in extracted.get("prerequisites") or [] for c in p.get("any_of") or [p.get("course")]])
        entry["metadata"]["source_files"].append(path.stem)

    # 3. WRITE: one JSON line per course, the format CourseCatalog.load_from_file reads
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as handle:
        for entry in courses.values():
            entry["description"] = "; ".join(entry.pop("topics"))  # 3a FINISH: BM25 search reads description
            handle.write(json.dumps(entry) + "\n")                  # 3b SAVE
            print(f"{entry['course_id']:10} {len(entry['metadata']['source_files'])} syllabi  "
                  f"{len(entry['teaches_skills']):3} skills  {entry['level']}")

    # 4. REPORT
    print(f"\n{len(courses)} courses -> {OUT_FILE}")


if __name__ == "__main__":
    main()
