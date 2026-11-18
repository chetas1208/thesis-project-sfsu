"""Demo of the handoff: a student's syllabi -> syllabus parser -> skills -> course recommender.

    python scripts/run_sfsu_demo.py                          # CSC 101, 215, 220
    python scripts/run_sfsu_demo.py CSC101 CSC215 CSC220     # syllabi by part of their file name

The parser's output (inputs/syllabi_parsed/, from llm_extract_syllabi.py) is handed to the course
recommender as the student's skills; the recommender builds the profile, finds the gaps for the
target job, and recommends courses.
"""
import json
import sys
import warnings
from pathlib import Path

STAGE2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2_DIR / "src"))
PARSED_DIR = STAGE2_DIR / "inputs" / "syllabi_parsed"

JOB = {
    "job_id": "backend-swe-intern",
    "title": "Backend Software Engineer Intern",
    "required_skills": ["Java", "Object-Oriented Programming", "Data Structures", "SQL", "Git"],
    "preferred_skills": ["Docker", "Unit Testing", "Linux"],
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    names = sys.argv[1:] or ["CSC101", "CSC215", "CSC220_DataStructures_Fall24"]

    # 1. The parser's output for each syllabus the student took
    skills = []
    for name in names:
        path = next(PARSED_DIR.glob(f"*{name}*.json"), None)
        if path is None:
            sys.exit(f"No parsed syllabus matching '{name}'. Run llm_extract_syllabi.py on it first.")
        extracted = json.loads(path.read_text(encoding="utf-8"))["extracted"]
        course_skills = [s["skill"] for s in extracted.get("skills") or []]
        print(f"{extracted['course_id']}: {', '.join(course_skills)}")
        skills += [s for s in course_skills if s not in skills]

    # 2. Hand off to the course recommender
    from skillllm_v2.pipeline.course_recommendation_pipeline import CourseRecommendationPipeline
    result = CourseRecommendationPipeline().run(
        user_input={"user_id": "demo_student", "skills": skills}, job_input=JOB, top_k=5)

    print(f"\nTarget job: {JOB['title']}")
    for gap in result["skill_gap_analysis"]["gaps"]:
        print(f"  gap: {gap['skill_name']} ({gap['gap_type']})")
    print("Recommended courses:")
    for r in result["recommendations"]:
        print(f"  #{r['rank']} {r['title']} - covers {', '.join(r['covered_missing_skills'])}")


if __name__ == "__main__":
    main()
