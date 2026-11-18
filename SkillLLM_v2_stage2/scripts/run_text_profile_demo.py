"""Demo of the handoff: a user's own description -> user-text parser -> skills -> course recommender.

    python scripts/llm_extract_user_text.py --text "I have 3 years of Java and know some Python." --name alex
    python scripts/run_text_profile_demo.py alex

The parser's output (inputs/users_parsed/<name>.json) is handed to the course recommender as the
user's profile; the recommender builds the graph, finds the gaps for the target job, and recommends
SFSU courses. Same target job as run_sfsu_demo.py.
"""
import json
import sys
import warnings
from pathlib import Path

STAGE2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STAGE2_DIR / "src"))
sys.path.insert(0, str(STAGE2_DIR / "scripts"))

from llm_extract_user_text import OUT_DIR, to_profile  # noqa: E402
from run_sfsu_demo import JOB  # noqa: E402

CATALOG = STAGE2_DIR / "inputs" / "courses" / "sfsu_catalog.jsonl"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings("ignore")
    name = sys.argv[1] if len(sys.argv) > 1 else "user"
    path = OUT_DIR / f"{name}.json"
    if not path.exists():
        sys.exit(f"No parsed user '{name}'. Run llm_extract_user_text.py --name {name} first.")
    record = json.loads(path.read_text(encoding="utf-8"))

    # 1. The parser's output for this user
    print(f"User text: {record['text']}\n")
    for s in record["extracted"]["skills"]:
        print(f"  {s['surface_form']:<22} {s['mastery_level']:<13} ({s['confidence']})  \"{s['evidence']}\"")
    if record["negated"]:
        print(f"  not counted (user said no): {', '.join(n['skill'] for n in record['negated'])}")
    if record["dropped_unverified"]:
        print(f"  dropped (quote not in text): {', '.join(d['skill'] for d in record['dropped_unverified'])}")

    # 2. Hand off to the course recommender
    from skillllm_v2.pipeline.course_recommendation_pipeline import CourseRecommendationPipeline
    result = CourseRecommendationPipeline(catalog_path=CATALOG).run(
        user_input=to_profile(record), job_input=JOB, top_k=5)

    print(f"\nTarget job: {JOB['title']}")
    for gap in result["skill_gap_analysis"]["gaps"]:
        print(f"  gap: {gap['skill_name']} ({gap['gap_type']})")
    print("Recommended courses:")
    for r in result["recommendations"]:
        print(f"  #{r['rank']} {r['title']} - covers {', '.join(r['covered_missing_skills'])}")


if __name__ == "__main__":
    main()
