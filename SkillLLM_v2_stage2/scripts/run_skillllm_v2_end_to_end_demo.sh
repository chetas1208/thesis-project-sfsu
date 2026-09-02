#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from skillllm_v2.match import CourseRecommender, GapDetector, JobMatcher
from skillllm_v2.pipeline import SkillLLMv2Pipeline

text = "I use Python, SQL, Docker, and communication skills to build machine learning services."
skills = SkillLLMv2Pipeline().extract(text)
profile = {"skills": [skill.get("canonical_skill_name") or skill["surface_form"] for skill in skills]}
job = {"job_id": "ml-platform-engineer", "required_skills": ["Python", "SQL", "Amazon Web Services"], "preferred_skills": ["Docker", "communication"], "semantic_role_fit": 0.8, "experience_level_fit": 0.7, "user_preference_fit": 0.8}
match = JobMatcher().score(profile, job)
gaps = GapDetector().detect(profile, job)
courses = CourseRecommender().recommend(gaps, [{"course_id": "aws-101", "course_name": "AWS Foundations", "teaches_skills": ["Amazon Web Services"], "prerequisite_skills": ["Python"]}], profile)
payload = {"input_text": text, "skills": skills, "profile": profile, "job_match": match, "gaps": gaps, "course_recommendations": courses}
Path("reports/skillllm_v2/end_to_end_demo.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
echo "Demo output: reports/skillllm_v2/end_to_end_demo.json"
