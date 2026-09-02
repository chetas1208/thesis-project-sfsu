#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p reports/skillllm_v2

LIMIT="${1:-10}"
PYTHONPATH=src python - "$LIMIT" <<'PY'
import json
import sys
from pathlib import Path

from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.match import CourseRecommender, GapDetector, JobMatcher
from skillllm_v2.pipeline import SkillLLMv2Pipeline
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split

limit = int(sys.argv[1])
docs = [doc for doc in load_skillspan_split(Path("data/source/skillspan_hf"), "dev", limit=None) if doc["gold_spans"]][:limit]
pipeline = SkillLLMv2Pipeline()
out_path = Path("reports/skillllm_v2/smoke_output.jsonl")
gold, pred = [], []
with out_path.open("w", encoding="utf-8") as handle:
    samples = docs[:limit]
    if not any(doc["gold_spans"] for doc in samples):
        samples.append({"id": "synthetic-smoke", "text": "Python, SQL, machine learning, AWS, communication, and leadership are required.", "gold_spans": []})
    for doc in samples:
        skills = pipeline.extract(doc["text"])
        handle.write(json.dumps({"id": doc["id"], "text": doc["text"], "skills": skills}) + "\n")
        gold.append(doc.get("gold_spans", []))
        pred.append(skills)

metrics = evaluate_extraction(gold, pred)
profile = {"skills": ["Python", "SQL"], "weak_skills": ["SQL"]}
job = {"job_id": "job-smoke", "required_skills": ["Python", "AWS"], "preferred_skills": ["SQL"], "semantic_role_fit": 0.7, "experience_level_fit": 0.8, "user_preference_fit": 0.7}
match = JobMatcher().score(profile, job)
gaps = GapDetector().detect(profile, job)
courses = CourseRecommender().recommend(gaps, [{"course_id": "c-aws", "course_name": "AWS Foundations", "teaches_skills": ["AWS"], "prerequisite_skills": ["Python"]}], profile)
payload = {"limit": limit, "documents": len(pred), "metrics": metrics, "job_match": match, "gaps": gaps, "courses": courses}
Path("reports/skillllm_v2/smoke_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Smoke output: reports/skillllm_v2/smoke_results.json"
