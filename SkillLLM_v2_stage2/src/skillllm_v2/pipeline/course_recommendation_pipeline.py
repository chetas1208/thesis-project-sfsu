from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from skillllm_v2.course.extract_course_skills import CourseSkillExtractor
from skillllm_v2.course.ingest_catalog import CourseCatalog
from skillllm_v2.course.prerequisite_parser import PrerequisiteParser
from skillllm_v2.extract.dictionary_extractor import DictionaryExtractor
from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker
from skillllm_v2.recommend.config import DEFAULT_CONFIG, RecommenderConfig
from skillllm_v2.recommend.course_ranker import CourseRanker
from skillllm_v2.recommend.course_retriever import CourseRetriever
from skillllm_v2.recommend.explanation_generator import CourseExplanationGenerator
from skillllm_v2.recommend.gap_vector import GapEngine
from skillllm_v2.recommend.learning_path_planner import LearningPathPlanner
from skillllm_v2.schema import normalize_text
from skillllm_v2.wrapper import SkillLLMv2Pipeline


class CourseRecommendationPipeline:
    """End-to-end Course Recommendation Pipeline for SkillLLM_v2.

    Ingestion -> SkillLLM extraction -> Canonical linking ->
    Knowledge graph construction -> Gap vector computation ->
    Hybrid retrieval -> Weighted ranking -> Learning path planning ->
    Evidence-backed explanation generation.
    """

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        linker: TaxonomyLinker | None = None,
        pipeline: SkillLLMv2Pipeline | None = None,
        config: RecommenderConfig | None = None,
    ) -> None:
        self.linker = linker or TaxonomyLinker()
        alias_file = Path("data/skillllm_v2/taxonomy/skill_aliases.jsonl")
        dict_extractor = DictionaryExtractor(alias_paths=[str(alias_file)] if alias_file.exists() else None)
        self.extractor_pipeline = pipeline or SkillLLMv2Pipeline(dictionary=dict_extractor, linker=self.linker)
        self.prereq_parser = PrerequisiteParser()
        self.course_extractor = CourseSkillExtractor(
            pipeline=self.extractor_pipeline,
            linker=self.linker,
            prerequisite_parser=self.prereq_parser,
        )
        self.gap_engine = GapEngine()
        self.config = config or DEFAULT_CONFIG
        self.ranker = CourseRanker(config=self.config)
        self.path_planner = LearningPathPlanner()
        self.explainer = CourseExplanationGenerator()

        default_cat = Path("inputs/courses/course_catalog.jsonl")
        self.catalog_path = Path(catalog_path) if catalog_path else default_cat
        self.catalog = CourseCatalog.load_from_file(self.catalog_path)
        self.enriched_courses = self.course_extractor.extract_from_catalog(self.catalog)

        # Build Graph
        self.graph = CourseGraphBuilder()
        for ec in self.enriched_courses:
            self.graph.ingest_course(ec)

    def set_config(self, config: RecommenderConfig) -> None:
        """Swap the recommender configuration (used by the tuning harness; no state besides the ranker changes)."""
        self.config = config
        self.ranker = CourseRanker(config=config)

    def run(
        self,
        user_input: str | dict[str, Any],
        job_input: str | dict[str, Any],
        top_k: int = 5,
    ) -> dict[str, Any]:
        # 1. Process user profile
        if isinstance(user_input, str):
            extracted_user = self.extractor_pipeline.extract(user_input)
            user_profile = {
                "user_id": "demo_user",
                "skills": extracted_user,
                "target_roles": [],
            }
        else:
            user_profile = dict(user_input)
            # If raw skills strings provided, link them
            if "skills" in user_profile and user_profile["skills"]:
                if isinstance(user_profile["skills"][0], str):
                    linked = []
                    for s in user_profile["skills"]:
                        match = self.linker.link(s)
                        linked.append({
                            "surface_form": s,
                            "canonical_skill_id": match["canonical_skill_id"] if match else normalize_text(s),
                            "canonical_skill_name": match["canonical_skill_name"] if match else s,
                            "confidence": 1.0,
                            "evidence": f"Declared user profile skill: {s}",
                            "source": "profile",
                        })
                    user_profile["skills"] = linked

        # Ingest user into graph
        self.graph.ingest_user_profile(user_profile)

        # 2. Process job description
        if isinstance(job_input, str):
            extracted_job = self.extractor_pipeline.extract(job_input)
            job = {
                "job_id": "target_role",
                "title": "Target Role",
                "required_skills": [c["surface_form"] for c in extracted_job if c.get("label") == "HARD_SKILL"],
                "preferred_skills": [c["surface_form"] for c in extracted_job if c.get("label") == "SOFT_SKILL"],
            }
        else:
            job = dict(job_input)

        # Ingest job into graph
        self.graph.ingest_job(job)

        # 3. Compute quantitative Gap Vector
        gap_vector = self.gap_engine.compute(user_profile, job, graph=self.graph)

        # 4. Candidate Retrieval
        retriever = CourseRetriever(catalog=self.catalog, graph=self.graph)
        candidates = retriever.retrieve(gap_vector=gap_vector, target_job=job, top_k=self.config.retrieval_pool)
        candidate_records = [c.course for c in candidates]

        # 5. Rule-weighted Course Ranking
        ranked_courses = self.ranker.rank(
            candidates=candidate_records,
            user_profile=user_profile,
            target_job=job,
            gap_vector=gap_vector,
            graph=self.graph,
            top_k=top_k,
            catalog_courses=self.catalog.list_courses(),
        )

        # 6. Learning Path Planning
        learning_path = self.path_planner.plan(
            recommended_courses=ranked_courses,
            user_profile=user_profile,
            target_job=job,
            gap_vector=gap_vector,
            graph=self.graph,
        )

        # 7. Minimum JSON Output Explanations
        recommendations = self.explainer.generate_recommendations_payload(
            ranked_courses=ranked_courses,
            user_profile=user_profile,
            target_job=job,
            gap_vector=gap_vector,
            graph=self.graph,
        )

        return {
            "user_profile": {
                "user_id": user_profile.get("user_id", "demo_user"),
                "known_skills_count": len(user_profile.get("skills", [])),
                "skills": [s.get("canonical_skill_name") or s.get("surface_form") for s in user_profile.get("skills", [])],
                "target_roles": user_profile.get("target_roles", []),
            },
            "target_job": {
                "job_id": job.get("job_id", "target_job"),
                "title": job.get("title", job.get("job_id")),
                "required_skills": job.get("required_skills", []),
                "preferred_skills": job.get("preferred_skills", []),
            },
            "skill_gap_analysis": gap_vector.to_dict(),
            "recommendations": recommendations,
            "learning_path": learning_path.to_dict(),
            "pipeline_metadata": {
                "total_catalog_courses": len(self.catalog),
                "graph_nodes": self.graph.graph.number_of_nodes(),
                "graph_edges": self.graph.graph.number_of_edges(),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillLLM_v2 Course Recommendation Pipeline")
    parser.add_argument("--catalog", default="inputs/courses/course_catalog.jsonl", help="Path to course catalog")
    parser.add_argument("--out", default="reports/skillllm_v2/demo/course_recommendations.json", help="Output path")
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()

    pipeline = CourseRecommendationPipeline(catalog_path=args.catalog)

    # Demo scenario: ML Platform Engineer target role
    user_text = "I have strong experience with Python, SQL, Git, and Linux administration."
    target_job = {
        "job_id": "ml-platform-engineer",
        "title": "Machine Learning Platform Engineer",
        "required_skills": [
            {"skill_name": "Python", "importance": 1.0, "source_sentence": "Must have senior-level Python engineering skills."},
            {"skill_name": "Docker", "importance": 1.0, "source_sentence": "Extensive experience containerizing distributed applications with Docker."},
            {"skill_name": "Machine Learning", "importance": 1.0, "source_sentence": "Understanding of machine learning workflows and scikit-learn models."},
            {"skill_name": "Amazon Web Services", "importance": 0.9, "source_sentence": "Hands-on experience deploying infrastructure on AWS."},
        ],
        "preferred_skills": [
            {"skill_name": "Kubernetes", "importance": 0.8, "source_sentence": "Kubernetes cluster orchestration preferred."},
            {"skill_name": "CI/CD", "importance": 0.7, "source_sentence": "Experience designing CI/CD automation pipelines."},
        ],
    }

    result = pipeline.run(user_input=user_text, job_input=target_job, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Course recommendation successfully written to: {out_path}")
    print(f"Total recommendations: {len(result['recommendations'])}")
    for r in result["recommendations"]:
        print(f"Rank {r['rank']}: {r['title']} (Score: {r['score']:.3f})")
        print(f"  Covers: {', '.join(r['covered_missing_skills'])}")
        print(f"  Prerequisites: {r['prerequisite_status']['explanation']}")
        print(f"  Explanation: {r['explanation_path'][0]}")


if __name__ == "__main__":
    main()
