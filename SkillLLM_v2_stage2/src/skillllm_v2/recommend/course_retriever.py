from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Sequence

from skillllm_v2.course.ingest_catalog import CourseCatalog, CourseRecord
from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder
from skillllm_v2.graph.schema import EdgeType
from skillllm_v2.recommend.gap_vector import GapVector
from skillllm_v2.schema import normalize_text


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()) if len(t) > 1]


@dataclass(slots=True)
class RetrievedCandidate:
    course_id: str
    course: CourseRecord
    retrieval_score: float
    retrieval_sources: list[str]  # e.g. ["graph", "bm25", "semantic"]
    covered_gap_skills: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "course_id": self.course_id,
            "title": self.course.title,
            "retrieval_score": round(self.retrieval_score, 4),
            "retrieval_sources": self.retrieval_sources,
            "covered_gap_skills": self.covered_gap_skills,
        }


class CourseRetriever:
    """Hybrid candidate course retriever combining graph traversal, BM25 search, and semantic similarity."""

    def __init__(
        self,
        catalog: CourseCatalog,
        graph: CourseGraphBuilder | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.catalog = catalog
        self.graph = graph
        self.k1 = k1
        self.b = b
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        self._courses = self.catalog.list_courses()
        self._doc_ids = [c.course_id for c in self._courses]
        self._docs = []
        for c in self._courses:
            text = f"{c.title} {c.description} {' '.join(c.learning_outcomes)} {' '.join(c.teaches_skills)}"
            tokens = _tokenize(text)
            self._docs.append(tokens)

        self._doc_lens = [len(doc) for doc in self._docs]
        self._avg_doc_len = sum(self._doc_lens) / max(1, len(self._doc_lens))
        self._doc_freq: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

    def retrieve_by_graph(self, gap_vector: GapVector) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        if not self.graph:
            # Fallback to catalog teaches_skills matching
            gap_ids = set(gap_vector.vector.keys())
            for c in self._courses:
                course_skills = {normalize_text(s) for s in c.teaches_skills + c.canonical_skills}
                covered = sorted(list(gap_ids & course_skills))
                if covered:
                    score = sum(gap_vector.get_gap_weight(s) for s in covered)
                    results[c.course_id] = {
                        "score": score,
                        "covered_skills": covered,
                        "source": "graph_fallback",
                    }
            return results

        gap_skills = set(gap_vector.vector.keys())
        for gap_skill in gap_skills:
            if not self.graph.graph.has_node(gap_skill):
                continue
            # Incoming TEACHES edges to this skill
            for course_id, _, data in self.graph.graph.in_edges(gap_skill, data=True):
                if data.get("type") == EdgeType.TEACHES.value:
                    if course_id not in results:
                        results[course_id] = {
                            "score": 0.0,
                            "covered_skills": [],
                            "source": "graph",
                        }
                    gap_w = gap_vector.get_gap_weight(gap_skill)
                    depth_mult = 1.0 if data.get("depth") != "introductory" else 0.8
                    results[course_id]["score"] += gap_w * depth_mult * float(data.get("confidence", 1.0))
                    results[course_id]["covered_skills"].append(gap_skill)

        return results

    def retrieve_by_bm25(self, query: str, top_k: int = 15) -> dict[str, float]:
        query_terms = _tokenize(query)
        if not query_terms or not self._doc_ids:
            return {}

        n_docs = len(self._doc_ids)
        scores: dict[str, float] = {}

        for i, (doc_id, tokens) in enumerate(zip(self._doc_ids, self._docs)):
            doc_len = self._doc_lens[i]
            score = 0.0
            term_counts: dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term in query_terms:
                df = self._doc_freq.get(term, 0)
                if df == 0:
                    continue
                tf = term_counts.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self._avg_doc_len)))
                score += idf * (tf * (self.k1 + 1.0)) / max(1e-6, denom)

            if score > 0.0:
                scores[doc_id] = score

        # Normalize BM25 scores to [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        return {k: v / max_score for k, v in scores.items()}

    def retrieve_by_semantic(self, target_job: dict[str, Any], gap_vector: GapVector) -> dict[str, float]:
        """Calculates semantic overlap between course outcomes and target job / gaps."""
        scores: dict[str, float] = {}
        target_tokens = set(_tokenize(f"{target_job.get('title', '')} {' '.join(g.skill_name for g in gap_vector.gaps)}"))
        if not target_tokens:
            return scores

        for c in self._courses:
            outcome_text = f"{c.title} {' '.join(c.learning_outcomes)}"
            outcome_tokens = set(_tokenize(outcome_text))
            intersection = target_tokens & outcome_tokens
            if intersection:
                jaccard = len(intersection) / len(target_tokens | outcome_tokens)
                scores[c.course_id] = min(1.0, jaccard * 3.0)  # scaled jaccard
        return scores

    def retrieve(
        self,
        gap_vector: GapVector,
        target_job: dict[str, Any],
        top_k: int = 15,
        graph_weight: float = 0.6,
        bm25_weight: float = 0.25,
        semantic_weight: float = 0.15,
    ) -> list[RetrievedCandidate]:
        """Executes hybrid retrieval combining graph, BM25, and semantic candidate sets."""
        graph_matches = self.retrieve_by_graph(gap_vector)
        query = f"{target_job.get('title', '')} {' '.join(g.skill_name for g in gap_vector.gaps)}"
        bm25_matches = self.retrieve_by_bm25(query)
        semantic_matches = self.retrieve_by_semantic(target_job, gap_vector)

        all_ids = set(graph_matches.keys()) | set(bm25_matches.keys()) | set(semantic_matches.keys())
        candidates: list[RetrievedCandidate] = []

        max_graph_score = max((m["score"] for m in graph_matches.values()), default=1.0)
        max_graph_score = max(max_graph_score, 1e-6)

        for cid in all_ids:
            course = self.catalog.get_course(cid)
            if not course:
                continue

            sources = []
            g_score = 0.0
            covered = []
            if cid in graph_matches:
                g_score = graph_matches[cid]["score"] / max_graph_score
                sources.append("graph")
                covered = graph_matches[cid]["covered_skills"]

            b_score = 0.0
            if cid in bm25_matches:
                b_score = bm25_matches[cid]
                sources.append("bm25")

            s_score = 0.0
            if cid in semantic_matches:
                s_score = semantic_matches[cid]
                sources.append("semantic")

            combined_score = (
                graph_weight * g_score +
                bm25_weight * b_score +
                semantic_weight * s_score
            )

            candidates.append(
                RetrievedCandidate(
                    course_id=cid,
                    course=course,
                    retrieval_score=combined_score,
                    retrieval_sources=sources,
                    covered_gap_skills=covered,
                )
            )

        candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        return candidates[:top_k]
