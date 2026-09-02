# SkillLLM_v2 Architecture

SkillLLM_v2 Stage-2 is a Context-Aware Graph-Augmented Skill Intelligence System.

## Why Mistral Is The Main Extractor

The completed SkillSpan benchmark selected Mistral Small 3.1 24B LoRA because it achieved the best completed exact span, token, and partial span F1 among feasible local models. Larger Gemma 26B attempts were not hardware-feasible on 2x24GB GPUs.

## Why Old Multi-Model Experiments Were Removed

The thesis system no longer needs active Qwen, Gemma, and Llama training or inference paths. Their results are retained as legacy benchmark evidence, while active code is centered on the final system behavior.

## Extraction Design

The wrapper uses candidate union:

1. Mistral LoRA extracts strict JSON skill spans.
2. GLiNER adds lightweight recall candidates.
3. Dictionary/taxonomy matching adds traceable aliases.
4. The merger deduplicates exact and alias-equivalent spans while preserving provenance.
5. Boundary refinement improves exact-span behavior.
6. Verification removes generic or unsupported spans.
7. Confidence calibration applies source-aware thresholds.

Every skill keeps surface form, label, offsets, source, confidence, evidence, canonical IDs, and metadata.

## Normalization Design

The taxonomy linker supports ESCO, O*NET, Lightcast/OpenSkills-style, and custom alias files. It starts with exact alias and normalized string matching, then falls back to fuzzy matching. BM25, dense retrieval, and reranking are explicit extension points.

## Graph Reasoning

The first graph implementation uses NetworkX with node types User, Profile, Job, Course, Skill, SkillCluster, and Occupation. Edges represent skill ownership, job requirements, course coverage, prerequisites, similarity, and cluster membership. The interfaces are intentionally simple enough to migrate to Neo4j later.

## CAG vs RAG vs Graph

CAG is used for small stable context: user profile summaries, course catalogues, and taxonomy aliases. RAG is reserved for large or changing data such as scraped job descriptions and live postings. Graph retrieval handles relational reasoning over user skills, jobs, missing skills, and courses.

## Matching And Recommendations

The job matcher scores required skill coverage, preferred skill coverage, role fit, experience fit, user preference fit, and critical missing skill penalties. The gap detector classifies missing critical, missing preferred, weak, outdated, uncertain, and transferable skill gaps. The course recommender scores gap coverage, prerequisite fit, level fit, goal alignment, availability, and workload penalties.

## Evaluation Plan

Extraction evaluation reports exact span F1, partial span F1, token F1, HARD_SKILL F1, SOFT_SKILL F1, false positive rate, unaligned rate, latency, and memory when available. Additional reports cover taxonomy linking, matching, course recommendation, smoke results, and error analysis.

## Thesis Research Questions

- Does hybrid candidate union improve recall without damaging exact span precision?
- Does boundary refinement improve exact span F1 while preserving token F1?
- Does taxonomy normalization improve downstream job matching and course recommendation?
- Does graph retrieval produce more explainable skill-gap recommendations than flat retrieval alone?
