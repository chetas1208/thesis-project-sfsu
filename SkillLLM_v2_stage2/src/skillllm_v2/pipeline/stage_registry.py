from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Stage:
    stage_name: str
    system_name: str
    inputs: list[str]
    outputs: list[str]
    required_before: list[str]
    metrics_file: str | None
    runner: Callable[..., dict]
    optional: bool = False


STAGE_ORDER = [
    "setup_stage2_repo",
    "legacy_cleanup_check",
    "schema_smoke",
    "component_unit_tests",
    "build_skillspan_bio_dataset",
    "train_bert_baseline",
    "eval_bert_baseline",
    "train_or_run_gliner_baseline",
    "eval_gliner_baseline",
    "build_mistral_sft_data",
    "train_or_verify_mistral_lora",
    "eval_raw_mistral_lora",
    "eval_mistral_boundary_refiner",
    "extraction_component_smoke",
    "merge_refine_verify_smoke",
    "taxonomy_linking_smoke",
    "eval_mistral_gliner_taxonomy",
    "build_verifier_dataset",
    "train_span_verifier",
    "verifier_eval_smoke",
    "build_linker_dataset",
    "train_taxonomy_reranker",
    "graph_smoke",
    "matching_recommendation_smoke",
    "eval_full_cag_graph",
    "extraction_ablation_full",
    "graph_matching_eval_full",
    "money_table_generation",
    "end_to_end_demo_smoke",
    "final_report_generation",
    "master_merge_readiness_check",
]


def build_registry() -> dict[str, Stage]:
    from skillllm_v2.pipeline import smoke_runner

    def stage(name: str, system: str, runner, outputs: list[str] | None = None, required: list[str] | None = None, metrics: str | None = None) -> Stage:
        return Stage(name, system, [], outputs or [], required or [], metrics, runner, False)

    return {
        "setup_stage2_repo": stage("setup_stage2_repo", "pipeline", smoke_runner.setup_stage2_repo),
        "legacy_cleanup_check": stage("legacy_cleanup_check", "pipeline", smoke_runner.legacy_cleanup_check, required=["setup_stage2_repo"]),
        "schema_smoke": stage("schema_smoke", "pipeline", smoke_runner.schema_smoke, required=["legacy_cleanup_check"]),
        "component_unit_tests": stage("component_unit_tests", "pipeline", smoke_runner.component_unit_tests, required=["schema_smoke"]),
        "build_skillspan_bio_dataset": stage("build_skillspan_bio_dataset", "BERT baseline", smoke_runner.build_skillspan_bio_dataset, outputs=["data/skillllm_v2/bio/test.jsonl"], required=["component_unit_tests"]),
        "train_bert_baseline": stage("train_bert_baseline", "BERT baseline", smoke_runner.train_bert_baseline, outputs=["artifacts/skillllm_v2/bert_baseline/model/model.json"], required=["build_skillspan_bio_dataset"]),
        "eval_bert_baseline": stage("eval_bert_baseline", "BERT baseline", smoke_runner.eval_bert_baseline, outputs=["artifacts/skillllm_v2/bert_baseline/metrics.json"], required=["train_bert_baseline"], metrics="artifacts/skillllm_v2/bert_baseline/metrics.json"),
        "train_or_run_gliner_baseline": stage("train_or_run_gliner_baseline", "GLiNER", smoke_runner.train_or_run_gliner_baseline, outputs=["artifacts/skillllm_v2/gliner_baseline/model/config.json"], required=["component_unit_tests"]),
        "eval_gliner_baseline": stage("eval_gliner_baseline", "GLiNER", smoke_runner.eval_gliner_baseline, outputs=["artifacts/skillllm_v2/gliner_baseline/metrics.json"], required=["train_or_run_gliner_baseline"], metrics="artifacts/skillllm_v2/gliner_baseline/metrics.json"),
        "build_mistral_sft_data": stage("build_mistral_sft_data", "Raw Mistral LoRA", smoke_runner.build_mistral_sft_data, outputs=["data/skillllm_v2/mistral_sft/train.jsonl"], required=["component_unit_tests"]),
        "train_or_verify_mistral_lora": stage("train_or_verify_mistral_lora", "Raw Mistral LoRA", smoke_runner.train_or_verify_mistral_lora, outputs=["artifacts/skillllm_v2/mistral_lora/model/adapter_status.json"], required=["build_mistral_sft_data"]),
        "eval_raw_mistral_lora": stage("eval_raw_mistral_lora", "Raw Mistral LoRA", smoke_runner.eval_raw_mistral_lora, outputs=["artifacts/skillllm_v2/mistral_lora/metrics.json"], required=["train_or_verify_mistral_lora"], metrics="artifacts/skillllm_v2/mistral_lora/metrics.json"),
        "eval_mistral_boundary_refiner": stage("eval_mistral_boundary_refiner", "Mistral + Boundary Refiner", smoke_runner.eval_mistral_boundary_refiner, outputs=["artifacts/skillllm_v2/mistral_boundary/metrics.json"], required=["eval_raw_mistral_lora"], metrics="artifacts/skillllm_v2/mistral_boundary/metrics.json"),
        "extraction_component_smoke": stage("extraction_component_smoke", "components", smoke_runner.extraction_component_smoke, required=["eval_mistral_boundary_refiner"]),
        "merge_refine_verify_smoke": stage("merge_refine_verify_smoke", "components", smoke_runner.merge_refine_verify_smoke, required=["extraction_component_smoke"]),
        "taxonomy_linking_smoke": stage("taxonomy_linking_smoke", "taxonomy", smoke_runner.taxonomy_linking_smoke, required=["merge_refine_verify_smoke"]),
        "eval_mistral_gliner_taxonomy": stage("eval_mistral_gliner_taxonomy", "Mistral + GLiNER + Taxonomy", smoke_runner.eval_mistral_gliner_taxonomy, outputs=["artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json"], required=["taxonomy_linking_smoke"], metrics="artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json"),
        "build_verifier_dataset": stage("build_verifier_dataset", "SkillLLM_v2 Full CAG/Graph", smoke_runner.build_verifier_dataset, outputs=["data/skillllm_v2/verifier_train.jsonl"], required=["eval_mistral_gliner_taxonomy"]),
        "train_span_verifier": stage("train_span_verifier", "SkillLLM_v2 Full CAG/Graph", smoke_runner.train_span_verifier, outputs=["artifacts/skillllm_v2/span_verifier/verifier_config.json"], required=["build_verifier_dataset"]),
        "verifier_eval_smoke": stage("verifier_eval_smoke", "SkillLLM_v2 Full CAG/Graph", smoke_runner.verifier_eval_smoke, required=["train_span_verifier"]),
        "build_linker_dataset": stage("build_linker_dataset", "SkillLLM_v2 Full CAG/Graph", smoke_runner.build_linker_dataset, outputs=["data/skillllm_v2/linker_dev.jsonl"], required=["taxonomy_linking_smoke"]),
        "train_taxonomy_reranker": stage("train_taxonomy_reranker", "SkillLLM_v2 Full CAG/Graph", smoke_runner.train_taxonomy_reranker, outputs=["artifacts/skillllm_v2/taxonomy_reranker/README.md"], required=["build_linker_dataset"]),
        "graph_smoke": stage("graph_smoke", "SkillLLM_v2 Full CAG/Graph", smoke_runner.graph_smoke, required=["train_taxonomy_reranker"]),
        "matching_recommendation_smoke": stage("matching_recommendation_smoke", "SkillLLM_v2 Full CAG/Graph", smoke_runner.matching_recommendation_smoke, required=["graph_smoke"]),
        "eval_full_cag_graph": stage("eval_full_cag_graph", "SkillLLM_v2 Full CAG/Graph", smoke_runner.eval_full_cag_graph, outputs=["artifacts/skillllm_v2/full_cag_graph/metrics.json"], required=["matching_recommendation_smoke"], metrics="artifacts/skillllm_v2/full_cag_graph/metrics.json"),
        "extraction_ablation_full": stage("extraction_ablation_full", "money table", smoke_runner.extraction_ablation_full, outputs=["reports/skillllm_v2/extraction_ablation.csv"], required=["eval_full_cag_graph"]),
        "graph_matching_eval_full": stage("graph_matching_eval_full", "money table", smoke_runner.graph_matching_eval_full, outputs=["reports/skillllm_v2/matching_eval.csv"], required=["eval_full_cag_graph"]),
        "money_table_generation": stage("money_table_generation", "money table", smoke_runner.money_table_generation, outputs=["reports/skillllm_v2/money_table.csv"], required=["extraction_ablation_full", "graph_matching_eval_full"]),
        "end_to_end_demo_smoke": stage("end_to_end_demo_smoke", "demo", smoke_runner.end_to_end_demo_smoke, outputs=["reports/skillllm_v2/demo/explanation.md"], required=["money_table_generation"]),
        "final_report_generation": stage("final_report_generation", "reports", smoke_runner.final_report_generation, outputs=["reports/skillllm_v2/thesis_results_summary.md"], required=["end_to_end_demo_smoke"]),
        "master_merge_readiness_check": stage("master_merge_readiness_check", "merge", smoke_runner.master_merge_readiness_check, required=["final_report_generation"]),
    }
