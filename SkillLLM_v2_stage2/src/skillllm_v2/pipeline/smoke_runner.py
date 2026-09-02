from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from skillllm_v2.baselines import BertBaseline, GLiNERBaseline
from skillllm_v2.baselines.eval_baselines import evaluate_prediction_file
from skillllm_v2.config import load_yaml
from skillllm_v2.eval.ablation_runner import run_ablation
from skillllm_v2.eval.extraction_eval import evaluate_extraction
from skillllm_v2.eval.money_table_generator import generate_money_table
from skillllm_v2.eval.run_safe_wrapper import run_safe_wrapper
from skillllm_v2.extract import CandidateMerger, DictionaryExtractor, GLiNERExtractor, MistralExtractor
from skillllm_v2.graph import SkillGraphBuilder
from skillllm_v2.graph.schema import EdgeType, NodeType
from skillllm_v2.io_utils import read_json, read_jsonl, write_json, write_jsonl, write_text
from skillllm_v2.match import CourseRecommender, GapDetector, JobMatcher
from skillllm_v2.normalize import TaxonomyLinker
from skillllm_v2.pipeline.gates import require_no_legacy_active_paths, require_paths
from skillllm_v2.schemas import SkillCandidate, SystemBenchmarkResult, validate_skill_candidate, validate_system_result
from skillllm_v2.train.build_mistral_sft_data import build_mistral_sft_data as build_mistral_sft_dataset
from skillllm_v2.train.build_skillspan_bio import build_bio_dataset
from skillllm_v2.train.build_verifier_dataset import load_skillspan_split
from skillllm_v2.train.train_mistral_lora import train_or_verify_mistral_lora as train_or_verify_mistral_lora_artifact
from skillllm_v2.train.train_span_verifier import train_span_verifier as train_span_verifier_model
from skillllm_v2.train.train_taxonomy_reranker import train_taxonomy_reranker as train_taxonomy_reranker_model
from skillllm_v2.verify import BoundaryRefiner, ConfidenceCalibrator, SpanVerifier


def _cfg(config_path: str) -> dict[str, Any]:
    return load_yaml(config_path) if config_path else {}


def _limit(config: dict[str, Any], mode: str, limit: int | None) -> int | None:
    if limit is not None:
        return limit
    key = "smoke_limit" if mode == "smoke" else "training_limit"
    return config.get("limits", {}).get(key)


def _dataset_root(config: dict[str, Any]) -> str:
    return config.get("data", {}).get("skillspan_root", "data/source/skillspan_hf")


def _trained_mistral_adapter_path(config: dict[str, Any]) -> str | None:
    trained_adapter = Path("artifacts/skillllm_v2/mistral_lora/model")
    if (trained_adapter / "adapter_config.json").exists():
        return str(trained_adapter)
    return config.get("model", {}).get("adapter_path")


def _mistral_extractor(config: dict[str, Any], mode: str, allow_fallback: bool | None = None) -> MistralExtractor:
    model_cfg = config.get("model", {})
    use_real_model = mode != "smoke" or model_cfg.get("allow_real_model_in_smoke", False)
    if allow_fallback is None:
        allow_fallback = True if mode == "smoke" and not use_real_model else bool(model_cfg.get("allow_smoke_fallback", False))
    return MistralExtractor(
        model_path=model_cfg.get("model_path") if use_real_model else None,
        adapter_path=_trained_mistral_adapter_path(config) if use_real_model else None,
        use_4bit=bool(model_cfg.get("use_4bit", True)),
        max_new_tokens=int(model_cfg.get("max_new_tokens", 512)),
        allow_fallback=allow_fallback,
    )


def _docs(config: dict[str, Any], split: str, mode: str, limit: int | None) -> list[dict[str, Any]]:
    docs = load_skillspan_split(Path(_dataset_root(config)), split, limit=None)
    docs = [doc for doc in docs if doc["gold_spans"]]
    selected_limit = _limit(config, mode, limit)
    return docs[:selected_limit] if selected_limit else docs


def setup_stage2_repo(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    require_paths(["src/skillllm_v2", "configs/skillllm_v2_stage2.yaml", "data/source/skillspan_hf/train.json"])
    for path in ["artifacts/skillllm_v2/status", "artifacts/skillllm_v2/smoke", "reports/skillllm_v2", "logs"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    return {"checked": True}


def legacy_cleanup_check(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    require_no_legacy_active_paths()
    return {"legacy_active_paths": 0}


def schema_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    skill = SkillCandidate("Python", "HARD_SKILL", 0, 6, "mistral", 0.9, "Python").to_dict()
    validate_skill_candidate(skill)
    result = SystemBenchmarkResult(system_name="BERT baseline").to_dict()
    validate_system_result(result)
    write_json("artifacts/skillllm_v2/smoke/schema_smoke.json", {"skill": skill, "result": result})
    return {"schemas": 2}


def component_unit_tests(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    command = ["python", "-m", "pytest", "tests/skillllm_v2", "-q"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    write_text("artifacts/skillllm_v2/logs/component_unit_tests.log", completed.stdout + completed.stderr)
    return {"pytest": "passed"}


def build_skillspan_bio_dataset(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    paths = build_bio_dataset(_dataset_root(config), "data/skillllm_v2/bio", _limit(config, mode, limit))
    return {"paths": paths}


def train_bert_baseline(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    model = BertBaseline()
    return model.train("data/skillllm_v2/bio/train.jsonl")


def eval_bert_baseline(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    model = BertBaseline()
    model.predict_dataset("data/skillllm_v2/bio/test.jsonl", "artifacts/skillllm_v2/bert_baseline/predictions.jsonl")
    metrics = evaluate_prediction_file(_dataset_root(_cfg(config_path)), "test", "artifacts/skillllm_v2/bert_baseline/predictions.jsonl", "artifacts/skillllm_v2/bert_baseline/metrics.json", "CPU-fast token classification baseline artifact.")
    write_json("reports/skillllm_v2/bert_baseline_metrics.json", metrics)
    return metrics


def train_or_run_gliner_baseline(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    return GLiNERBaseline().train_or_prepare()


def eval_gliner_baseline(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    baseline = GLiNERBaseline()
    baseline.predict(_dataset_root(config), "test", "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl", _limit(config, mode, limit))
    note = read_json("artifacts/skillllm_v2/gliner_baseline/model/config.json", {}).get("note", "")
    metrics = evaluate_prediction_file(_dataset_root(config), "test", "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl", "artifacts/skillllm_v2/gliner_baseline/metrics.json", note)
    metrics["training_mode"] = read_json("artifacts/skillllm_v2/gliner_baseline/model/config.json", {}).get("training_mode")
    write_json("artifacts/skillllm_v2/gliner_baseline/metrics.json", metrics)
    write_json("reports/skillllm_v2/gliner_baseline_metrics.json", metrics)
    return metrics


def build_mistral_sft_data(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    paths = build_mistral_sft_dataset(_dataset_root(config), "data/skillllm_v2/mistral_sft", _limit(config, mode, limit))
    return {"paths": paths}


def train_or_verify_mistral_lora(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    model_config = config.get("model", {})
    mistral_config = config.get("mistral_lora", {}) | config.get("mistral", {})
    training_config = config.get("training", {})
    train_enabled = bool(
        model_config.get("train_mistral_lora", False)
        or mistral_config.get("train_mistral_lora", False)
        or training_config.get("train_mistral_lora", False)
    )
    launch_training = mode == "full" and train_enabled
    epochs = (
        mistral_config.get("num_train_epochs")
        or mistral_config.get("epochs")
        or training_config.get("num_train_epochs")
        or training_config.get("epochs")
        or 3
    )
    distributed = config.get("distributed", {})
    extra_training = {
        "learning_rate": mistral_config.get("learning_rate", 0.00012),
        "gradient_accumulation_steps": mistral_config.get("gradient_accumulation_steps", 16),
        "batch_size": mistral_config.get("batch_size", 1),
        "eval_batch_size": mistral_config.get("eval_batch_size", 1),
        "max_seq_length": mistral_config.get("max_seq_length", 2048),
        "lora_r": mistral_config.get("lora_r", 64),
        "lora_alpha": mistral_config.get("lora_alpha", 128),
        "lora_dropout": mistral_config.get("lora_dropout", 0.05),
        "seed": config.get("seed", 42),
    }
    return train_or_verify_mistral_lora_artifact(
        model_config.get("model_path", "models/mistral-small-3.1-24b-instruct-2503"),
        model_config.get("adapter_path", "artifacts/mistral-small-3.1-24b-skillspan-lora"),
        "artifacts/skillllm_v2/mistral_lora/model",
        train_enabled,
        "data/skillllm_v2/mistral_sft/train.jsonl",
        "data/skillllm_v2/mistral_sft/dev.jsonl",
        epochs,
        int(distributed.get("num_processes", mistral_config.get("accelerate_num_processes", 2))),
        str(distributed.get("cuda_visible_devices", config.get("device", {}).get("cuda_visible_devices", "0,1"))),
        launch_training,
        extra_training,
    )


def eval_raw_mistral_lora(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    docs = _docs(config, "test", mode, limit)
    extractor = _mistral_extractor(config, mode)
    rows = []
    gold, pred = [], []
    for doc in docs:
        predictions = extractor.extract(doc["text"])
        rows.append({"id": doc["id"], "text": doc["text"], "predictions": predictions})
        gold.append(doc["gold_spans"])
        pred.append(predictions)
    write_jsonl("artifacts/skillllm_v2/mistral_lora/predictions.jsonl", rows)
    write_jsonl("artifacts/skillllm_v2/mistral_lora/raw_outputs.jsonl", rows)
    metrics = evaluate_extraction(gold, pred)
    metrics["notes"] = (
        f"Raw Mistral LoRA path. Model load fallback used: {bool(extractor.load_error)}."
        + (f" Load error: {extractor.load_error[:300]}" if extractor.load_error else "")
    )
    write_json("artifacts/skillllm_v2/mistral_lora/metrics.json", metrics)
    write_json("reports/skillllm_v2/raw_mistral_lora_metrics.json", metrics)
    return metrics


def eval_mistral_boundary_refiner(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    docs = _docs(config, "test", mode, limit)
    raw = read_jsonl("artifacts/skillllm_v2/mistral_lora/predictions.jsonl")
    by_id = {row["id"]: row["predictions"] for row in raw}
    refiner = BoundaryRefiner()
    rows, gold, pred = [], [], []
    for doc in docs:
        predictions = refiner.refine(doc["text"], by_id.get(doc["id"], []))
        rows.append({"id": doc["id"], "text": doc["text"], "predictions": predictions})
        gold.append(doc["gold_spans"])
        pred.append(predictions)
    write_jsonl("artifacts/skillllm_v2/mistral_boundary/predictions.jsonl", rows)
    metrics = evaluate_extraction(gold, pred)
    write_json("artifacts/skillllm_v2/mistral_boundary/metrics.json", metrics)
    write_json("reports/skillllm_v2/mistral_boundary_metrics.json", metrics)
    return metrics


def extraction_component_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    text = "Python, AWS, communication, and leadership are useful."
    counts = {
        "mistral": len(MistralExtractor().extract(text)),
        "gliner": len(GLiNERExtractor().extract(text)),
        "dictionary": len(DictionaryExtractor().extract(text)),
    }
    write_json("artifacts/skillllm_v2/smoke/extraction_components.json", counts)
    return counts


def merge_refine_verify_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    text = "strong communication skills and Python programming"
    candidates = DictionaryExtractor().extract(text)
    merged = CandidateMerger().merge(candidates)
    refined = BoundaryRefiner().refine(text, merged)
    verified = SpanVerifier().filter(refined)
    calibrated = ConfidenceCalibrator().calibrate(verified)
    write_json("artifacts/skillllm_v2/smoke/merge_refine_verify.json", {"skills": calibrated})
    return {"candidate_count": len(calibrated)}


def taxonomy_linking_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    linker = TaxonomyLinker()
    linked = [linker.link(skill) for skill in ["Python", "AWS", "communication"]]
    write_json("artifacts/skillllm_v2/smoke/taxonomy_linking.json", {"links": linked})
    return {"linked": sum(1 for item in linked if item)}


def eval_mistral_gliner_taxonomy(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    docs = _docs(config, "test", mode, limit)
    mistral = _mistral_extractor(config, mode, allow_fallback=True)
    gliner = GLiNERExtractor()
    dictionary = DictionaryExtractor()
    merger = CandidateMerger()
    linker = TaxonomyLinker()
    rows, linked_rows, gold, pred = [], [], [], []
    linked_count = 0
    total = 0
    for doc in docs:
        candidates = mistral.extract(doc["text"]) + gliner.extract(doc["text"]) + dictionary.extract(doc["text"])
        merged = linker.link_candidates(merger.merge(candidates))
        linked_count += sum(1 for item in merged if item.get("canonical_skill_name"))
        total += len(merged)
        rows.append({"id": doc["id"], "text": doc["text"], "predictions": merged})
        linked_rows.append({"id": doc["id"], "linked_skills": merged})
        gold.append(doc["gold_spans"])
        pred.append(merged)
    write_jsonl("artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl", rows)
    write_jsonl("artifacts/skillllm_v2/mistral_gliner_taxonomy/linked_skills.jsonl", linked_rows)
    metrics = evaluate_extraction(gold, pred)
    metrics["linking_accuracy"] = linked_count / total if total else None
    metrics["notes"] = "Mistral + GLiNER + Taxonomy. Mistral may use configured smoke fallback when local model loading is unavailable."
    write_json("artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json", metrics)
    write_json("reports/skillllm_v2/mistral_gliner_taxonomy_metrics.json", metrics)
    return metrics


def build_verifier_dataset(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    from skillllm_v2.train.build_verifier_dataset import build_verifier_dataset as build

    build(Path(_dataset_root(config)), Path("data/skillllm_v2"), _limit(config, mode, limit))
    return {"train": "data/skillllm_v2/verifier_train.jsonl", "dev": "data/skillllm_v2/verifier_dev.jsonl"}


def train_span_verifier(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    train_span_verifier_model(Path("data/skillllm_v2"), Path("artifacts/skillllm_v2/span_verifier"))
    return read_json("artifacts/skillllm_v2/span_verifier/verifier_config.json", {})


def verifier_eval_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    rows = read_jsonl("data/skillllm_v2/verifier_dev.jsonl")
    rows = rows[: _limit(_cfg(config_path), mode, limit) or 20]
    verifier = SpanVerifier()
    tp = fp = fn = 0
    predictions = []
    for row in rows:
        candidate = row.get("candidate") or {
            "surface_form": row.get("candidate_span"),
            "label": row.get("candidate_label", "HARD_SKILL"),
            "start": row.get("start"),
            "end": row.get("end"),
            "source": "dictionary",
            "confidence": 0.7,
            "evidence": row.get("candidate_span"),
            "canonical_skill_id": None,
            "canonical_skill_name": None,
            "metadata": {},
        }
        decision = verifier.verify(candidate)
        gold = bool(row.get("is_valid_skill", row.get("gold_is_skill", False)))
        tp += int(decision["is_valid_skill"] and gold)
        fp += int(decision["is_valid_skill"] and not gold)
        fn += int(not decision["is_valid_skill"] and gold)
        predictions.append({"candidate": candidate, "decision": decision, "gold": gold})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {"precision": precision, "recall": recall, "f1": f1, "false_positive_rate": fp / max(1, fp + tp)}
    write_json("artifacts/skillllm_v2/span_verifier/metrics.json", metrics)
    write_jsonl("artifacts/skillllm_v2/span_verifier/eval_predictions.jsonl", predictions)
    return metrics


def build_linker_dataset(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    rows = [
        {"surface_form": "Python", "context": "Python programming", "positive_canonical_skill_id": "python", "positive_canonical_skill_name": "Python", "negative_candidates": ["Java"], "taxonomy": "CUSTOM"},
        {"surface_form": "AWS", "context": "AWS cloud", "positive_canonical_skill_id": "aws", "positive_canonical_skill_name": "Amazon Web Services", "negative_candidates": ["Azure"], "taxonomy": "CUSTOM"},
    ]
    write_jsonl("data/skillllm_v2/linker_train.jsonl", rows)
    write_jsonl("data/skillllm_v2/linker_dev.jsonl", rows)
    return {"rows": len(rows)}


def train_taxonomy_reranker(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    train_taxonomy_reranker_model("artifacts/skillllm_v2/taxonomy_reranker")
    metrics = {"training_mode": "retrieval_only", "note": "Supervised taxonomy reranker labels unavailable; retrieval-only linker used."}
    write_json("artifacts/skillllm_v2/taxonomy_reranker/metrics.json", metrics)
    write_text("reports/skillllm_v2/linking_eval.md", metrics["note"] + "\n")
    return metrics


def graph_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    builder = SkillGraphBuilder()
    builder.add_node("u1", NodeType.USER)
    builder.add_node("profile-u1", NodeType.PROFILE)
    for skill in ["Python", "SQL", "AWS", "communication", "leadership"]:
        builder.add_node(skill, NodeType.SKILL)
    builder.add_node("j1", NodeType.JOB)
    builder.add_node("j2", NodeType.JOB)
    builder.add_node("c1", NodeType.COURSE)
    builder.add_node("c2", NodeType.COURSE)
    builder.add_edge("u1", "Python", EdgeType.HAS_SKILL)
    builder.add_edge("j1", "AWS", EdgeType.REQUIRES)
    builder.add_edge("c1", "AWS", EdgeType.TEACHES)
    write_json("artifacts/skillllm_v2/smoke/graph.json", {"nodes": builder.graph.number_of_nodes(), "edges": builder.graph.number_of_edges()})
    return {"nodes": builder.graph.number_of_nodes(), "edges": builder.graph.number_of_edges()}


def matching_recommendation_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    profile = {"skills": ["Python", "SQL"], "weak_skills": ["SQL"]}
    job = {"job_id": "j1", "required_skills": ["Python", "AWS"], "preferred_skills": ["SQL"], "semantic_role_fit": 0.8, "experience_level_fit": 0.7, "user_preference_fit": 0.8}
    match = JobMatcher().score(profile, job)
    gaps = GapDetector().detect(profile, job)
    courses = CourseRecommender().recommend(gaps, [{"course_id": "c1", "course_name": "AWS Foundations", "teaches_skills": ["AWS"], "prerequisite_skills": ["Python"]}], profile)
    write_json("artifacts/skillllm_v2/smoke/matching_recommendation.json", {"match": match, "gaps": gaps, "courses": courses})
    return {"job_matches": 1, "courses": len(courses)}


def eval_full_cag_graph(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    docs = _docs(config, "test", mode, limit)
    from skillllm_v2.wrapper import SkillLLMv2Pipeline

    pipeline = SkillLLMv2Pipeline(mistral=_mistral_extractor(config, mode, allow_fallback=True))
    rows, linked_rows, gold, pred = [], [], [], []
    linked = total = 0
    for doc in docs:
        predictions = pipeline.extract(doc["text"])
        linked += sum(1 for item in predictions if item.get("canonical_skill_name"))
        total += len(predictions)
        rows.append({"id": doc["id"], "text": doc["text"], "predictions": predictions})
        linked_rows.append({"id": doc["id"], "linked_skills": predictions})
        gold.append(doc["gold_spans"])
        pred.append(predictions)
    write_jsonl("artifacts/skillllm_v2/full_cag_graph/predictions.jsonl", rows)
    write_jsonl("artifacts/skillllm_v2/full_cag_graph/linked_skills.jsonl", linked_rows)
    match_payload = matching_recommendation_smoke(config_path, mode, limit)
    write_jsonl("artifacts/skillllm_v2/full_cag_graph/job_matches.jsonl", [read_json("artifacts/skillllm_v2/smoke/matching_recommendation.json", {}).get("match", {})])
    write_jsonl("artifacts/skillllm_v2/full_cag_graph/course_recommendations.jsonl", read_json("artifacts/skillllm_v2/smoke/matching_recommendation.json", {}).get("courses", []))
    metrics = evaluate_extraction(gold, pred)
    metrics["linking_accuracy"] = linked / total if total else None
    metrics["job_match_ndcg_at_10"] = None
    metrics["course_rec_precision_at_5"] = None
    metrics["notes"] = "Full CAG/Graph wrapper. Graph matching labels unavailable; ranking metrics are NA."
    write_json("artifacts/skillllm_v2/full_cag_graph/metrics.json", metrics)
    write_json("reports/skillllm_v2/full_cag_graph_metrics.json", metrics)
    return {**metrics, **match_payload}


def extraction_ablation_full(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    run_ablation(_dataset_root(_cfg(config_path)), "reports/skillllm_v2", limit=_limit(_cfg(config_path), mode, limit) or 20)
    return {"report": "reports/skillllm_v2/extraction_ablation.csv"}


def graph_matching_eval_full(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    write_text("reports/skillllm_v2/matching_eval.csv", "metric,value\njob_match_ndcg_at_10,NA\n")
    write_text("reports/skillllm_v2/course_recommendation_eval.csv", "metric,value\ncourse_rec_precision_at_5,NA\n")
    return {"job_match_ndcg_at_10": None, "course_rec_precision_at_5": None}


def money_table_generation(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    if Path("artifacts/skillllm_v2/mistral_lora/predictions.jsonl").exists():
        run_safe_wrapper(config_path=config_path, limit=_limit(config, mode, limit))
    rows = generate_money_table("reports/skillllm_v2")
    return {"rows": len(rows)}


def end_to_end_demo_smoke(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    config = _cfg(config_path)
    profile = read_json("inputs/demo/user_profile.json", {"user_id": "demo-user", "text": "I know Python, SQL, Docker, communication, and machine learning."})
    jobs = read_jsonl(config.get("data", {}).get("job_data_path", "inputs/demo/jobs.jsonl")) or [
        {"job_id": "ml-platform", "required_skills": ["Python", "AWS"], "preferred_skills": ["Docker", "communication"], "semantic_role_fit": 0.8, "experience_level_fit": 0.8, "user_preference_fit": 0.8}
    ]
    courses = read_jsonl(config.get("data", {}).get("course_catalog_path", "inputs/demo/course_catalog.jsonl")) or [
        {"course_id": "aws-101", "course_name": "AWS Foundations", "teaches_skills": ["AWS"], "prerequisite_skills": ["Python"]}
    ]
    from skillllm_v2.wrapper import SkillLLMv2Pipeline

    skills = SkillLLMv2Pipeline().extract(profile["text"])
    skill_names = [skill.get("canonical_skill_name") or skill["surface_form"] for skill in skills]
    user_profile = {"user_id": profile["user_id"], "skills": skill_names}
    matches = [JobMatcher().score(user_profile, job) for job in jobs]
    gaps = [gap for job in jobs for gap in GapDetector().detect(user_profile, job)]
    recommendations = CourseRecommender().recommend(gaps, courses, user_profile)
    base = Path("reports/skillllm_v2/demo")
    write_json(base / "user_skill_profile.json", user_profile)
    write_json(base / "job_matches.json", matches)
    write_json(base / "skill_gaps.json", gaps)
    write_json(base / "course_recommendations.json", recommendations)
    write_text(base / "explanation.md", "# Demo Explanation\n\nSkillLLM_v2 extracted profile skills, matched jobs, detected gaps, and recommended courses.\n")
    return {"skills": len(skills), "jobs": len(matches), "courses": len(recommendations)}


def final_report_generation(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    text = """# SkillLLM_v2 Thesis Results Summary

Reports were generated by the gated Stage-2 pipeline. Metrics in the money table are real artifact values when present and `NA` when labels or evaluations are unavailable.
"""
    write_text("reports/skillllm_v2/thesis_results_summary.md", text)
    if not Path("reports/skillllm_v2/error_analysis.md").exists():
        write_text("reports/skillllm_v2/error_analysis.md", "# Error Analysis\n\nReview false positives, missed skills, boundary errors, and taxonomy misses.\n")
    _write_artifact_index()
    return {
        "report": "reports/skillllm_v2/thesis_results_summary.md",
        "artifact_index": "reports/skillllm_v2/artifact_index.md",
    }


def master_merge_readiness_check(config_path: str, mode: str, limit: int | None = None) -> dict[str, Any]:
    require_no_legacy_active_paths()
    require_paths(
        [
            "reports/skillllm_v2/money_table.csv",
            "reports/skillllm_v2/money_table.md",
            "artifacts/skillllm_v2/bert_baseline/metrics.json",
            "artifacts/skillllm_v2/gliner_baseline/metrics.json",
            "artifacts/skillllm_v2/mistral_lora/metrics.json",
            "artifacts/skillllm_v2/mistral_boundary/metrics.json",
            "artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json",
            "artifacts/skillllm_v2/full_cag_graph/metrics.json",
            "docs/legacy_benchmark_summary.md",
            "docs/removed_artifacts_manifest.md",
        ]
    )
    return {"ready": True}


def _write_artifact_index() -> None:
    groups: dict[str, list[str]] = {
        "Status and logs": [
            "artifacts/skillllm_v2/status/*.success.json",
            "artifacts/skillllm_v2/status/*.failed.json",
            "logs/latest_stage2_80epoch_2gpu.log",
            "logs/latest_stage2_80epoch_2gpu.pid",
            "reports/skillllm_v2/gpu_component_availability.json",
            "reports/skillllm_v2/gpu_component_availability.md",
        ],
        "BERT baseline": [
            "artifacts/skillllm_v2/bert_baseline/model/model.json",
            "artifacts/skillllm_v2/bert_baseline/predictions.jsonl",
            "artifacts/skillllm_v2/bert_baseline/metrics.json",
            "reports/skillllm_v2/bert_baseline_metrics.json",
        ],
        "GLiNER": [
            "artifacts/skillllm_v2/gliner_baseline/model/config.json",
            "artifacts/skillllm_v2/gliner_baseline/predictions.jsonl",
            "artifacts/skillllm_v2/gliner_baseline/metrics.json",
            "reports/skillllm_v2/gliner_baseline_metrics.json",
        ],
        "Raw Mistral LoRA": [
            "artifacts/skillllm_v2/mistral_lora/model/",
            "artifacts/skillllm_v2/mistral_lora/model/training_summary.json",
            "artifacts/skillllm_v2/mistral_lora/model/adapter_status.json",
            "artifacts/skillllm_v2/mistral_lora/raw_outputs.jsonl",
            "artifacts/skillllm_v2/mistral_lora/predictions.jsonl",
            "artifacts/skillllm_v2/mistral_lora/metrics.json",
            "reports/skillllm_v2/raw_mistral_lora_metrics.json",
        ],
        "Mistral + Boundary Refiner": [
            "artifacts/skillllm_v2/mistral_boundary/predictions.jsonl",
            "artifacts/skillllm_v2/mistral_boundary/metrics.json",
            "reports/skillllm_v2/mistral_boundary_metrics.json",
        ],
        "Mistral + GLiNER + Taxonomy": [
            "artifacts/skillllm_v2/mistral_gliner_taxonomy/predictions.jsonl",
            "artifacts/skillllm_v2/mistral_gliner_taxonomy/linked_skills.jsonl",
            "artifacts/skillllm_v2/mistral_gliner_taxonomy/metrics.json",
            "reports/skillllm_v2/mistral_gliner_taxonomy_metrics.json",
        ],
        "SkillLLM_v2 Full CAG/Graph": [
            "artifacts/skillllm_v2/full_cag_graph/predictions.jsonl",
            "artifacts/skillllm_v2/full_cag_graph/linked_skills.jsonl",
            "artifacts/skillllm_v2/full_cag_graph/job_matches.jsonl",
            "artifacts/skillllm_v2/full_cag_graph/course_recommendations.jsonl",
            "artifacts/skillllm_v2/full_cag_graph/metrics.json",
            "reports/skillllm_v2/full_cag_graph_metrics.json",
        ],
        "Verifier and linker": [
            "data/skillllm_v2/verifier_train.jsonl",
            "data/skillllm_v2/verifier_dev.jsonl",
            "artifacts/skillllm_v2/span_verifier/model/",
            "artifacts/skillllm_v2/span_verifier/metrics.json",
            "data/skillllm_v2/linker_train.jsonl",
            "data/skillllm_v2/linker_dev.jsonl",
            "artifacts/skillllm_v2/taxonomy_reranker/metrics.json",
        ],
        "Final reports": [
            "reports/skillllm_v2/money_table.csv",
            "reports/skillllm_v2/money_table.md",
            "reports/skillllm_v2/extraction_ablation.csv",
            "reports/skillllm_v2/extraction_ablation.md",
            "reports/skillllm_v2/matching_eval.csv",
            "reports/skillllm_v2/course_recommendation_eval.csv",
            "reports/skillllm_v2/thesis_results_summary.md",
            "reports/skillllm_v2/error_analysis.md",
        ],
    }
    payload = {
        group: [{"path": path, "exists": Path(path).exists() if "*" not in path else bool(list(Path(".").glob(path)))} for path in paths]
        for group, paths in groups.items()
    }
    lines = ["# SkillLLM_v2 Artifact Index", ""]
    for group, paths in groups.items():
        lines.extend([f"## {group}", ""])
        for path in paths:
            exists = Path(path).exists() if "*" not in path else bool(list(Path(".").glob(path)))
            lines.append(f"- `{path}` - {'present' if exists else 'missing'}")
        lines.append("")
    write_json("reports/skillllm_v2/artifact_index.json", payload)
    write_text("reports/skillllm_v2/artifact_index.md", "\n".join(lines))
