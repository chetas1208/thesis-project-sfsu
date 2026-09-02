import argparse
from pathlib import Path

from common import (
    deterministic_split,
    ensure_benchmark_env,
    ensure_dir,
    flatten_resume_annotation,
    load_config,
    load_env_file,
    normalize_record,
    read_json_records,
    resolve_path,
    validate_record,
    write_jsonl,
)
from model_adapters.context import export_context_dataset
from model_adapters.gliner import export_gliner_dataset
from model_adapters.gpt5 import export_gpt5_prompts
from model_adapters.skill_llm import export_skill_llm_dataset
from model_adapters.token_classification import export_token_classification_dataset


def discover_dataset_files(raw_dir: Path) -> dict[str, Path] | None:
    search_roots = [raw_dir / "annotated", raw_dir]
    last_candidates = {}
    for root in search_roots:
        if not root.exists():
            continue
        candidates = {}
        for file_path in sorted(root.glob("*")):
            if file_path.suffix not in {".json", ".jsonl"}:
                continue
            stem = file_path.stem.lower()
            candidates[stem] = file_path
        if {"train", "dev", "test"} <= set(candidates):
            return {"train": candidates["train"], "dev": candidates["dev"], "test": candidates["test"]}
        if {"train", "validation", "test"} <= set(candidates):
            return {"train": candidates["train"], "dev": candidates["validation"], "test": candidates["test"]}
        for single_name in ("all", "dataset", "full", "data", "annotated"):
            if single_name in candidates:
                return {"all": candidates[single_name]}
        last_candidates = candidates
    if len(last_candidates) == 1:
        key = next(iter(last_candidates))
        return {"all": last_candidates[key]}
    return None


def normalize_split(records: list[dict], split_name: str, config: dict) -> list[dict]:
    normalized = []
    for index, record in enumerate(records):
        record_id = f"{split_name}-resume-{index + 1:06d}"
        if record.get("sentences"):
            normalized_records = flatten_resume_annotation(
                record,
                record_id_prefix=record_id,
                collapse_to_skill=config["collapse_labels_to_skill"],
            )
        else:
            if record.get("sections") and not record.get("tokens"):
                raise ValueError(
                    "Found resume-style records without sentence annotations. "
                    "Annotate them into the canonical resume schema before running prepare_data."
                )
            normalized_records = [
                normalize_record(
                    record,
                    record_id=f"{split_name}-{index + 1:06d}",
                    collapse_to_skill=config["collapse_labels_to_skill"],
                )
            ]
        for normalized_record in normalized_records:
            errors = validate_record(normalized_record, config["labels"])
            if errors:
                joined = "; ".join(errors)
                raise ValueError(f"Validation failed for {normalized_record['id']}: {joined}")
            normalized.append(normalized_record)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical and model-specific benchmark datasets.")
    parser.add_argument("--config", default=None, help="Optional path to benchmark.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate discovery and print actions without writing data.")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    ensure_benchmark_env(config)

    raw_dir = resolve_path(config["paths"]["raw_data_dir"])
    canonical_dir = ensure_dir(resolve_path(config["paths"]["canonical_dir"]))
    model_inputs_root = ensure_dir(resolve_path(config["paths"]["model_inputs_dir"]))

    discovered = discover_dataset_files(raw_dir)
    if not discovered:
        print(f"No dataset found in {raw_dir}. The scaffold is ready and waiting for your dataset.")
        return

    if "all" in discovered:
        all_records = read_json_records(discovered["all"])
        normalized_all = normalize_split(all_records, "all", config)
        splits = deterministic_split(normalized_all, seed=config["seed"], ratios=config["split_ratios"])
    else:
        splits = {
            split_name: normalize_split(read_json_records(path), split_name, config)
            for split_name, path in discovered.items()
        }

    if args.dry_run:
        for split_name, records in splits.items():
            print(f"{split_name}: {len(records)} normalized records")
        return

    write_jsonl(canonical_dir / "train.jsonl", splits["train"])
    write_jsonl(canonical_dir / "dev.jsonl", splits["dev"])
    write_jsonl(canonical_dir / "test.jsonl", splits["test"])

    export_token_classification_dataset(splits, model_inputs_root / "nnose")
    export_token_classification_dataset(splits, model_inputs_root / "jobbert")
    export_gliner_dataset(splits, model_inputs_root / "gliner")
    export_skill_llm_dataset(splits, model_inputs_root / "skill_llm")
    export_context_dataset(splits, model_inputs_root / "context")
    prompt_template = resolve_path(config["paths"]["prompt_template"]).read_text(encoding="utf-8")
    export_gpt5_prompts(splits, model_inputs_root / "gpt5", prompt_template)

    print(f"Canonical data written to {canonical_dir}")
    print(f"Model inputs written under {model_inputs_root}")


if __name__ == "__main__":
    main()
