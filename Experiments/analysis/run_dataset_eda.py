import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_annotation_queue import default_input_dir, discover_source_files, load_source_records, queue_record
from common import deterministic_split, ensure_dir, load_config, load_env_file, read_json_records, resolve_path, write_json
from prepare_benchmark_data import discover_dataset_files, normalize_split


def numeric_summary(values):
    values = list(values)
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": min(sorted_values),
        "max": max(sorted_values),
        "mean": statistics.fmean(sorted_values),
        "median": statistics.median(sorted_values),
        "p90": sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.9))],
    }


def write_csv(path: Path, fieldnames, rows):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_raw_eda(config, output_dir: Path):
    source_dir = default_input_dir()
    source_files = discover_source_files(source_dir)
    report_lines = ["# Raw Dataset EDA", ""]
    if not source_files:
        payload = {"status": "no_source_data", "source_dir": str(source_dir), "resume_count": 0}
        write_json(output_dir / "raw_summary.json", payload)
        (output_dir / "raw_report.md").write_text("# Raw Dataset EDA\n\nNo source data found.\n", encoding="utf-8")
        return

    source_counts = Counter()
    role_counts = Counter()
    text_lengths = []
    section_counts = []
    readiness_rows = []
    skipped_records = 0
    total_records = 0

    running_index = 1
    for source_file in source_files:
        file_total = 0
        file_queued = 0
        for record in load_source_records(source_file):
            total_records += 1
            file_total += 1
            queued = queue_record(record, source_file, running_index)
            running_index += 1
            if queued is None:
                skipped_records += 1
                continue
            file_queued += 1
            source_counts[queued["source"]] += 1
            if queued.get("role_hint"):
                role_counts[queued["role_hint"]] += 1
            text_lengths.append(len(queued["text"]))
            section_counts.append(len(queued.get("sections", [])))
        readiness_rows.append(
            {
                "input_file": str(source_file),
                "records_total": file_total,
                "records_queueable": file_queued,
                "records_skipped": file_total - file_queued,
            }
        )

    summary = {
        "status": "ok",
        "source_dir": str(source_dir),
        "resume_count": sum(source_counts.values()),
        "source_file_count": len(source_files),
        "skipped_records": skipped_records,
        "text_length_chars": numeric_summary(text_lengths),
        "section_count": numeric_summary(section_counts),
        "role_hint_coverage": (sum(role_counts.values()) / sum(source_counts.values())) if source_counts else 0.0,
    }
    write_json(output_dir / "raw_summary.json", summary)
    write_csv(output_dir / "raw_source_counts.csv", ["source", "resume_count"], [{"source": key, "resume_count": value} for key, value in source_counts.most_common()])
    write_csv(output_dir / "raw_role_counts.csv", ["role_hint", "resume_count"], [{"role_hint": key, "resume_count": value} for key, value in role_counts.most_common()])
    write_csv(output_dir / "raw_parsing_readiness.csv", ["input_file", "records_total", "records_queueable", "records_skipped"], readiness_rows)

    report_lines.extend(
        [
            f"- Source files: {len(source_files)}",
            f"- Queueable resumes: {sum(source_counts.values())}",
            f"- Skipped records: {skipped_records}",
            f"- Role hint coverage: {summary['role_hint_coverage']:.2%}",
            f"- Mean raw text length (chars): {summary['text_length_chars']['mean']:.1f}",
        ]
    )
    (output_dir / "raw_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def load_annotated_splits(config):
    canonical_dir = resolve_path(config["paths"]["canonical_dir"])
    canonical_files = {
        split: canonical_dir / f"{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    if all(path.exists() for path in canonical_files.values()):
        return {split: read_json_records(path) for split, path in canonical_files.items()}

    raw_dir = resolve_path(config["paths"]["raw_data_dir"])
    discovered = discover_dataset_files(raw_dir)
    if not discovered:
        return {}

    if "all" in discovered:
        records = read_json_records(discovered["all"])
        normalized = normalize_split(records, "all", config)
        return deterministic_split(normalized, seed=config["seed"], ratios=config["split_ratios"])

    return {split: normalize_split(read_json_records(path), split, config) for split, path in discovered.items()}


def run_annotated_eda(config, output_dir: Path):
    splits = load_annotated_splits(config)
    if not splits:
        write_json(output_dir / "annotated_summary.json", {"status": "no_annotated_data"})
        (output_dir / "annotated_report.md").write_text("# Annotated Dataset EDA\n\nNo annotated or canonical data found.\n", encoding="utf-8")
        return

    all_records = [record for records in splits.values() for record in records]
    unique_resume_ids = {record.get("resume_id") or record["id"] for record in all_records}
    total_spans = sum(len(record.get("spans", [])) for record in all_records)
    normalized_skill_mentions = []
    source_by_split = []
    role_by_split = []
    section_stats = defaultdict(lambda: {"sentences": 0, "skill_spans": 0})
    skills_per_resume = Counter()
    skills_per_sentence = []
    skill_counter = Counter()

    for split_name, records in splits.items():
        for record in records:
            source_by_split.append({"split": split_name, "source": record.get("source") or "unknown"})
            role_by_split.append({"split": split_name, "role_hint": record.get("role_hint") or "unknown"})
            section_name = record.get("section_name") or "unknown"
            section_stats[section_name]["sentences"] += 1
            section_stats[section_name]["skill_spans"] += len(record.get("spans", []))
            skills_per_resume[record.get("resume_id") or record["id"]] += len(record.get("spans", []))
            skills_per_sentence.append(len(record.get("spans", [])))
            for item in record.get("normalized_skills", []):
                skill_name = item.get("canonical_name") or item.get("text")
                normalized_skill_mentions.append(skill_name)
                skill_counter[skill_name] += 1

    summary = {
        "status": "ok",
        "resume_count": len(unique_resume_ids),
        "sentence_count": len(all_records),
        "skill_span_count": total_spans,
        "normalized_skill_mentions": len(normalized_skill_mentions),
        "unique_normalized_skills": len(skill_counter),
        "skills_per_resume": numeric_summary(skills_per_resume.values()),
        "skills_per_sentence": numeric_summary(skills_per_sentence),
        "long_tail_distribution": {
            "head_ge_10": sum(1 for value in skill_counter.values() if value >= 10),
            "mid_3_to_9": sum(1 for value in skill_counter.values() if 3 <= value <= 9),
            "tail_1_to_2": sum(1 for value in skill_counter.values() if value <= 2),
        },
    }
    write_json(output_dir / "annotated_summary.json", summary)
    write_csv(
        output_dir / "annotated_top_skills.csv",
        ["skill", "count"],
        [{"skill": skill, "count": count} for skill, count in skill_counter.most_common(100)],
    )
    write_csv(
        output_dir / "annotated_section_density.csv",
        ["section_name", "sentence_count", "skill_span_count", "avg_spans_per_sentence"],
        [
            {
                "section_name": section_name,
                "sentence_count": payload["sentences"],
                "skill_span_count": payload["skill_spans"],
                "avg_spans_per_sentence": (payload["skill_spans"] / payload["sentences"]) if payload["sentences"] else 0.0,
            }
            for section_name, payload in sorted(section_stats.items())
        ],
    )

    split_source_counts = Counter((row["split"], row["source"]) for row in source_by_split)
    split_role_counts = Counter((row["split"], row["role_hint"]) for row in role_by_split)
    split_skill_counts = Counter()
    for split_name, records in splits.items():
        for record in records:
            for item in record.get("normalized_skills", []):
                skill_name = item.get("canonical_name") or item.get("text")
                split_skill_counts[(split_name, skill_name)] += 1
    write_csv(
        output_dir / "annotated_split_balance.csv",
        ["dimension", "split", "value", "count"],
        [
            *[
                {"dimension": "source", "split": split, "value": value, "count": count}
                for (split, value), count in split_source_counts.most_common()
            ],
            *[
                {"dimension": "role_hint", "split": split, "value": value, "count": count}
                for (split, value), count in split_role_counts.most_common()
            ],
            *[
                {"dimension": "top_skill", "split": split, "value": value, "count": count}
                for (split, value), count in split_skill_counts.most_common(100)
            ],
        ],
    )
    report_lines = [
        "# Annotated Dataset EDA",
        "",
        f"- Resumes: {summary['resume_count']}",
        f"- Sentences: {summary['sentence_count']}",
        f"- Skill spans: {summary['skill_span_count']}",
        f"- Unique normalized skills: {summary['unique_normalized_skills']}",
        f"- Mean skills per sentence: {summary['skills_per_sentence']['mean']:.2f}",
    ]
    (output_dir / "annotated_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run dataset EDA for raw and annotated benchmark data.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["raw", "annotated", "all"], default="all")
    args = parser.parse_args()

    load_env_file()
    config = load_config(args.config)
    output_dir = ensure_dir(resolve_path(config["paths"]["results_dir"]) / "analysis" / "dataset")

    if args.mode in {"raw", "all"}:
        run_raw_eda(config, output_dir)
    if args.mode in {"annotated", "all"}:
        run_annotated_eda(config, output_dir)

    print(f"Dataset EDA written under {output_dir}")


if __name__ == "__main__":
    main()
