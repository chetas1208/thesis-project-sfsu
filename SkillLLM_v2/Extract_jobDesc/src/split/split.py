import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_records(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def export_conll(records: Sequence[Dict[str, object]], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as handle:
        for record in records:
            tokens = record.get("tokens", [])
            tags = record.get("ner_tags", [])
            for token, tag in zip(tokens, tags):
                handle.write(f"{token} {tag}\n")
            handle.write("\n")


def split_job_ids(job_ids: List[str], seed: int) -> Tuple[set, set, set]:
    shuffled = sorted(set(job_ids))
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.70)
    dev_end = int(total * 0.85)
    return set(shuffled[:train_end]), set(shuffled[train_end:dev_end]), set(shuffled[dev_end:])


def write_jsonl(records: Sequence[Dict[str, object]], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_split(input_file: str, out_dir: str, seed: int = 42) -> Dict[str, int]:
    ensure_dir(out_dir)
    records = read_records(input_file)
    train_jobs, dev_jobs, test_jobs = split_job_ids([str(record["job_id"]) for record in records], seed)

    train_records = [record for record in records if record["job_id"] in train_jobs]
    dev_records = [record for record in records if record["job_id"] in dev_jobs]
    test_records = [record for record in records if record["job_id"] in test_jobs]

    write_jsonl(train_records, os.path.join(out_dir, "train.jsonl"))
    write_jsonl(dev_records, os.path.join(out_dir, "dev.jsonl"))
    write_jsonl(test_records, os.path.join(out_dir, "test.jsonl"))

    export_conll(train_records, os.path.join(out_dir, "train.conll"))
    export_conll(dev_records, os.path.join(out_dir, "dev.conll"))
    export_conll(test_records, os.path.join(out_dir, "test.conll"))

    return {
        "train_records": len(train_records),
        "dev_records": len(dev_records),
        "test_records": len(test_records),
        "train_jobs": len(train_jobs),
        "dev_jobs": len(dev_jobs),
        "test_jobs": len(test_jobs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/dev/test splits by job_id.")
    parser.add_argument("--input-file", default="data/silver/skillspan_style_silver_bio.jsonl")
    parser.add_argument("--out-dir", default="data/silver")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_split(args.input_file, args.out_dir, seed=args.seed)
    print(json.dumps(summary, indent=2))
