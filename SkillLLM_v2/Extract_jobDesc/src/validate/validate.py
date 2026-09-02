import argparse
import json
import os
from collections import Counter
from typing import Dict, List


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def record_error(report: Dict[str, object], error_type: str, message: str) -> None:
    report["error_counts"][error_type] += 1
    if len(report["errors"]) < report["max_errors"]:
        report["errors"].append(message)


def spans_overlap(left: Dict[str, object], right: Dict[str, object]) -> bool:
    return not (int(left["end"]) <= int(right["start"]) or int(right["end"]) <= int(left["start"]))


def run_validation(input_file: str, report_file: str) -> Dict[str, object]:
    ensure_parent(report_file)
    report: Dict[str, object] = {
        "total_sentences": 0,
        "valid_sentences": 0,
        "invalid_sentences": 0,
        "error_counts": Counter(),
        "errors": [],
        "max_errors": 100,
    }

    with open(input_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            report["total_sentences"] += 1
            valid = True
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                record_error(report, "json_decode_error", f"Line {line_number}: {exc}")
                report["invalid_sentences"] += 1
                continue

            sentence = str(record.get("sentence", ""))
            spans = list(record.get("spans", []))
            tokens = list(record.get("tokens", []))
            tags = list(record.get("ner_tags", []))
            offsets = list(record.get("token_offsets", []))

            if sentence.strip() and not tokens:
                valid = False
                record_error(report, "empty_tokens", f"{record.get('id')}: empty tokens for non-empty sentence")

            if len(tokens) != len(tags):
                valid = False
                record_error(report, "token_tag_length_mismatch", f"{record.get('id')}: tokens={len(tokens)} tags={len(tags)}")

            if offsets and len(offsets) != len(tokens):
                valid = False
                record_error(report, "token_offset_length_mismatch", f"{record.get('id')}: offsets={len(offsets)} tokens={len(tokens)}")

            spans_sorted = sorted(spans, key=lambda item: (int(item["start"]), int(item["end"])))
            for index, span in enumerate(spans_sorted):
                start = int(span["start"])
                end = int(span["end"])
                text = str(span["text"])
                if start < 0 or end > len(sentence) or start >= end:
                    valid = False
                    record_error(report, "invalid_span_bounds", f"{record.get('id')}: invalid bounds {start}:{end}")
                    continue

                if sentence[start:end] != text:
                    valid = False
                    record_error(report, "span_text_mismatch", f"{record.get('id')}: expected '{text}' got '{sentence[start:end]}'")

                if index > 0 and spans_overlap(spans_sorted[index - 1], span):
                    valid = False
                    record_error(report, "overlapping_spans", f"{record.get('id')}: overlapping spans detected")

            if offsets:
                for token, offset in zip(tokens, offsets):
                    start = int(offset["start"])
                    end = int(offset["end"])
                    if sentence[start:end] != token:
                        valid = False
                        record_error(report, "token_offset_mismatch", f"{record.get('id')}: token '{token}' does not align")

            reconstructed_tags = ["O"] * len(tokens)
            if offsets:
                for span in spans_sorted:
                    covered = [
                        idx
                        for idx, offset in enumerate(offsets)
                        if int(offset["start"]) >= int(span["start"]) and int(offset["end"]) <= int(span["end"])
                    ]
                    if covered:
                        reconstructed_tags[covered[0]] = f"B-{span['label']}"
                        for idx in covered[1:]:
                            reconstructed_tags[idx] = f"I-{span['label']}"

            if offsets and reconstructed_tags != tags:
                valid = False
                record_error(report, "bio_mismatch", f"{record.get('id')}: BIO tags do not match spans")

            if valid:
                report["valid_sentences"] += 1
            else:
                report["invalid_sentences"] += 1

    report["error_counts"] = dict(report["error_counts"])
    report.pop("max_errors", None)
    with open(report_file, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate span and BIO consistency.")
    parser.add_argument("--input-file", default="data/silver/skillspan_style_silver_bio.jsonl")
    parser.add_argument("--report-file", default="reports/validation_report.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_validation(args.input_file, args.report_file)
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))
