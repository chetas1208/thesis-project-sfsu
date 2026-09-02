import argparse
import json
import os
import re
from typing import Dict, List


TOKEN_PATTERN = re.compile(
    r"C\+\+|C#|Node\.js|\.NET|A/B|[\w]+(?:[./+#-][\w]+)*|[^\w\s]",
    re.UNICODE,
)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def tokenize(sentence: str) -> List[Dict[str, object]]:
    return [
        {"text": match.group(), "start": match.start(), "end": match.end()}
        for match in TOKEN_PATTERN.finditer(sentence)
    ]


def align_spans_to_tokens(tokens: List[Dict[str, object]], spans: List[Dict[str, object]]) -> List[str]:
    ner_tags = ["O"] * len(tokens)

    for span in spans:
        span_start = int(span["start"])
        span_end = int(span["end"])
        label = str(span["label"])
        token_indices = [
            index
            for index, token in enumerate(tokens)
            if int(token["start"]) >= span_start and int(token["end"]) <= span_end
        ]

        if not token_indices:
            continue

        ner_tags[token_indices[0]] = f"B-{label}"
        for index in token_indices[1:]:
            ner_tags[index] = f"I-{label}"

    return ner_tags


def run_conversion(input_file: str, output_file: str) -> Dict[str, int]:
    ensure_parent(output_file)
    stats = {"records_seen": 0, "records_written": 0}

    with open(input_file, "r", encoding="utf-8") as source, open(
        output_file, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue

            stats["records_seen"] += 1
            record = json.loads(line)
            sentence = str(record["sentence"])
            spans = list(record.get("spans", []))
            token_dicts = tokenize(sentence)

            enriched_record = dict(record)
            enriched_record["tokens"] = [token["text"] for token in token_dicts]
            enriched_record["token_offsets"] = [
                {"start": int(token["start"]), "end": int(token["end"])} for token in token_dicts
            ]
            enriched_record["ner_tags"] = align_spans_to_tokens(token_dicts, spans)

            sink.write(json.dumps(enriched_record, ensure_ascii=False) + "\n")
            stats["records_written"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert span annotations into BIO tags.")
    parser.add_argument("--input-file", default="data/silver/skillspan_style_silver.jsonl")
    parser.add_argument("--output-file", default="data/silver/skillspan_style_silver_bio.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_conversion(args.input_file, args.output_file)
    print(json.dumps(summary, indent=2))
