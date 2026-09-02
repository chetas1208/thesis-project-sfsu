import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Tuple


BOUNDARY_PATTERN = re.compile(r"[.!?]+(?:['\"\)\]]+)?")


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def normalize_fragment(text: str) -> str:
    return text.strip()


def iter_paragraph_spans(text: str) -> Iterable[Tuple[int, int]]:
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n|$)", text):
        yield match.start(), match.end()


def split_paragraph(text: str, base_start: int) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    cursor = 0

    for match in BOUNDARY_PATTERN.finditer(text):
        end = match.end()
        fragment = text[cursor:end]
        cleaned = normalize_fragment(fragment)
        if cleaned:
            exact_start = base_start + cursor + fragment.find(cleaned)
            exact_end = exact_start + len(cleaned)
            spans.append((exact_start, exact_end, cleaned))
        cursor = end

    remainder = text[cursor:]
    cleaned_remainder = normalize_fragment(remainder)
    if cleaned_remainder:
        exact_start = base_start + cursor + remainder.find(cleaned_remainder)
        exact_end = exact_start + len(cleaned_remainder)
        spans.append((exact_start, exact_end, cleaned_remainder))

    return spans


def get_sentences(text: str) -> List[Tuple[int, int, str]]:
    sentences: List[Tuple[int, int, str]] = []
    for paragraph_start, paragraph_end in iter_paragraph_spans(text):
        paragraph = text[paragraph_start:paragraph_end]
        for start, end, sentence in split_paragraph(paragraph, paragraph_start):
            if len(sentence) <= 1 and sentence not in {"C", "R"}:
                continue
            if all(char in ".*" for char in sentence):
                continue
            sentences.append((start, end, sentence))
    return sentences


def run_segment(input_file: str, output_file: str) -> Dict[str, int]:
    ensure_parent(output_file)
    stats = {
        "jobs_seen": 0,
        "sentences_written": 0,
        "jobs_with_no_sentences": 0,
    }

    with open(input_file, "r", encoding="utf-8") as source, open(
        output_file, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue

            stats["jobs_seen"] += 1
            job = json.loads(line)
            cleaned_text = str(job.get("cleaned_text", ""))
            sentences = get_sentences(cleaned_text)
            if not sentences:
                stats["jobs_with_no_sentences"] += 1

            for index, (start, end, sentence_text) in enumerate(sentences, start=1):
                assert cleaned_text[start:end] == sentence_text
                record = {
                    "job_id": job.get("job_id", ""),
                    "sentence_id": f"{job.get('job_id', '')}_sent_{index:04d}",
                    "sentence_text": sentence_text,
                    "sentence_start_char_in_doc": start,
                    "sentence_end_char_in_doc": end,
                    "sentence_start": start,
                    "sentence_end": end,
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "url": job.get("url", ""),
                    "posted_date": job.get("posted_date", ""),
                    "source_repo": job.get("source_repo", ""),
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["sentences_written"] += 1

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment cleaned job descriptions into sentences.")
    parser.add_argument("--input-file", default="data/processed/jobs_deduped.jsonl")
    parser.add_argument("--output-file", default="data/processed/job_sentences.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_segment(args.input_file, args.output_file)
    print(json.dumps(summary, indent=2))
