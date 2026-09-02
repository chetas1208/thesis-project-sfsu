import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests


HARD_SKILLS = {
    "python": ["python"],
    "sql": ["sql"],
    "excel": ["excel", "ms excel", "microsoft excel"],
    "google sheets": ["google sheets"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "pyspark": ["pyspark"],
    "spark": ["spark", "apache spark"],
    "databricks": ["databricks"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab"],
    "linux": ["linux"],
    "unix": ["unix"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript"],
    "react": ["react", "react.js"],
    "node.js": ["node.js", "nodejs"],
    "html": ["html"],
    "css": ["css"],
    "c++": ["c++"],
    "c#": ["c#"],
    "go": ["go", "golang"],
    ".net": [".net", "dotnet"],
    "flask": ["flask"],
    "django": ["django"],
    "fastapi": ["fastapi"],
    "spring boot": ["spring boot"],
    "rest api": ["rest api", "restful api", "restful apis"],
    "graphql": ["graphql"],
    "mongodb": ["mongodb"],
    "postgresql": ["postgresql", "postgres"],
    "mysql": ["mysql"],
    "redis": ["redis"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision"],
    "data science": ["data science"],
    "data analysis": ["data analysis", "data analytics"],
    "data engineering": ["data engineering"],
    "data modeling": ["data modeling", "data modelling"],
    "data warehousing": ["data warehousing", "data warehouse"],
    "etl": ["etl", "elt"],
    "statistics": ["statistics", "statistical analysis"],
    "a/b testing": ["a/b testing", "ab testing"],
    "forecasting": ["forecasting"],
    "automation": ["automation"],
    "scripting": ["scripting"],
    "api development": ["api development", "api design"],
    "microservices": ["microservices"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous delivery"],
    "terraform": ["terraform"],
    "airflow": ["airflow", "apache airflow"],
    "hadoop": ["hadoop"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery"],
    "looker": ["looker"],
    "matlab": ["matlab"],
    "r": ["r programming", "r language", "rstudio"],
    "bi tools": ["bi tools", "business intelligence tools"],
    "prompt engineering": ["prompt engineering"],
    "llms": ["llm", "llms", "large language models"],
}

SOFT_SKILLS = {
    "communication": ["communication", "written communication", "verbal communication", "communication skills"],
    "leadership": ["leadership", "leadership skills"],
    "time management": ["time management"],
    "problem solving": ["problem solving", "problem-solving"],
    "teamwork": ["teamwork"],
    "collaboration": ["collaboration", "collaborative"],
    "adaptability": ["adaptability", "adaptable"],
    "creativity": ["creativity", "creative thinking"],
    "critical thinking": ["critical thinking"],
    "attention to detail": ["attention to detail", "detail-oriented"],
    "interpersonal skills": ["interpersonal skills"],
    "mentoring": ["mentoring", "coaching"],
    "project management": ["project management"],
    "stakeholder management": ["stakeholder management"],
    "analytical skills": ["analytical skills", "analytical mindset"],
    "presentation skills": ["presentation skills", "presenting"],
    "organization": ["organizational skills", "organisational skills"],
    "initiative": ["initiative", "self-starter"],
    "prioritization": ["prioritization", "prioritisation"],
}

SOFT_CONTEXT_REQUIRED = {
    "communication",
    "leadership",
    "time management",
    "problem solving",
    "teamwork",
    "collaboration",
    "adaptability",
    "critical thinking",
    "creativity",
    "interpersonal skills",
    "project management",
    "analytical skills",
    "organization",
}

HARD_CUE_PATTERN = re.compile(
    r"(experience|proficiency|knowledge|familiarity|expertise|background|hands[- ]on|working knowledge|understanding)\s+(in|with|of|using)",
    re.IGNORECASE,
)
SOFT_CUE_PATTERN = re.compile(
    r"(excellent|strong|solid|effective|outstanding|good)\s+.*?(communication|leadership|collaboration|teamwork|problem|analytical|organizational|detail)",
    re.IGNORECASE,
)
LIST_TRIGGER_PATTERN = re.compile(
    r"(experience|proficiency|knowledge|familiarity|skills?|tools?|technologies?|expertise)\s*(in|with|of|using)?\s*:?$",
    re.IGNORECASE,
)


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def looks_like_word_pattern(alias: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9 ./#&+\-]*", alias))


def build_alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias)
    if alias == " r ":
        return re.compile(r"(?<![A-Za-z0-9])R(?![A-Za-z0-9])")
    if looks_like_word_pattern(alias):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def compile_entries(skill_map: Dict[str, Sequence[str]], label: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for normalized, aliases in skill_map.items():
        for alias in aliases:
            entries.append(
                {
                    "label": label,
                    "normalized_skill": normalized,
                    "alias": alias,
                    "pattern": build_alias_pattern(alias.lower()),
                }
            )
    entries.sort(key=lambda item: len(str(item["alias"])), reverse=True)
    return entries


COMPILED_ENTRIES = compile_entries(HARD_SKILLS, "hard_skill") + compile_entries(SOFT_SKILLS, "soft_skill")


def sentence_window(sentence: str, start: int, end: int, radius: int = 40) -> str:
    return sentence[max(0, start - radius) : min(len(sentence), end + radius)]


def in_skill_context(sentence: str, start: int, end: int, label: str) -> bool:
    window = sentence_window(sentence, start, end)
    if label == "hard_skill":
        return bool(HARD_CUE_PATTERN.search(window)) or "," in window or "/" in window
    return bool(SOFT_CUE_PATTERN.search(window)) or "skills" in window.lower() or "ability" in window.lower()


def base_confidence(label: str, contextual: bool) -> float:
    if label == "hard_skill":
        return 0.96 if contextual else 0.91
    return 0.93 if contextual else 0.84


def exact_text(sentence: str, start: int, end: int) -> str:
    return sentence[start:end]


def build_span(sentence: str, start: int, end: int, label: str, normalized_skill: str, source_method: str, confidence: float) -> Dict[str, object]:
    return {
        "text": exact_text(sentence, start, end),
        "label": label,
        "start": start,
        "end": end,
        "confidence": round(confidence, 4),
        "source_method": source_method,
        "normalized_skill": normalized_skill,
    }


def find_dictionary_spans(sentence: str) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    for entry in COMPILED_ENTRIES:
        pattern = entry["pattern"]
        for match in pattern.finditer(sentence):
            start, end = match.start(), match.end()
            label = str(entry["label"])
            normalized_skill = str(entry["normalized_skill"])
            contextual = in_skill_context(sentence, start, end, label)
            if label == "soft_skill" and normalized_skill in SOFT_CONTEXT_REQUIRED and not contextual:
                continue

            method = "dictionary+alignment+context" if contextual else "dictionary+alignment"
            spans.append(
                build_span(
                    sentence,
                    start,
                    end,
                    label,
                    normalized_skill,
                    method,
                    base_confidence(label, contextual),
                )
            )
    return spans


def expand_list_items(sentence: str, existing: List[Dict[str, object]]) -> List[Dict[str, object]]:
    existing_keys = {(span["start"], span["end"], span["label"]) for span in existing}
    new_spans: List[Dict[str, object]] = []
    fragments = re.split(r"[,;/]", sentence)

    search_start = 0
    for fragment in fragments:
        piece = fragment.strip()
        if not piece:
            search_start += len(fragment) + 1
            continue

        index = sentence.find(piece, search_start)
        if index == -1:
            search_start += len(fragment) + 1
            continue

        search_start = index + len(piece)
        if not LIST_TRIGGER_PATTERN.search(piece.split(" ")[0] if " " in piece else piece):
            pass

        for entry in COMPILED_ENTRIES:
            if entry["alias"].lower() != piece.lower():
                continue

            label = str(entry["label"])
            normalized_skill = str(entry["normalized_skill"])
            contextual = in_skill_context(sentence, index, index + len(piece), label)
            if label == "soft_skill" and normalized_skill in SOFT_CONTEXT_REQUIRED and not contextual:
                continue

            key = (index, index + len(piece), label)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            new_spans.append(
                build_span(
                    sentence,
                    index,
                    index + len(piece),
                    label,
                    normalized_skill,
                    "rule+dictionary+alignment",
                    base_confidence(label, True),
                )
            )
    return new_spans


def dedupe_spans(spans: List[Dict[str, object]]) -> List[Dict[str, object]]:
    best_by_key: Dict[Tuple[int, int, str], Dict[str, object]] = {}
    for span in spans:
        key = (int(span["start"]), int(span["end"]), str(span["label"]))
        current = best_by_key.get(key)
        if current is None or float(span.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            best_by_key[key] = span
    return list(best_by_key.values())


def overlap_score(span: Dict[str, object]) -> Tuple[float, int]:
    return (float(span.get("confidence", 0.0)), int(span["end"]) - int(span["start"]))


def resolve_overlaps(spans: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ordered = sorted(
        spans,
        key=lambda span: (int(span["start"]), -(int(span["end"]) - int(span["start"])), -float(span.get("confidence", 0.0))),
    )
    resolved: List[Dict[str, object]] = []
    for span in ordered:
        conflict_index = None
        for index, saved in enumerate(resolved):
            overlaps = not (int(span["end"]) <= int(saved["start"]) or int(saved["end"]) <= int(span["start"]))
            if overlaps:
                conflict_index = index
                break
        if conflict_index is None:
            resolved.append(span)
            continue

        saved = resolved[conflict_index]
        if overlap_score(span) > overlap_score(saved):
            resolved[conflict_index] = span

    return sorted(resolved, key=lambda item: (int(item["start"]), int(item["end"])))


def align_llm_prediction(sentence: str, prediction: Dict[str, object]) -> Optional[Dict[str, object]]:
    text = str(prediction.get("text", "")).strip()
    label = str(prediction.get("label", "")).strip()
    normalized_skill = str(prediction.get("normalized_skill", text.lower())).strip() or text.lower()
    if label not in {"hard_skill", "soft_skill"} or not text:
        return None

    start = sentence.lower().find(text.lower())
    if start == -1:
        return None

    end = start + len(text)
    if sentence[start:end].lower() != text.lower():
        return None

    return build_span(
        sentence,
        start,
        end,
        label,
        normalized_skill,
        "llm+alignment",
        float(prediction.get("confidence", 0.78) or 0.78),
    )


def call_llm(sentence: str, title: str, company: str, timeout_seconds: int = 30) -> List[Dict[str, object]]:
    base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    if not base_url or not api_key or not model:
        return []

    prompt = (
        "Extract only explicit skill mentions from the sentence.\n"
        "Return JSON only as an array of objects with keys: text, label, normalized_skill, confidence.\n"
        "Allowed labels: hard_skill, soft_skill.\n"
        "Rules: exact substrings only, no paraphrases, no inferred skills, no job titles unless clearly a skill.\n\n"
        f"Title: {title}\n"
        f"Company: {company}\n"
        f"Sentence: {sentence}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a careful skill span extractor."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except Exception:
        return []
    return []


def label_sentence(sentence: str, title: str, company: str, use_llm: bool = False) -> List[Dict[str, object]]:
    spans = find_dictionary_spans(sentence)
    spans.extend(expand_list_items(sentence, spans))

    if use_llm:
        llm_predictions = call_llm(sentence, title, company)
        for prediction in llm_predictions:
            span = align_llm_prediction(sentence, prediction)
            if span is not None:
                spans.append(span)

    return resolve_overlaps(dedupe_spans(spans))


def run_weak_labeling(input_file: str, output_file: str, use_llm: bool = False) -> Dict[str, int]:
    ensure_parent(output_file)
    stats = {
        "sentences_seen": 0,
        "sentences_with_spans": 0,
        "total_spans": 0,
    }

    with open(input_file, "r", encoding="utf-8") as source, open(
        output_file, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if not line.strip():
                continue

            stats["sentences_seen"] += 1
            sent_data = json.loads(line)
            sentence = str(sent_data.get("sentence_text", ""))
            title = str(sent_data.get("title", ""))
            company = str(sent_data.get("company", ""))
            spans = label_sentence(sentence, title, company, use_llm=use_llm)
            if spans:
                stats["sentences_with_spans"] += 1
                stats["total_spans"] += len(spans)

            record = {
                "id": sent_data.get("sentence_id", ""),
                "job_id": sent_data.get("job_id", ""),
                "sentence": sentence,
                "spans": spans,
                "title": title,
                "company": company,
                "location": sent_data.get("location", ""),
                "url": sent_data.get("url", ""),
                "posted_date": sent_data.get("posted_date", ""),
                "source_repo": sent_data.get("source_repo", ""),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create weak skill labels for job-description sentences.")
    parser.add_argument("--input-file", default="data/processed/job_sentences.jsonl")
    parser.add_argument("--output-file", default="data/silver/skillspan_style_silver.jsonl")
    parser.add_argument("--use-llm", action="store_true", help="Enable optional OpenAI-compatible LLM supplementation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_weak_labeling(args.input_file, args.output_file, use_llm=args.use_llm)
    print(json.dumps(summary, indent=2))
