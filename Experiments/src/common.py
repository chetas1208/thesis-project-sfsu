import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "benchmark.json"


def load_env_file(path: Path | None = None) -> None:
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: str | Path | None = None) -> Dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_benchmark_env(config: Dict) -> None:
    cache_root = resolve_path(config["paths"]["cache_root"])
    def _env_path(name: str, default: Path) -> str:
        raw_value = os.environ.get(name)
        if raw_value:
            return str(resolve_path(raw_value))
        return str(default)

    hf_home = Path(_env_path("HF_HOME", cache_root / "huggingface"))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ["TRANSFORMERS_CACHE"] = _env_path("TRANSFORMERS_CACHE", hf_home / "transformers")
    os.environ["HUGGINGFACE_HUB_CACHE"] = _env_path("HUGGINGFACE_HUB_CACHE", hf_home / "hub")
    os.environ["PIP_CACHE_DIR"] = _env_path("PIP_CACHE_DIR", cache_root / "pip")
    os.environ["TMPDIR"] = _env_path("TMPDIR", cache_root / "tmp")
    os.environ["TMP"] = os.environ["TMPDIR"]
    os.environ["TEMP"] = os.environ["TMPDIR"]
    os.environ["WANDB_DIR"] = _env_path("WANDB_DIR", cache_root / "wandb")
    os.environ["WANDB_CACHE_DIR"] = _env_path("WANDB_CACHE_DIR", cache_root / "wandb")
    os.environ["OPENAI_CACHE_DIR"] = _env_path("OPENAI_CACHE_DIR", cache_root / "openai")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json_records(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.split("\n") if line.strip()]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def first_present(mapping: Dict, keys: Sequence[str], default=None):
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def write_json(path: Path, payload: Dict | List) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def rebuild_text_from_tokens(tokens: Sequence[Dict]) -> str:
    pieces = []
    for token in tokens:
        pieces.append(token["text"])
        if token.get("ws", False):
            pieces.append(" ")
    return "".join(pieces)


def derive_tokens_from_text(text: str, token_texts: Sequence[str]) -> List[Dict]:
    tokens: List[Dict] = []
    cursor = 0
    for token_text in token_texts:
        start = text.find(token_text, cursor)
        if start < 0:
            raise ValueError(f"Could not align token '{token_text}' in text.")
        end = start + len(token_text) - 1
        next_index = end + 1
        ws = next_index < len(text) and text[next_index].isspace()
        cursor = next_index
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        tokens.append({"text": token_text, "start": start, "end": end, "ws": ws})
    return tokens


def normalize_tokens(record: Dict) -> List[Dict]:
    raw_tokens = record.get("tokens") or []
    if not raw_tokens:
        raise ValueError("Record is missing tokens.")
    if isinstance(raw_tokens[0], str):
        text = record.get("text")
        if not text:
            raise ValueError("String tokens require record.text.")
        return derive_tokens_from_text(text, raw_tokens)
    normalized = []
    for index, token in enumerate(raw_tokens):
        if "text" not in token:
            raise ValueError(f"Token {index} is missing text.")
        normalized_token = dict(token)
        if "start" not in normalized_token or "end" not in normalized_token:
            text = record.get("text")
            if not text:
                raise ValueError("Token offsets are missing and record.text is unavailable.")
            rebuilt = derive_tokens_from_text(text, [item["text"] for item in raw_tokens])
            return rebuilt
        if "ws" not in normalized_token:
            text = record.get("text")
            if text:
                next_index = int(normalized_token["end"]) + 1
                normalized_token["ws"] = next_index < len(text) and text[next_index].isspace()
            else:
                normalized_token["ws"] = index < len(raw_tokens) - 1
        normalized.append(
            {
                "text": str(normalized_token["text"]),
                "start": int(normalized_token["start"]),
                "end": int(normalized_token["end"]),
                "ws": bool(normalized_token["ws"]),
            }
        )
    return normalized


def normalize_spans(record: Dict, tokens: Sequence[Dict], collapse_to_skill: bool) -> List[Dict]:
    text = record.get("text") or rebuild_text_from_tokens(tokens)
    raw_spans = record.get("spans")
    if raw_spans is None:
        raw_spans = record.get("entities") or []
    normalized = []
    for span in raw_spans:
        if "start" in span and "end" in span:
            start = int(span["start"])
            end = int(span["end"])
        elif "token_start" in span and "token_end" in span:
            token_start = int(span["token_start"])
            token_end = int(span["token_end"])
            start = int(tokens[token_start]["start"])
            end = int(tokens[token_end]["end"])
        elif "start_token" in span and "end_token" in span:
            token_start = int(span["start_token"])
            token_end = int(span["end_token"])
            start = int(tokens[token_start]["start"])
            end = int(tokens[token_end]["end"])
        else:
            raise ValueError(f"Unsupported span format: {span}")
        raw_label = first_present(span, ("label", "entity_label"), "SKILL")
        label = "SKILL" if collapse_to_skill else str(raw_label).upper()
        normalized.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": span.get("text", text[start : end + 1]),
            }
        )
    normalized.sort(key=lambda item: (item["start"], item["end"]))
    return normalized


def normalize_normalized_skills(record: Dict, spans: Sequence[Dict]) -> List[Dict]:
    raw_items = record.get("normalized_skills") or []
    normalized = []
    seen = set()
    for item in raw_items:
        payload = {
            "text": str(first_present(item, ("text", "surface_form"), "")),
            "canonical_name": first_present(item, ("canonical_name", "canonical", "name"), None),
            "taxonomy": first_present(item, ("taxonomy",), None),
            "skill_id": first_present(item, ("skill_id", "id"), None),
        }
        if not payload["text"]:
            continue
        if payload["canonical_name"] in ("", None):
            payload["canonical_name"] = payload["text"]
        dedupe_key = (
            payload["text"],
            payload["canonical_name"],
            payload["taxonomy"],
            payload["skill_id"],
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(payload)

    if normalized:
        return normalized

    for span in spans:
        dedupe_key = (span["text"], span["text"], None, None)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(
            {
                "text": span["text"],
                "canonical_name": span["text"],
                "taxonomy": None,
                "skill_id": None,
            }
        )
    return normalized


def normalize_record(record: Dict, record_id: str, collapse_to_skill: bool = True) -> Dict:
    tokens = normalize_tokens(record)
    text = record.get("text") or rebuild_text_from_tokens(tokens)
    spans = normalize_spans(record, tokens, collapse_to_skill)
    sentence_id = str(first_present(record, ("id", "sentence_id"), record_id))
    resume_id = first_present(record, ("resume_id", "document_id", "resumeId"), None)
    normalized = {
        "id": sentence_id,
        "resume_id": str(resume_id) if resume_id not in (None, "") else None,
        "source": first_present(record, ("source",), None),
        "role_hint": first_present(record, ("role_hint", "role", "category"), None),
        "section_name": first_present(record, ("section_name", "section"), None),
        "text": text,
        "tokens": tokens,
        "spans": spans,
        "normalized_skills": normalize_normalized_skills(record, spans),
    }
    return normalized


def flatten_resume_annotation(record: Dict, record_id_prefix: str, collapse_to_skill: bool = True) -> List[Dict]:
    sentences = record.get("sentences") or []
    if not sentences:
        raise ValueError(
            "Resume-style annotations must include a non-empty 'sentences' array before benchmark preparation."
        )

    resume_id = str(first_present(record, ("resume_id", "id", "document_id"), record_id_prefix))
    source = first_present(record, ("source",), None)
    role_hint = first_present(record, ("role_hint", "role", "category"), None)
    document_text = first_present(record, ("text",), None)

    normalized_sentences = []
    for index, sentence in enumerate(sentences, start=1):
        payload = dict(sentence)
        payload.setdefault("resume_id", resume_id)
        payload.setdefault("source", source)
        payload.setdefault("role_hint", role_hint)
        payload.setdefault("sentence_id", f"{resume_id}_s{index}")
        normalized = normalize_record(
            payload,
            record_id=str(payload["sentence_id"]),
            collapse_to_skill=collapse_to_skill,
        )
        normalized["resume_id"] = resume_id
        normalized["source"] = source
        normalized["role_hint"] = role_hint
        normalized["section_name"] = first_present(payload, ("section_name", "section"), None)
        if document_text:
            normalized["document_text"] = document_text
        normalized_sentences.append(normalized)
    return normalized_sentences


def validate_record(record: Dict, labels: Sequence[str]) -> List[str]:
    errors: List[str] = []
    if "id" not in record:
        errors.append("missing id")
    if "text" not in record:
        errors.append("missing text")
    tokens = record.get("tokens") or []
    spans = record.get("spans") or []
    text = record.get("text", "")
    previous_end = -1
    for index, token in enumerate(tokens):
        for field in ("text", "start", "end", "ws"):
            if field not in token:
                errors.append(f"token {index} missing {field}")
        start = int(token["start"])
        end = int(token["end"])
        if start > end:
            errors.append(f"token {index} has start > end")
        if start <= previous_end:
            errors.append(f"token {index} overlaps previous token")
        previous_end = end
        if text and text[start : end + 1] != token["text"]:
            errors.append(f"token {index} text does not match offsets")
    last_span_end = -1
    for index, span in enumerate(spans):
        for field in ("start", "end", "label", "text"):
            if field not in span:
                errors.append(f"span {index} missing {field}")
        start = int(span["start"])
        end = int(span["end"])
        if start > end:
            errors.append(f"span {index} has start > end")
        if start <= last_span_end:
            errors.append(f"span {index} overlaps previous span")
        last_span_end = end
        if text and (start < 0 or end >= len(text)):
            errors.append(f"span {index} exceeds text bounds")
        if span["label"] not in labels:
            errors.append(f"span {index} has unsupported label '{span['label']}'")
        if text and text[start : end + 1] != span["text"]:
            errors.append(f"span {index} text does not match offsets")
    return errors


def deterministic_split(records: List[Dict], seed: int, ratios: Dict[str, float]) -> Dict[str, List[Dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * ratios["train"])
    dev_count = int(total * ratios["dev"])
    test_count = total - train_count - dev_count
    if total >= 3:
        train_count = max(1, train_count)
        dev_count = max(1, dev_count)
        test_count = max(1, total - train_count - dev_count)
        overflow = train_count + dev_count + test_count - total
        if overflow > 0:
            train_count = max(1, train_count - overflow)
            test_count = total - train_count - dev_count
    train_split = shuffled[:train_count]
    dev_split = shuffled[train_count : train_count + dev_count]
    test_split = shuffled[train_count + dev_count :]
    return {"train": train_split, "dev": dev_split, "test": test_split}


def bio_tags_from_spans(tokens: Sequence[Dict], spans: Sequence[Dict]) -> List[str]:
    tags = ["O"] * len(tokens)
    for span in spans:
        covered = [
            idx
            for idx, token in enumerate(tokens)
            if int(token["start"]) >= int(span["start"]) and int(token["end"]) <= int(span["end"])
        ]
        if not covered:
            continue
        tags[covered[0]] = "B"
        for idx in covered[1:]:
            tags[idx] = "I"
    return tags


def labeled_bio_tags_from_spans(tokens: Sequence[Dict], spans: Sequence[Dict], label: str = "SKILL") -> List[str]:
    tags = ["O"] * len(tokens)
    for span in spans:
        covered = [
            idx
            for idx, token in enumerate(tokens)
            if int(token["start"]) >= int(span["start"]) and int(token["end"]) <= int(span["end"])
        ]
        if not covered:
            continue
        tags[covered[0]] = f"B-{label}"
        for idx in covered[1:]:
            tags[idx] = f"I-{label}"
    return tags


def spans_from_bio_tags(tokens: Sequence[Dict], tags: Sequence[str], label: str = "SKILL") -> List[Dict]:
    spans: List[Dict] = []
    current_start = None
    current_end = None
    for idx, tag in enumerate(tags):
        normalized_tag = tag.split("-", 1)[0]
        if normalized_tag == "B":
            if current_start is not None:
                spans.append(
                    {
                        "start": int(tokens[current_start]["start"]),
                        "end": int(tokens[current_end]["end"]),
                        "label": label,
                        "text": "",
                    }
                )
            current_start = idx
            current_end = idx
        elif normalized_tag == "I" and current_start is not None:
            current_end = idx
        else:
            if current_start is not None:
                spans.append(
                    {
                        "start": int(tokens[current_start]["start"]),
                        "end": int(tokens[current_end]["end"]),
                        "label": label,
                        "text": "",
                    }
                )
                current_start = None
                current_end = None
    if current_start is not None:
        spans.append(
            {
                "start": int(tokens[current_start]["start"]),
                "end": int(tokens[current_end]["end"]),
                "label": label,
                "text": "",
            }
        )
    return spans


def finalize_span_text(text: str, spans: Sequence[Dict]) -> List[Dict]:
    finalized = []
    for span in spans:
        updated = dict(span)
        updated["text"] = text[int(updated["start"]) : int(updated["end"]) + 1]
        finalized.append(updated)
    return finalized


def parse_conll_prediction_file(path: Path, canonical_records: Sequence[Dict]) -> List[Dict]:
    content = path.read_text(encoding="utf-8").strip()
    blocks = [block for block in content.split("\n\n") if block.strip()]
    if len(blocks) != len(canonical_records):
        raise ValueError(
            f"Prediction block count mismatch for {path}: {len(blocks)} vs {len(canonical_records)} canonical records."
        )
    predictions = []
    for record, block in zip(canonical_records, blocks):
        tags: List[str] = []
        for line in block.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            tags.append(parts[2].strip())
        spans = finalize_span_text(record["text"], spans_from_bio_tags(record["tokens"], tags))
        predictions.append({"id": record["id"], "predicted_spans": spans})
    return predictions


def exact_span_metrics(gold_records: Sequence[Dict], prediction_records: Sequence[Dict]) -> Dict[str, float]:
    gold_map = {
        record["id"]: {
            (int(span["start"]), int(span["end"]), span["label"])
            for span in record.get("spans", [])
        }
        for record in gold_records
    }
    pred_map = {
        record["id"]: {
            (int(span["start"]), int(span["end"]), span["label"])
            for span in record.get("predicted_spans", [])
        }
        for record in prediction_records
    }
    tp = fp = fn = 0
    for record_id, gold_spans in gold_map.items():
        pred_spans = pred_map.get(record_id, set())
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_run_metadata(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    write_json(path, payload)


def read_optional_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
