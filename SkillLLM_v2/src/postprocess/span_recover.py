from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from src.data.dataset_utils import normalize_label


PRIORITY = {"exact": 3, "normalized": 2, "fuzzy": 1}


@dataclass
class RecoveryTrace:
    method: str
    score: float
    accepted: bool
    reason: str
    start: int | None = None
    end: int | None = None


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_normalized_index(text: str, lowercase: bool = False) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    previous_was_space = False

    for index, char in enumerate(text):
        if char.isspace():
            if normalized_chars and not previous_was_space:
                normalized_chars.append(" ")
                index_map.append(index)
            previous_was_space = True
            continue
        previous_was_space = False
        normalized_chars.append(char.lower() if lowercase else char)
        index_map.append(index)

    while normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        index_map.pop()

    return "".join(normalized_chars), index_map


class SpanRecoverer:
    def __init__(self, threshold: float = 0.92, debug_dir: str | None = None) -> None:
        self.threshold = threshold
        self.debug_dir = Path(debug_dir) if debug_dir else None
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def recover_example(
        self,
        example_id: str,
        text: str,
        predicted_skills: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        recovered: list[dict[str, Any]] = []
        trace_log: dict[str, Any] = {"id": example_id, "text": text, "attempts": [], "rejected": []}

        for item in predicted_skills:
            candidate, attempt_traces = self._recover_single(text, item)
            trace_log["attempts"].append({"item": item, "traces": [trace.__dict__ for trace in attempt_traces]})
            if candidate is None:
                trace_log["rejected"].append(item)
                continue
            recovered.append(candidate)

        recovered = self._collapse_duplicates(recovered)
        recovered = self._resolve_overlaps(recovered)

        if self.debug_dir is not None and trace_log["rejected"]:
            debug_file = self.debug_dir / f"{example_id}.json"
            debug_file.write_text(json.dumps(trace_log, indent=2, ensure_ascii=False), encoding="utf-8")

        return recovered, trace_log

    def _recover_single(self, source_text: str, item: dict[str, Any]) -> tuple[dict[str, Any] | None, list[RecoveryTrace]]:
        traces: list[RecoveryTrace] = []
        raw_text = item.get("text")
        label = normalize_label(item.get("label"))
        if not isinstance(raw_text, str) or not raw_text.strip() or label is None:
            traces.append(RecoveryTrace(method="rejected", score=0.0, accepted=False, reason="missing_text_or_label"))
            return None, traces

        candidate_text = raw_text.strip()

        candidate = self._recover_from_indices(source_text, candidate_text, label, item)
        if candidate is not None:
            traces.append(RecoveryTrace(method="exact", score=1.0, accepted=True, reason="valid_indices", start=candidate["start"], end=candidate["end"]))
            return candidate, traces
        traces.append(RecoveryTrace(method="exact", score=0.0, accepted=False, reason="invalid_or_missing_indices"))

        exact_match = self._find_exact_match(source_text, candidate_text, item.get("start"))
        if exact_match is not None:
            start, end = exact_match
            traces.append(RecoveryTrace(method="exact", score=1.0, accepted=True, reason="exact_substring_search", start=start, end=end))
            return self._build_recovered_skill(source_text, label, start, end, "exact", 1.0), traces
        traces.append(RecoveryTrace(method="exact", score=0.0, accepted=False, reason="exact_substring_not_found"))

        normalized_match = self._find_normalized_match(source_text, candidate_text)
        if normalized_match is not None:
            start, end, score, reason = normalized_match
            traces.append(RecoveryTrace(method="normalized", score=score, accepted=True, reason=reason, start=start, end=end))
            return self._build_recovered_skill(source_text, label, start, end, "normalized", score), traces
        traces.append(RecoveryTrace(method="normalized", score=0.0, accepted=False, reason="normalized_match_not_found"))

        fuzzy_match = self._find_fuzzy_match(source_text, candidate_text)
        if fuzzy_match is not None:
            start, end, score = fuzzy_match
            accepted = score >= self.threshold
            traces.append(RecoveryTrace(method="fuzzy", score=score, accepted=accepted, reason="fuzzy_ngram_search", start=start, end=end))
            if accepted:
                return self._build_recovered_skill(source_text, label, start, end, "fuzzy", score), traces
        else:
            traces.append(RecoveryTrace(method="fuzzy", score=0.0, accepted=False, reason="fuzzy_match_not_found"))

        traces.append(RecoveryTrace(method="rejected", score=0.0, accepted=False, reason="no_grounded_span"))
        return None, traces

    def _recover_from_indices(self, source_text: str, candidate_text: str, label: str, item: dict[str, Any]) -> dict[str, Any] | None:
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            return None
        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            return None
        if start < 0 or end > len(source_text) or start >= end:
            return None
        if source_text[start:end] != candidate_text:
            return None
        return self._build_recovered_skill(source_text, label, start, end, "exact", 1.0)

    def _build_recovered_skill(self, source_text: str, label: str, start: int, end: int, recovery_type: str, score: float) -> dict[str, Any]:
        return {
            "text": source_text[start:end],
            "label": label,
            "start": start,
            "end": end,
            "recovery_type": recovery_type,
            "recovery_score": round(score, 4),
        }

    def _find_exact_match(self, source_text: str, candidate_text: str, preferred_start: Any) -> tuple[int, int] | None:
        matches = [(match.start(), match.end()) for match in re.finditer(re.escape(candidate_text), source_text)]
        if not matches:
            return None
        if preferred_start is not None:
            try:
                preferred = int(preferred_start)
            except (TypeError, ValueError):
                preferred = None
            if preferred is not None:
                matches.sort(key=lambda item: abs(item[0] - preferred))
        return matches[0]

    def _find_normalized_match(self, source_text: str, candidate_text: str) -> tuple[int, int, float, str] | None:
        normalized_source, source_index = build_normalized_index(source_text, lowercase=False)
        normalized_candidate = normalize_whitespace(candidate_text)
        match_start = normalized_source.find(normalized_candidate)
        if match_start != -1:
            match_end = match_start + len(normalized_candidate)
            start = source_index[match_start]
            end = source_index[match_end - 1] + 1
            return start, end, 1.0, "whitespace_normalized_search"

        normalized_source_lower, source_index_lower = build_normalized_index(source_text, lowercase=True)
        normalized_candidate_lower = normalize_whitespace(candidate_text).lower()
        match_start = normalized_source_lower.find(normalized_candidate_lower)
        if match_start != -1:
            match_end = match_start + len(normalized_candidate_lower)
            start = source_index_lower[match_start]
            end = source_index_lower[match_end - 1] + 1
            score = fuzz.ratio(source_text[start:end].lower(), candidate_text.lower()) / 100.0
            return start, end, score, "case_insensitive_search"

        return None

    def _find_fuzzy_match(self, source_text: str, candidate_text: str) -> tuple[int, int, float] | None:
        token_matches = list(re.finditer(r"\S+", source_text))
        if not token_matches:
            return None

        target_token_count = max(1, len(candidate_text.split()))
        min_window = max(1, target_token_count - 2)
        max_window = min(len(token_matches), target_token_count + 3)

        best: tuple[int, int, float] | None = None
        normalized_candidate = normalize_whitespace(candidate_text).lower()

        for window_size in range(min_window, max_window + 1):
            for start_index in range(0, len(token_matches) - window_size + 1):
                end_index = start_index + window_size - 1
                start = token_matches[start_index].start()
                end = token_matches[end_index].end()
                candidate_span = source_text[start:end]
                score = fuzz.ratio(normalize_whitespace(candidate_span).lower(), normalized_candidate) / 100.0
                if best is None or score > best[2]:
                    best = (start, end, score)

        return best

    def _collapse_duplicates(self, recovered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in recovered:
            key = (item["label"], item["start"], item["end"])
            current = deduped.get(key)
            if current is None:
                deduped[key] = item
                continue
            if PRIORITY[item["recovery_type"]] > PRIORITY[current["recovery_type"]]:
                deduped[key] = item
                continue
            if item["recovery_score"] > current["recovery_score"]:
                deduped[key] = item
        return list(deduped.values())

    def _resolve_overlaps(self, recovered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(
            recovered,
            key=lambda item: (
                -PRIORITY[item["recovery_type"]],
                -item["recovery_score"],
                -(item["end"] - item["start"]),
                item["start"],
            ),
        )
        kept: list[dict[str, Any]] = []
        for item in ordered:
            if any(self._overlaps_same_label(item, existing) for existing in kept):
                continue
            kept.append(item)
        return sorted(kept, key=lambda item: (item["start"], item["end"], item["label"]))

    @staticmethod
    def _overlaps_same_label(left: dict[str, Any], right: dict[str, Any]) -> bool:
        if left["label"] != right["label"]:
            return False
        return left["start"] < right["end"] and right["start"] < left["end"]
