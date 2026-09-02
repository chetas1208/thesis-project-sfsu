from __future__ import annotations

import re
from typing import Any


def _find_matches(text: str, needle: str, ignore_case: bool) -> list[tuple[int, int]]:
    flags = re.IGNORECASE if ignore_case else 0
    return [(match.start(), match.end()) for match in re.finditer(re.escape(needle), text, flags)]


def _choose_match(
    matches: list[tuple[int, int]],
    used: set[tuple[int, int]],
    last_start: int | None,
) -> tuple[int, int] | None:
    available = [match for match in matches if match not in used]
    if not available:
        return None
    if last_start is None:
        return min(available, key=lambda item: (item[0], item[1]))

    forward = [match for match in available if match[0] >= last_start]
    if forward:
        return min(forward, key=lambda item: (item[0] - last_start, item[0], item[1]))
    return min(available, key=lambda item: (abs(item[0] - last_start), item[0], item[1]))


def align_predicted_skills(text: str, skills: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    aligned: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    last_start: int | None = None
    unaligned = 0

    for item in skills:
        candidate_text = item["text"]
        match = _choose_match(_find_matches(text, candidate_text, ignore_case=False), used, last_start)
        alignment_method = "exact"
        if match is None:
            match = _choose_match(_find_matches(text, candidate_text, ignore_case=True), used, last_start)
            alignment_method = "case_insensitive"

        if match is None:
            aligned.append(
                {
                    "text": candidate_text,
                    "label": item["label"],
                    "start": None,
                    "end": None,
                    "alignment_method": "unaligned",
                }
            )
            unaligned += 1
            continue

        used.add(match)
        last_start = match[0]
        aligned.append(
            {
                "text": text[match[0] : match[1]],
                "label": item["label"],
                "start": match[0],
                "end": match[1],
                "alignment_method": alignment_method,
            }
        )
    return aligned, unaligned
