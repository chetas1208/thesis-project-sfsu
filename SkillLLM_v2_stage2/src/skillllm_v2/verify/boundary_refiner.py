"""boundary_refiner.py

Generates non-destructive boundary-edit *candidates* for each extracted skill
span. The refiner never directly replaces a Raw Mistral (or any other source)
span -- it proposes variants (original, trimmed-left, trimmed-right,
head-skill, split-on-conjunction, taxonomy-normalized) and hands them to
:class:`skillllm_v2.verify.boundary_selector.BoundarySelector`, which decides
whether any variant has strong enough evidence to replace the original.

Previously this module unconditionally replaced every span with a trimmed /
split version, which dropped Exact Span F1 from 0.675 (Raw Mistral) to 0.570
and Soft Skill F1 from 0.610 to 0.426 (see
``reports/skillllm_v2/wrapper_error_analysis.md``). The selector defaults to
"preserve original" unless a candidate has exact alias/taxonomy evidence.
"""
from __future__ import annotations

import re
from typing import Any

from skillllm_v2.normalize.skill_alias_builder import build_alias_index, load_alias_records
from skillllm_v2.schema import normalize_text, validate_skill
from skillllm_v2.verify.boundary_selector import BoundarySelector

LEADING_FILLERS = [
    "experience with",
    "knowledge of",
    "familiarity with",
    "proficiency in",
    "strong",
    "excellent",
    "team",
]
TRAILING_FILLERS = ["programming", "skills", "skill", "experience", "knowledge", "tools", "technologies", "models"]
SPLIT_DROP_TERMS = {"team", "skills", "skill", "technologies", "tools"}


def _variant(candidate: dict[str, Any], spans: list[tuple[int, int, str]], edit_type: str, edit_reason: str, confidence_delta: float) -> dict[str, Any]:
    return {
        "original_surface_form": candidate["surface_form"],
        "candidate_surface_form": " and ".join(s[2] for s in spans) if len(spans) > 1 else spans[0][2],
        "original_start": candidate["start"],
        "original_end": candidate["end"],
        "candidate_start": spans[0][0],
        "candidate_end": spans[-1][1],
        "edit_type": edit_type,
        "edit_reason": edit_reason,
        "confidence_delta": confidence_delta,
        "source": "boundary_refiner",
        "spans": [{"start": s[0], "end": s[1], "surface_form": s[2]} for s in spans],
    }


class BoundaryRefiner:
    def __init__(self, alias_paths: list[str] | None = None) -> None:
        self.alias_index = build_alias_index(load_alias_records(alias_paths))

    def refine(self, text: str, candidates: list[dict[str, Any]], selector: BoundarySelector | None = None) -> list[dict[str, Any]]:
        selector = selector or BoundarySelector()
        output: list[dict[str, Any]] = []
        for candidate in candidates:
            validate_skill(candidate)
            variants = self.generate_variants(text, candidate)
            output.extend(selector.select(text, candidate, variants))
        return output

    def generate_variants(self, text: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        if candidate.get("start") is None or candidate.get("end") is None:
            return [self._original(candidate)]
        variants = [self._original(candidate)]
        variants.extend(self._trim_leading(text, candidate))
        variants.extend(self._trim_trailing(text, candidate))
        variants.extend(self._head_skill(text, candidate))
        variants.extend(self._split_conjunction(text, candidate))
        variants.extend(self._taxonomy_normalized(text, candidate))
        variants.extend(self._realign_word_boundary(text, candidate))
        return variants

    def _original(self, candidate: dict[str, Any]) -> dict[str, Any]:
        span = (candidate["start"], candidate["end"], candidate["surface_form"])
        return _variant(candidate, [span], "original", "preserve_original", 0.0)

    def _trim_leading(self, text: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        start, end = candidate["start"], candidate["end"]
        surface = text[start:end]
        for filler in LEADING_FILLERS:
            match = re.match(rf"^{re.escape(filler)}\s+", surface, re.IGNORECASE)
            if match:
                new_start = start + match.end()
                new_surface = text[new_start:end]
                if new_surface.strip():
                    return [_variant(candidate, [(new_start, end, new_surface)], "trim_leading_filler", f"strip leading filler '{filler}'", -0.01)]
        return []

    def _trim_trailing(self, text: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        start, end = candidate["start"], candidate["end"]
        surface = text[start:end]
        for filler in TRAILING_FILLERS:
            match = re.search(rf"\s+{re.escape(filler)}$", surface, re.IGNORECASE)
            if match and len(surface[: match.start()].strip()) >= 2:
                new_end = start + match.start()
                new_surface = text[start:new_end]
                return [_variant(candidate, [(start, new_end, new_surface)], "trim_trailing_filler", f"strip trailing filler '{filler}'", -0.03)]
        return []

    def _head_skill(self, text: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        start, end = candidate["start"], candidate["end"]
        surface = text[start:end]
        tokens = surface.split()
        if len(tokens) < 2 or len(tokens[0]) < 2:
            return []
        head_start = start
        head_end = start + len(tokens[0])
        return [_variant(candidate, [(head_start, head_end, text[head_start:head_end])], "head_skill", "use first token as head skill", -0.04)]

    def _split_conjunction(self, text: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        start, end = candidate["start"], candidate["end"]
        surface = text[start:end]
        if " and " not in surface.lower() and "/" not in surface:
            return []
        parts = [part.strip(" ,.;:") for part in re.split(r"\s+and\s+|/", surface, flags=re.IGNORECASE) if part.strip(" ,.;:")]
        if len(parts) < 2:
            return []
        spans: list[tuple[int, int, str]] = []
        cursor = start
        for part in parts:
            for filler in LEADING_FILLERS:
                match = re.match(rf"^{re.escape(filler)}\s+", part, re.IGNORECASE)
                if match:
                    part = part[match.end():]
                    break
            normalized = normalize_text(part)
            if not part or normalized in SPLIT_DROP_TERMS:
                continue
            local = text.find(part, cursor)
            if local < 0:
                local = text.find(part, start)
            if local < 0:
                continue
            spans.append((local, local + len(part), part))
            cursor = local + len(part)
        if len(spans) < 2:
            return []
        return [_variant(candidate, spans, "split_conjunction", "split on conjunction/slash into separate skills", -0.05)]

    def _taxonomy_normalized(self, text: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        start, end = candidate["start"], candidate["end"]
        new_start, new_end = start, end
        applied: list[str] = []
        for filler in LEADING_FILLERS:
            match = re.match(rf"^{re.escape(filler)}\s+", text[new_start:end], re.IGNORECASE)
            if match:
                new_start += match.end()
                applied.append(f"leading:{filler}")
                break
        for filler in TRAILING_FILLERS:
            match = re.search(rf"\s+{re.escape(filler)}$", text[new_start:new_end], re.IGNORECASE)
            if match and len(text[new_start : new_start + match.start()].strip()) >= 2:
                new_end = new_start + match.start()
                applied.append(f"trailing:{filler}")
                break
        if len(applied) < 2:
            return []
        candidate_surface = text[new_start:new_end]
        if normalize_text(candidate_surface) not in self.alias_index:
            return []
        return [_variant(candidate, [(new_start, new_end, candidate_surface)], "taxonomy_normalized", f"core span matches taxonomy alias ({', '.join(applied)})", 0.05)]

    def _realign_word_boundary(self, text: str, candidate: dict[str, Any]) -> list[dict[str, Any]]:
        start, end = candidate["start"], candidate["end"]
        surface = candidate["surface_form"]
        # Check if current span already sits cleanly at word boundaries
        is_wb = (start == 0 or not text[start - 1].isalnum()) and (end >= len(text) or not text[end].isalnum())
        if is_wb:
            return []
        # Find exact case standalone word in text
        matches = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(surface)}\b", text)]
        if not matches:
            matches = [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(surface)}\b", text, re.IGNORECASE)]
        if matches:
            ns, ne = matches[0]
            return [_variant(candidate, [(ns, ne, text[ns:ne])], "realign_word_boundary", "realign subword span to standalone word boundary", 0.10)]
        return []
