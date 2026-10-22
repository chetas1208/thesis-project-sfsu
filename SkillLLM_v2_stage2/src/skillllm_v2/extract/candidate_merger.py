from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from skillllm_v2.config import load_yaml
from skillllm_v2.normalize.skill_alias_builder import build_alias_index, load_alias_records
from skillllm_v2.schema import SkillCandidate, normalize_text, validate_skill
from skillllm_v2.source_aware_policy import candidate_sources, is_anchor_candidate, source_key

DEFAULT_TUNED_CONFIG_PATH = "artifacts/skillllm_v2/candidate_merger/tuned_config.yaml"

# A merged candidate is "add-only" (i.e. not proposed by Raw Mistral at all)
# when its source set does not include "mistral". Such candidates can only
# ever ADD recall on top of Raw Mistral -- they are kept only when both (a)
# their confidence clears a per-source-combination bar, and (b) (optionally)
# their surface form has exact alias-table evidence. Candidates that include
# "mistral" in their source set are always kept unchanged (the merger never
# removes a Raw Mistral candidate).
DEFAULT_CONFIG: dict[str, Any] = {
    "min_confidence_new_span": {
        "gliner": 0.90,
        "dictionary": 0.85,
        # key order matches sorted({"dictionary", "gliner"}) -> "dictionary+gliner"
        "dictionary+gliner": 0.65,
    },
    "default_min_confidence_new_span": 0.80,
    "require_alias_evidence_for_new_span": True,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_merger_config(tuned_config_path: str | None = DEFAULT_TUNED_CONFIG_PATH) -> dict[str, Any]:
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    if tuned_config_path and Path(tuned_config_path).exists():
        tuned = load_yaml(tuned_config_path)
        config = _deep_merge(config, tuned.get("candidate_merger", tuned))
    return config


class CandidateMerger:
    def __init__(
        self,
        alias_equivalents: dict[str, str] | None = None,
        alias_paths: list[str] | None = None,
        config: dict[str, Any] | None = None,
        tuned_config_path: str | None = DEFAULT_TUNED_CONFIG_PATH,
    ) -> None:
        self.alias_equivalents = {normalize_text(k): normalize_text(v) for k, v in (alias_equivalents or {"AWS": "Amazon Web Services"}).items()}
        self.config = config or load_merger_config(tuned_config_path)
        self.alias_index = build_alias_index(load_alias_records(alias_paths))

    def merge(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid = []
        for candidate in candidates:
            validate_skill(candidate)
            valid.append(dict(candidate))
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for candidate in valid:
            groups[(self._canonical_key(candidate), candidate["label"])].append(candidate)

        merged = [self._merge_group(group) for group in groups.values()]
        merged.sort(key=lambda item: (item["start"] if item["start"] is not None else 10**9, item["end"] or 10**9, -item["confidence"]))
        deduped = self._drop_subsumed(merged)
        return self._apply_add_only_policy(deduped)

    def _apply_add_only_policy(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        anchor_spans = [
            (c["start"], c["end"])
            for c in candidates
            if is_anchor_candidate(c) and c["start"] is not None and c["end"] is not None
        ]
        output: list[dict[str, Any]] = []
        for candidate in candidates:
            if is_anchor_candidate(candidate):
                output.append(candidate)
                continue
            min_confidence = self.config["min_confidence_new_span"].get(source_key(candidate), self.config["default_min_confidence_new_span"])
            if float(candidate["confidence"]) < min_confidence:
                continue
            if self.config["require_alias_evidence_for_new_span"] and normalize_text(candidate["surface_form"]) not in self.alias_index:
                continue
            # An add-only candidate that overlaps an anchor span is competing
            # with (not adding to) Raw Mistral's prediction for that text --
            # re-drawing the anchor's boundaries is BoundaryRefiner's job, not
            # the merger's. Keeping it would only ever create a duplicate
            # overlapping span, which costs exact-span precision with no
            # offsetting recall gain.
            if self._overlaps_anchor_span(candidate, anchor_spans):
                continue
            output.append(candidate)
        return output

    @staticmethod
    def _overlaps_anchor_span(candidate: dict[str, Any], anchor_spans: list[tuple[int, int]]) -> bool:
        start, end = candidate["start"], candidate["end"]
        if start is None or end is None:
            return False
        return any(start < a_end and a_start < end for a_start, a_end in anchor_spans)

    def _canonical_key(self, candidate: dict[str, Any]) -> str:
        if candidate.get("canonical_skill_id"):
            return f"id:{candidate['canonical_skill_id']}"
        name = normalize_text(candidate.get("canonical_skill_name") or candidate["surface_form"])
        return self.alias_equivalents.get(name, name)

    def _merge_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        best = max(group, key=lambda item: (len(candidate_sources(item)), item["confidence"]))
        sources = sorted({source for item in group for source in candidate_sources(item)})
        evidences = list(dict.fromkeys(item["evidence"] for item in group if item.get("evidence")))
        confidence = min(0.99, max(item["confidence"] for item in group) + 0.05 * (len(sources) - 1))
        output = dict(best)
        output["confidence"] = confidence
        output["metadata"] = dict(best.get("metadata") or {})
        output["metadata"]["sources"] = sources
        output["metadata"]["evidence"] = evidences
        output["metadata"]["merged_count"] = len(group)
        return output

    def _drop_subsumed(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            start, end = candidate["start"], candidate["end"]
            if start is None or end is None:
                kept.append(candidate)
                continue
            duplicate = False
            for existing in kept:
                estart, eend = existing["start"], existing["end"]
                if estart is None or eend is None or existing["label"] != candidate["label"]:
                    continue
                if estart <= start and end <= eend and normalize_text(candidate["surface_form"]) != normalize_text(existing["surface_form"]):
                    duplicate = candidate["confidence"] <= existing["confidence"]
                    if duplicate:
                        break
            if not duplicate:
                kept.append(candidate)
        return kept
