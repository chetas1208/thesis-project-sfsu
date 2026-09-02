from __future__ import annotations

import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
from typing import Any

from skillllm_v2.normalize.skill_alias_builder import DEFAULT_OUTPUT as DEFAULT_ALIAS_TABLE_PATH
from skillllm_v2.normalize.skill_alias_builder import load_alias_records
from skillllm_v2.schema import normalize_text

# Priority used to pick an alias-table entry's reported `taxonomy` and
# `canonical_skill_name` when multiple alias rows share a `canonical_skill_id`
# (e.g. an external taxonomy export and a SkillSpan-gold-derived row both map
# to "python"). Lower number wins.
TAXONOMY_PRIORITY = {"ESCO": 0, "ONET": 1, "LIGHTCAST": 2, "CUSTOM": 3, "SKILLSPAN_GOLD": 4}


class TaxonomyLinker:
    def __init__(
        self,
        taxonomy_paths: list[str] | None = None,
        alias_table_paths: list[str] | None = None,
        fuzzy_threshold: float = 0.88,
        dense_model_name: str | None = None,
    ) -> None:
        self.taxonomy_paths = [Path(path) for path in taxonomy_paths or []]
        self.alias_table_paths = alias_table_paths if alias_table_paths is not None else [DEFAULT_ALIAS_TABLE_PATH]
        self.fuzzy_threshold = fuzzy_threshold
        self.dense_model_name = dense_model_name
        self._dense_model: Any | None = None
        self._dense_vectors: Any | None = None
        self.entries = self._load_entries()
        if not self.entries:
            self.entries = [
                _entry("python", "Python", "CUSTOM", ["Python"]),
                _entry("aws", "Amazon Web Services", "CUSTOM", ["AWS", "Amazon Web Services"]),
                _entry("communication", "communication", "CUSTOM", ["communication"]),
                _entry("machine-learning", "machine learning", "CUSTOM", ["machine learning"]),
            ]
        self._bm25_docs = [_terms(" ".join(entry["aliases"] + [entry["canonical_skill_name"]])) for entry in self.entries]
        self._bm25_avg_len = sum(len(doc) for doc in self._bm25_docs) / max(1, len(self._bm25_docs))
        self._bm25_doc_freq: dict[str, int] = {}
        for doc in self._bm25_docs:
            for term in set(doc):
                self._bm25_doc_freq[term] = self._bm25_doc_freq.get(term, 0) + 1

    def link(self, surface: str) -> dict[str, Any] | None:
        norm = normalize_text(surface)
        for entry in self.entries:
            if norm in entry["aliases_norm"]:
                return self._result(entry, 1.0, "alias")
        for entry in self.entries:
            if norm == normalize_text(entry["canonical_skill_name"]):
                return self._result(entry, 0.96, "normalized")
        best = None
        best_score = 0.0
        for entry in self.entries:
            for alias in entry["aliases_norm"]:
                score = SequenceMatcher(None, norm, alias).ratio()
                if score > best_score:
                    best = entry
                    best_score = score
        if best and best_score >= self.fuzzy_threshold:
            return self._result(best, best_score, "fuzzy")
        bm25 = self._bm25(surface)
        if bm25 is not None:
            return bm25
        dense = self._dense(surface)
        if dense is not None:
            return dense
        return None

    def link_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for candidate in candidates:
            item = dict(candidate)
            match = self.link(candidate["surface_form"])
            if match:
                item["canonical_skill_id"] = match["canonical_skill_id"]
                item["canonical_skill_name"] = match["canonical_skill_name"]
                item["metadata"] = dict(candidate.get("metadata") or {})
                item["metadata"]["taxonomy_link"] = match
            output.append(item)
        return output

    def _load_entries(self) -> list[dict[str, Any]]:
        entries = self._load_legacy_taxonomy_files()
        if not entries:
            entries = self._load_alias_table_entries()
        return entries

    def _load_alias_table_entries(self) -> list[dict[str, Any]]:
        records = load_alias_records(self.alias_table_paths)
        return _group_alias_records_into_entries(records)

    def _load_legacy_taxonomy_files(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in self.taxonomy_paths:
            if not path.exists():
                continue
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("skills", [])
            elif path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            else:
                rows = [{"canonical_skill_name": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                name = row.get("canonical_skill_name") or row.get("name") or row.get("skill")
                if not name:
                    continue
                aliases = row.get("aliases") or row.get("alias") or name
                if isinstance(aliases, str):
                    alias_list = [part.strip() for part in aliases.replace(";", "|").replace(",", "|").split("|") if part.strip()]
                else:
                    alias_list = [str(value) for value in aliases]
                entries.append(_entry(row.get("canonical_skill_id") or row.get("id") or normalize_text(name), name, row.get("taxonomy") or "CUSTOM", alias_list))
        return entries

    def _result(self, entry: dict[str, Any], confidence: float, method: str) -> dict[str, Any]:
        return {
            "canonical_skill_id": entry["canonical_skill_id"],
            "canonical_skill_name": entry["canonical_skill_name"],
            "taxonomy": entry["taxonomy"],
            "link_confidence": float(confidence),
            "link_method": method,
        }

    def _bm25(self, surface: str) -> dict[str, Any] | None:
        query_terms = _terms(surface)
        if not query_terms:
            return None
        docs = self._bm25_docs
        doc_count = len(docs)
        avg_len = self._bm25_avg_len
        best_entry = None
        best_score = 0.0
        for entry, terms in zip(self.entries, docs):
            score = 0.0
            for term in query_terms:
                doc_freq = self._bm25_doc_freq.get(term, 0)
                if doc_freq == 0:
                    continue
                idf = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
                freq = terms.count(term)
                denom = freq + 1.5 * (1 - 0.75 + 0.75 * len(terms) / max(1.0, avg_len))
                score += idf * (freq * 2.5 / denom) if denom else 0.0
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry and best_score > 0.0:
            confidence = min(0.84, 0.55 + best_score / (best_score + 3.0))
            return self._result(best_entry, confidence, "bm25")
        return None

    def _dense(self, surface: str) -> dict[str, Any] | None:
        if not self.dense_model_name:
            return None
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        if self._dense_model is None:
            self._dense_model = SentenceTransformer(self.dense_model_name)
            labels = [entry["canonical_skill_name"] for entry in self.entries]
            self._dense_vectors = self._dense_model.encode(labels, normalize_embeddings=True)
        query = self._dense_model.encode([surface], normalize_embeddings=True)[0]
        scores = np.dot(self._dense_vectors, query)
        index = int(scores.argmax())
        score = float(scores[index])
        if score >= 0.72:
            return self._result(self.entries[index], score, "dense")
        return None


def _terms(value: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.-]+", normalize_text(value))


def _entry(identifier: str, name: str, taxonomy: str, aliases: list[str]) -> dict[str, Any]:
    return {
        "canonical_skill_id": identifier,
        "canonical_skill_name": name,
        "taxonomy": taxonomy,
        "aliases": aliases,
        "aliases_norm": {normalize_text(alias) for alias in aliases + [name]},
    }


def _group_alias_records_into_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group one-alias-per-row ``skill_aliases.jsonl`` records into multi-alias
    canonical entries.

    Each ``canonical_skill_id`` may appear in several rows (e.g. a manually
    curated alias and a SkillSpan-gold-derived alias both resolving to
    "python"). All aliases for a given ``canonical_skill_id`` are merged into
    one entry; the entry's reported ``taxonomy`` and ``canonical_skill_name``
    come from the highest-priority row (ESCO > ONET > LIGHTCAST > CUSTOM >
    SKILLSPAN_GOLD), per ``TAXONOMY_PRIORITY``.
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        cid = record["canonical_skill_id"]
        if cid not in groups:
            groups[cid] = {
                "canonical_skill_name": record["canonical_skill_name"],
                "taxonomy": record["taxonomy"],
                "taxonomy_priority": TAXONOMY_PRIORITY.get(record["taxonomy"], 99),
                "aliases": [],
            }
            order.append(cid)
        group = groups[cid]
        group["aliases"].append(record["alias"])
        priority = TAXONOMY_PRIORITY.get(record["taxonomy"], 99)
        if priority < group["taxonomy_priority"]:
            group["taxonomy_priority"] = priority
            group["taxonomy"] = record["taxonomy"]
            group["canonical_skill_name"] = record["canonical_skill_name"]
    return [_entry(cid, groups[cid]["canonical_skill_name"], groups[cid]["taxonomy"], groups[cid]["aliases"]) for cid in order]
