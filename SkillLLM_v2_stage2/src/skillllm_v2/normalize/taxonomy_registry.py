"""taxonomy_registry.py

Wires the per-taxonomy loaders (:func:`load_esco`, :func:`load_onet`,
:func:`load_lightcast`) and the generated alias table into ONE linker the
inference pipeline can use.

* With no external taxonomy configured, :func:`build_taxonomy_linker` returns a
  plain :class:`TaxonomyLinker` (identical to what ``wrapper.py`` always built).
* With one or more of ``esco_path`` / ``onet_path`` / ``lightcast_path`` that
  exist on disk, it returns a :class:`ChainedTaxonomyLinker`: every linker is
  queried and the highest ``link_confidence`` wins; ties are broken by taxonomy
  priority (ESCO > ONET > LIGHTCAST > alias table), matching
  ``taxonomy_linker.TAXONOMY_PRIORITY``.

Linking is attach-only: it never removes or rewrites a candidate's span.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from skillllm_v2.normalize.esco_loader import load_esco
from skillllm_v2.normalize.lightcast_loader import load_lightcast
from skillllm_v2.normalize.onet_loader import load_onet
from skillllm_v2.normalize.taxonomy_linker import TaxonomyLinker


class ChainedTaxonomyLinker:
    """Query several :class:`TaxonomyLinker` instances; best confidence wins (priority breaks ties)."""

    def __init__(self, linkers: list[tuple[str, TaxonomyLinker]]) -> None:
        if not linkers:
            raise ValueError("ChainedTaxonomyLinker needs at least one linker")
        self.linkers = linkers

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.linkers]

    @property
    def entries(self) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for _, linker in self.linkers:
            merged.extend(linker.entries)
        return merged

    def link(self, surface: str) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for _, linker in self.linkers:  # earlier == higher priority
            match = linker.link(surface)
            if match is not None and (best is None or match["link_confidence"] > best["link_confidence"]):
                best = match
        return best

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


def build_taxonomy_linker(
    taxonomy_paths: list[str] | None = None,
    alias_table_paths: list[str] | None = None,
    fuzzy_threshold: float = 0.88,
    esco_path: str | None = None,
    onet_path: str | None = None,
    lightcast_path: str | None = None,
) -> TaxonomyLinker | ChainedTaxonomyLinker:
    base = TaxonomyLinker(
        [str(p) for p in taxonomy_paths or []],
        alias_table_paths=alias_table_paths,
        fuzzy_threshold=fuzzy_threshold,
    )
    chained: list[tuple[str, TaxonomyLinker]] = []
    for name, loader, path in (("ESCO", load_esco, esco_path), ("ONET", load_onet, onet_path), ("LIGHTCAST", load_lightcast, lightcast_path)):
        if path and Path(path).exists():
            chained.append((name, loader(str(path))))
    if not chained:
        return base
    chained.append(("ALIAS_TABLE", base))
    return ChainedTaxonomyLinker(chained)
