from __future__ import annotations

from .taxonomy_linker import TaxonomyLinker


def load_esco(path: str) -> TaxonomyLinker:
    return TaxonomyLinker([path])
