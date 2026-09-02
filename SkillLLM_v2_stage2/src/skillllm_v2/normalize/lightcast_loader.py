from __future__ import annotations

from .taxonomy_linker import TaxonomyLinker


def load_lightcast(path: str) -> TaxonomyLinker:
    return TaxonomyLinker([path])
