"""Shared skill-key canonicalisation for :class:`JobMatcher` and :class:`GapDetector`.

Both classes used to compare skills with exact string equality, so ``"python" != "Python"`` and ``"SQL" !=
"structured query language"`` (the extractor emits canonical names, jobs and catalogs use surface names).
This module is the single place that decides "are these the same skill":

* :func:`skill_key` -- casefold + whitespace / punctuation normalisation that PRESERVES symbol-bearing names
  (``c++``, ``c#``, ``.net``, ``node.js`` stay distinct from ``c``, ``net``, ``node js``). It delegates to
  ``skillllm_v2.schema.graph_key`` (the graph key function) when that exists and falls back to an identical
  local implementation otherwise, so the graph and the matcher agree on node identity.
* :class:`SkillKeyResolver` -- turns a skill (string or dict) into a *set* of keys: the literal key, the key of
  its alias-resolved canonical name (via :class:`skillllm_v2.normalize.AliasResolver`) and, for dicts, the
  canonical id/name. Two skills are the same skill when their key sets intersect. For dicts that carry a
  canonical id/name, that canonical identity takes precedence over a conflicting surface alias.

Alias resolution is exact-alias evidence only (no fuzzy guessing). ``AliasResolver`` normalises with
``normalize_text`` which erases ``+ # .``; the resolver therefore (a) never alias-resolves a symbol-bearing
surface and (b) rejects an alias hit that differs from the surface only by those symbols (``"C"`` -> ``"C++"``).
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Iterable

from skillllm_v2.schema import normalize_text

try:  # the graph work owns the canonical graph key; use it when present
    from skillllm_v2.schema import graph_key as _graph_key  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - local fallback with identical semantics
    _GRAPH_TOKEN = re.compile(r"(?<![^\W_])\.[^\W_]+(?:\.[^\W_]*[^\W\d_][^\W_]*)*|[^\W_]+(?:\.[^\W\d_][^\W_]*)*[+#]*")

    def _graph_key(value: str) -> str:
        return " ".join(_GRAPH_TOKEN.findall(value.lower().strip()))


_SPACE_BEFORE_SYMBOL = re.compile(r"(?<=\w)\s+(?=[+#])")
_ID_NAME_FIELDS = ("canonical_skill_id",)
_CANON_NAME_FIELDS = ("canonical_skill_name",)
_SURFACE_FIELDS = ("surface_form", "name", "skill_name", "skill")


def skill_key(value: Any) -> str:
    """Case-, whitespace- and punctuation-insensitive key that keeps ``c++``, ``c#``, ``.net``, ``node.js`` distinct."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return _graph_key(_SPACE_BEFORE_SYMBOL.sub("", text))


def skill_name(item: Any) -> str:
    """Display name of a skill given as a string or a dict (surface first, then canonical name / id)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for field in _SURFACE_FIELDS + _CANON_NAME_FIELDS + _ID_NAME_FIELDS:
            value = item.get(field)
            if value:
                return str(value)
    return str(item)


_DEFAULT_RESOLVERS: dict[str, Any] = {}


def default_alias_resolver() -> Any | None:
    """``AliasResolver`` over the repo alias table plus the packaged ``resources/offline_aliases.jsonl`` (cached per working directory); ``None`` if unavailable."""
    cwd = os.getcwd()
    if cwd in _DEFAULT_RESOLVERS:
        return _DEFAULT_RESOLVERS[cwd]
    resolver: Any | None = None
    try:
        from pathlib import Path

        from skillllm_v2.normalize import AliasResolver, TaxonomyLinker  # lazy: keeps match importable without the linker
        from skillllm_v2.normalize.skill_alias_builder import DEFAULT_OUTPUT

        packaged = Path(__file__).resolve().parents[1] / "resources" / "offline_aliases.jsonl"  # cwd-independent supplement
        tables = [str(p) for p in (Path(DEFAULT_OUTPUT), packaged) if p.exists()]
        aliases: dict[str, str] = {}
        for entry in TaxonomyLinker(alias_table_paths=tables).entries:
            name = entry["canonical_skill_name"]
            for alias in list(entry["aliases"]) + [name]:
                aliases.setdefault(normalize_text(alias), name)  # first source wins, as in the inference pipeline
        resolver = AliasResolver(aliases) if aliases else None
    except Exception:  # the alias table is an enhancement; matching still works on literal keys
        resolver = None
    _DEFAULT_RESOLVERS[cwd] = resolver
    return resolver


class SkillKeyResolver:
    """Skill -> frozenset of keys. See the module docstring for the semantics."""

    def __init__(self, alias_resolver: Any | None = None, *, aliases: dict[str, str] | None = None, use_default_aliases: bool = True) -> None:
        self._alias: Any | None = alias_resolver
        if self._alias is None and aliases:
            from skillllm_v2.normalize import AliasResolver

            self._alias = AliasResolver(aliases)
        self._use_default = use_default_aliases and self._alias is None
        self._memo: dict[str, tuple[str, str | None]] = {}

    @property
    def alias_resolver(self) -> Any | None:
        if self._alias is None and self._use_default:
            self._alias = default_alias_resolver()
            self._use_default = False
        return self._alias

    # ------------------------------------------------------------------ single names
    def _name_keys(self, name: str) -> tuple[str, str | None]:
        """``(literal_key, alias_key_or_None)``; ``alias_key`` is set only for a real alias hit."""
        hit = self._memo.get(name)
        if hit is not None:
            return hit
        literal = skill_key(name)
        alias_key: str | None = None
        resolver = self.alias_resolver
        if resolver is not None and literal and literal == normalize_text(name):  # symbol-free surface only
            try:
                resolved = resolver.resolve(name)
            except Exception:
                resolved = name
            candidate = skill_key(resolved)
            collapsed_symbols = normalize_text(str(resolved)) == normalize_text(name) and candidate != literal
            if candidate and candidate != literal and not collapsed_symbols:
                alias_key = candidate
        result = (literal, alias_key)
        self._memo[name] = result
        return result

    def keys_for_name(self, name: str) -> frozenset[str]:
        literal, alias_key = self._name_keys(name)
        return frozenset(k for k in (literal, alias_key) if k)

    def primary_for_name(self, name: str) -> str:
        literal, alias_key = self._name_keys(name)
        return alias_key or literal

    # ------------------------------------------------------------------ skills (str | dict)
    def _id_key(self, item: dict[str, Any], surface_literals: set[str]) -> str | None:
        raw = next((item.get(f) for f in _ID_NAME_FIELDS if item.get(f)), None)
        if not raw:
            return None
        key = skill_key(raw)
        # An id equal to normalize_text(surface) of a symbol-bearing name ("c" for "C++") is a lossy id: ignore it.
        for surface in surface_literals:
            if key == normalize_text(surface) and surface != key:
                return None
        return key or None

    def keys(self, item: Any) -> frozenset[str]:
        """All keys under which this skill can be recognised (canonical id/name take precedence over a conflicting surface)."""
        if isinstance(item, dict):
            surfaces = [str(item[f]) for f in _SURFACE_FIELDS if item.get(f)]
            canon_names = [str(item[f]) for f in _CANON_NAME_FIELDS if item.get(f)]
            canon: set[str] = set()
            for name in canon_names:
                canon |= self.keys_for_name(name)
            id_key = self._id_key(item, set(surfaces + canon_names))
            if id_key:
                canon.add(id_key)
            surface_keys: set[str] = set()
            for surface in surfaces:
                literal, alias_key = self._name_keys(surface)
                if canon and alias_key and alias_key not in canon:
                    continue  # the surface resolves to a different skill than the declared canonical one: canonical wins
                surface_keys |= {k for k in (literal, alias_key) if k}
            return frozenset(canon | surface_keys)
        return self.keys_for_name(str(item)) if item is not None else frozenset()

    def primary(self, item: Any) -> str:
        """Stable identity key of a skill: canonical id/name first, then alias-resolved surface, then the literal surface."""
        if isinstance(item, dict):
            for field in _CANON_NAME_FIELDS:
                if item.get(field):
                    return self.primary_for_name(str(item[field]))
            surfaces = [str(item[f]) for f in _SURFACE_FIELDS if item.get(f)]
            id_key = self._id_key(item, set(surfaces))
            if id_key:
                return id_key
            return self.primary_for_name(surfaces[0]) if surfaces else ""
        return self.primary_for_name(str(item)) if item is not None else ""

    # ------------------------------------------------------------------ collections
    def index(self, *collections: Iterable[Any] | None) -> frozenset[str]:
        """Union of the keys of every skill in the given collections (``None`` collections are empty)."""
        out: set[str] = set()
        for collection in collections:
            for item in _as_list(collection):
                out |= self.keys(item)
        return frozenset(out)

    def unique(self, items: Iterable[Any] | None) -> list[tuple[str, frozenset[str]]]:
        """``[(display_name, keys)]`` in first-seen order, one entry per distinct skill (by primary key)."""
        seen: set[str] = set()
        out: list[tuple[str, frozenset[str]]] = []
        for item in _as_list(items):
            keys = self.keys(item)
            primary = self.primary(item) or skill_name(item)
            if not keys or primary in seen:
                continue
            seen.add(primary)
            out.append((skill_name(item), keys))
        return out


def _as_list(items: Iterable[Any] | None) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, (str, dict)):
        return [items]
    return list(items)
