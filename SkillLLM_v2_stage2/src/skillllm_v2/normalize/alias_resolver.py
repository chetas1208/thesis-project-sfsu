from __future__ import annotations

from skillllm_v2.schema import normalize_text


class AliasResolver:
    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = {normalize_text(k): v for k, v in (aliases or {}).items()}

    def resolve(self, surface: str) -> str:
        normalized = normalize_text(surface)
        return self.aliases.get(normalized, surface)
