"""Neo4j connection settings: a tiny ``.env`` parser, env-var precedence and credential redaction.

Resolution order for every ``NEO4J_*`` key (highest first):

1. real process environment variables;
2. the nearest ``.env`` file found walking UP from the current working directory (nearest directory first, up to the
   filesystem root), then walking UP from this package's location. A nearer file wins per key; a key missing from a
   nearer file may still come from a farther one.

Only the four ``NEO4J_*`` keys are ever read out of a ``.env`` file; nothing else from the file is kept in memory.

Security rules implemented here and relied on by every other module:

* the password (and the full URI, which may embed credentials or identify the instance) never appears in ``repr``,
  ``str``, log records or exception messages: use :func:`redact` / :func:`redact_error` on anything that is shown;
* only TLS URI schemes are accepted (``neo4j+s``, ``bolt+s``); an unencrypted or self-signed-skipping scheme is refused.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

ENV_KEYS = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
ALLOWED_SCHEMES = ("neo4j+s://", "bolt+s://")
REDACTED = "***REDACTED***"

_URI_USERINFO = re.compile(r"((?:neo4j|bolt)(?:\+s|\+ssc)?://)[^/\s@]*@", re.IGNORECASE)
_URI_ANY = re.compile(r"(?:neo4j|bolt)(?:\+s|\+ssc)?://[^\s'\"<>]+", re.IGNORECASE)
_KV_SECRET = re.compile(r"(?i)(password|passwd|pwd|token|secret|credentials?)(\s*[=:]\s*)([^\s,;'\"]+)")


class SettingsError(ValueError):
    """Configuration problem (missing key, insecure URI). The message never contains a secret."""


# --------------------------------------------------------------------------- redaction
def redact(text: object, secrets: Iterable[str | None] = ()) -> str:
    """Return ``text`` as a string with every known secret, URI and ``password=...`` pair masked."""
    out = str(text)
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        out = out.replace(secret, REDACTED)
    out = _URI_ANY.sub(f"<uri:{REDACTED}>", out)
    out = _URI_USERINFO.sub(lambda m: m.group(1) + REDACTED + "@", out)
    out = _KV_SECRET.sub(lambda m: m.group(1) + m.group(2) + REDACTED, out)
    return out


def redact_error(exc: BaseException, secrets: Iterable[str | None] = ()) -> str:
    """``ExcType: message`` with secrets removed (the exception's chained causes are NOT included)."""
    return f"{type(exc).__name__}: {redact(exc, secrets)}"[:500]


# --------------------------------------------------------------------------- .env parsing
def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (``export`` prefix, single/double quotes and ``#`` comments supported)."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        values[key] = value
    return values


def _walk_up(start: Path) -> list[Path]:
    start = start.resolve()
    return [start, *start.parents]


def find_dotenv_files(search_from: Iterable[Path] | None = None) -> list[Path]:
    """Every ``.env`` file on the walk up from each start directory, nearest first, de-duplicated.

    Default starts: the current working directory, then this package's directory.
    """
    starts = list(search_from) if search_from is not None else [Path.cwd(), Path(__file__).resolve().parent]
    found: list[Path] = []
    for start in starts:
        for directory in _walk_up(Path(start)):
            candidate = directory / ".env"
            try:
                if candidate.is_file() and candidate not in found:
                    found.append(candidate)
            except OSError:
                continue
    return found


def load_neo4j_env(
    env: Mapping[str, str] | None = None,
    search_from: Iterable[Path] | None = None,
) -> dict[str, str]:
    """Merge ``NEO4J_*`` values: real environment first, then ``.env`` files nearest-first. Missing file is fine."""
    environ = os.environ if env is None else env
    merged: dict[str, str] = {}
    for key in ENV_KEYS:
        if environ.get(key):
            merged[key] = str(environ[key])
    for path in find_dotenv_files(search_from):
        try:
            parsed = parse_dotenv(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for key in ENV_KEYS:
            if key not in merged and parsed.get(key):
                merged[key] = parsed[key]
    return merged


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Neo4jSettings:
    """Connection settings. ``repr``/``str`` never show the password or the URI."""

    uri: str = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)
    database: str | None = None
    connection_timeout: float = 5.0  # seconds to open a connection (a dead host must not stall callers)
    acquisition_timeout: float = 8.0
    query_timeout: float = 30.0  # per transaction
    max_retries: int = 2  # extra attempts on transient errors
    retry_backoff: float = 0.5  # seconds; doubled per attempt

    def __post_init__(self) -> None:
        if not any(self.uri.lower().startswith(scheme) for scheme in ALLOWED_SCHEMES):
            raise SettingsError("NEO4J_URI must use a TLS scheme (neo4j+s:// or bolt+s://)")

    @property
    def host(self) -> str:
        """Hostname of the URI (an instance id is identifying, so it is redacted from errors too)."""
        return re.sub(r"^[a-z0-9+]+://", "", self.uri, flags=re.IGNORECASE).split("/")[0].split("@")[-1].split(":")[0]

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.password, self.uri, self.username, self.host) if s)

    def redact(self, text: object) -> str:
        return redact(text, self.secrets)

    def __repr__(self) -> str:
        return f"Neo4jSettings(database={self.database!r}, uri={REDACTED}, username={REDACTED}, password={REDACTED})"

    __str__ = __repr__

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        search_from: Iterable[Path] | None = None,
        **overrides: object,
    ) -> "Neo4jSettings":
        values = load_neo4j_env(env, search_from)
        missing = [k for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD") if not values.get(k)]
        if missing:
            raise SettingsError(f"missing Neo4j setting(s): {', '.join(missing)} (set env vars or a .env file)")
        return cls(
            uri=values["NEO4J_URI"],
            username=values["NEO4J_USERNAME"],
            password=values["NEO4J_PASSWORD"],
            database=values.get("NEO4J_DATABASE") or None,
            **overrides,  # type: ignore[arg-type]
        )

    @classmethod
    def available(cls, env: Mapping[str, str] | None = None, search_from: Iterable[Path] | None = None) -> bool:
        """True if URI, username and password are configured and the URI is TLS. Never raises."""
        try:
            cls.from_env(env, search_from)
            return True
        except SettingsError:
            return False
