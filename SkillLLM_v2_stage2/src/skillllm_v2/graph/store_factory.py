"""Backend selection: ``get_graph_store(config, backend='auto'|'local'|'neo4j'|'mirror')``.

* ``local``   NetworkX only (default everywhere in the pipeline; nothing changes unless a caller opts in).
* ``neo4j``   Neo4j only. Failures are returned in ``StoreResult.error`` (``ok=False``); nothing falls back.
* ``auto``    try Neo4j, on ANY connection / auth / timeout / query failure answer from the local graph instead
              (``fell_back=True``, redacted ``error``, one warning). Never raises because Neo4j is down or not
              configured. Writes go to Neo4j when reachable, else to the local graph.
* ``mirror``  the local graph is always built and written; writes are also synced to Neo4j when it is reachable;
              reads prefer Neo4j and fall back to local mid-flight when a query fails.

A health probe result is cached for ``health_ttl_seconds`` (default 30 s) in both directions, so a dead database costs
one timeout per TTL window rather than one per call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Mapping

from .local_store import LocalGraphStore
from .neo4j_store import Neo4jStore
from .store import (
    PROJECT,
    EdgeRecord,
    GraphStore,
    GraphStoreError,
    NodeRecord,
    StoreResult,
    validate_project,
)
from .store_settings import Neo4jSettings, SettingsError, redact_error

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "local", "neo4j", "mirror")


class FallbackStore:
    """Neo4j-preferring composition over a local store (modes ``auto`` and ``mirror``)."""

    def __init__(
        self,
        primary: Neo4jStore | None,
        local: LocalGraphStore,
        *,
        mirror: bool = False,
        health_ttl: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        unavailable_reason: str | None = None,
    ) -> None:
        self.primary = primary
        self.local = local
        self.mirror = mirror
        self.name = "mirror" if mirror else "auto"
        self.project = local.project
        self.health_ttl = float(health_ttl)
        self._clock = clock
        self._up_until = 0.0
        self._down_until = 0.0
        self.last_error: str | None = unavailable_reason
        self.fell_back = False
        self._warned = False
        self.probes = 0  # number of real health probes sent (visible to tests / diagnostics)
        if mirror:
            local.ensure_loaded()  # mirror: the local graph is always built

    def __repr__(self) -> str:
        return f"FallbackStore(mode={self.name!r}, primary={self.primary!r}, last_error={self.last_error!r})"

    # ------------------------------------------------------------------ health cache
    def _mark_down(self, error: str | None) -> None:
        self._down_until = self._clock() + self.health_ttl
        self._up_until = 0.0
        self.last_error = error or "neo4j unavailable"
        if not self._warned:
            self._warned = True
            logger.warning("neo4j unavailable, using local graph (%s)", self.last_error)

    def _primary_ok(self) -> bool:
        if self.primary is None:
            return False
        now = self._clock()
        if now < self._down_until:
            return False
        if now < self._up_until:
            return True
        self.probes += 1
        res = self.primary.health()
        if res.ok and (res.value or {}).get("ok"):
            self._up_until = now + self.health_ttl
            self._down_until = 0.0
            self.last_error = None
            self._warned = False
            return True
        self._mark_down(res.error)
        return False

    def _local_answer(self, res: StoreResult) -> StoreResult:
        res.fell_back = True
        res.error = self.last_error or "neo4j unavailable"
        self.fell_back = True
        return res

    # ------------------------------------------------------------------ dispatch
    def _read(self, op: str, *args: Any, **kwargs: Any) -> StoreResult:
        if self._primary_ok():
            res = getattr(self.primary, op)(*args, **kwargs)
            if res.ok:
                return res
            self._mark_down(res.error)  # query failed mid-flight: answer locally, skip Neo4j for one TTL
        return self._local_answer(getattr(self.local, op)(*args, **kwargs))

    def _write(self, op: str, items: Iterable[Any], build_id: str, project: str | None) -> StoreResult:
        items = list(items)  # may be a generator: both backends need it
        if self.mirror:
            local_res = getattr(self.local, op)(items, build_id, project)
            if self._primary_ok():
                res = getattr(self.primary, op)(items, build_id, project)
                if res.ok:
                    res.value = {**(res.value or {}), "local": local_res.value}
                    return res
                self._mark_down(res.error)
            return self._local_answer(local_res)
        if self._primary_ok():
            res = getattr(self.primary, op)(items, build_id, project)
            if res.ok:
                return res
            self._mark_down(res.error)
        return self._local_answer(getattr(self.local, op)(items, build_id, project))

    # ------------------------------------------------------------------ GraphStore API
    def upsert_nodes(self, nodes: Iterable[NodeRecord], build_id: str, project: str | None = None) -> StoreResult:
        return self._write("upsert_nodes", nodes, build_id, project)

    def upsert_edges(self, edges: Iterable[EdgeRecord], build_id: str, project: str | None = None) -> StoreResult:
        return self._write("upsert_edges", edges, build_id, project)

    def user_skills(self, user_id: str) -> StoreResult:
        return self._read("user_skills", user_id)

    def job_skills(self, job_id: str, strength: str = "required") -> StoreResult:
        return self._read("job_skills", job_id, strength)

    def skill_gap(self, user_id: str, job_id: str) -> StoreResult:
        return self._read("skill_gap", user_id, job_id)

    def courses_teaching(self, skill_id: str) -> StoreResult:
        return self._read("courses_teaching", skill_id)

    def prerequisite_chain(self, course_id: str, max_depth: int = 4) -> StoreResult:
        return self._read("prerequisite_chain", course_id, max_depth)

    def explain_path(self, source: str, target: str, max_hops: int = 4) -> StoreResult:
        return self._read("explain_path", source, target, max_hops)

    def stats(self) -> StoreResult:
        return self._read("stats")

    def health(self) -> StoreResult:
        up = self._primary_ok()
        value = {"ok": True, "neo4j": up, "local": True, "mode": self.name}
        if up:
            return StoreResult(value, "neo4j")
        return StoreResult(value, "local", fell_back=True, error=self.last_error or "neo4j unavailable")

    def clear_namespace(self, project: str | None = None) -> StoreResult:
        proj = validate_project(self.project if project is None else project)
        local_res = self.local.clear_namespace(proj)
        if self.primary is not None and self._primary_ok():
            res = self.primary.clear_namespace(proj)
            if res.ok:
                res.value = {**res.value, "local_nodes_deleted": local_res.value["nodes_deleted"]}
                return res
            self._mark_down(res.error)
        return self._local_answer(local_res)

    def sync_to_neo4j(self, build_id: str) -> StoreResult:
        """Push the whole local graph (nodes, then edges) into Neo4j; used by ``mirror`` to catch up after an outage."""
        from .neo4j_adapters import records_from_local_store

        if not self._primary_ok():
            return self._local_answer(StoreResult({"synced": False}, "local"))
        nodes, edges = records_from_local_store(self.local)
        res = self.primary.upsert_nodes(nodes, build_id, self.project)
        if not res.ok:
            self._mark_down(res.error)
            return self._local_answer(StoreResult({"synced": False}, "local"))
        res2 = self.primary.upsert_edges(edges, build_id, self.project)
        if not res2.ok:
            self._mark_down(res2.error)
            return self._local_answer(StoreResult({"synced": False}, "local"))
        return StoreResult({"synced": True, **res.value, **res2.value}, "neo4j")

    def close(self) -> None:
        if self.primary is not None:
            self.primary.close()
        self.local.close()

    def __enter__(self) -> "FallbackStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _graph_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    inner = config.get("graph") if isinstance(config.get("graph"), Mapping) else config
    return dict(inner)


def get_graph_store(
    config: Mapping[str, Any] | None = None,
    backend: str | None = None,
    *,
    settings: Neo4jSettings | None = None,
    local: LocalGraphStore | None = None,
    local_loader: Callable[[LocalGraphStore], None] | None = None,
    driver_factory: Callable[[Neo4jSettings], Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
    env: Mapping[str, str] | None = None,
) -> GraphStore:
    """Build a store. ``backend`` (argument) > ``config['graph']['backend']`` > ``'auto'``.

    ``config`` may be the whole pipeline config (its ``graph`` section is used) or the section itself. Recognised
    keys: ``backend``, ``project``, ``health_ttl_seconds``, ``connection_timeout``, ``query_timeout``, ``max_retries``.
    ``local_loader(store)`` populates the local graph lazily (``mirror``: immediately).
    """
    cfg = _graph_cfg(config)
    chosen = (backend or cfg.get("backend") or "auto").lower()
    if chosen not in BACKENDS:
        raise ValueError(f"graph backend must be one of {BACKENDS}, not {chosen!r}")
    project = validate_project(cfg.get("project", PROJECT))
    local_store = local or LocalGraphStore(project, loader=local_loader)
    if chosen == "local":
        return local_store

    overrides: dict[str, Any] = {}
    for src, dst in (("connection_timeout", "connection_timeout"), ("query_timeout", "query_timeout"),
                     ("max_retries", "max_retries")):
        if cfg.get(src) is not None:
            overrides[dst] = cfg[src]
    reason: str | None = None
    primary: Neo4jStore | None = None
    try:
        s = settings or Neo4jSettings.from_env(env=env, **overrides)
        primary = Neo4jStore(s, project, driver_factory=driver_factory)
    except SettingsError as exc:
        if chosen == "neo4j":
            raise
        reason = redact_error(exc)
    except GraphStoreError as exc:  # pragma: no cover - construction does not connect
        reason = str(exc)
    if chosen == "neo4j":
        assert primary is not None
        return primary
    store = FallbackStore(primary, local_store, mirror=(chosen == "mirror"),
                          health_ttl=float(cfg.get("health_ttl_seconds", 30.0)), clock=clock,
                          unavailable_reason=reason)
    if primary is None:
        store.fell_back = True
        logger.warning("neo4j not configured, using local graph (%s)", reason)
    return store
