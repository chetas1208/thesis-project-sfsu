"""Neo4j implementation of :class:`GraphStore` (tested on Aura, Neo4j 5.x).

Schema (see docs/NEO4J_GRAPH.md)::

    (:User {id, project, build_id, ...})-[:HAS_SKILL {confidence, source, ...}]->(:Skill {id, name, taxonomy, aliases})
    (:Job)-[:REQUIRES|PREFERS {importance, requirement_strength}]->(:Skill)
    (:Course {title, level, ...})-[:TEACHES {depth, confidence, evidence}]->(:Skill)
    (:Course)-[:REQUIRES_PREREQUISITE {level, strictness, evidence}]->(:Skill)
    (:Skill)-[:RELATED_TO {relation_type, weight}]->(:Skill)

Safety properties:

* every node and relationship carries ``project`` and ``build_id``; writes ``MERGE`` on ``{id, project}`` under a
  uniqueness constraint on ``id`` per label, so a node of another namespace with the same id is never modified (the
  constraint rejects the write instead);
* only values are ever passed as Cypher parameters; label / relationship-type identifiers come from the fixed
  whitelists in :mod:`skillllm_v2.graph.store` and the one integer path-length literal is range-checked;
* ``clear_namespace`` deletes only ``project = <ns>`` in bounded batches, never anything else, never DROP DATABASE;
* driver exceptions are caught and re-raised/returned as redacted :class:`GraphStoreError` subclasses without the
  original cause (``from None``), so a credential or URI cannot reach a traceback, log or ``StoreResult``;
* the connection pool is closed by :meth:`close`, the context manager and a ``weakref.finalize`` safety net.
"""

from __future__ import annotations

import logging
import time
import weakref
from typing import Any, Callable, Iterable

from .skill_graph_explainer import REVERSE_TRAVERSABLE, explain_skill_path
from .store import (
    LABELS,
    MAX_NODES,
    PROJECT,
    REL_ENDPOINTS,
    REL_TYPES,
    EdgeRecord,
    GraphAuthError,
    GraphConnectionError,
    GraphQueryError,
    GraphStoreError,
    GraphTimeoutError,
    NodeRecord,
    SizeGuardError,
    StoreResult,
    chain_from_primitives,
    label_of,
    normalize_path_result,
    validate_project,
)
from .store_settings import Neo4jSettings, redact_error

logger = logging.getLogger(__name__)

_RESERVED = {"id", "project", "build_id", "created_build_id"}
_PATH_TYPES = ("HAS_SKILL", "REQUIRES", "PREFERS", "TEACHES", "REQUIRES_PREREQUISITE", "RELATED_TO")
_MAX_PATH_LEN = 10
_PATH_LIMIT = 2000


def _clean_props(props: dict[str, Any]) -> dict[str, Any]:
    """Keep only values Neo4j can store as properties (primitives and homogeneous lists of strings/numbers)."""
    out: dict[str, Any] = {}
    for key, value in props.items():
        if key in _RESERVED or value is None or value == "":
            continue
        if isinstance(value, (bool, int, float, str)):
            out[str(key)] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value) and value:
            out[str(key)] = [str(v) for v in value]
    return out


def classify_error(exc: BaseException, secrets: Iterable[str | None] = ()) -> GraphStoreError:
    """Map any driver/network exception to a redacted :class:`GraphStoreError` subclass (by class name, so it works
    with real neo4j exceptions, fakes and plain ``OSError``)."""
    names = {c.__name__ for c in type(exc).__mro__}
    code = str(getattr(exc, "code", "") or "")
    msg = redact_error(exc, secrets)
    if names & {"AuthError", "Unauthorized", "Forbidden", "AuthConfigurationError", "TokenExpired"} or "Security.Unauthorized" in code:
        cls: type[GraphStoreError] = GraphAuthError
    elif "TransactionTimedOut" in code or names & {"TimeoutError", "socket.timeout"} or "timeout" in msg.lower() or "timed out" in msg.lower():
        cls = GraphTimeoutError
    elif names & {"ServiceUnavailable", "SessionExpired", "ConnectionResetError", "ConnectionRefusedError", "OSError",
                  "DriverError", "ConnectionError", "BrokenPipeError", "gaierror"}:
        cls = GraphConnectionError
    else:
        cls = GraphQueryError
    return cls(msg)


def _retryable(exc: BaseException) -> bool:
    names = {c.__name__ for c in type(exc).__mro__}
    if names & {"AuthError", "Unauthorized", "Forbidden", "ConstraintError", "CypherSyntaxError"}:
        return False
    return bool(names & {"ServiceUnavailable", "SessionExpired", "TransientError", "ConnectionResetError", "BrokenPipeError"})


def _default_driver_factory(settings: Neo4jSettings) -> Any:
    import neo4j  # lazy: the offline package must work without the driver

    return neo4j.GraphDatabase.driver(
        settings.uri,
        auth=(settings.username, settings.password),
        connection_timeout=settings.connection_timeout,
        connection_acquisition_timeout=settings.acquisition_timeout,
        max_connection_lifetime=1800,
        max_transaction_retry_time=settings.query_timeout,
    )


def _close_driver(box: list[Any]) -> None:
    drv = box[0] if box else None
    if drv is not None:
        box[0] = None
        try:
            drv.close()
        except Exception:  # pragma: no cover - best effort
            pass


class Neo4jStore:
    name = "neo4j"

    def __init__(
        self,
        settings: Neo4jSettings,
        project: str = PROJECT,
        *,
        driver_factory: Callable[[Neo4jSettings], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        batch_size: int = 500,
        max_nodes: int = MAX_NODES,
    ) -> None:
        self.settings = settings
        self.project = validate_project(project)
        self._factory = driver_factory or _default_driver_factory
        self._sleep = sleep
        self.batch_size = max(1, int(batch_size))
        self.max_nodes = int(max_nodes)
        self._driver_box: list[Any] = [None]
        self._finalizer = weakref.finalize(self, _close_driver, self._driver_box)
        self.last_error: str | None = None

    def __repr__(self) -> str:
        return f"Neo4jStore(project={self.project!r}, database={self.settings.database!r})"

    # ------------------------------------------------------------------ plumbing
    def _driver(self) -> Any:
        if self._driver_box[0] is None:
            try:
                self._driver_box[0] = self._factory(self.settings)
            except BaseException as exc:  # noqa: BLE001 - re-raised redacted below
                raise classify_error(exc, self.settings.secrets) from None
        return self._driver_box[0]

    def close(self) -> None:
        _close_driver(self._driver_box)

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _run(self, work: Callable[[Any], Any], *, write: bool, retries: int | None = None) -> Any:
        """Run ``work(tx)`` in a managed transaction with timeout, bounded retry/backoff and redacted errors."""
        retries = self.settings.max_retries if retries is None else retries
        try:
            import neo4j

            work = neo4j.unit_of_work(timeout=self.settings.query_timeout)(work)
        except Exception:  # driver absent (fake factory) or API drift: run without a per-tx timeout
            pass
        attempt = 0
        while True:
            try:
                drv = self._driver()
                kwargs = {"database": self.settings.database} if self.settings.database else {}
                with drv.session(**kwargs) as session:
                    return session.execute_write(work) if write else session.execute_read(work)
            except GraphStoreError:
                raise
            except BaseException as exc:  # noqa: BLE001
                if attempt < retries and _retryable(exc):
                    delay = self.settings.retry_backoff * (2 ** attempt)
                    logger.warning("neo4j transient error (%s); retry %d/%d in %.1fs",
                                   type(exc).__name__, attempt + 1, retries, delay)
                    self._sleep(delay)
                    attempt += 1
                    continue
                raise classify_error(exc, self.settings.secrets) from None

    def _exec(self, work: Callable[[Any], Any], *, write: bool, retries: int | None = None,
              post: Callable[[Any], Any] | None = None) -> StoreResult:
        t0 = time.perf_counter()
        try:
            value = self._run(work, write=write, retries=retries)
            if post is not None:
                value = post(value)
        except SizeGuardError:
            raise
        except GraphStoreError as exc:
            self.last_error = str(exc)
            return StoreResult(value=None, backend="neo4j", error=str(exc), ok=False,
                               elapsed_ms=(time.perf_counter() - t0) * 1000.0)
        self.last_error = None
        return StoreResult(value=value, backend="neo4j", elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    @staticmethod
    def _rows(result: Any) -> list[dict[str, Any]]:
        rows = []
        for record in result:
            rows.append(record.data() if hasattr(record, "data") else dict(record))
        return rows

    # ------------------------------------------------------------------ schema
    def ensure_schema(self) -> StoreResult:
        """Create uniqueness constraints on (label, id) and supporting indexes, all ``IF NOT EXISTS``."""
        statements: list[str] = []
        for label in LABELS:
            statements.append(f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.id IS UNIQUE")
            statements.append(f"CREATE INDEX {label.lower()}_project IF NOT EXISTS FOR (n:`{label}`) ON (n.project)")
        statements.append("CREATE INDEX skill_name IF NOT EXISTS FOR (n:`Skill`) ON (n.name)")
        for rel in REL_TYPES:
            statements.append(f"CREATE INDEX rel_{rel.lower()}_project IF NOT EXISTS FOR ()-[r:`{rel}`]-() ON (r.project)")

        t0 = time.perf_counter()
        try:
            for stmt in statements:  # schema changes need their own transaction each
                def one(tx: Any, _s: str = stmt) -> None:
                    tx.run(_s).consume()
                self._run(one, write=True)
        except GraphStoreError as exc:
            return StoreResult(value=None, backend="neo4j", error=str(exc), ok=False,
                               elapsed_ms=(time.perf_counter() - t0) * 1000.0)
        return StoreResult(value={"statements": len(statements)}, backend="neo4j",
                           elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    def schema_info(self) -> StoreResult:
        def work(tx: Any) -> dict[str, Any]:
            cons = self._rows(tx.run("SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
                                     "RETURN name, type, entityType, labelsOrTypes, properties ORDER BY name"))
            idx = self._rows(tx.run("SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state "
                                    "RETURN name, type, entityType, labelsOrTypes, properties, state ORDER BY name"))
            return {"constraints": cons, "indexes": [i for i in idx if i.get("type") != "LOOKUP"]}

        return self._exec(work, write=False)

    # ------------------------------------------------------------------ writes
    def _node_counts(self) -> tuple[int, int]:
        def work(tx: Any) -> tuple[int, int]:
            row = self._rows(tx.run("MATCH (n) RETURN count(n) AS total, "
                                    "count(CASE WHEN n.project = $project THEN 1 END) AS ours", {"project": self.project}))[0]
            return int(row["total"]), int(row["ours"])

        return self._run(work, write=False)

    def upsert_nodes(self, nodes: Iterable[NodeRecord], build_id: str, project: str | None = None) -> StoreResult:
        proj = validate_project(self.project if project is None else project)
        records = list(nodes)
        if len(records) > self.max_nodes:
            raise SizeGuardError(f"planned graph has {len(records)} nodes (> hard limit {self.max_nodes}); refusing to write")
        t0 = time.perf_counter()
        try:
            total, ours = self._node_counts()
            outside = total - ours
            if outside + len({r.id for r in records}) > self.max_nodes:
                raise SizeGuardError(f"database would hold {outside + len(records)} nodes (> hard limit {self.max_nodes}); refusing to write")
        except GraphStoreError as exc:
            if isinstance(exc, SizeGuardError):
                raise
            return StoreResult(None, "neo4j", error=str(exc), ok=False, elapsed_ms=(time.perf_counter() - t0) * 1000.0)
        by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
        for rec in records:
            by_label[rec.label].append({"id": rec.id, "props": _clean_props(rec.props)})
        written = 0
        for label, rows in by_label.items():
            query = (
                "UNWIND $rows AS row "
                f"MERGE (n:`{label}` {{id: row.id, project: $project}}) "
                "ON CREATE SET n.created_build_id = $build_id "
                "SET n += row.props, n.build_id = $build_id "
                "RETURN count(n) AS n"
            )
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]

                def work(tx: Any, _q: str = query, _b: list[dict[str, Any]] = batch) -> int:
                    return int(self._rows(tx.run(_q, {"rows": _b, "project": proj, "build_id": build_id}))[0]["n"])

                res = self._exec(work, write=True)
                if not res.ok:
                    res.value = {"nodes": written}
                    return res
                written += res.value
        return StoreResult({"nodes": written}, "neo4j", elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    def upsert_edges(self, edges: Iterable[EdgeRecord], build_id: str, project: str | None = None) -> StoreResult:
        proj = validate_project(self.project if project is None else project)
        t0 = time.perf_counter()
        by_type: dict[str, list[dict[str, Any]]] = {t: [] for t in REL_TYPES}
        for rec in edges:
            by_type[rec.type].append({"src": rec.src, "dst": rec.dst, "props": _clean_props(rec.props)})
        written = skipped = 0
        for rel, rows in by_type.items():
            s_label, d_label = REL_ENDPOINTS[rel]
            query = (
                "UNWIND $rows AS row "
                f"MATCH (a:`{s_label}` {{id: row.src, project: $project}}) "
                f"MATCH (b:`{d_label}` {{id: row.dst, project: $project}}) "
                f"MERGE (a)-[r:`{rel}`]->(b) "
                "ON CREATE SET r.created_build_id = $build_id "
                "SET r += row.props, r.project = $project, r.build_id = $build_id "
                "RETURN count(r) AS n"
            )
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i:i + self.batch_size]

                def work(tx: Any, _q: str = query, _b: list[dict[str, Any]] = batch) -> int:
                    return int(self._rows(tx.run(_q, {"rows": _b, "project": proj, "build_id": build_id}))[0]["n"])

                res = self._exec(work, write=True)
                if not res.ok:
                    res.value = {"edges": written, "skipped": skipped}
                    return res
                written += res.value
                skipped += len(batch) - res.value
        return StoreResult({"edges": written, "skipped": skipped}, "neo4j", elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    def clear_namespace(self, project: str | None = None, batch_size: int | None = None) -> StoreResult:
        """Delete ONLY nodes (and, with DETACH, their relationships) whose ``project`` equals the namespace."""
        proj = validate_project(self.project if project is None else project)
        size = int(batch_size or max(self.batch_size, 1000))
        t0 = time.perf_counter()
        deleted = 0
        for label in LABELS:
            query = (f"MATCH (n:`{label}` {{project: $project}}) WITH n LIMIT $batch "
                     "DETACH DELETE n RETURN count(*) AS deleted")
            while True:
                def work(tx: Any, _q: str = query) -> int:
                    return int(self._rows(tx.run(_q, {"project": proj, "batch": size}))[0]["deleted"])

                res = self._exec(work, write=True)
                if not res.ok:
                    res.value = {"nodes_deleted": deleted}
                    return res
                deleted += res.value
                if res.value < size:
                    break
        return StoreResult({"nodes_deleted": deleted, "project": proj}, "neo4j", elapsed_ms=(time.perf_counter() - t0) * 1000.0)

    # ------------------------------------------------------------------ reads
    def _q_user_skills(self, tx: Any, user_id: str) -> list[str]:
        rows = self._rows(tx.run(
            "MATCH (u:`User` {id: $id, project: $project})-[:`HAS_SKILL`]->(s:`Skill` {project: $project}) "
            "RETURN DISTINCT s.id AS id ORDER BY id", {"id": user_id, "project": self.project}))
        return [r["id"] for r in rows]

    _JOB_PATTERNS = {"required": "REQUIRES", "preferred": "PREFERS", "all": "REQUIRES|PREFERS"}

    def _q_job_skills(self, tx: Any, job_id: str, strength: str) -> list[str]:
        pattern = self._JOB_PATTERNS[strength]  # whitelist lookup: not user text
        rows = self._rows(tx.run(
            f"MATCH (j:`Job` {{id: $id, project: $project}})-[:{pattern}]->(s:`Skill` {{project: $project}}) "
            "RETURN DISTINCT s.id AS id ORDER BY id", {"id": job_id, "project": self.project}))
        return [r["id"] for r in rows]

    def user_skills(self, user_id: str) -> StoreResult:
        return self._exec(lambda tx: self._q_user_skills(tx, user_id), write=False)

    def job_skills(self, job_id: str, strength: str = "required") -> StoreResult:
        if strength not in self._JOB_PATTERNS:
            raise ValueError("strength must be required|preferred|all")
        return self._exec(lambda tx: self._q_job_skills(tx, job_id, strength), write=False)

    def skill_gap(self, user_id: str, job_id: str) -> StoreResult:
        def work(tx: Any) -> dict[str, Any]:
            have = set(self._q_user_skills(tx, user_id))
            required = self._q_job_skills(tx, job_id, "required")
            preferred = self._q_job_skills(tx, job_id, "preferred")
            matched = [s for s in required if s in have]
            return {
                "user_id": user_id, "job_id": job_id, "required": required, "matched": matched,
                "missing": [s for s in required if s not in have], "preferred": preferred,
                "preferred_missing": [s for s in preferred if s not in have],
                "coverage": (len(matched) / len(required)) if required else None,
            }

        return self._exec(work, write=False)

    def courses_teaching(self, skill_id: str) -> StoreResult:
        def work(tx: Any) -> list[dict[str, Any]]:
            return self._rows(tx.run(
                "MATCH (c:`Course` {project: $project})-[r:`TEACHES`]->(s:`Skill` {id: $id, project: $project}) "
                "RETURN c.id AS course_id, r.depth AS depth, r.confidence AS confidence ORDER BY course_id",
                {"id": skill_id, "project": self.project}))

        return self._exec(work, write=False)

    def prerequisite_chain(self, course_id: str, max_depth: int = 4) -> StoreResult:
        def work(tx: Any) -> dict[str, Any]:
            exists = self._rows(tx.run("MATCH (c:`Course` {id: $id, project: $project}) RETURN c.id AS id",
                                       {"id": course_id, "project": self.project}))
            if not exists:
                return {"course_id": course_id, "levels": [], "max_depth": int(max_depth), "truncated": False, "skill_ids": []}

            def prereqs(course_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
                out: dict[str, list[dict[str, Any]]] = {}
                rows = self._rows(tx.run(
                    "UNWIND $ids AS cid MATCH (c:`Course` {id: cid, project: $project})"
                    "-[r:`REQUIRES_PREREQUISITE`]->(s:`Skill` {project: $project}) "
                    "RETURN c.id AS cid, s.id AS skill_id, r.level AS level, r.strictness AS strictness",
                    {"ids": list(course_ids), "project": self.project}))
                for row in rows:
                    out.setdefault(row["cid"], []).append(
                        {"skill_id": row["skill_id"], "level": row["level"], "strictness": row["strictness"]})
                return out

            def teachers(skill_ids: list[str]) -> dict[str, list[str]]:
                rows = self._rows(tx.run(
                    "UNWIND $ids AS sid MATCH (c:`Course` {project: $project})-[:`TEACHES`]->(s:`Skill` {id: sid, project: $project}) "
                    "RETURN s.id AS skill_id, collect(DISTINCT c.id) AS courses",
                    {"ids": list(skill_ids), "project": self.project}))
                return {r["skill_id"]: list(r["courses"]) for r in rows}

            return chain_from_primitives(course_id, max_depth, prereqs, teachers)

        return self._exec(work, write=False)

    def explain_path(self, source: str, target: str, max_hops: int = 4) -> StoreResult:
        """Shortest explanatory path via bounded variable-length Cypher, depth by depth, hop cap ``max_hops``.

        Mirrors ``explain_skill_path``: stored edges are followed forward; edges that point into a skill (and
        HAS_SKILL ...) may also be followed backwards (``REVERSE_TRAVERSABLE``); RELATED_TO only forward. All
        shortest paths at the first depth that has any are fetched (at most 2000) and the final path is then chosen
        by the *same* deterministic BFS the local store uses, so equal-length ties resolve identically.
        ``found=False, truncated=True`` means no path within ``max_hops`` (a longer one may exist).
        """
        cap = int(max_hops)
        if not 1 <= cap <= _MAX_PATH_LEN:
            raise ValueError(f"max_hops must be in 1..{_MAX_PATH_LEN}")
        s_label, t_label = label_of(source), label_of(target)
        if s_label is None or t_label is None:
            return StoreResult(normalize_path_result(source, target, False, [], [], cap, False, reason="unknown node"), "neo4j")
        types = list(_PATH_TYPES)
        reversible = sorted(REVERSE_TRAVERSABLE)
        pattern_types = "|".join(f"`{t}`" for t in _PATH_TYPES)  # fixed whitelist

        def work(tx: Any) -> dict[str, Any]:
            known = self._rows(tx.run(
                f"MATCH (a:`{s_label}` {{id: $s, project: $project}}) MATCH (b:`{t_label}` {{id: $t, project: $project}}) "
                "RETURN a.id AS a, b.id AS b", {"s": source, "t": target, "project": self.project}))
            if not known:
                return normalize_path_result(source, target, False, [], [], cap, False, reason="unknown node")
            if source == target:
                return normalize_path_result(source, target, True, [source], [], cap, False)
            for depth in range(1, cap + 1):
                rows = self._rows(tx.run(
                    f"MATCH p = (a:`{s_label}` {{id: $s, project: $project}})-[rs:{pattern_types}*{int(depth)}]-"
                    f"(b:`{t_label}` {{id: $t, project: $project}}) "
                    "WHERE all(i IN range(0, size(rs) - 1) WHERE type(rs[i]) IN $types AND "
                    "(startNode(rs[i]) = nodes(p)[i] OR type(rs[i]) IN $reversible)) "
                    "RETURN [r IN rs | {type: type(r), src: startNode(r).id, dst: endNode(r).id}] AS rels "
                    f"LIMIT {_PATH_LIMIT}",
                    {"s": source, "t": target, "project": self.project, "types": types, "reversible": reversible}))
                if rows:
                    return self._pick_path(source, target, rows, cap)
            return normalize_path_result(source, target, False, [], [], cap, True,
                                         reason=f"no path within max_hops={cap}")

        return self._exec(work, write=False)

    @staticmethod
    def _pick_path(source: str, target: str, rows: list[dict[str, Any]], cap: int) -> dict[str, Any]:
        import networkx as nx

        g = nx.MultiDiGraph()
        for row in rows:
            for rel in row["rels"]:
                g.add_edge(rel["src"], rel["dst"], key=rel["type"], type=rel["type"])
        res = explain_skill_path(g, source, target, max_hops=cap)
        if not res["found"]:  # cannot happen (rows are valid paths); stay honest if it does
            return normalize_path_result(source, target, False, [], [], cap, True, reason="no path reconstructed")
        return normalize_path_result(source, target, True, res["path"], res["edges"], cap, False)

    def stats(self) -> StoreResult:
        def work(tx: Any) -> dict[str, Any]:
            nodes = {label: int(self._rows(tx.run(f"MATCH (n:`{label}` {{project: $project}}) RETURN count(n) AS c",
                                                   {"project": self.project}))[0]["c"]) for label in LABELS}
            edges = {rel: int(self._rows(tx.run(f"MATCH ()-[r:`{rel}` {{project: $project}}]->() RETURN count(r) AS c",
                                                 {"project": self.project}))[0]["c"]) for rel in REL_TYPES}
            return {"project": self.project, "nodes": nodes, "edges": edges,
                    "total_nodes": sum(nodes.values()), "total_edges": sum(edges.values())}

        return self._exec(work, write=False)

    def health(self) -> StoreResult:
        def work(tx: Any) -> dict[str, Any]:
            rows = self._rows(tx.run("RETURN 1 AS ok"))
            return {"ok": bool(rows and rows[0]["ok"] == 1), "backend": "neo4j", "database": self.settings.database}

        return self._exec(work, write=False, retries=0)
