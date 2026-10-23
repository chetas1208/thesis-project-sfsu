"""Graph store tests. No live database: Neo4j is replaced by a fake driver; real neo4j exception classes are used."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

neo4j_exc = pytest.importorskip("neo4j.exceptions")
pytest.importorskip("networkx")

from skillllm_v2.graph import neo4j_store as neo4j_store_mod  # noqa: E402
from skillllm_v2.graph.local_store import LocalGraphStore  # noqa: E402
from skillllm_v2.graph.neo4j_adapters import (  # noqa: E402
    CuratedSkillIndex,
    assert_safe_plan,
    build_curated_plan,
    plan_content_hash,
    records_from_graph,
)
from skillllm_v2.graph.neo4j_store import Neo4jStore  # noqa: E402
from skillllm_v2.graph.store import (  # noqa: E402
    MAX_NODES,
    EdgeRecord,
    GraphPlan,
    GraphStore,
    GraphStoreError,
    NodeRecord,
    SizeGuardError,
    StoreResult,
    validate_project,
)
from skillllm_v2.graph.store_factory import FallbackStore, get_graph_store  # noqa: E402
from skillllm_v2.graph.store_settings import (  # noqa: E402
    Neo4jSettings,
    SettingsError,
    find_dotenv_files,
    load_neo4j_env,
    parse_dotenv,
    redact,
)

SECRET = "s3cr3t-PASSWORD-xyz"
URI = "neo4j+s://abcd1234.databases.neo4j.io"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_from_real_credentials(monkeypatch):
    """No test in this file may reach the real database: hide NEO4J_* env vars and any real .env from default lookups."""
    for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        monkeypatch.delenv(key, raising=False)
    from skillllm_v2.graph import store_settings

    original = store_settings.find_dotenv_files
    monkeypatch.setattr(store_settings, "find_dotenv_files", lambda search_from=None: original(search_from) if search_from is not None else [])


def settings(**kw) -> Neo4jSettings:
    return Neo4jSettings(uri=URI, username="neo4j-user", password=SECRET, database="db1", retry_backoff=0.5, **kw)


# --------------------------------------------------------------------------- fake driver
class FakeResult(list):
    def consume(self):
        return None


class FakeTx:
    def __init__(self, driver: "FakeDriver") -> None:
        self.d = driver

    def run(self, query, parameters=None, **kw):
        params = dict(parameters or {}, **kw)
        self.d.log.append((query, params))
        return FakeResult(self.d.respond(query, params))


class FakeSession:
    def __init__(self, driver: "FakeDriver") -> None:
        self.d = driver

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _go(self, fn, *a):
        self.d.attempts += 1
        if self.d.fail:
            err = self.d.fail.pop(0) if isinstance(self.d.fail, list) else self.d.fail
            raise err
        return fn(FakeTx(self.d), *a)

    execute_read = _go
    execute_write = _go


class FakeDriver:
    def __init__(self, fail=None, total=0, ours=0) -> None:
        self.log: list[tuple[str, dict]] = []
        self.fail = fail  # exception instance (always) or list (consumed one per attempt)
        self.attempts = 0
        self.closed = False
        self.total, self.ours = total, ours
        self.sessions: list[dict] = []

    def session(self, **kw):
        self.sessions.append(kw)
        return FakeSession(self)

    def close(self):
        self.closed = True

    def respond(self, query, params):
        if query.startswith("RETURN 1"):
            return [{"ok": 1}]
        if "count(CASE WHEN" in query:
            return [{"total": self.total, "ours": self.ours}]
        if query.startswith("UNWIND $rows") and "MERGE (n:" in query:
            return [{"n": len(params["rows"])}]
        if query.startswith("UNWIND $rows") and "MERGE (a)-[r" in query:
            return [{"n": len(params["rows"])}]
        if "DETACH DELETE" in query:
            return [{"deleted": min(self.ours, params["batch"])}] if self.ours else [{"deleted": 0}]
        return []


def factory_for(driver: FakeDriver):
    return lambda s: driver


def store_with(driver: FakeDriver, **kw) -> Neo4jStore:
    return Neo4jStore(settings(), driver_factory=factory_for(driver), sleep=lambda s: None, **kw)


def nodes_edges():
    nodes = [NodeRecord("User", "user:u1", {"name": "u1", "synthetic": True, "source": "demo"}),
             NodeRecord("Skill", "skill:python", {"name": "Python", "source": "curated_aliases"}),
             NodeRecord("Skill", "skill:aws", {"name": "AWS", "source": "curated_aliases"}),
             NodeRecord("Job", "job:j1", {"name": "j1", "source": "demo"}),
             NodeRecord("Course", "course:c1", {"name": "C1", "level": "beginner", "source": "demo"}),
             NodeRecord("Course", "course:c2", {"name": "C2", "level": "advanced", "source": "demo"})]
    edges = [EdgeRecord("HAS_SKILL", "user:u1", "skill:python", {"confidence": 1.0, "source": "demo"}),
             EdgeRecord("REQUIRES", "job:j1", "skill:python", {"importance": 1.0}),
             EdgeRecord("REQUIRES", "job:j1", "skill:aws", {"importance": 1.0}),
             EdgeRecord("TEACHES", "course:c1", "skill:python", {"depth": "beginner", "confidence": 1.0}),
             EdgeRecord("TEACHES", "course:c2", "skill:aws", {"depth": "advanced", "confidence": 0.9}),
             EdgeRecord("REQUIRES_PREREQUISITE", "course:c2", "skill:python", {"level": "beginner", "strictness": "strict"})]
    return nodes, edges


# --------------------------------------------------------------------------- .env parsing / settings
def test_parse_dotenv_quotes_comments_export():
    text = "# c\nNEO4J_URI='neo4j+s://h'\nexport NEO4J_USERNAME=\"u\"\nNEO4J_PASSWORD=p # trailing\n\nbad line\nX=1\n"
    assert parse_dotenv(text) == {"NEO4J_URI": "neo4j+s://h", "NEO4J_USERNAME": "u", "NEO4J_PASSWORD": "p", "X": "1"}


def _write_env(directory: Path, **kv: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / ".env"
    f.write_text("\n".join(f"{k}={v}" for k, v in kv.items()))
    return f


def test_env_file_in_parent_dir_is_found(tmp_path):
    _write_env(tmp_path, NEO4J_URI=URI, NEO4J_USERNAME="u", NEO4J_PASSWORD="pw", NEO4J_DATABASE="d")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert find_dotenv_files([deep]) == [tmp_path / ".env"]
    s = Neo4jSettings.from_env(env={}, search_from=[deep])
    assert (s.username, s.password, s.database) == ("u", "pw", "d")


def test_nearer_env_file_wins_and_missing_keys_fall_through(tmp_path):
    _write_env(tmp_path, NEO4J_URI=URI, NEO4J_USERNAME="far-user", NEO4J_PASSWORD="far-pw", NEO4J_DATABASE="far-db")
    near = tmp_path / "proj"
    _write_env(near, NEO4J_USERNAME="near-user", NEO4J_PASSWORD="near-pw")
    merged = load_neo4j_env(env={}, search_from=[near])
    assert merged["NEO4J_USERNAME"] == "near-user" and merged["NEO4J_PASSWORD"] == "near-pw"
    assert merged["NEO4J_URI"] == URI and merged["NEO4J_DATABASE"] == "far-db"  # not in the nearer file -> parent file


def test_real_env_overrides_file(tmp_path):
    _write_env(tmp_path, NEO4J_URI=URI, NEO4J_USERNAME="file-user", NEO4J_PASSWORD="file-pw")
    s = Neo4jSettings.from_env(env={"NEO4J_PASSWORD": "env-pw", "NEO4J_USERNAME": "env-user"}, search_from=[tmp_path])
    assert (s.username, s.password, s.uri) == ("env-user", "env-pw", URI)


def test_missing_env_file_is_ok_but_settings_error_is_raised(tmp_path):
    assert load_neo4j_env(env={}, search_from=[tmp_path]) == {}
    assert Neo4jSettings.available(env={}, search_from=[tmp_path]) is False
    with pytest.raises(SettingsError) as info:
        Neo4jSettings.from_env(env={}, search_from=[tmp_path])
    assert "NEO4J_URI" in str(info.value)


def test_only_neo4j_keys_are_read_from_files(tmp_path):
    _write_env(tmp_path, NEO4J_URI=URI, OTHER_SECRET="nope", AWS_KEY="nope")
    assert set(load_neo4j_env(env={}, search_from=[tmp_path])) == {"NEO4J_URI"}


def test_non_tls_uri_refused():
    with pytest.raises(SettingsError):
        Neo4jSettings(uri="bolt://localhost:7687", username="u", password="p")
    with pytest.raises(SettingsError):
        Neo4jSettings(uri="neo4j://host", username="u", password="p")


def test_settings_repr_and_str_hide_credentials():
    s = settings()
    for text in (repr(s), str(s), f"{s}", repr(Neo4jStore(s))):
        assert SECRET not in text and "abcd1234" not in text and "neo4j-user" not in text


def test_redact_masks_uri_password_and_pairs():
    msg = f"failed {URI} user=neo4j+s://bob:{SECRET}@host/x password={SECRET} token: abc123"
    out = redact(msg, [SECRET])
    assert SECRET not in out and "abcd1234" not in out and "bob" not in out and "abc123" not in out


# --------------------------------------------------------------------------- Cypher shape
def test_node_merge_is_parameterised_idempotent_cypher():
    d = FakeDriver()
    store = store_with(d)
    nodes, _ = nodes_edges()
    res = store.upsert_nodes(nodes, "b1")
    assert res.ok and res.backend == "neo4j" and res.value == {"nodes": 6}
    writes = [(q, p) for q, p in d.log if q.startswith("UNWIND $rows")]
    assert {q.split("MERGE (n:`")[1].split("`")[0] for q, _ in writes} == {"User", "Skill", "Job", "Course"}
    for q, p in writes:
        assert "MERGE (n:" in q and "CREATE (" not in q and "{id: row.id, project: $project}" in q
        assert set(p) == {"rows", "project", "build_id"} and p["project"] == "skillllm_v2" and p["build_id"] == "b1"
        for row in p["rows"]:  # values travel as parameters, never inside the query text
            assert row["id"] not in q
    assert not any("python" in q.lower() or "u1" in q for q, _ in writes)


def test_edge_merge_matches_endpoints_and_never_creates_nodes():
    d = FakeDriver()
    store = store_with(d)
    _, edges = nodes_edges()
    res = store.upsert_edges(edges, "b1")
    assert res.ok and res.value == {"edges": 6, "skipped": 0}
    qs = [q for q, _ in d.log if "MERGE (a)-[r" in q]
    assert len(qs) == 4  # HAS_SKILL, REQUIRES, TEACHES, REQUIRES_PREREQUISITE
    for q in qs:
        assert q.count("MATCH (a:") == 1 and q.count("MATCH (b:") == 1 and "MERGE (a)-[r:" in q and "MERGE (n" not in q
        assert "r.project = $project" in q and "r.build_id = $build_id" in q


def test_writes_are_batched():
    d = FakeDriver()
    store = store_with(d, batch_size=500)
    nodes = [NodeRecord("Skill", f"skill:s{i}", {"name": f"s{i}", "source": "curated_aliases"}) for i in range(1201)]
    store.upsert_nodes(nodes, "b")
    sizes = [len(p["rows"]) for q, p in d.log if q.startswith("UNWIND $rows")]
    assert sizes == [500, 500, 201]


def test_schema_statements_are_if_not_exists_and_constraints_per_label():
    d = FakeDriver()
    store = store_with(d)
    assert store.ensure_schema().ok
    stmts = [q for q, _ in d.log]
    assert all("IF NOT EXISTS" in s for s in stmts)
    for label in ("User", "Job", "Skill", "Course"):
        assert any(f"FOR (n:`{label}`) REQUIRE n.id IS UNIQUE" in s for s in stmts)
    assert not any(s.startswith("DROP") for s in stmts)


def test_explain_path_uses_bounded_varlength_with_params_and_whitelisted_types():
    d = FakeDriver()
    store = store_with(d)
    # first query: endpoints exist; later queries return no rows -> truncated at the cap
    seen = []

    def respond(query, params):
        seen.append(query)
        return [{"a": "user:u1", "b": "job:j1"}] if "RETURN a.id" in query else []

    d.respond = respond
    res = store.explain_path("user:u1", "job:j1", max_hops=3)
    assert res.ok and res.value["found"] is False and res.value["truncated"] is True
    var = [q for q in seen if "rs:" in q]
    assert [q.split("*")[1].split("]")[0] for q in var] == ["1", "2", "3"]
    assert all("$s" in q and "$t" in q and "user:u1" not in q for q in var)


def test_explain_path_rejects_unbounded_hops():
    with pytest.raises(ValueError):
        store_with(FakeDriver()).explain_path("user:u1", "job:j1", max_hops=99)


def test_pick_path_matches_local_bfs_tiebreak():
    rows = [{"rels": [{"type": "HAS_SKILL", "src": "user:u", "dst": "skill:b"}, {"type": "REQUIRES", "src": "job:j", "dst": "skill:b"}]},
            {"rels": [{"type": "HAS_SKILL", "src": "user:u", "dst": "skill:a"}, {"type": "REQUIRES", "src": "job:j", "dst": "skill:a"}]}]
    out = Neo4jStore._pick_path("user:u", "job:j", rows, 4)
    assert out["found"] and out["path"] == ["user:u", "skill:a", "job:j"]  # sorted-neighbour BFS, same as local
    assert [e["direction"] for e in out["edges"]] == ["forward", "reverse"]


# --------------------------------------------------------------------------- namespace safety
def test_clear_namespace_only_touches_own_project_in_batches():
    d = FakeDriver(ours=2500)

    def respond(query, params):
        d.log_deleted = getattr(d, "log_deleted", 0)
        n = min(2500 - d.log_deleted, params["batch"])
        d.log_deleted += n
        return [{"deleted": n}]

    d.respond = respond
    store = store_with(d, batch_size=1000)
    res = store.clear_namespace()
    assert res.ok and res.value["nodes_deleted"] == 2500 and res.value["project"] == "skillllm_v2"
    for q, p in d.log:
        assert "{project: $project}" in q and "DETACH DELETE" in q and "LIMIT $batch" in q
        assert p["project"] == "skillllm_v2" and "DROP" not in q.upper().replace("DETACH", "")
        assert not q.startswith("MATCH (n) ") and "MATCH (n)" not in q.replace("MATCH (n:", "")


def test_clear_refuses_foreign_namespaces():
    d = FakeDriver()
    store = store_with(d)
    for bad in ("production", "", "skillllm_v2; MATCH (n) DETACH DELETE n", "SKILLLLM_V2", "skillllm_v2_testX!"):
        with pytest.raises(GraphStoreError):
            store.clear_namespace(bad)
    assert d.log == []
    assert validate_project("skillllm_v2_test") and validate_project("skillllm_v2_test_ab12")


def test_writes_refuse_foreign_namespace():
    store = store_with(FakeDriver())
    with pytest.raises(GraphStoreError):
        store.upsert_nodes(nodes_edges()[0], "b", project="someone_else")


# --------------------------------------------------------------------------- size guard
def test_size_guard_refuses_oversized_plan_without_connecting():
    called = []
    store = Neo4jStore(settings(), driver_factory=lambda s: called.append(1) or FakeDriver())
    big = [NodeRecord("Skill", f"skill:s{i}", {"name": "x", "source": "curated_aliases"}) for i in range(MAX_NODES + 1)]
    with pytest.raises(SizeGuardError):
        store.upsert_nodes(big, "b")
    assert called == []


def test_size_guard_counts_what_is_already_in_the_database():
    d = FakeDriver(total=MAX_NODES - 2, ours=0)
    store = store_with(d)
    with pytest.raises(SizeGuardError):
        store.upsert_nodes(nodes_edges()[0], "b")
    assert not any(q.startswith("UNWIND $rows") for q, _ in d.log)  # nothing written


def test_plan_size_guard_and_counts():
    plan = GraphPlan()
    nodes, edges = nodes_edges()
    plan.extend(nodes, edges)
    assert plan.counts()["total_nodes"] == 6 and plan.counts()["edges"]["TEACHES"] == 2
    plan.check_size(100)
    with pytest.raises(SizeGuardError):
        plan.check_size(5)


# --------------------------------------------------------------------------- errors, redaction, retry
@pytest.mark.parametrize("exc", [
    neo4j_exc.AuthError(f"unauthorized for {URI} password={SECRET} user neo4j-user"),
    neo4j_exc.ServiceUnavailable(f"Unable to retrieve routing information from {URI} ({SECRET})"),
    TimeoutError(f"timed out talking to {URI} with {SECRET}"),
    ConnectionRefusedError(f"refused {URI}:{SECRET}"),
    neo4j_exc.ClientError(f"boom {SECRET}"),
])
def test_errors_are_redacted_and_chained_cause_suppressed(exc, caplog):
    d = FakeDriver(fail=exc)
    store = store_with(d)
    with caplog.at_level(logging.DEBUG):
        res = store.user_skills("user:u1")
    assert res.ok is False and res.backend == "neo4j" and res.value is None
    assert SECRET not in res.error and "abcd1234" not in res.error
    assert all(SECRET not in r.getMessage() and "abcd1234" not in r.getMessage() for r in caplog.records)
    assert SECRET not in repr(res) and SECRET not in repr(store)


def test_raised_wrapper_error_has_no_cause_and_no_secret():
    d = FakeDriver(fail=neo4j_exc.AuthError(f"bad {SECRET}"))
    store = store_with(d)
    with pytest.raises(GraphStoreError) as info:
        store._run(lambda tx: [], write=False)
    err = info.value
    assert SECRET not in str(err) and err.__cause__ is None and err.__suppress_context__ is True


def test_driver_factory_failure_is_redacted():
    def boom(s):
        raise neo4j_exc.ServiceUnavailable(f"cannot reach {s.uri} with {s.password}")

    store = Neo4jStore(settings(), driver_factory=boom, sleep=lambda s: None)
    res = store.health()
    assert not res.ok and SECRET not in res.error and "abcd1234" not in res.error


def test_transient_errors_retry_with_exponential_backoff_then_succeed():
    sleeps = []
    d = FakeDriver(fail=[neo4j_exc.ServiceUnavailable("x"), neo4j_exc.SessionExpired("y")])
    store = Neo4jStore(settings(), driver_factory=factory_for(d), sleep=sleeps.append)
    res = store.health()
    assert res.ok is False or res.ok  # health uses retries=0; use a normal read below
    d.fail = [neo4j_exc.ServiceUnavailable("x"), neo4j_exc.SessionExpired("y")]
    sleeps.clear()
    d.attempts = 0
    d.respond = lambda q, p: [{"id": "skill:python"}]
    res = store.user_skills("user:u1")
    assert res.ok and res.value == ["skill:python"] and d.attempts == 3 and sleeps == [0.5, 1.0]


def test_auth_errors_are_never_retried():
    sleeps = []
    d = FakeDriver(fail=neo4j_exc.AuthError("nope"))
    store = Neo4jStore(settings(), driver_factory=factory_for(d), sleep=sleeps.append)
    assert not store.user_skills("user:u1").ok
    assert d.attempts == 1 and sleeps == []


def test_retries_are_bounded():
    sleeps = []
    d = FakeDriver(fail=neo4j_exc.ServiceUnavailable("down"))
    store = Neo4jStore(settings(), driver_factory=factory_for(d), sleep=sleeps.append)
    assert not store.user_skills("user:u1").ok
    assert d.attempts == 3 and len(sleeps) == 2  # 1 try + max_retries=2


def test_pool_closed_on_close_and_context_exit():
    d = FakeDriver()
    with store_with(d) as store:
        assert store.health().ok
    assert d.closed
    d2 = FakeDriver()
    s2 = store_with(d2)
    s2.health()
    s2.close()
    assert d2.closed


def test_session_uses_configured_database_and_timeouts():
    d = FakeDriver()
    store = store_with(d)
    store.health()
    assert d.sessions == [{"database": "db1"}]
    assert settings().connection_timeout > 0 and settings().query_timeout > 0


def test_default_driver_factory_passes_timeouts_and_tls(monkeypatch):
    import neo4j

    captured = {}
    monkeypatch.setattr(neo4j.GraphDatabase, "driver", staticmethod(lambda uri, **kw: captured.update(uri=uri, **kw) or FakeDriver()))
    Neo4jStore(settings()).health()
    assert captured["uri"].startswith("neo4j+s://") and captured["auth"] == ("neo4j-user", SECRET)
    assert captured["connection_timeout"] == 5.0 and "connection_acquisition_timeout" in captured


# --------------------------------------------------------------------------- local store semantics
def loaded_local() -> LocalGraphStore:
    local = LocalGraphStore()
    nodes, edges = nodes_edges()
    local.upsert_nodes(nodes, "b1")
    local.upsert_edges(edges, "b1")
    return local


def test_local_store_satisfies_protocol_and_answers():
    local = loaded_local()
    assert isinstance(local, GraphStore) and isinstance(store_with(FakeDriver()), GraphStore)
    assert local.user_skills("user:u1").value == ["skill:python"]
    gap = local.skill_gap("user:u1", "job:j1").value
    assert gap["missing"] == ["skill:aws"] and gap["matched"] == ["skill:python"] and gap["coverage"] == 0.5
    assert [r["course_id"] for r in local.courses_teaching("skill:python").value] == ["course:c1"]
    chain = local.prerequisite_chain("course:c2").value
    assert chain["levels"][0]["skills"][0]["taught_by"] == ["course:c1"] and not chain["truncated"]
    path = local.explain_path("user:u1", "job:j1", 4).value
    assert path["found"] and path["hops"] == 2 and path["path"] == ["user:u1", "skill:python", "job:j1"]
    assert local.explain_path("user:u1", "course:c2", 1).value["truncated"] is True
    res = local.user_skills("user:u1")
    assert res.backend == "local" and res.fell_back is False and res.error is None


def test_local_unknown_ids_are_empty_not_leaky():
    local = loaded_local()
    assert local.user_skills("us").value == [] and local.courses_teaching("skill:nope").value == []
    assert local.explain_path("user:u1", "job:zzz").value["reason"] == "unknown node"


def test_local_upsert_is_idempotent_and_skips_dangling_edges():
    local = loaded_local()
    before = local.stats().value
    nodes, edges = nodes_edges()
    local.upsert_nodes(nodes, "b2")
    local.upsert_edges(edges, "b2")
    assert local.stats().value == before
    res = local.upsert_edges([EdgeRecord("HAS_SKILL", "user:ghost", "skill:python", {})], "b2")
    assert res.value == {"edges": 0, "skipped": 1}
    assert local.stats().value == before


def test_local_clear_namespace_only_removes_matching_project():
    local = LocalGraphStore()
    nodes, _ = nodes_edges()
    local.upsert_nodes(nodes, "b1")
    local.upsert_nodes([NodeRecord("Skill", "skill:t1", {"name": "t"})], "t", project="skillllm_v2_test")
    assert local.clear_namespace("skillllm_v2_test").value["nodes_deleted"] == 1
    assert local.stats().value["total_nodes"] == 6


# --------------------------------------------------------------------------- fallback / mirror
def local_for_fallback() -> LocalGraphStore:
    nodes, edges = nodes_edges()

    def load(store: LocalGraphStore) -> None:
        store.upsert_nodes(nodes, "b1")
        store.upsert_edges(edges, "b1")

    return LocalGraphStore(loader=load)


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.mark.parametrize("exc", [
    neo4j_exc.AuthError(f"bad credentials {SECRET}"),
    TimeoutError(f"timeout {SECRET}"),
    neo4j_exc.ServiceUnavailable(f"unavailable {URI}"),
])
def test_auto_falls_back_to_local_on_auth_timeout_unavailable(exc, caplog):
    d = FakeDriver(fail=exc)
    store = get_graph_store({"graph": {"backend": "auto"}}, settings=settings(), local=local_for_fallback(),
                            driver_factory=factory_for(d))
    with caplog.at_level(logging.DEBUG):
        res = store.user_skills("user:u1")
    assert res.value == ["skill:python"] and res.backend == "local" and res.fell_back is True
    assert res.error and SECRET not in res.error and "abcd1234" not in res.error
    assert isinstance(store, FallbackStore) and store.fell_back
    assert all(SECRET not in r.getMessage() for r in caplog.records)
    assert SECRET not in repr(store)


def test_auto_uses_neo4j_when_healthy():
    d = FakeDriver()
    d.respond = lambda q, p: [{"ok": 1}] if q.startswith("RETURN 1") else [{"id": "skill:python"}]
    store = get_graph_store(backend="auto", settings=settings(), local=local_for_fallback(), driver_factory=factory_for(d))
    res = store.user_skills("user:u1")
    assert res.backend == "neo4j" and res.fell_back is False and res.error is None


def test_health_result_is_cached_so_dead_db_does_not_slow_every_call():
    clock = Clock()
    d = FakeDriver(fail=neo4j_exc.ServiceUnavailable("down"))
    store = get_graph_store(backend="auto", settings=settings(), local=local_for_fallback(),
                            driver_factory=factory_for(d), clock=clock, config={"health_ttl_seconds": 10})
    for _ in range(5):
        assert store.user_skills("user:u1").fell_back
    assert store.probes == 1 and d.attempts == 1
    clock.t += 11  # TTL expired: one more probe
    store.user_skills("user:u1")
    assert store.probes == 2


def test_health_cache_for_healthy_db_probes_once_per_ttl():
    clock = Clock()
    d = FakeDriver()
    d.respond = lambda q, p: [{"ok": 1}] if q.startswith("RETURN 1") else [{"id": "skill:python"}]
    store = get_graph_store(backend="auto", settings=settings(), local=local_for_fallback(),
                            driver_factory=factory_for(d), clock=clock)
    for _ in range(4):
        store.user_skills("user:u1")
    assert store.probes == 1


def test_query_failure_mid_flight_falls_back_for_reads():
    d = FakeDriver()

    def respond(q, p):
        if q.startswith("RETURN 1"):
            return [{"ok": 1}]
        raise neo4j_exc.ClientError(f"query exploded {SECRET}")

    d.respond = respond
    store = get_graph_store(backend="mirror", settings=settings(), local=local_for_fallback(), driver_factory=factory_for(d))
    res = store.courses_teaching("skill:python")
    assert res.backend == "local" and res.fell_back and [r["course_id"] for r in res.value] == ["course:c1"]
    assert SECRET not in res.error


def test_explicit_neo4j_backend_without_credentials_is_a_config_error(monkeypatch):
    with pytest.raises(SettingsError):
        get_graph_store(backend="neo4j", env={})


def test_auto_without_credentials_falls_back_with_reason(monkeypatch):
    store = get_graph_store(backend="auto", local=local_for_fallback(), env={})
    res = store.user_skills("user:u1")
    assert res.fell_back and res.backend == "local" and "missing Neo4j setting" in res.error


def test_mirror_builds_local_always_and_syncs_to_neo4j_when_reachable():
    d = FakeDriver()
    local = LocalGraphStore()
    store = get_graph_store(backend="mirror", settings=settings(), local=local, driver_factory=factory_for(d))
    nodes, edges = nodes_edges()
    r1 = store.upsert_nodes(nodes, "b1")
    r2 = store.upsert_edges(edges, "b1")
    assert r1.backend == "neo4j" and not r1.fell_back and r2.backend == "neo4j"
    assert local.stats().value["total_nodes"] == 6 and local.stats().value["total_edges"] == 6
    assert sum(len(p["rows"]) for q, p in d.log if q.startswith("UNWIND $rows") and "MERGE (n:" in q) == 6


def test_mirror_writes_local_when_neo4j_is_down_and_can_sync_later():
    clock = Clock()
    d = FakeDriver(fail=neo4j_exc.ServiceUnavailable("down"))
    local = LocalGraphStore()
    store = get_graph_store(backend="mirror", settings=settings(), local=local, driver_factory=factory_for(d),
                            clock=clock, config={"health_ttl_seconds": 5})
    nodes, edges = nodes_edges()
    r1 = store.upsert_nodes(nodes, "b1")
    r2 = store.upsert_edges(edges, "b1")
    assert r1.backend == "local" and r1.fell_back and r2.backend == "local"
    assert local.stats().value["total_nodes"] == 6
    d.fail = None  # database back
    d.respond = FakeDriver.respond.__get__(d)  # healthy default behaviour again
    clock.t += 6
    res = store.sync_to_neo4j("b2")
    assert res.backend == "neo4j" and res.value["synced"] is True and res.value["nodes"] == 6 and res.value["edges"] == 6


def test_mirror_reads_prefer_neo4j():
    d = FakeDriver()
    d.respond = lambda q, p: [{"ok": 1}] if q.startswith("RETURN 1") else [{"id": "skill:from-neo4j"}]
    store = get_graph_store(backend="mirror", settings=settings(), local=local_for_fallback(), driver_factory=factory_for(d))
    assert store.user_skills("user:u1").value == ["skill:from-neo4j"]


def test_local_backend_is_plain_local_store():
    store = get_graph_store(backend="local")
    assert isinstance(store, LocalGraphStore)
    with pytest.raises(ValueError):
        get_graph_store(backend="postgres")


def test_fallback_clear_namespace_is_namespaced():
    d = FakeDriver()
    local = loaded_local()
    store = get_graph_store(backend="mirror", settings=settings(), local=local, driver_factory=factory_for(d))
    with pytest.raises(GraphStoreError):
        store.clear_namespace("prod")
    store.clear_namespace()
    assert local.stats().value["total_nodes"] == 0


# --------------------------------------------------------------------------- adapters / data safety
def test_curated_plan_is_synthetic_only_and_includes_rewritten_scenarios():
    plan, prov = build_curated_plan(REPO)
    assert_safe_plan(plan)
    assert "benchmark_scenarios_other_source" not in prov["skipped"]
    assert prov["included"]["benchmark_scenarios_synthetic"]["scenarios"] == 48
    assert any(n.id == "job:job-sc-42" for n in plan.nodes.values())
    assert {n.props["source"] for n in plan.nodes.values()} <= {"curated_aliases", "synthetic_catalog", "benchmark_synthetic", "demo", "derived_skill"}
    for n in plan.nodes.values():
        assert not ({"text", "description", "learning_outcomes", "source_sentence"} & set(n.props))
        assert all(len(str(v)) <= 300 for v in n.props.values())
    counts = plan.counts()
    assert counts["dangling_edges"] == 0 and counts["total_nodes"] < MAX_NODES
    assert plan_content_hash(plan) == plan_content_hash(build_curated_plan(REPO)[0])  # deterministic => idempotent build_id


def test_curated_plan_refuses_real_posting_excerpt_scenarios(tmp_path):
    import shutil
    (tmp_path / "inputs" / "courses").mkdir(parents=True)
    shutil.copy(REPO / "inputs" / "courses" / "course_catalog_large.jsonl", tmp_path / "inputs" / "courses")
    rows = [json.loads(line) for line in (REPO / "inputs" / "courses" / "benchmark_scenarios.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["source"] = "real_posting_excerpt"
    (tmp_path / "inputs" / "courses" / "benchmark_scenarios.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(GraphStoreError):
        build_curated_plan(tmp_path)


def test_assert_safe_plan_rejects_free_text_and_foreign_sources():
    plan = GraphPlan()
    plan.add_node(NodeRecord("Job", "job:x", {"name": "x", "source": "demo", "text": "a posting"}))
    with pytest.raises(GraphStoreError):
        assert_safe_plan(plan)
    plan = GraphPlan()
    plan.add_node(NodeRecord("Job", "job:x", {"name": "x" * 400, "source": "demo"}))
    with pytest.raises(GraphStoreError):
        assert_safe_plan(plan)
    plan = GraphPlan()
    plan.add_node(NodeRecord("Job", "job:x", {"name": "x", "source": "Extract_jobDesc"}))
    with pytest.raises(GraphStoreError):
        assert_safe_plan(plan)


def test_records_from_graph_is_read_only_and_drops_free_text():
    from skillllm_v2.graph.course_graph_builder import CourseGraphBuilder

    cb = CourseGraphBuilder()
    cb.ingest_course({"course_id": "c9", "title": "T", "description": "LONG DESCRIPTION", "teaches_skills": ["Python"],
                      "prerequisites": ["SQL"], "learning_outcomes": ["Able to do things"], "level": "beginner"})
    snapshot = (cb.graph.number_of_nodes(), cb.graph.number_of_edges(), dict(cb.graph.nodes["c9"]))
    nodes, edges, skipped = records_from_graph(cb.graph, source="synthetic_catalog", synthetic=True, index=CuratedSkillIndex.load())
    assert snapshot == (cb.graph.number_of_nodes(), cb.graph.number_of_edges(), dict(cb.graph.nodes["c9"]))
    course = next(n for n in nodes if n.id == "course:c9")
    assert "description" not in course.props and "learning_outcomes" not in course.props and course.props["level"] == "beginner"
    assert {e.type for e in edges} == {"TEACHES", "REQUIRES_PREREQUISITE"} and skipped["edges_dangling"] == 0
    assert any(n.id == "skill:python" for n in nodes)


def test_no_secrets_in_repo_files():
    """The .env.example is placeholders only and no tracked source file embeds a connection URI with credentials."""
    example = REPO.parent / ".env.example"
    assert example.exists()
    text = example.read_text()
    assert "<your-password>" in text and "<your-instance-id>" in text  # placeholders only
    for path in list((REPO / "src" / "skillllm_v2" / "graph").glob("*.py")) + [REPO / "scripts" / "build_neo4j_graph.py"]:
        content = path.read_text()
        assert "databases.neo4j.io" not in content and "NEO4J_PASSWORD=" not in content


# --------------------------------------------------------------------------- pipeline wiring
def test_pipeline_graph_backend_default_local_records_provenance():
    from skillllm_v2.pipeline.inference import SkillLLMV2Pipeline

    p = SkillLLMV2Pipeline.__new__(SkillLLMV2Pipeline)
    p.cfg = {"graph": {"max_hops": 4, "backend": "local"}}
    courses, info = p._graph_store_lookup(["Python"])
    assert courses == {} and info == {"requested": "local", "backend": "local", "fell_back": False, "error": None, "queries": 0}


def test_pipeline_graph_backend_auto_falls_back_without_failing_the_stage(monkeypatch, tmp_path):
    from skillllm_v2.pipeline.inference import SkillLLMV2Pipeline

    d = FakeDriver(fail=neo4j_exc.AuthError(f"bad password {SECRET} for {URI}"))
    monkeypatch.setattr(neo4j_store_mod, "_default_driver_factory", lambda s: d)
    monkeypatch.setenv("NEO4J_URI", URI)
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j-user")
    monkeypatch.setenv("NEO4J_PASSWORD", SECRET)
    p = SkillLLMV2Pipeline.__new__(SkillLLMV2Pipeline)
    p.cfg = {"graph": {"max_hops": 4, "backend": "auto"}}
    p.root = REPO
    courses, info = p._graph_store_lookup(["Docker", "Kubernetes"])
    assert info["requested"] == "auto" and info["backend"] == "local" and info["fell_back"] is True
    assert SECRET not in str(info) and "abcd1234" not in str(info)
    assert "Docker" in courses and all(isinstance(c, str) for c in courses["Docker"])  # answered by the local curated graph
    p.close_graph_store()


def test_stage_result_dict_has_catalog_courses_only_when_present():
    from skillllm_v2.pipeline.inference_types import GraphArtifact, StageRecord

    base = dict(user_node="u", job_node=None, user_skills=[], job_required_skills=[], graph_missing_skills=[], explanations=[],
                node_count=0, edge_count=0, record=StageRecord(name="graph"))
    assert "catalog_courses" not in GraphArtifact(**base).to_dict()
    assert GraphArtifact(**base, catalog_courses={"Docker": ["devops-001"]}).to_dict()["catalog_courses"] == {"Docker": ["devops-001"]}


def test_stage_graph_records_backend_in_stage_record_and_default_is_unchanged(monkeypatch):
    """End-to-end through stage_graph: default local, then auto with a dead Neo4j (fake driver), never failing the stage."""
    import test_inference_pipeline as tip  # reuse its helpers

    p = tip.make_pipeline()
    ext = p.run_extraction_only(tip.PROFILE_TEXT)
    jobs = p._prepare_jobs([tip.JOB], set(tip.STAGE_ORDER), set(), ext.trace)
    base = p.stage_graph("u1", ext.enriched_skills, jobs, jobs[0])
    assert base.record.provenance["graph_backend"]["backend"] == "local" and base.catalog_courses == {}

    d = FakeDriver(fail=neo4j_exc.ServiceUnavailable(f"down {URI} {SECRET}"))
    monkeypatch.setattr(neo4j_store_mod, "_default_driver_factory", lambda s: d)
    monkeypatch.setenv("NEO4J_URI", URI)
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j-user")
    monkeypatch.setenv("NEO4J_PASSWORD", SECRET)
    p.cfg["graph"]["backend"] = "auto"
    g = p.stage_graph("u1", ext.enriched_skills, jobs, jobs[0])
    info = g.record.provenance["graph_backend"]
    assert g.record.status == "ok" and info["requested"] == "auto" and info["fell_back"] is True and info["backend"] == "local"
    assert SECRET not in str(g.record.to_dict()) and "abcd1234" not in str(g.record.to_dict())
    assert g.graph_missing_skills == base.graph_missing_skills and g.explanations == base.explanations  # graph answer unchanged
    assert "Amazon Web Services" in g.catalog_courses  # catalogue courses answered by the local curated graph
    p.close_graph_store()


def test_bare_hostname_in_driver_messages_is_redacted():
    s = settings()
    assert s.host == "abcd1234.databases.neo4j.io"
    assert "abcd1234" not in s.redact("Cannot resolve address abcd1234.databases.neo4j.io:7687")
    d = FakeDriver(fail=ValueError("Cannot resolve address abcd1234.databases.neo4j.io:7687"))
    res = store_with(d).user_skills("user:u1")
    assert not res.ok and "abcd1234" not in res.error
