"""Live Neo4j tests. Skipped unless NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD are available (environment variables or the
nearest .env, see store_settings) AND ``RUN_LIVE_NEO4J=1``.

    RUN_LIVE_NEO4J=1 PYTHONPATH=src python -m pytest tests/live -q -p no:cacheprovider

Every write goes to the throwaway namespace ``project='skillllm_v2_test'`` with ids that cannot collide with the real
graph (``*:__t_*``); the namespace is cleared before and after, and the main namespace is only ever *counted*.
"""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("networkx")

from skillllm_v2.graph.local_store import LocalGraphStore  # noqa: E402
from skillllm_v2.graph.neo4j_store import Neo4jStore  # noqa: E402
from skillllm_v2.graph.store import PROJECT, TEST_PROJECT, EdgeRecord, NodeRecord  # noqa: E402
from skillllm_v2.graph.store_factory import get_graph_store  # noqa: E402
from skillllm_v2.graph.store_settings import Neo4jSettings  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_NEO4J") != "1" or not Neo4jSettings.available(),
    reason="live Neo4j test: set RUN_LIVE_NEO4J=1 and NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD (env or .env)",
)


def assert_no_secret(text: object, *secrets: str) -> None:
    """Fail WITHOUT echoing either operand (pytest's assert introspection would print the secret)."""
    blob = str(text)
    if any(secret and secret in blob for secret in secrets):
        pytest.fail("a credential, URI or host leaked into an error/repr (value deliberately not shown)")


def _records():
    nodes = [
        NodeRecord("User", "user:__t_u", {"name": "tu", "synthetic": True, "source": "demo"}),
        NodeRecord("Job", "job:__t_j", {"name": "tj", "synthetic": True, "source": "demo"}),
        NodeRecord("Skill", "skill:__t_py", {"name": "py", "source": "demo"}),
        NodeRecord("Skill", "skill:__t_aws", {"name": "aws", "source": "demo", "aliases": ["a", "b"]}),
        NodeRecord("Course", "course:__t_c1", {"name": "c1", "level": "beginner", "duration_hours": 10.0, "source": "demo"}),
        NodeRecord("Course", "course:__t_c2", {"name": "c2", "level": "advanced", "source": "demo"}),
    ]
    edges = [
        EdgeRecord("HAS_SKILL", "user:__t_u", "skill:__t_py", {"confidence": 0.9, "source": "demo"}),
        EdgeRecord("REQUIRES", "job:__t_j", "skill:__t_py", {"importance": 1.0}),
        EdgeRecord("REQUIRES", "job:__t_j", "skill:__t_aws", {"importance": 1.0}),
        EdgeRecord("TEACHES", "course:__t_c1", "skill:__t_py", {"depth": "beginner", "confidence": 1.0}),
        EdgeRecord("TEACHES", "course:__t_c2", "skill:__t_aws", {"depth": "advanced", "confidence": 0.8}),
        EdgeRecord("REQUIRES_PREREQUISITE", "course:__t_c2", "skill:__t_py", {"level": "beginner", "strictness": "strict"}),
    ]
    return nodes, edges


@pytest.fixture()
def store():
    s = Neo4jStore(Neo4jSettings.from_env(), project=TEST_PROJECT)
    assert s.clear_namespace(TEST_PROJECT).ok
    yield s
    s.clear_namespace(TEST_PROJECT)
    s.close()


def test_connectivity(store):
    res = store.health()
    assert res.ok and res.backend == "neo4j" and res.value["ok"] is True and res.error is None


def test_schema_is_idempotent(store):
    assert store.ensure_schema().ok and store.ensure_schema().ok
    info = store.schema_info()
    names = {c["name"] for c in info.value["constraints"]}
    assert {"user_id_unique", "job_id_unique", "skill_id_unique", "course_id_unique"} <= names


def test_write_read_clear_round_trip_in_throwaway_namespace(store):
    assert store.ensure_schema().ok
    main_before = Neo4jStore(Neo4jSettings.from_env(), project=PROJECT).stats().value
    nodes, edges = _records()
    assert store.upsert_nodes(nodes, "live-t1").value == {"nodes": 6}
    assert store.upsert_edges(edges, "live-t1").value == {"edges": 6, "skipped": 0}
    stats = store.stats().value
    assert stats["total_nodes"] == 6 and stats["total_edges"] == 6

    # parity with the local store on every read
    local = LocalGraphStore(project=TEST_PROJECT)
    local.upsert_nodes(nodes, "live-t1")
    local.upsert_edges(edges, "live-t1")
    assert store.user_skills("user:__t_u").value == local.user_skills("user:__t_u").value == ["skill:__t_py"]
    assert store.skill_gap("user:__t_u", "job:__t_j").value == local.skill_gap("user:__t_u", "job:__t_j").value
    assert store.courses_teaching("skill:__t_py").value == local.courses_teaching("skill:__t_py").value
    assert store.prerequisite_chain("course:__t_c2").value == local.prerequisite_chain("course:__t_c2").value
    assert store.explain_path("user:__t_u", "job:__t_j").value == local.explain_path("user:__t_u", "job:__t_j").value
    assert store.explain_path("user:__t_u", "job:nope").value["reason"] == "unknown node"

    # idempotent: a second identical write changes no counts
    store.upsert_nodes(nodes, "live-t2")
    store.upsert_edges(edges, "live-t2")
    assert store.stats().value == stats

    # dangling edge is skipped, never creates a node
    res = store.upsert_edges([EdgeRecord("HAS_SKILL", "user:__t_ghost", "skill:__t_py", {})], "live-t2")
    assert res.value == {"edges": 0, "skipped": 1} and store.stats().value["total_nodes"] == 6

    cleared = store.clear_namespace(TEST_PROJECT)
    assert cleared.value["nodes_deleted"] == 6 and store.stats().value["total_nodes"] == 0
    main_after = Neo4jStore(Neo4jSettings.from_env(), project=PROJECT).stats().value
    assert main_after == main_before  # the real namespace was never touched


def test_wrong_password_falls_back_without_leaking_secrets():
    real = Neo4jSettings.from_env()
    bad = Neo4jSettings(uri=real.uri, username=real.username, password="definitely-wrong-password", database=real.database)
    store = get_graph_store(backend="auto", settings=bad, local=LocalGraphStore(project=TEST_PROJECT))
    t0 = time.perf_counter()
    res = store.user_skills("user:anyone")
    assert res.backend == "local" and res.fell_back and res.error
    assert_no_secret(res.error, real.password, real.uri, real.host, "definitely-wrong-password")
    assert_no_secret(repr(store), real.password, real.uri, real.host)
    assert time.perf_counter() - t0 < 30
    store.close()


def test_unreachable_host_falls_back_quickly():
    real = Neo4jSettings.from_env()
    dead = Neo4jSettings(uri="neo4j+s://unreachable.invalid", username=real.username, password=real.password,
                         database=real.database, connection_timeout=3.0, acquisition_timeout=3.0)
    store = get_graph_store(backend="mirror", settings=dead, local=LocalGraphStore(project=TEST_PROJECT))
    t0 = time.perf_counter()
    first = store.user_skills("user:anyone")
    second = store.user_skills("user:anyone")
    elapsed = time.perf_counter() - t0
    assert first.backend == "local" and first.fell_back and second.fell_back
    assert_no_secret(first.error, real.password, dead.host)
    assert store.probes == 1 and elapsed < 25  # the dead database was probed once, not per call
    store.close()
