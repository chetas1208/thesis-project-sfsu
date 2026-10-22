# Neo4j graph backend (with a local NetworkX fallback)

The SkillLLM_v2 knowledge graph (users, jobs, skills, courses) can live in a Neo4j instance (built and tested on a
Neo4j 5.27 Aura instance). The local NetworkX graph is always buildable offline and is the source of truth the Neo4j
backend is checked against and falls back to. Nothing changes unless you opt in: the pipeline default is
`graph.backend: local`.

## Architecture

```
                         GraphStore (graph/store.py: Protocol, StoreResult, records, size guard)
                                     |
        +----------------------------+-----------------------------+
        |                            |                             |
 LocalGraphStore              Neo4jStore                    FallbackStore (store_factory.py)
 (local_store.py)             (neo4j_store.py)              mode 'auto' | 'mirror'
 SkillGraphBuilder +          Cypher, MERGE, retry,         Neo4j preferred, local on any failure,
 SkillGraphRetriever +        timeouts, redaction           cached health probe (TTL)
 explain_skill_path

 neo4j_adapters.py: CourseGraphBuilder / SkillGraphBuilder graphs (read-only) -> NodeRecord / EdgeRecord -> GraphPlan
 store_settings.py: .env parser, env precedence, redaction, TLS-only settings
 scripts/build_neo4j_graph.py: --dry-run | --build | --verify | --clear --yes
```

Every operation returns a `StoreResult(value, backend, fell_back, error, elapsed_ms, ok)`. `backend` is `'neo4j'` or
`'local'` (who produced `value`), `fell_back` is True when the preferred backend failed and the other answered, and
`error` is a redacted string.

Operations: `upsert_nodes`, `upsert_edges`, `user_skills(user)`, `job_skills(job, strength)`, `skill_gap(user, job)`,
`courses_teaching(skill)`, `prerequisite_chain(course, max_depth)`, `explain_path(source, target, max_hops)`, `stats`,
`health`, `clear_namespace`, `close`. Ids are global and prefixed: `user:`, `job:`, `skill:`, `course:`.

```python
from skillllm_v2.graph.store_factory import get_graph_store
store = get_graph_store(backend="auto")           # 'auto' | 'local' | 'neo4j' | 'mirror'; or get_graph_store(cfg["graph"])
res = store.courses_teaching("skill:docker")
res.value, res.backend, res.fell_back, res.error
store.close()
```

## Schema

```
(:User {id, project, build_id, name, experience_level, source, synthetic})
(:Job  {id, project, build_id, name, title, source, synthetic})
(:Skill {id, project, build_id, name, taxonomy, aliases[], alias_count, source, synthetic})
(:Course {id, project, build_id, name, title, level, duration_hours, provider, rating, cost_usd, source, synthetic})

(:User)  -[:HAS_SKILL {confidence, source, last_seen?, mastery_level?, evidence?}]->  (:Skill)
(:Job)   -[:REQUIRES  {importance, requirement_strength, source}]->                   (:Skill)
(:Job)   -[:PREFERS   {importance, requirement_strength, source}]->                   (:Skill)
(:Course)-[:TEACHES   {depth, confidence, outcome_index, evidence, source}]->         (:Skill)
(:Course)-[:REQUIRES_PREREQUISITE {level, strictness, evidence, source}]->            (:Skill)
(:Skill) -[:RELATED_TO {relation_type, weight, source}]->                             (:Skill)
```

`PREFERS` is added to the five types of the project description because the graph builders distinguish required from
preferred skills. `MISSING_SKILL` is not stored: it is derived (`skill_gap`). Every node and relationship carries
`project` (namespace, default `skillllm_v2`) and `build_id` (a content hash of the plan: identical data gives an identical
id, so a rebuild changes nothing); `created_build_id` is set once on creation.

Constraints and indexes (all `IF NOT EXISTS`): `REQUIRE n.id IS UNIQUE` per label, an index on `project` per label and per
relationship type, and an index on `Skill.name`.

Cypher examples (parameters, not literals, in the real queries):

```cypher
// skills of a user and the gap to a job
MATCH (u:User {id: $user, project: $project})-[:HAS_SKILL]->(s:Skill) RETURN s.id
MATCH (j:Job {id: $job, project: $project})-[:REQUIRES]->(s:Skill)
WHERE NOT EXISTS { MATCH (:User {id: $user})-[:HAS_SKILL]->(s) } RETURN s.id AS missing

// courses that teach a skill
MATCH (c:Course)-[r:TEACHES]->(s:Skill {id: $skill}) RETURN c.id, r.depth, r.confidence ORDER BY c.id

// prerequisite chain: level 1 = skills the course requires, level k+1 = what the teaching courses require (cycle safe)
MATCH (c:Course {id: $course})-[:REQUIRES_PREREQUISITE]->(s:Skill)<-[:TEACHES]-(c2:Course) RETURN s.id, c2.id

// explanation path, depth by depth (1..max_hops), hop cap; reverse traversal only for edges that point into a skill
MATCH p = (a:User {id: $s})-[rs:HAS_SKILL|REQUIRES|PREFERS|TEACHES|REQUIRES_PREREQUISITE|RELATED_TO*3]-(b:Job {id: $t})
WHERE all(i IN range(0, size(rs)-1) WHERE startNode(rs[i]) = nodes(p)[i] OR type(rs[i]) IN $reversible)
RETURN [r IN rs | {type: type(r), src: startNode(r).id, dst: endNode(r).id}]
```

`explain_path` mirrors `explain_skill_path`: forward edges always; `HAS_SKILL`, `MISSING_SKILL`, `REQUIRES`, `PREFERS`,
`TEACHES`, `REQUIRES_PREREQUISITE`, `BELONGS_TO` also backwards; `RELATED_TO` forward only. All shortest paths of the first
depth that has any (max 2000) are fetched and the final path is picked by the same deterministic BFS the local store uses,
so ties resolve identically. Semantics of the hop cap in the store API: a path longer than `max_hops` is **not** returned;
`found=False, truncated=True` ("search stopped at the cap"). Local can prove "no path at all" (`truncated=False`); Neo4j
reports `truncated=True` for any miss within the cap (the only parity difference we observed).

## Configuration and credentials

Environment variables (never put them in a config file, test or doc):

| variable | meaning |
|---|---|
| `NEO4J_URI` | TLS only: `neo4j+s://<id>.databases.neo4j.io` (also `bolt+s://`); anything else is refused |
| `NEO4J_USERNAME`, `NEO4J_PASSWORD` | credentials |
| `NEO4J_DATABASE` | database name (Aura: the instance id); optional |
| `RUN_LIVE_NEO4J=1` | enables `tests/live` |

Resolution order for each key: real environment variables, then the **nearest `.env` found walking up** from the current
working directory and from the package location, up to the filesystem root (a nearer file wins per key; a key missing from
a nearer file can come from a farther one). The repository uses ONE global `<repo root>/.env` (gitignored, mode 600);
`.env.example` at the repo root has placeholders only. A `.env` next to a sub-project overrides the global one. Only the four
`NEO4J_*` keys are read from a file; the file is parsed by a small local parser (no `python-dotenv` needed). A missing file
is fine (the store then reports "not configured" and `auto`/`mirror` use the local graph).

The `neo4j` driver is optional: `pip install 'skillllm[graph]'` (`neo4j>=5.20`). The offline package works without it.

## Fallback semantics

| backend | behaviour |
|---|---|
| `local` (default) | NetworkX only |
| `neo4j` | Neo4j only; failures come back as `StoreResult(ok=False, error=...)`, no fallback. Missing credentials raise `SettingsError` (a configuration error, not an outage) |
| `auto` | try Neo4j; on any connection, auth, timeout or query failure answer from the local graph (`fell_back=True`, redacted `error`, one warning). Never raises because Neo4j is down or not configured. Writes go to Neo4j when reachable, else local |
| `mirror` | the local graph is always built and written; writes are also synced to Neo4j when it is reachable (`sync_to_neo4j` catches up after an outage); reads prefer Neo4j and fall back to local mid-flight if a query fails |

A health probe result is cached for `health_ttl_seconds` (default 30) in both directions: a dead database costs one probe
(bounded by `connection_timeout`, default 5 s) per window, not one per call. Measured: wrong password or unresolvable host
costs ~0.05-0.1 s once, then ~0.05 ms per call. Transient errors (`ServiceUnavailable`, `SessionExpired`, `TransientError`)
are retried with exponential backoff (0.5 s, 1 s; `max_retries=2`); auth errors and constraint violations never are. Every
transaction has a timeout (`query_timeout`, 30 s); the connection pool is closed by `close()`, the context manager, and a
`weakref.finalize` safety net.

### Pipeline (stage 9)

```yaml
inference:
  graph:
    max_hops: 4
    backend: local        # local | auto | neo4j | mirror
```

or `SkillLLMV2Pipeline({"inference": {"graph": {"backend": "auto"}}})`. With a non-local backend, stage 9 additionally reads
the courses teaching each missing skill from the catalogue graph (`GraphArtifact.catalog_courses`, only present then) and
records `provenance["graph_backend"] = {requested, backend, fell_back, error (redacted), queries}` in the stage record. The
user's own skills and the target job stay in the local in-memory graph: **nothing about a user or job from the pipeline is
ever written to Neo4j.** The stage never fails because of the store. `pipeline.close_graph_store()` closes the pool.

## Building, verifying, clearing, rebuilding

```bash
cd SkillLLM_v2_stage2
PYTHONPATH=src python scripts/build_neo4j_graph.py --dry-run          # planned counts per label/type + size guard; no connection
PYTHONPATH=src python scripts/build_neo4j_graph.py --build            # local graph + constraints/indexes + idempotent MERGE writes
PYTHONPATH=src python scripts/build_neo4j_graph.py --verify --json-out parity.json   # counts + >=50 parity queries + timings
PYTHONPATH=src python scripts/build_neo4j_graph.py --clear            # prints counts only
PYTHONPATH=src python scripts/build_neo4j_graph.py --clear --yes      # deletes ONLY project='skillllm_v2' (batched DETACH DELETE)
```

Rebuild = `--build` again (idempotent: counts unchanged, only `build_id` changes if the data changed) or `--clear --yes` then
`--build`. Heavy runs go through the server guardrails launcher: `guardrails run cpu -- /opt/miniconda/bin/python <abs script>`.

Live build (this repository's data): 42 User, 43 Job, 275 Skill, 173 Course nodes (533); 206 HAS_SKILL, 136 REQUIRES,
71 PREFERS, 309 TEACHES, 125 REQUIRES_PREREQUISITE, 10 RELATED_TO relationships (857). The 10 RELATED_TO edges are skill
pairs taught together by at least two catalogue courses (`relation_type='co_taught'`, weight = Jaccard of the course sets).
Parity on 100 sampled queries: 100/100 identical answers; latency (median / p95, ms): local 0.05-1.2 / 0.1-3.3, Neo4j
(Aura, from this server) 164-390 / 170-625, i.e. 100-1000x slower per call than in-process NetworkX, dominated by the network
round trip. Use Neo4j for shared/persistent graph queries and the local graph for hot paths.

## Safety

* **Credentials never leak.** `repr`/`str` of settings and stores show nothing; every driver exception is caught and re-raised
  or returned as a `GraphStoreError` subclass (`GraphAuthError`, `GraphConnectionError`, `GraphTimeoutError`,
  `GraphQueryError`) whose message has the password, username, URI, host, `user:pass@` userinfo and `password=...` pairs masked,
  raised `from None` so the original exception (and its message) is not chained into tracebacks. Logs carry only redacted text.
  The test suite asserts this for auth, unavailable, timeout, refused and generic errors, including `caplog`.
* **Parameterised Cypher only.** All values travel as parameters. Label and relationship-type identifiers come from fixed
  whitelists (`LABELS`, `REL_TYPES`, `REL_ENDPOINTS`) and the single path-length literal is range-checked (1..10).
* **TLS only.** The URI scheme must be `neo4j+s` or `bolt+s`.
* **Namespace-scoped writes.** `MERGE (n:L {id, project})` under a uniqueness constraint on `id`: a node of another namespace with
  the same id is never modified (the write fails instead). Edges `MATCH` both endpoints in the namespace and never create nodes
  (dangling edges are skipped and counted). `clear_namespace(project)` only accepts `skillllm_v2` and `skillllm_v2_test[_x]`
  and deletes `MATCH (n:L {project: $project}) ... LIMIT $batch DETACH DELETE n` in batches. No `DROP`/`CREATE DATABASE`, no
  unscoped deletes.
* **Size guard.** A plan above 50,000 nodes (or a database that would exceed it counting existing nodes) is refused before any
  write (`SizeGuardError`). Aura free tier is about 200k nodes / 400k relationships; the current graph uses 533 / 857.

## What was uploaded, and what deliberately was not

Uploaded (`neo4j_adapters.build_curated_plan`, checked by `assert_safe_plan`: allowed sources only, no free-text keys, no string
over 300 chars, no corpus path markers):

* skill names, taxonomy and aliases (max 25 per skill) from the curated `resources/offline_aliases.jsonl` (162 skills, 286 alias
  rows; source `curated_aliases`);
* the synthetic course catalogue `inputs/courses/course_catalog_large.jsonl` (171 courses, all `source: synthetic`): ids, title,
  level, duration, provider, rating, cost, TEACHES / REQUIRES_PREREQUISITE edges with `depth`, `level`, `strictness` and a
  <=200-char `evidence` string (a templated learning-outcome sentence of the fictional course);
* the 41 synthetic benchmark scenarios of `inputs/courses/benchmark_scenarios.jsonl`: user ids, `experience_level`, skill
  names, job ids/titles, required/preferred skill names;
* `inputs/demo/*`: 2 demo courses, 2 demo jobs, one demo user (only the curated skills the profile text mentions).

Deliberately NOT uploaded: the 7 benchmark scenarios whose `source` is `real_posting_excerpt` (sc-42..sc-48) and, for every
scenario, its job text and gold labels; any real job posting (`SkillLLM_v2/Extract_jobDesc`), resume (`Resume+Job Data`),
SkillSpan sentence or `data/source` content; course descriptions and learning-outcome lists as node properties; the demo
profile text; per-user data from the pipeline (see above); derived per-corpus document frequencies
(`metadata.real_posting_df`).

## Limits and caveats

* Skill ids are the curated canonical id when the name is a known alias, else a `graph_key` of the name; 113 catalogue/scenario
  skills are not in the curated alias set and become `source: derived_skill` nodes.
* The local store keys edges by type (one edge per source, target, type), as `SkillGraphBuilder` does; Neo4j `MERGE` matches.
  `CourseGraphBuilder` ids use `normalize_text`, which collides `C`, `C++`, `C#`; the adapter canonicalises through the curated
  alias table instead, but a name outside it still goes through `graph_key`.
* `explain_path` in Neo4j enumerates paths per depth (at most 2000 per depth): fine for this graph, not for millions of edges.
* No Aura free-tier idle pause handling beyond the generic retry: a paused instance looks like an unreachable host and falls back.

## Tests

* `tests/skillllm_v2/test_graph_store.py`: fake driver (no network); real `neo4j.exceptions` classes. Covers MERGE shape and
  parameterisation, batching, schema DDL, namespace-only clear, foreign-namespace refusal, size guard, redaction (errors, logs,
  repr), retry/backoff, pool close, `.env` parsing (parent dir found, nearer file wins, env overrides file, missing file ok),
  fallback on auth/timeout/unavailable, health-cache TTL, mirror semantics, curated-plan data safety, stage-9 wiring.
* `tests/live/test_neo4j_live.py`: skipped unless credentials are available and `RUN_LIVE_NEO4J=1`; connectivity, a write/read/clear
  round trip in the throwaway namespace `skillllm_v2_test` (main namespace only counted), wrong-password and unreachable-host
  fallback. Run: `RUN_LIVE_NEO4J=1 PYTHONPATH=src python -m pytest tests/live -q -p no:cacheprovider`.
