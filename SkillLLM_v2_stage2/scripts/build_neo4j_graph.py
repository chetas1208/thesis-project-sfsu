#!/usr/bin/env python3
"""Build / verify / clear the SkillLLM_v2 knowledge graph in Neo4j (synthetic + curated data only).

    PYTHONPATH=src python scripts/build_neo4j_graph.py --dry-run     # counts + size guard, no connection
    PYTHONPATH=src python scripts/build_neo4j_graph.py --build       # constraints, indexes, idempotent MERGE writes
    PYTHONPATH=src python scripts/build_neo4j_graph.py --verify      # local-vs-Neo4j parity + timing
    PYTHONPATH=src python scripts/build_neo4j_graph.py --clear --yes # delete ONLY project='skillllm_v2'

Credentials come from NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD (+ NEO4J_DATABASE): real environment variables, else the
nearest ``.env`` found walking up from the cwd / package. Nothing secret is ever printed. See docs/NEO4J_GRAPH.md.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from skillllm_v2.graph.local_store import LocalGraphStore  # noqa: E402
from skillllm_v2.graph.neo4j_adapters import (  # noqa: E402
    assert_safe_plan,
    build_curated_plan,
    load_plan_into,
    plan_content_hash,
)
from skillllm_v2.graph.neo4j_store import Neo4jStore  # noqa: E402
from skillllm_v2.graph.store import MAX_NODES, PROJECT, GraphStoreError, SizeGuardError  # noqa: E402
from skillllm_v2.graph.store_settings import Neo4jSettings, SettingsError, redact  # noqa: E402


def out(msg: str = "") -> None:
    print(redact(msg))


def print_counts(title: str, counts: dict[str, Any]) -> None:
    out(f"{title}")
    for label, n in counts["nodes"].items():
        out(f"  node  {label:<22}{n:>7}")
    for rel, n in counts["edges"].items():
        out(f"  edge  {rel:<22}{n:>7}")
    out(f"  total nodes={counts['total_nodes']} edges={counts['total_edges']}")


def make_plan(args: argparse.Namespace):
    plan, prov = build_curated_plan(args.root)
    assert_safe_plan(plan)
    build_id = args.build_id or plan_content_hash(plan)
    return plan, prov, build_id


def cmd_dry_run(args: argparse.Namespace) -> int:
    plan, prov, build_id = make_plan(args)
    counts = plan.counts()
    out(f"DRY RUN (no connection). project={args.project} build_id={build_id}")
    print_counts("planned graph:", counts)
    try:
        plan.check_size(args.max_nodes)
        out(f"size guard: OK ({counts['total_nodes']} nodes <= {args.max_nodes})")
    except SizeGuardError as exc:
        out(f"size guard: REFUSED - {exc}")
        return 2
    out("included: " + json.dumps(prov["included"], sort_keys=True))
    out("skipped: " + json.dumps(prov["skipped"], sort_keys=True))
    out("not uploaded: " + "; ".join(prov["not_uploaded"]))
    return 0


def connect(args: argparse.Namespace) -> Neo4jStore:
    try:
        settings = Neo4jSettings.from_env()
    except SettingsError as exc:
        out(f"cannot connect: {exc}")
        raise SystemExit(3)
    return Neo4jStore(settings, args.project, max_nodes=args.max_nodes)


def cmd_build(args: argparse.Namespace) -> int:
    plan, prov, build_id = make_plan(args)
    counts = plan.counts()
    try:
        plan.check_size(args.max_nodes)
    except SizeGuardError as exc:
        out(f"size guard: REFUSED - {exc}")
        return 2
    local = LocalGraphStore(args.project)
    load_plan_into(local, plan, build_id)  # the local graph is always built (fallback source of truth)
    out(f"local graph built: {local.stats().value['total_nodes']} nodes, {local.stats().value['total_edges']} edges")
    with connect(args) as store:
        health = store.health()
        if not health.ok:
            out(f"neo4j unreachable, nothing written: {health.error}")
            return 4
        schema = store.ensure_schema()
        if not schema.ok:
            out(f"schema setup failed: {schema.error}")
            return 4
        before = store.stats().value
        out(f"neo4j before: nodes={before['total_nodes']} edges={before['total_edges']}")
        r1 = store.upsert_nodes(plan.nodes.values(), build_id)
        if not r1.ok:
            out(f"node write failed: {r1.error}")
            return 4
        r2 = store.upsert_edges(plan.edges.values(), build_id)
        if not r2.ok:
            out(f"edge write failed: {r2.error}")
            return 4
        after = store.stats().value
        out(f"wrote nodes={r1.value['nodes']} edges={r2.value['edges']} skipped_edges={r2.value['skipped']} "
            f"in {r1.elapsed_ms + r2.elapsed_ms:.0f} ms (build_id={build_id})")
        print_counts("neo4j after:", after)
        ok = (after["nodes"] == counts["nodes"] and after["edges"] == counts["edges"])
        out(f"counts match plan: {ok}")
        if before["total_nodes"]:
            out(f"idempotent re-run (counts unchanged vs before): {before['nodes'] == after['nodes'] and before['edges'] == after['edges']}")
        info = store.schema_info()
        if info.ok:
            out("constraints:")
            for c in info.value["constraints"]:
                out(f"  {c['name']:<20} {c['type']:<22} {c['labelsOrTypes']} {c['properties']}")
            out("indexes:")
            for i in info.value["indexes"]:
                out(f"  {i['name']:<28} {i['type']:<8} {i['entityType']:<13} {i['labelsOrTypes']} {i['properties']} {i['state']}")
        return 0 if ok else 5


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def _comparable(op: str, value: Any) -> Any:
    """explain_path with no path: only 'unknown node' vs 'no path' matters (local can prove unreachable, Neo4j reports
    'truncated' for any miss within the hop cap)."""
    if op == "explain_path" and isinstance(value, dict) and not value.get("found"):
        return {"found": False, "unknown_node": value.get("reason") == "unknown node"}
    return value


def cmd_verify(args: argparse.Namespace) -> int:
    plan, prov, build_id = make_plan(args)
    local = LocalGraphStore(args.project)
    load_plan_into(local, plan, build_id)
    rng = random.Random(args.seed)
    users = sorted(n for n in plan.nodes if n.startswith("user:"))
    jobs = sorted(n for n in plan.nodes if n.startswith("job:"))
    skills = sorted(n for n in plan.nodes if n.startswith("skill:"))
    courses = sorted(n for n in plan.nodes if n.startswith("course:"))
    k = args.samples
    queries: list[tuple[str, tuple]] = []
    for u in rng.sample(users, min(k, len(users))):
        queries.append(("user_skills", (u,)))
    for _ in range(k):
        queries.append(("skill_gap", (rng.choice(users), rng.choice(jobs))))
    for s in rng.sample(skills, min(k, len(skills))):
        queries.append(("courses_teaching", (s,)))
    prereq_courses = sorted({e.src for e in plan.edges.values() if e.type == "REQUIRES_PREREQUISITE"})
    for c in rng.sample(prereq_courses, min(k - 2, len(prereq_courses))) + ["course:aws-101", "course:prog-001"]:
        queries.append(("prerequisite_chain", (c, 4)))
    for _ in range(k - 6):
        queries.append(("explain_path", (rng.choice(users), rng.choice(jobs), 4)))
    for _ in range(3):
        queries.append(("explain_path", (rng.choice(users), rng.choice(courses), 4)))
    for _ in range(2):
        queries.append(("explain_path", (rng.choice(skills), rng.choice(skills), 4)))
    queries.append(("explain_path", (users[0], "job:does-not-exist", 4)))
    out(f"verify: {len(queries)} queries (seed={args.seed})")

    with connect(args) as store:
        health = store.health()
        if not health.ok:
            out(f"neo4j unreachable: {health.error}")
            return 4
        lstats, nstats = local.stats().value, store.stats().value
        out(f"count parity: nodes {'MATCH' if lstats['nodes'] == nstats['nodes'] else 'DIFF'}, "
            f"edges {'MATCH' if lstats['edges'] == nstats['edges'] else 'DIFF'} "
            f"(local {lstats['total_nodes']}/{lstats['total_edges']}, neo4j {nstats['total_nodes']}/{nstats['total_edges']})")
        report: dict[str, Any] = {"count_parity": {"local": lstats, "neo4j": nstats}, "ops": {}, "differences": []}
        per_op: dict[str, dict[str, list[float]]] = {}
        diffs = 0
        trunc_diffs = 0
        for op, a in queries:
            lres = getattr(local, op)(*a)
            nres = getattr(store, op)(*a)
            if not nres.ok:
                out(f"neo4j error on {op}: {nres.error}")
                return 4
            lv, nv = lres.value, nres.value
            same = _comparable(op, json.loads(json.dumps(lv))) == _comparable(op, json.loads(json.dumps(nv)))
            if op == "explain_path" and not lv.get("found") and not nv.get("found") and lv.get("truncated") != nv.get("truncated"):
                trunc_diffs += 1  # both: no path within the cap; local can prove "unreachable", Neo4j says "truncated"
            bucket = per_op.setdefault(op, {"local": [], "neo4j": [], "match": []})
            bucket["local"].append(lres.elapsed_ms)
            bucket["neo4j"].append(nres.elapsed_ms)
            bucket["match"].append(1.0 if same else 0.0)
            if not same:
                diffs += 1
                report["differences"].append({"op": op, "args": list(a), "local": lv, "neo4j": nv})
        out(f"{'op':<20}{'n':>4}{'match':>7}  {'local med/p95 ms':>20}  {'neo4j med/p95 ms':>20}")
        for op, b in per_op.items():
            n = len(b["match"])
            out(f"{op:<20}{n:>4}{sum(b['match']):>7.0f}  "
                f"{statistics.median(b['local']):>9.3f}/{_pct(b['local'], .95):<9.3f}  "
                f"{statistics.median(b['neo4j']):>9.1f}/{_pct(b['neo4j'], .95):<9.1f}")
            report["ops"][op] = {"n": n, "matches": int(sum(b["match"])),
                                 "local_ms": {"median": statistics.median(b["local"]), "p95": _pct(b["local"], .95)},
                                 "neo4j_ms": {"median": statistics.median(b["neo4j"]), "p95": _pct(b["neo4j"], .95)}}
        out(f"queries={len(queries)} differences={diffs} (explain_path no-path cases with differing 'truncated' wording: {trunc_diffs})")
        for d in report["differences"][:5]:
            out("DIFF " + json.dumps(d, default=str)[:600])
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report, indent=2, default=str))
            out(f"wrote {args.json_out}")
        return 0 if diffs == 0 and lstats["nodes"] == nstats["nodes"] and lstats["edges"] == nstats["edges"] else 5


def cmd_clear(args: argparse.Namespace) -> int:
    with connect(args) as store:
        stats = store.stats()
        if not stats.ok:
            out(f"neo4j unreachable: {stats.error}")
            return 4
        print_counts(f"namespace project='{args.project}' currently holds:", stats.value)
        if not args.yes:
            out("not clearing: pass --yes to delete exactly these (only project-tagged data is touched)")
            return 1
        res = store.clear_namespace(args.project)
        if not res.ok:
            out(f"clear failed: {res.error}")
            return 4
        out(f"deleted {res.value['nodes_deleted']} nodes (and their relationships) in {res.elapsed_ms:.0f} ms")
        after = store.stats().value
        out(f"remaining in namespace: nodes={after['total_nodes']} edges={after['total_edges']}")
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--clear", action="store_true")
    p.add_argument("--yes", action="store_true", help="confirm --clear")
    p.add_argument("--root", default=str(REPO))
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--build-id", default=None)
    p.add_argument("--max-nodes", type=int, default=MAX_NODES)
    p.add_argument("--samples", type=int, default=20, help="queries per kind for --verify (>= 10)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)
    args.samples = max(10, args.samples)
    try:
        if args.dry_run:
            return cmd_dry_run(args)
        if args.build:
            return cmd_build(args)
        if args.verify:
            return cmd_verify(args)
        return cmd_clear(args)
    except GraphStoreError as exc:
        out(f"error: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
