from __future__ import annotations

from collections import deque
from typing import Any

# Edge types that point INTO a skill-like node (User->Skill, Job->Skill, Course->Skill ...). Read backwards they are
# still true statements ("skill s is REQUIRED BY job j"), so an explanatory path may cross them against their
# direction: User -HAS_SKILL-> s <-REQUIRES- Job. Every other edge type (RELATED_TO, SIMILAR_TO, unknown types) is only
# followed forward, exactly as stored.
REVERSE_TRAVERSABLE = frozenset(
    {"HAS_SKILL", "MISSING_SKILL", "REQUIRES", "PREFERS", "TEACHES", "REQUIRES_PREREQUISITE", "BELONGS_TO"}
)


def _edge_types(graph: Any, u: Any, v: Any) -> list[str]:
    """Distinct edge-type labels of the edges u -> v (u-v for an undirected graph), sorted."""
    data = graph.get_edge_data(u, v)
    if data is None:
        return []
    if graph.is_multigraph():
        return sorted({str(d.get("type", "")) for d in data.values()})
    return [str(data.get("type", ""))]


def _steps(graph: Any, node: Any) -> list[tuple[Any, str, str]]:
    """``(neighbour, edge type, direction)`` moves from ``node``, deterministic order.

    ``direction`` is ``"forward"`` (edge stored node -> neighbour), ``"reverse"`` (stored neighbour -> node, only for
    REVERSE_TRAVERSABLE types) or ``"undirected"`` (the graph has no direction information).
    """
    out: list[tuple[Any, str, str]] = []
    if graph.is_directed():
        for nbr in graph.successors(node):
            for etype in _edge_types(graph, node, nbr):
                out.append((nbr, etype, "forward"))
        for nbr in graph.predecessors(node):
            for etype in _edge_types(graph, nbr, node):
                if etype in REVERSE_TRAVERSABLE:
                    out.append((nbr, etype, "reverse"))
    else:
        for nbr in graph.neighbors(node):
            for etype in _edge_types(graph, node, nbr):
                out.append((nbr, etype, "undirected"))
    return sorted(out, key=lambda m: (str(m[0]), m[1], m[2]))


def _bfs(graph: Any, source: Any, target: Any) -> list[tuple[Any, str, str]] | None:
    """Shortest explanatory path as a list of hops ``(node, edge type, direction)`` after ``source``; None if none."""
    if source == target:
        return []
    parent: dict[Any, tuple[Any, str, str]] = {}
    seen = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nbr, etype, direction in _steps(graph, node):
            if nbr in seen:
                continue
            seen.add(nbr)
            parent[nbr] = (node, etype, direction)
            if nbr == target:
                hops: list[tuple[Any, str, str]] = []
                cur = target
                while cur != source:
                    prev, et, di = parent[cur]
                    hops.append((cur, et, di))
                    cur = prev
                return list(reversed(hops))
            queue.append(nbr)
    return None


def _render(source: Any, hops: list[tuple[Any, str, str]]) -> str:
    parts = [str(source)]
    for node, etype, direction in hops:
        if direction == "reverse":
            parts.append(f"<-[{etype}]- {node}")
        elif direction == "forward":
            parts.append(f"-[{etype}]-> {node}")
        else:
            parts.append(f"-[{etype}]- {node}")
    return " ".join(parts)


def _no_path(source: str, target: str, reason: str) -> dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "path": [],
        "edges": [],
        "hops": None,
        "found": False,
        "exceeds_max_hops": False,
        "truncated": False,
        "endpoint": None,
        "reason": reason,
        "explanation": "No graph path found.",
        "confidence": 0.0,
    }


def explain_skill_path(graph: Any, source: str, target: str, max_hops: int = 4, strict_max_hops: bool = False) -> dict[str, Any]:
    """Explain how ``source`` connects to ``target`` in a skill graph.

    The search is a breadth-first search over an *explanatory* view of the graph: stored edges can be followed
    forward, and edges that point into a skill (HAS_SKILL, REQUIRES, TEACHES ...) can also be followed backwards, so
    ``user -> skill <- job`` connects a user to a job even though no directed path exists. Every returned hop reports
    its real stored direction (``edges[i]["direction"]``: ``forward`` / ``reverse`` / ``undirected``) and relation, and
    ``explanation`` renders it, e.g. ``u1 -[HAS_SKILL]-> python <-[REQUIRES]- j1``. For graphs without direction
    (``nx.Graph``) every hop is ``undirected``. Other edge types stay directed, so on a purely directed graph a path
    against the arrows is still "no path".

    ``max_hops`` is a confidence knob, not a truncation: a path is never cut. If the shortest path is longer than
    ``max_hops`` it is returned whole with ``exceeds_max_hops=True`` and a lower ``confidence`` (0.7). With
    ``strict_max_hops=True`` such a path is not returned at all (``path == []``, ``reason`` says why). ``target`` is
    always the requested target; ``endpoint`` is the real last node of ``path``. Unknown nodes give no path.

    Returned keys (superset of the legacy ``source/target/path/explanation/confidence``): ``edges``, ``hops``,
    ``found``, ``exceeds_max_hops``, ``truncated`` (always False), ``endpoint``, ``reason``.
    """
    if not hasattr(graph, "is_directed"):
        legacy = getattr(graph, "shortest_path", None)  # a non-networkx graph object exposing shortest_path()
        try:
            nodes = list(legacy(source, target)) if legacy else []
        except Exception:
            nodes = []
        if not nodes:
            return _no_path(source, target, "no path")
        hops = [(n, "", "unknown") for n in nodes[1:]]
    else:
        if not graph.has_node(source) or not graph.has_node(target):
            return _no_path(source, target, "unknown node")
        found = _bfs(graph, source, target)
        if found is None:
            return _no_path(source, target, "no path")
        hops = found
        nodes = [source] + [h[0] for h in hops]
    n_hops = len(hops)
    exceeds = n_hops > max_hops
    if exceeds and strict_max_hops:
        result = _no_path(source, target, f"shortest path has {n_hops} hops, more than max_hops={max_hops}")
        result["hops"] = n_hops
        return result
    previous = source
    edges = []
    for node, etype, direction in hops:
        edges.append({"from": previous, "to": node, "relation": etype, "direction": direction})
        previous = node
    explanation = _render(source, hops) if hasattr(graph, "is_directed") else " -> ".join(nodes)
    if exceeds:
        explanation += f"  [{n_hops} hops, longer than max_hops={max_hops}]"
    return {
        "source": source,
        "target": target,
        "path": nodes,
        "edges": edges,
        "hops": n_hops,
        "found": True,
        "exceeds_max_hops": exceeds,
        "truncated": False,
        "endpoint": nodes[-1],
        "reason": "",
        "explanation": explanation,
        "confidence": 0.7 if exceeds else 1.0,
    }


def _first_edge(graph: Any, source: str, target: str, types: tuple[str, ...]) -> dict[str, Any] | None:
    """Data of the first edge source -> target whose type is in ``types`` (deterministic: sorted by type), else None."""
    if not (graph.has_node(source) and graph.has_node(target)):
        return None
    data = graph.get_edge_data(source, target)
    if not data:
        return None
    candidates = list(data.values()) if graph.is_multigraph() else [data]
    for etype in types:
        for item in candidates:
            if item.get("type") == etype:
                return dict(item)
    return None


def narrate_skill_trace(graph: Any, user_id: str, job_id: str, course_id: str, skill_id: str) -> dict[str, Any]:
    """Faithful Job-skill / User-skill / Course-skill narrative: only edges that exist are asserted.

    Drop-in for the templated ``CourseGraphBuilder.trace_explanation_path`` (which asserts "required" and "teaches"
    even when the edge is missing): same keys, plus ``relations_present``. Each of the three clauses states the real
    relation found in the graph, or says explicitly that the graph holds no such relation.
    """
    skill_name = graph.nodes[skill_id].get("canonical_skill_name", skill_id) if graph.has_node(skill_id) else skill_id
    job_edge = _first_edge(graph, job_id, skill_id, ("REQUIRES", "PREFERS"))
    has_edge = _first_edge(graph, user_id, skill_id, ("HAS_SKILL",))
    missing_edge = _first_edge(graph, user_id, skill_id, ("MISSING_SKILL",))
    teaches_edge = _first_edge(graph, course_id, skill_id, ("TEACHES",))
    job_title = graph.nodes[job_id].get("title", job_id) if graph.has_node(job_id) else job_id
    course_title = graph.nodes[course_id].get("title", course_id) if graph.has_node(course_id) else course_id

    if job_edge is not None:
        verb = "requires" if job_edge.get("type") == "REQUIRES" else "prefers"
        job_clause = f"Target role '{job_title}' {verb} skill '{skill_name}'"
    else:
        job_clause = f"The graph has no REQUIRES or PREFERS edge from role '{job_title}' to skill '{skill_name}'"
    if has_edge is not None:
        user_clause = "user has documented proficiency (HAS_SKILL)"
    elif missing_edge is not None:
        user_clause = "user is recorded as missing this skill (MISSING_SKILL)"
    else:
        user_clause = "the graph records no HAS_SKILL or MISSING_SKILL edge between the user and this skill"
    if teaches_edge is not None:
        course_clause = f"Course '{course_title}' teaches '{skill_name}'"
        evidence = teaches_edge.get("learning_outcome_evidence", "")
        if evidence:
            course_clause += f" ('{evidence}')"
    else:
        course_clause = f"The graph has no TEACHES edge from course '{course_title}' to skill '{skill_name}'"
    return {
        "target_skill_id": skill_id,
        "skill_name": skill_name,
        "job_requirement": job_edge,
        "user_possession": has_edge or missing_edge,
        "course_evidence": teaches_edge,
        "relations_present": {"job_requires_or_prefers": job_edge is not None, "user_has_skill": has_edge is not None,
                              "user_missing_skill": missing_edge is not None, "course_teaches": teaches_edge is not None},
        "narrative_path": f"{job_clause} -> {user_clause} -> {course_clause}",
    }
