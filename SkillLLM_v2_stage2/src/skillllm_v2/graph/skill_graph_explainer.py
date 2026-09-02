from __future__ import annotations

from typing import Any


def explain_skill_path(graph: Any, source: str, target: str, max_hops: int = 4) -> dict[str, Any]:
    try:
        path = graph.shortest_path(source, target) if hasattr(graph, "shortest_path") else None
    except Exception:
        path = None
    if path is None:
        try:
            import networkx as nx

            path = nx.shortest_path(graph, source, target)
        except Exception:
            return {
                "source": source,
                "target": target,
                "path": [],
                "explanation": "No graph path found.",
                "confidence": 0.0,
            }
    trimmed = path[: max_hops + 1]
    return {
        "source": source,
        "target": target,
        "path": trimmed,
        "explanation": " -> ".join(trimmed),
        "confidence": 1.0 if len(path) <= max_hops + 1 else 0.7,
    }
