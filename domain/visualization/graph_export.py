"""Graph export utilities — Pure domain logic."""

import json
import random
from typing import Optional


def prepare_graph_for_export(
    sources: list[int],
    targets: list[int],
    weights: list[float],
    types: Optional[list[bool]] = None,
    total_nodes: Optional[int] = None,
    max_nodes: int = 5000,
    max_edges: int = 50000,
) -> dict:
    random.seed(42)
    if types is None:
        types = [True] * len(sources)
    if total_nodes is None:
        total_nodes = max(max(sources), max(targets)) + 1 if sources else 0

    nodes = [{"id": i, "degree": 0} for i in range(total_nodes)]
    edges = []
    for s, t, w, ty in zip(sources, targets, weights, types):
        if s < total_nodes and t < total_nodes:
            nodes[s]["degree"] += 1
            nodes[t]["degree"] += 1
            edges.append({
                "source": s, "target": t,
                "weight": round(abs(w), 6),
                "type": "excitatory" if ty else "inhibitory",
            })

    if len(nodes) > max_nodes:
        indices = set(random.sample(range(len(nodes)), max_nodes))
        nodes = [n for i, n in enumerate(nodes) if i in indices]
        old_to_new = {old: new for new, old in enumerate(sorted(indices))}
        for n in nodes:
            n["id"] = old_to_new[n["id"]]
        edges = [
            {**e, "source": old_to_new[e["source"]], "target": old_to_new[e["target"]]}
            for e in edges
            if e["source"] in old_to_new and e["target"] in old_to_new
        ]

    if len(edges) > max_edges:
        edges = random.sample(edges, max_edges)

    return {
        "nodes": nodes, "edges": edges,
        "metrics": {
            "total_nodes": total_nodes, "total_edges": len(edges),
            "displayed_nodes": len(nodes), "displayed_edges": len(edges),
        },
    }


def export_graph_json(graph_data: dict, output_path: str) -> str:
    with open(output_path, "w") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    return output_path
