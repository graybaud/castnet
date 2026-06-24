"""Routing congestion model (Rent's rule) — Pure domain logic."""

import math
from domain.hardware.constants import PROCESS_NODES


def rents_rule(n_nodes: int, K: float = 2.5, p: float = 0.6) -> tuple[float, float]:
    """Rent's rule: estimates number of terminals from number of nodes.

    Returns (num_terminals, K).
    """
    return K * (n_nodes ** p), K


def wire_length_distribution(
    n_nodes: int,
    L_avg_um: float,
    p: float = 0.6,
    num_bins: int = 10,
) -> dict[float, float]:
    """Wire length probability distribution based on Rent's exponent."""
    if n_nodes <= 1:
        return {L_avg_um: 1.0}

    alpha = 4.0 - 2.0 * p
    L_min = L_avg_um * 0.3
    L_max = L_avg_um * 10.0

    bins = {}
    for i in range(num_bins):
        l_start = L_min * (L_max / L_min) ** (i / num_bins)
        l_end = L_min * (L_max / L_min) ** ((i + 1) / num_bins)
        l_mid = (l_start + l_end) / 2.0
        prob = l_mid ** (-alpha)
        bins[round(l_mid, 2)] = prob

    total = sum(bins.values())
    return {l: round(p / total, 4) for l, p in bins.items()}


def estimate_congestion_overhead(
    edges: int,
    n_nodes: int,
    metal_layers: int = 8,
    utilization_target: float = 0.50,
) -> tuple[float, float]:
    """Estimate routing congestion overhead.

    Returns (overhead_factor, congestion_pct).
    """
    tracks = math.sqrt(n_nodes) * utilization_target * metal_layers
    if tracks >= edges:
        return 1.0, 0.0
    overhead = math.sqrt(edges / tracks)
    congestion_pct = max(0.0, (edges - tracks) / edges * 100.0)
    return round(overhead, 3), round(congestion_pct, 1)


def estimate_with_congestion(
    nodes: int,
    edges: int,
    process: str = "130nm",
    transistors_per_node: int = 5,
    congestion_margin: float = 0.25,
) -> dict:
    """Full area estimation including routing congestion."""
    if process not in PROCESS_NODES:
        process = "130nm"

    p = PROCESS_NODES[process]
    t_area = nodes * transistors_per_node * p["transistor_area"] / 1e6
    ideal_edge = edges * p["edge_pitch"] * p["edge_length_avg"] / 1e6
    overhead, cong_pct = estimate_congestion_overhead(edges, nodes, p["metal_layers"])
    realistic_edge = ideal_edge * overhead * (1.0 + congestion_margin)
    active = t_area + realistic_edge
    total = active * p["routing_factor"]

    return {
        "process": process,
        "nodes": nodes,
        "edges": edges,
        "transistors_per_node": transistors_per_node,
        "transistor_area_mm2": round(t_area, 4),
        "ideal_routing_area_mm2": round(ideal_edge, 4),
        "realistic_routing_area_mm2": round(realistic_edge, 4),
        "active_area_mm2": round(active, 4),
        "total_area_mm2": round(total, 2),
        "chip_side_mm": round(math.sqrt(total), 2),
        "congestion_overhead_factor": overhead,
        "congestion_pct": cong_pct,
        "congestion_margin": congestion_margin,
        "routing_dominance_pct": round(realistic_edge / active * 100, 1) if active > 0 else 0.0,
        "metal_layers": p["metal_layers"],
        "edge_pitch_um": p["edge_pitch"],
        "edge_length_avg_um": p["edge_length_avg"],
    }
