"""Chip area estimation — Pure domain."""

import math
from domain.hardware.constants import PROCESS_NODES, TRANSISTORS_PER_NODE


def estimate_chip_area(nodes: int, edges: int, process: str = "130nm") -> dict:
    """Estimate chip area for a given graph and process node.
    
    Args:
        nodes: total number of neurons
        edges: total number of active connections
        process: process node name (e.g. "130nm")
    
    Returns:
        dict with area estimates in mm2.
    """
    if process not in PROCESS_NODES:
        process = "130nm"
    
    p = PROCESS_NODES[process]
    total_transistors = nodes * TRANSISTORS_PER_NODE
    
    active_area = total_transistors / p["density"]
    total_area = active_area * p["routing_factor"]
    chip_side = math.sqrt(total_area)
    
    return {
        "process": process,
        "total_nodes": nodes,
        "total_edges": edges,
        "total_transistors": int(total_transistors),
        "active_area_mm2": round(active_area, 2),
        "total_area_mm2": round(total_area, 2),
        "chip_side_mm": round(chip_side, 2),
        "mask_cost_range": f"${p['mask_cost_min']//1000}k-${p['mask_cost_max']//1000}k",
        "availability": p["availability"],
    }


def estimate_all_processes(nodes: int, edges: int) -> list[dict]:
    """Estimate chip area for all available process nodes."""
    results = []
    for name in PROCESS_NODES:
        results.append(estimate_chip_area(nodes, edges, name))
    results.sort(key=lambda x: x["chip_side_mm"])
    return results
