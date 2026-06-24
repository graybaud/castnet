"""SPICE netlist generator — Pure domain logic.

Generates a SPICE netlist from a CastNet graph for circuit simulation.
No PyTorch dependency — works with plain Python dicts.
"""

import math
from typing import Optional


V_REF = 1.2
I_REF = 1e-6
V_THRESHOLD = 0.6

NEURON_SUBCKT = """
.subckt neuron in out vdd vss
B1 out vss V=2.5 * (V(in) > {VTH})
Rout out vss 10k
.ends neuron
"""


def weight_to_resistance(weight: float) -> float:
    """Convert a synaptic weight to a resistance value in ohms.

    R = V_REF / (|W| * I_REF), clamped to [1k, 100M].
    """
    if abs(weight) < 1e-8:
        return float("inf")
    r = V_REF / (abs(weight) * I_REF + 1e-12)
    return max(1e3, min(r, 100e6))


def generate_spice_netlist(
    sources: list[int],
    targets: list[int],
    weights: list[float],
    types: Optional[list[bool]] = None,
    total_nodes: Optional[int] = None,
    max_nodes: Optional[int] = None,
) -> str:
    """Generate a complete SPICE netlist from graph data.

    Args:
        sources: list of source node indices
        targets: list of target node indices
        weights: list of connection weights
        types: list of connection types (True=excitatory, False=inhibitory)
        total_nodes: total number of nodes in the graph
        max_nodes: limit the number of nodes to export

    Returns:
        SPICE netlist as a string.
    """
    if types is None:
        types = [True] * len(sources)

    if total_nodes is None:
        total_nodes = max(targets) + 1 if targets else 0

    if max_nodes is not None and max_nodes < total_nodes:
        filtered = [
            (s, t, w, ty)
            for s, t, w, ty in zip(sources, targets, weights, types)
            if s < max_nodes and t < max_nodes
        ]
        if filtered:
            sources, targets, weights, types = zip(*filtered)
            sources, targets, weights, types = (
                list(sources), list(targets), list(weights), list(types)
            )
        else:
            sources, targets, weights, types = [], [], [], []
        total_nodes = max_nodes

    lines = [
        "* CastNet SPICE - 250nm",
        "",
        NEURON_SUBCKT.replace("{VTH}", str(V_THRESHOLD)),
        "",
        "Vdd vdd 0 DC 2.5",
        "Vss vss 0 DC 0",
        "",
    ]

    # Instantiate neurons
    for n in range(total_nodes):
        lines.append(f"Xneuron_{n} node_{n} out_{n} vdd vss neuron")
    lines.append("")

    # Synaptic connections as resistors
    edge_count = 0
    for i in range(len(sources)):
        w = weights[i]
        if abs(w) < 1e-8:
            continue
        r = weight_to_resistance(w)
        if types[i]:
            lines.append(f"R_{edge_count:06d} in_{sources[i]} node_{targets[i]} {r:.1f}")
        else:
            lines.append(f"R_{edge_count:06d} node_{targets[i]} vss {r:.1f}")
        edge_count += 1
    lines.append("")

    # Input voltage sources for first 10 neurons
    for n in range(min(10, total_nodes)):
        lines.append(f"Vin_{n} in_{n} 0 DC 0 PULSE(0 2.5 10n 1n 1n 50n 100n)")
    lines.append("")

    # Simulation command
    lines.append(".tran 1n 500n")
    lines.append(".end")

    return "\n".join(lines)


def save_spice_netlist(
    netlist: str,
    output_path: str,
) -> tuple[str, int, int]:
    """Save a SPICE netlist to a file and return stats.

    Returns:
        (output_path, num_resistors, num_neurons)
    """
    with open(output_path, "w") as f:
        f.write(netlist)

    # Count components
    num_resistors = netlist.count("R_")
    num_neurons = netlist.count("Xneuron_")

    return output_path, num_resistors, num_neurons
