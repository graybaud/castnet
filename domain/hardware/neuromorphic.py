"""Neuromorphic hardware comparison — Pure domain logic."""

from domain.hardware.routing import estimate_with_congestion


EXISTING_CHIPS = {
    "Loihi 2 (Intel, 2021)": {
        "process": "Intel 4 (~7nm)",
        "die_area_mm2": 31,
        "neurons": 1_000_000,
        "synapses": 120_000_000,
        "synapses_per_neuron": 120,
        "power_w": 1.0,
        "energy_per_syn_op_pj": 0.5,
        "neuron_density_per_mm2": 32_258,
        "synapse_density_per_mm2": 3_870_968,
        "on_chip_learning": True,
        "programmable": True,
        "architecture": "Manycore mesh",
        "target": "Research",
    },
    "TrueNorth (IBM, 2014)": {
        "process": "28nm",
        "die_area_mm2": 430,
        "neurons": 1_000_000,
        "synapses": 256_000_000,
        "synapses_per_neuron": 256,
        "power_w": 0.07,
        "energy_per_syn_op_pj": 26,
        "neuron_density_per_mm2": 2_325,
        "synapse_density_per_mm2": 595_348,
        "on_chip_learning": False,
        "programmable": False,
        "architecture": "4,096 cores",
        "target": "DNN inference",
    },
    "SpiNNaker 2 (2022)": {
        "process": "22nm FDX",
        "die_area_mm2": 100,
        "neurons": 10_000,
        "synapses": 10_000_000,
        "synapses_per_neuron": 1_000,
        "power_w": 1.0,
        "energy_per_syn_op_pj": 10,
        "neuron_density_per_mm2": 100,
        "synapse_density_per_mm2": 100_000,
        "on_chip_learning": True,
        "programmable": True,
        "architecture": "ARM cores",
        "target": "Neuroscience",
    },
}


def compare_castnet(nodes: int, edges: int, process: str = "130nm") -> dict:
    """Compare CastNet chip estimates against existing neuromorphic hardware.
    
    Args:
        nodes: total number of neurons
        edges: total number of active connections
        process: process node name (e.g. "130nm")
    
    Returns:
        dict with castnet entry and comparison table.
    """
    r = estimate_with_congestion(nodes, edges, process)
    nd = nodes / r["total_area_mm2"] if r["total_area_mm2"] > 0 else 0
    sd = edges / r["total_area_mm2"] if r["total_area_mm2"] > 0 else 0
    area = r["total_area_mm2"]

    castnet_entry = {
        "process": process,
        "die_area_mm2": round(area, 1),
        "neurons": nodes,
        "synapses": edges,
        "synapses_per_neuron": round(edges / nodes, 1) if nodes > 0 else 0,
        "power_w": "See thermal",
        "energy_per_syn_op_pj": "N/A",
        "neuron_density_per_mm2": round(nd, 0),
        "synapse_density_per_mm2": round(sd, 0),
        "on_chip_learning": False,
        "programmable": False,
        "architecture": "Sparse FFN graph",
        "target": "LLM inference",
    }

    comparison = []
    for name, chip in EXISTING_CHIPS.items():
        vs_castnet = (
            castnet_entry["synapse_density_per_mm2"] / chip["synapse_density_per_mm2"]
            if chip["synapse_density_per_mm2"] > 0
            else 0
        )
        comparison.append({
            "name": name,
            **chip,
            "synapse_density_vs_castnet": round(vs_castnet, 1),
        })

    comparison.append({
        "name": f"CastNet ({process})",
        **castnet_entry,
        "synapse_density_vs_castnet": 1.0,
    })

    return {
        "castnet": castnet_entry,
        "comparison": comparison,
        "congestion_details": r,
    }
