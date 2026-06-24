"""Thermal dissipation model — Pure domain logic."""

from domain.hardware.constants import (
    ENERGY_PER_SWITCHING,
    WIRE_RESISTANCE_PER_UM,
    VDD,
    THERMAL_RESISTANCE,
    T_JUNCTION_MAX,
    T_AMBIENT,
)


def estimate_switching_power(
    nodes: int,
    process: str = "130nm",
    activity_factor: float = 0.05,
    frequency_mhz: float = 100.0,
) -> float:
    """Dynamic power from neuron switching."""
    energy = ENERGY_PER_SWITCHING.get(process, 0.5e-12)
    return energy * frequency_mhz * 1e6 * nodes * activity_factor


def estimate_wire_power(
    edges: int,
    process: str = "130nm",
    edge_length_avg_um: float = 10.0,
    activity_factor: float = 0.05,
    frequency_mhz: float = 100.0,
) -> float:
    """Dynamic power from wire charging/discharging."""
    r_per_um = WIRE_RESISTANCE_PER_UM.get(process, 0.1)
    v = VDD.get(process, 1.2)
    r_total = r_per_um * edge_length_avg_um
    p_per_wire = (v * v) / r_total if r_total > 0 else 0.0
    return p_per_wire * edges * activity_factor * (frequency_mhz / 1000.0)


def estimate_leakage_power(
    nodes: int,
    leakage_per_transistor_w: float = 1e-9,
    transistors_per_node: int = 5,
) -> float:
    """Static leakage power."""
    return nodes * transistors_per_node * leakage_per_transistor_w


def estimate_thermal(
    nodes: int,
    edges: int,
    process: str = "130nm",
    frequency_mhz: float = 100.0,
    activity_factor: float = 0.05,
    package: str = "bga",
    edge_length_avg_um: float = 10.0,
) -> dict:
    """Full thermal estimation for a chip.

    Returns:
        dict with power breakdown, junction temperature, and safety assessment.
    """
    p_sw = estimate_switching_power(nodes, process, activity_factor, frequency_mhz)
    p_w = estimate_wire_power(edges, process, edge_length_avg_um, activity_factor, frequency_mhz)
    p_lk = estimate_leakage_power(nodes)
    p_total = p_sw + p_w + p_lk

    theta_ja = THERMAL_RESISTANCE.get(package, 15.0)
    t_junction = T_AMBIENT + p_total * theta_ja
    temp_margin = T_JUNCTION_MAX - t_junction

    # Estimate die area for power density
    from domain.hardware.chip_area import estimate_chip_area
    area_info = estimate_chip_area(nodes, edges, process)
    est_area_mm2 = area_info["total_area_mm2"]
    p_density = p_total / (est_area_mm2 / 100.0) if est_area_mm2 > 0 else 0.0

    return {
        "process": process,
        "frequency_mhz": frequency_mhz,
        "activity_factor": activity_factor,
        "package": package,
        "switching_power_w": round(p_sw, 6),
        "wire_power_w": round(p_w, 6),
        "leakage_power_w": round(p_lk, 6),
        "total_power_w": round(p_total, 4),
        "wire_power_dominance_pct": round(p_w / p_total * 100, 1) if p_total > 0 else 0.0,
        "theta_ja_c_per_w": theta_ja,
        "t_ambient_c": T_AMBIENT,
        "t_junction_c": round(t_junction, 2),
        "t_junction_max_c": T_JUNCTION_MAX,
        "temp_margin_c": round(temp_margin, 2),
        "is_thermally_safe": temp_margin > 20,
        "est_area_mm2": round(est_area_mm2, 2),
        "power_density_w_per_cm2": round(p_density, 2),
        "recommendation": (
            "Passive" if p_density < 5
            else "Active" if p_density < 25
            else "Liquid" if p_density < 100
            else "Exceeds limits"
        ),
    }


def thermal_sweep(
    nodes: int,
    edges: int,
    process: str = "130nm",
) -> list[dict]:
    """Sweep frequency and activity factor for thermal analysis."""
    results = []
    for f in [10, 50, 100, 200, 500]:
        for a in [0.01, 0.05, 0.10, 0.25]:
            results.append(estimate_thermal(nodes, edges, process, frequency_mhz=f, activity_factor=a))
    return results
