"""Domain Hardware — Pure estimation functions."""

from domain.hardware.constants import (
    PROCESS_NODES,
    TRANSISTORS_PER_NODE,
    DEFECT_DENSITY,
    CLUSTERING_ALPHA,
    ENERGY_PER_SWITCHING,
    WIRE_RESISTANCE_PER_UM,
    VDD,
    THERMAL_RESISTANCE,
    T_JUNCTION_MAX,
    T_AMBIENT,
)
from domain.hardware.yield_model import (
    murphy_yield,
    seeds_yield,
    negative_binomial_yield,
    chips_per_wafer,
    estimate_yield,
)
from domain.hardware.chip_area import (
    estimate_chip_area,
    estimate_all_processes,
)
from domain.hardware.thermal import (
    estimate_switching_power,
    estimate_wire_power,
    estimate_leakage_power,
    estimate_thermal,
    thermal_sweep,
)
from domain.hardware.routing import (
    rents_rule,
    wire_length_distribution,
    estimate_congestion_overhead,
    estimate_with_congestion,
)

__all__ = [
    # Constants
    "PROCESS_NODES",
    "TRANSISTORS_PER_NODE",
    "DEFECT_DENSITY",
    "CLUSTERING_ALPHA",
    "ENERGY_PER_SWITCHING",
    "WIRE_RESISTANCE_PER_UM",
    "VDD",
    "THERMAL_RESISTANCE",
    "T_JUNCTION_MAX",
    "T_AMBIENT",
    # Yield
    "murphy_yield",
    "seeds_yield",
    "negative_binomial_yield",
    "chips_per_wafer",
    "estimate_yield",
    # Chip area
    "estimate_chip_area",
    "estimate_all_processes",
    # Thermal
    "estimate_switching_power",
    "estimate_wire_power",
    "estimate_leakage_power",
    "estimate_thermal",
    "thermal_sweep",
    # Routing
    "rents_rule",
    "wire_length_distribution",
    "estimate_congestion_overhead",
    "estimate_with_congestion",
]
