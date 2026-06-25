"""Hardware physical constants — Pure domain."""

TRANSISTORS_PER_NODE = 5
ROUTING_FACTOR_DEFAULT = 2.0

PROCESS_NODES = {
    "90nm":  {"density": 200_000, "transistor_area": 0.005, "edge_pitch": 0.09, "edge_length_avg": 7,  "metal_layers": 10, "routing_factor": 2.0, "mask_cost_min": 300_000, "mask_cost_max": 500_000, "era": 2004, "availability": "Limited"},
    "130nm": {"density": 100_000, "transistor_area": 0.02,  "edge_pitch": 0.13, "edge_length_avg": 10, "metal_layers": 8,  "routing_factor": 2.0, "mask_cost_min": 100_000, "mask_cost_max": 300_000, "era": 2002, "availability": "Very wide"},
    "180nm": {"density": 50_000,  "transistor_area": 0.04,  "edge_pitch": 0.18, "edge_length_avg": 14, "metal_layers": 6,  "routing_factor": 2.0, "mask_cost_min": 50_000,  "mask_cost_max": 150_000, "era": 2000, "availability": "Very wide"},
    "250nm": {"density": 25_000,  "transistor_area": 0.08,  "edge_pitch": 0.25, "edge_length_avg": 20, "metal_layers": 5,  "routing_factor": 2.0, "mask_cost_min": 30_000,  "mask_cost_max": 80_000,  "era": 1998, "availability": "Wide"},
    "350nm": {"density": 12_000,  "transistor_area": 0.16,  "edge_pitch": 0.35, "edge_length_avg": 28, "metal_layers": 4,  "routing_factor": 2.0, "mask_cost_min": 20_000,  "mask_cost_max": 50_000,  "era": 1996, "availability": "Available"},
    "500nm": {"density": 5_000,   "transistor_area": 0.35,  "edge_pitch": 0.50, "edge_length_avg": 40, "metal_layers": 3,  "routing_factor": 2.0, "mask_cost_min": 10_000,  "mask_cost_max": 30_000,  "era": 1994, "availability": "Niche"},
    "700nm": {"density": 2_500,   "transistor_area": 0.50,  "edge_pitch": 0.70, "edge_length_avg": 56, "metal_layers": 2,  "routing_factor": 2.0, "mask_cost_min": 5_000,   "mask_cost_max": 15_000,  "era": 1992, "availability": "Rare"},
    "1um":   {"density": 1_200,   "transistor_area": 0.80,  "edge_pitch": 1.00, "edge_length_avg": 80, "metal_layers": 2,  "routing_factor": 2.0, "mask_cost_min": 3_000,   "mask_cost_max": 10_000,  "era": 1990, "availability": "Very rare"},
}

ENERGY_PER_SWITCHING = {"130nm": 0.5e-12, "180nm": 0.8e-12, "350nm": 2.0e-12}
WIRE_RESISTANCE_PER_UM = {"130nm": 0.1, "180nm": 0.08, "350nm": 0.05}
VDD = {"130nm": 1.2, "180nm": 1.8, "350nm": 3.3}
THERMAL_RESISTANCE = {"qfn": 30.0, "bga": 15.0, "custom_sink": 5.0, "ideal": 2.0}
T_JUNCTION_MAX = 125.0
T_AMBIENT = 25.0
DEFECT_DENSITY = {"130nm": 0.3, "180nm": 0.2, "350nm": 0.15, "500nm": 0.1, "1um": 0.05}
CLUSTERING_ALPHA = {"130nm": 1.0, "180nm": 0.8, "350nm": 0.6, "500nm": 0.5, "1um": 0.4}
WAFER_DIAMETER_MM = 200

TRANSISTORS_PER_EDGE = 0
