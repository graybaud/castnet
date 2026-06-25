"""Tests for castnet.hardware.constants."""
from domain.hardware.constants import (
    PROCESS_NODES, TRANSISTORS_PER_NODE, TRANSISTORS_PER_EDGE,
    ENERGY_PER_SWITCHING, DEFECT_DENSITY, CLUSTERING_ALPHA
)

class TestConstants:
    def test_process_nodes_present(self):
        assert "130nm" in PROCESS_NODES
        assert "350nm" in PROCESS_NODES
        assert "1um" in PROCESS_NODES

    def test_all_nodes_have_required_keys(self):
        required = ["density", "transistor_area", "edge_pitch", "edge_length_avg", "metal_layers"]
        for name, node in PROCESS_NODES.items():
            for key in required:
                assert key in node, f"{name} missing {key}"

    def test_transistors_per_node(self):
        assert TRANSISTORS_PER_NODE == 5

    def test_transistors_per_edge_zero(self):
        assert TRANSISTORS_PER_EDGE == 0  # resistive wires, no transistors

    def test_energy_switching_130nm(self):
        assert ENERGY_PER_SWITCHING["130nm"] == 0.5e-12

    def test_defect_density_order(self):
        # Smaller nodes = more defects
        assert DEFECT_DENSITY["130nm"] > DEFECT_DENSITY["1um"]
