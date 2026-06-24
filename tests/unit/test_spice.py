"""Unit tests for SPICE netlist generator."""

import os
import tempfile
import pytest
from domain.hardware.spice import (
    weight_to_resistance,
    generate_spice_netlist,
    save_spice_netlist,
    V_REF,
    I_REF,
)


class TestWeightToResistance:
    def test_positive_weight(self):
        r = weight_to_resistance(0.5)
        expected = V_REF / (0.5 * I_REF)
        assert r == pytest.approx(expected, rel=1e-5)

    def test_zero_weight(self):
        r = weight_to_resistance(0.0)
        assert r == float("inf")

    def test_very_small_weight(self):
        r = weight_to_resistance(1e-12)
        assert r > 0

    def test_negative_weight(self):
        r = weight_to_resistance(-0.5)
        assert r > 0


class TestGenerateSpiceNetlist:
    def test_basic_graph(self):
        sources = [0, 1]
        targets = [1, 2]
        weights = [0.5, -0.3]
        types = [True, False]
        total_nodes = 3

        netlist = generate_spice_netlist(
            sources, targets, weights, types, total_nodes
        )

        assert "* CastNet SPICE" in netlist
        assert "Xneuron_0" in netlist
        assert "Xneuron_1" in netlist
        assert "Xneuron_2" in netlist
        assert "R_000000" in netlist  # excitatory
        assert "R_000001" in netlist  # inhibitory
        assert ".tran 1n 500n" in netlist
        assert ".end" in netlist

    def test_max_nodes(self):
        sources = [0, 1, 2, 3]
        targets = [1, 2, 3, 4]
        weights = [0.1, 0.2, 0.3, 0.4]
        total_nodes = 5

        netlist = generate_spice_netlist(
            sources, targets, weights, total_nodes=total_nodes, max_nodes=3
        )

        assert "Xneuron_0" in netlist
        assert "Xneuron_1" in netlist
        assert "Xneuron_2" in netlist
        assert "Xneuron_3" not in netlist

    def test_empty_graph(self):
        netlist = generate_spice_netlist([], [], [], total_nodes=0)
        assert ".end" in netlist

    def test_no_types_defaults_to_excitatory(self):
        sources = [0]
        targets = [1]
        weights = [0.5]
        netlist = generate_spice_netlist(sources, targets, weights, total_nodes=2)
        assert "R_000000 in_0 node_1" in netlist


class TestSaveSpiceNetlist:
    def test_save_and_stats(self):
        netlist = generate_spice_netlist([0, 1], [1, 2], [0.5, -0.3], total_nodes=3)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".spice", delete=False
        ) as f:
            path = f.name

        try:
            out_path, n_resistors, n_neurons = save_spice_netlist(netlist, path)
            assert out_path == path
            assert n_resistors == 2
            assert n_neurons == 3
            assert os.path.exists(path)
            with open(path) as f:
                saved = f.read()
            assert saved == netlist
        finally:
            os.unlink(path)
