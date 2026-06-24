"""Unit tests for thermal model."""

import pytest
from domain.hardware.thermal import (
    estimate_switching_power,
    estimate_wire_power,
    estimate_leakage_power,
    estimate_thermal,
    thermal_sweep,
)


class TestSwitchingPower:
    def test_known_process(self):
        p = estimate_switching_power(1000, "130nm", activity_factor=0.05, frequency_mhz=100)
        assert p > 0

    def test_unknown_process_falls_back(self):
        p = estimate_switching_power(1000, "mystery_node")
        assert p > 0

    def test_zero_nodes(self):
        p = estimate_switching_power(0)
        assert p == 0.0

    def test_zero_activity(self):
        p = estimate_switching_power(1000, activity_factor=0.0)
        assert p == 0.0


class TestWirePower:
    def test_known_process(self):
        p = estimate_wire_power(5000, "130nm")
        assert p > 0

    def test_zero_edges(self):
        p = estimate_wire_power(0)
        assert p == 0.0


class TestLeakagePower:
    def test_basic(self):
        p = estimate_leakage_power(1000)
        assert p > 0

    def test_zero_nodes(self):
        p = estimate_leakage_power(0)
        assert p == 0.0


class TestEstimateThermal:
    def test_full_estimation(self):
        result = estimate_thermal(1000, 5000, "130nm")
        assert "total_power_w" in result
        assert "t_junction_c" in result
        assert "is_thermally_safe" in result
        assert result["total_power_w"] > 0

    def test_recommendation_passive(self):
        result = estimate_thermal(100, 500, "500nm", frequency_mhz=10, activity_factor=0.01)
        assert result["recommendation"] in ("Passive", "Active", "Liquid", "Exceeds limits")


class TestThermalSweep:
    def test_returns_20_results(self):
        results = thermal_sweep(1000, 5000, "130nm")
        assert len(results) == 20
        for r in results:
            assert "total_power_w" in r
