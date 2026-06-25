"""Tests for castnet.hardware.thermal_model."""
from domain.hardware.thermal import (
    estimate_switching_power, estimate_wire_power,
    estimate_leakage_power, estimate_thermal, thermal_sweep
)

class TestSwitchingPower:
    def test_basic(self):
        p = estimate_switching_power(1000, "130nm", activity_factor=0.05, frequency_mhz=100)
        assert p > 0

    def test_zero_activity(self):
        p = estimate_switching_power(1000, "130nm", activity_factor=0.0)
        assert p == 0.0


class TestWirePower:
    def test_basic(self):
        p = estimate_wire_power(1000, "130nm", edge_length_avg_um=10, activity_factor=0.05, frequency_mhz=100)
        assert p > 0

    def test_zero_edges(self):
        p = estimate_wire_power(0, "130nm")
        assert p == 0.0


class TestLeakagePower:
    def test_basic(self):
        p = estimate_leakage_power(1000, leakage_per_transistor_w=1e-9, transistors_per_node=5)
        assert p > 0


class TestEstimateThermal:
    def test_basic(self):
        result = estimate_thermal(1000, 10000, "130nm")
        assert result["total_power_w"] > 0
        assert result["t_junction_c"] > result["t_ambient_c"]
        assert "recommendation" in result

    def test_safety(self):
        # Small chip should be safe
        result = estimate_thermal(100, 1000, "130nm", frequency_mhz=10, activity_factor=0.01)
        assert result["is_thermally_safe"]

    def test_package_affects_temp(self):
        r_qfn = estimate_thermal(1000, 10000, package="qfn")
        r_bga = estimate_thermal(1000, 10000, package="bga")
        assert r_qfn["t_junction_c"] > r_bga["t_junction_c"]  # QFN has higher thermal resistance


class TestThermalSweep:
    def test_basic(self):
        results = thermal_sweep(1000, 10000, "130nm")
        assert len(results) == 20  # 5 freqs × 5 activities
        assert all("total_power_w" in r for r in results)
