"""Unit tests for routing congestion model."""

import pytest
from domain.hardware.routing import (
    rents_rule,
    wire_length_distribution,
    estimate_congestion_overhead,
    estimate_with_congestion,
)


class TestRentsRule:
    def test_basic(self):
        terminals, K = rents_rule(1000)
        assert terminals > 0
        assert K == 2.5

    def test_single_node(self):
        terminals, K = rents_rule(1)
        assert terminals > 0


class TestWireLengthDistribution:
    def test_basic(self):
        dist = wire_length_distribution(1000, 10.0)
        assert len(dist) > 0
        total = sum(dist.values())
        assert total == pytest.approx(1.0, abs=0.1)

    def test_single_node(self):
        dist = wire_length_distribution(1, 10.0)
        assert 10.0 in dist


class TestCongestionOverhead:
    def test_low_congestion(self):
        # Few edges on many nodes = no congestion
        overhead, cong_pct = estimate_congestion_overhead(100, 10000, metal_layers=8)
        assert overhead == 1.0
        assert cong_pct == 0.0

    def test_high_congestion(self):
        # Many edges on few nodes = congested
        overhead, cong_pct = estimate_congestion_overhead(1000000, 100)
        assert overhead > 1.0

    def test_moderate_congestion(self):
        overhead, cong_pct = estimate_congestion_overhead(5000, 1000, metal_layers=8)
        assert overhead >= 1.0


class TestEstimateWithCongestion:
    def test_full_estimation(self):
        result = estimate_with_congestion(1000, 5000, "130nm")
        assert "total_area_mm2" in result
        assert "chip_side_mm" in result
        assert "congestion_overhead_factor" in result
        assert result["total_area_mm2"] > 0

    def test_unknown_process_falls_back(self):
        result = estimate_with_congestion(1000, 5000, "future_node")
        assert result["process"] == "130nm"
