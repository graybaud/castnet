"""Tests for castnet.hardware.routing_congestion."""
from domain.hardware.routing import (
    rents_rule, wire_length_distribution,
    estimate_congestion_overhead, estimate_with_congestion
)

class TestRentsRule:
    def test_basic(self):
        P, K = rents_rule(1000)
        assert P > 0
        assert K == 2.5

    def test_scaling(self):
        P1, _ = rents_rule(100)
        P2, _ = rents_rule(10000)
        assert P2 > P1  # more nodes = more pins


class TestWireLengthDistribution:
    def test_basic(self):
        dist = wire_length_distribution(1000, 10.0)
        assert len(dist) == 10
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01

    def test_single_node(self):
        dist = wire_length_distribution(1, 10.0)
        assert dist == {10.0: 1.0}


class TestCongestionOverhead:
    def test_no_congestion(self):
        overhead, pct = estimate_congestion_overhead(100, 10000, metal_layers=8)
        assert overhead == 1.0
        assert pct == 0.0

    def test_congestion(self):
        overhead, pct = estimate_congestion_overhead(1_000_000, 100, metal_layers=2)
        assert overhead > 1.0
        assert pct > 0


class TestEstimateWithCongestion:
    def test_basic(self):
        result = estimate_with_congestion(25600, 15_300_000, "130nm")
        assert result["chip_side_mm"] > 0
        assert result["total_area_mm2"] > 0
        assert "congestion_overhead_factor" in result

    def test_unknown_process_fallback(self):
        result = estimate_with_congestion(100, 1000, "nonexistent")
        assert result["process"] == "130nm"  # fallback

    def test_routing_dominance(self):
        result = estimate_with_congestion(100, 1_000_000, "130nm")
        assert result["routing_dominance_pct"] > 90
