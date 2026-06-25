"""Tests for castnet.hardware.yield_model."""
from domain.hardware.yield_model import (
    murphy_yield, seeds_yield, negative_binomial_yield,
    chips_per_wafer, estimate_yield, yield_vs_size_sweep
)

class TestYieldModels:
    def test_murphy_bounds(self):
        y = murphy_yield(40, 0.3)
        assert 0 < y <= 1.0

    def test_seeds_bounds(self):
        y = seeds_yield(40, 0.3)
        assert 0 < y <= 1.0

    def test_neg_binomial_bounds(self):
        y = negative_binomial_yield(40, 0.3, 1.0)
        assert 0 < y <= 1.0

    def test_small_chip_high_yield(self):
        y = negative_binomial_yield(1, 0.3, 1.0)
        assert y > 0.95

    def test_large_chip_low_yield(self):
        y = negative_binomial_yield(600, 0.3, 1.0)
        assert y < 0.5


class TestChipsPerWafer:
    def test_small_chip(self):
        n = chips_per_wafer(5)
        assert n > 1000

    def test_large_chip(self):
        n = chips_per_wafer(50)
        assert n < 20

    def test_too_large(self):
        n = chips_per_wafer(300)
        assert n == 0


class TestEstimateYield:
    def test_basic(self):
        result = estimate_yield(40, process="130nm")
        assert result["yield_neg_binomial_pct"] > 0
        assert result["gross_dies_per_wafer"] > 0
        assert result["cost_per_working_die_usd"] > 0

    def test_strategy_monolithic(self):
        result = estimate_yield(10, process="130nm")
        assert result["strategy"] == "monolithic"

    def test_strategy_chiplet(self):
        result = estimate_yield(400, process="130nm")
        assert result["strategy"] == "chiplet"
        assert result["chiplet_yield_4x"] is not None


class TestYieldSweep:
    def test_basic(self):
        results = yield_vs_size_sweep("130nm", max_side=10)
        assert len(results) == 10
        # Yield should decrease with size
        assert results[0]["yield_pct"] > results[-1]["yield_pct"]
