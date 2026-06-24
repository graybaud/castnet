"""Unit tests for hardware estimation — Pure domain."""

import math
import pytest
from domain.hardware.constants import PROCESS_NODES
from domain.hardware.yield_model import (
    murphy_yield,
    seeds_yield,
    negative_binomial_yield,
    chips_per_wafer,
    estimate_yield,
)
from domain.hardware.chip_area import estimate_chip_area, estimate_all_processes


class TestMurphyYield:

    def test_no_defects(self):
        assert murphy_yield(100.0, 0.0) == 1.0

    def test_small_area(self):
        y = murphy_yield(0.01, 0.3)
        assert 0.95 < y <= 1.0

    def test_large_area(self):
        y = murphy_yield(500.0, 1.0)
        assert y < 0.5

    def test_zero_area(self):
        y = murphy_yield(0.0, 0.3)
        assert y == 1.0


class TestSeedsYield:

    def test_no_defects(self):
        assert seeds_yield(100.0, 0.0) == 1.0

    def test_decays_with_area(self):
        y1 = seeds_yield(10.0, 0.3)
        y2 = seeds_yield(100.0, 0.3)
        assert y1 > y2


class TestNegativeBinomialYield:

    def test_reduces_to_seeds_when_alpha_zero(self):
        y_nb = negative_binomial_yield(100.0, 0.3, alpha=0.0)
        y_seeds = seeds_yield(100.0, 0.3)
        assert math.isclose(y_nb, y_seeds, rel_tol=1e-6)

    def test_clustering_changes_yield(self):
        # Alpha different => yield different (sauf cas triviaux)
        y1 = negative_binomial_yield(100.0, 0.3, alpha=1.0)
        y5 = negative_binomial_yield(100.0, 0.3, alpha=5.0)
        y10 = negative_binomial_yield(100.0, 0.3, alpha=10.0)
        # Les yields doivent etre tous differents
        assert y1 != y5
        assert y5 != y10
        # Et tous dans [0, 1]
        for y in [y1, y5, y10]:
            assert 0.0 <= y <= 1.0

    def test_small_area_high_yield(self):
        # Petite surface => yield eleve quel que soit le modele
        y_murphy = murphy_yield(1.0, 0.3)
        y_seeds = seeds_yield(1.0, 0.3)
        y_nb = negative_binomial_yield(1.0, 0.3, alpha=2.0)
        # Tous proches de 1.0 pour une petite puce
        assert y_murphy > 0.95
        assert y_seeds > 0.95
        assert y_nb > 0.95

    def test_large_area_low_yield(self):
        # Grande surface => yield bas
        y = negative_binomial_yield(500.0, 1.0, alpha=2.0)
        assert y < 0.5

    def test_alpha_zero_is_seeds(self):
        y_nb = negative_binomial_yield(100.0, 0.3, alpha=0.0)
        y_seeds = seeds_yield(100.0, 0.3)
        assert y_nb == pytest.approx(y_seeds, rel=1e-6)


class TestChipsPerWafer:

    def test_small_chip(self):
        n = chips_per_wafer(5.0)
        assert n > 500

    def test_large_chip(self):
        n = chips_per_wafer(50.0)
        assert n < 20

    def test_chip_bigger_than_wafer(self):
        n = chips_per_wafer(300.0)
        assert n == 0

    def test_zero_side(self):
        n = chips_per_wafer(0.0)
        assert n == 0


class TestEstimateYield:

    def test_known_process(self):
        result = estimate_yield(area_mm2=50.0, side_mm=7.0, process="130nm")
        assert "yield_neg_binomial_pct" in result
        assert 0 <= result["yield_neg_binomial_pct"] <= 100

    def test_unknown_process_falls_back(self):
        result = estimate_yield(50.0, 7.0, process="mystery_node")
        assert result["process"] == "mystery_node"


class TestChipArea:

    def test_known_process(self):
        result = estimate_chip_area(1000, 5000, "130nm")
        assert result["total_nodes"] == 1000
        assert result["total_edges"] == 5000
        assert result["total_area_mm2"] > 0

    def test_unknown_process_falls_back(self):
        result = estimate_chip_area(1000, 5000, "future_3nm")
        assert result["process"] == "130nm"

    def test_zero_nodes(self):
        result = estimate_chip_area(0, 0)
        assert result["total_transistors"] == 0
        assert result["total_area_mm2"] == 0.0


class TestEstimateAllProcesses:

    def test_returns_all_processes(self):
        results = estimate_all_processes(1000, 5000)
        assert len(results) == len(PROCESS_NODES)

    def test_sorted_by_size(self):
        results = estimate_all_processes(1000, 5000)
        sides = [r["chip_side_mm"] for r in results]
        assert sides == sorted(sides)
