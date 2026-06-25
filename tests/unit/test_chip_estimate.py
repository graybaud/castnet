"""Tests for castnet.hardware.chip_estimate."""
import math
from domain.hardware.chip_area import (
    estimate_chip_area, estimate_all_processes,
)


class TestEstimateAllProcesses:
    def test_basic(self):
        results = estimate_all_processes(100, 1000)
        assert len(results) == 8
        assert all("process" in r for r in results)

    def test_130nm_present(self):
        results = estimate_all_processes(100, 1000)
        processes = [r["process"] for r in results]
        assert "130nm" in processes


class TestEstimateChipArea:
    def test_basic_estimation(self):
        result = estimate_chip_area(100, 1000)
        assert result["total_nodes"] == 100
        assert result["total_edges"] == 1000

    def test_all_processes(self):
        results = estimate_all_processes(25600, 15_300_000)
        assert len(results) == 8
        assert results[0]["chip_side_mm"] > 0


