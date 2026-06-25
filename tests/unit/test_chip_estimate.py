"""Tests for castnet.hardware.chip_estimate."""
import math
from domain.hardware.chip_area import (
    estimate_chip_area_v1, estimate_chip_area_v2, estimate_all_processes,
    estimate_chip_area, estimate_system_configs
)

class TestChipEstimateV1:
    def test_basic(self):
        results = estimate_chip_area_v1(100, 1000)
        assert len(results) == 8  # 8 process nodes
        assert all("chip_side_mm" in r for r in results)
        assert all(r["chip_side_mm"] > 0 for r in results)

    def test_ordering(self):
        results = estimate_chip_area_v1(100, 1000)
        sides = [r["chip_side_mm"] for r in results]
        assert sides == sorted(sides)  # sorted by size

    def test_zero_nodes(self):
        results = estimate_chip_area_v1(0, 1000)
        assert results[0]["total_area_mm2"] >= 0


class TestChipEstimateV2:
    def test_basic(self):
        results = estimate_chip_area_v2(100, 1000)
        assert len(results) == 8
        assert all("edge_dominance_pct" in r for r in results)

    def test_edge_dominance(self):
        # With many edges, routing should dominate
        results = estimate_chip_area_v2(100, 1_000_000)
        r130 = [r for r in results if r["process"] == "130nm"][0]
        assert r130["edge_dominance_pct"] > 90


class TestEstimateAllProcesses:
    def test_basic(self):
        results = estimate_all_processes(100, 1000)
        assert len(results) == 8
        assert all("density_per_mm2" in r for r in results)

    def test_130nm_present(self):
        results = estimate_all_processes(100, 1000)
        processes = [r["process"] for r in results]
        assert "130nm" in processes


class TestEstimateChipArea:
    def test_v1_method(self):
        result = estimate_chip_area(100, 1000, method="v1")
        assert "total_nodes" in result
        assert result["total_nodes"] == 100
        assert result["total_active_edges"] == 1000

    def test_v2_method(self):
        result = estimate_chip_area(100, 1000, method="v2")
        assert result["method"] == "v2"
        assert len(result["process_estimates"]) == 8

    def test_default_130nm_summary(self):
        result = estimate_chip_area(25600, 15_300_000)
        assert result["chip_side_mm"] > 0
        assert result["total_area_mm2"] > 0


class TestSystemConfigs:
    def test_basic(self):
        chip_results = estimate_all_processes(25600, 15_300_000)
        configs = estimate_system_configs(chip_results, n_experts=64)
        assert len(configs) > 0
        assert all("feasible" in c for c in configs)

    def test_feasible_configs_exist(self):
        chip_results = estimate_all_processes(25600, 15_300_000)
        configs = estimate_system_configs(chip_results, n_experts=64)
        feasible = [c for c in configs if c["feasible"] == "YES"]
        assert len(feasible) > 0
