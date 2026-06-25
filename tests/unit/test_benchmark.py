"""Unit tests for magnitude pruning benchmark."""

import torch
import pytest
from domain.metrics.benchmark import (
    apply_magnitude_pruning,
    apply_magnitude_pruning_all_layers,
    compute_magnitude_benchmark,
)


class TestApplyMagnitudePruning:
    def test_basic(self):
        W = torch.tensor([[1.0, 0.05], [0.02, 3.0]])
        pruned, sp = apply_magnitude_pruning(W, threshold=0.1)
        assert pruned[0, 0] == 1.0
        assert pruned[0, 1] == 0.0
        assert pruned[1, 0] == 0.0
        assert pruned[1, 1] == 3.0
        assert 0.0 < sp < 1.0

    def test_zero_threshold(self):
        W = torch.randn(10, 10)
        pruned, sp = apply_magnitude_pruning(W, threshold=0.0)
        assert sp == 0.0

    def test_high_threshold(self):
        W = torch.tensor([[0.01, 0.02], [0.03, 0.04]])
        pruned, sp = apply_magnitude_pruning(W, threshold=1.0)
        assert sp == 1.0
        assert pruned.sum() == 0.0

    def test_all_zeros(self):
        W = torch.zeros(5, 5)
        pruned, sp = apply_magnitude_pruning(W, threshold=0.1)
        assert sp == 1.0

    def test_negative_weights(self):
        W = torch.tensor([[-1.0, 2.0], [0.05, -3.0]])
        pruned, sp = apply_magnitude_pruning(W, threshold=0.1)
        assert pruned[0, 0] == -1.0
        assert pruned[1, 0] == 0.0


class TestApplyMagnitudePruningAllLayers:
    def test_multiple_layers(self):
        weights = {
            "L0": torch.tensor([[1.0, 0.05], [0.02, 3.0]]),
            "L1": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        }
        pruned, sparsities = apply_magnitude_pruning_all_layers(weights, threshold=0.1)
        assert len(pruned) == 2
        assert "overall" in sparsities
        assert 0.0 < sparsities["overall"] < 1.0

    def test_empty_dict(self):
        pruned, sparsities = apply_magnitude_pruning_all_layers({}, threshold=0.1)
        assert sparsities["overall"] == 0.0


class TestComputeMagnitudeBenchmark:
    def test_default_thresholds(self):
        W = torch.randn(10, 10) * 0.5
        results = compute_magnitude_benchmark({"L0": W})
        assert len(results) > 0
        for r in results:
            assert "threshold" in r
            assert "sparsity_pct" in r
            assert 0.0 <= r["sparsity_pct"] <= 100.0

    def test_custom_thresholds(self):
        W = torch.randn(10, 10)
        results = compute_magnitude_benchmark({"L0": W}, thresholds=[0.05, 0.5])
        assert len(results) == 2

    def test_sparsity_increases_with_threshold(self):
        W = torch.randn(20, 20) * 0.3
        results = compute_magnitude_benchmark({"L0": W}, thresholds=[0.01, 0.10, 0.50])
        sp = [r["sparsity_pct"] for r in results]
        assert sp[0] <= sp[1] <= sp[2]
