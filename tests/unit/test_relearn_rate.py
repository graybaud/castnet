"""Unit tests for relearning rate metric."""

import torch
from domain.metrics.relearn_rate import compute_weight_change


class TestComputeWeightChange:
    def test_no_change(self):
        W = torch.randn(5, 10)
        before = {"L0": W.clone()}
        after = {"L0": W.clone()}
        result = compute_weight_change(before, after)
        assert result["avg_relative_weight_change"] == 0.0

    def test_full_change(self):
        before = {"L0": torch.ones(5, 10)}
        after = {"L0": torch.ones(5, 10) * 2.0}
        result = compute_weight_change(before, after)
        assert result["avg_relative_weight_change"] > 0.0

    def test_partial_change(self):
        W = torch.randn(5, 10)
        before = {"L0": W.clone()}
        after = {"L0": W.clone()}
        after["L0"][0, 0] += 0.5
        result = compute_weight_change(before, after)
        assert result["avg_relative_weight_change"] > 0.0

    def test_multiple_layers(self):
        before = {
            "L0": torch.randn(5, 10),
            "L1": torch.randn(3, 5),
        }
        after = {k: v.clone() for k, v in before.items()}
        after["L0"][0, 0] += 1.0
        result = compute_weight_change(before, after)
        assert "per_layer" in result
        assert result["per_layer"]["L0"] > 0
        assert result["per_layer"]["L1"] == 0.0

    def test_empty(self):
        result = compute_weight_change({}, {})
        assert result["avg_relative_weight_change"] == 0.0
        assert result["total_params"] == 0

    def test_layer_missing_in_after(self):
        before = {"L0": torch.randn(5, 10)}
        after = {}
        result = compute_weight_change(before, after)
        assert result["total_params"] == 0
