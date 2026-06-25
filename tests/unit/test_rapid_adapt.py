"""Unit tests for rapid adaptation metric."""

import pytest
from domain.metrics.rapid_adapt import compute_adaptation_rate


class TestComputeAdaptationRate:
    def test_basic(self):
        # 50 steps, monotonically decreasing from 5.0 to 0.1
        loss_curve = [5.0 - i * 0.1 for i in range(50)]
        result = compute_adaptation_rate(loss_curve, window_size=20)
        assert result["total_steps"] == 50
        assert result["initial_loss"] > result["final_loss"]
        assert result["loss_ratio"] < 1.0

    def test_short_curve(self):
        loss_curve = [5.0, 4.0, 3.0]
        result = compute_adaptation_rate(loss_curve, window_size=20)
        assert result["loss_ratio"] == 1.0

    def test_flat_curve(self):
        loss_curve = [2.0] * 50
        result = compute_adaptation_rate(loss_curve, window_size=20)
        assert result["initial_loss"] == 2.0
        assert result["final_loss"] == 2.0

    def test_increasing_curve(self):
        # 50 steps, monotonically increasing from 0.1 to 5.0
        loss_curve = [0.1 + i * 0.1 for i in range(50)]
        result = compute_adaptation_rate(loss_curve, window_size=20)
        assert result["loss_ratio"] > 1.0

    def test_empty_curve(self):
        result = compute_adaptation_rate([], window_size=20)
        assert result["initial_loss"] is None
        assert result["final_loss"] is None
