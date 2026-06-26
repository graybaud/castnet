"""Tests for score diagnostics."""

import torch
import pytest
from domain.scoring.diagnose import diagnose_scores


class TestDiagnoseScores:

    def test_empty(self):
        result = diagnose_scores({})
        assert result["total_layers"] == 0
        assert result["all_zero"] is True

    def test_mixed(self):
        scores = {
            "fc1": torch.rand(5, 10),
            "fc2": torch.zeros(10, 5),
        }
        result = diagnose_scores(scores)
        assert result["total_layers"] == 2
        assert result["zero_layers"] == 1
        assert result["zero_layer_names"] == ["fc2"]

    def test_all_non_zero(self):
        scores = {"fc": torch.ones(3, 4)}
        result = diagnose_scores(scores)
        assert result["zero_layers"] == 0
        assert result["all_zero"] is False

    def test_all_zero(self):
        scores = {"fc": torch.zeros(3, 4)}
        result = diagnose_scores(scores)
        assert result["zero_layers"] == 1
        assert result["all_zero"] is True

    def test_non_zero_stats(self):
        scores = {"fc": torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])}
        result = diagnose_scores(scores)
        layer = result["non_zero_layers"][0]
        assert layer["non_zero"] == 5
        assert layer["non_zero_pct"] == pytest.approx(83.3, abs=0.2)
