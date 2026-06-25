"""
test_normalization.py — Tests for normalize_scores.
Migrated from legacy.
"""

import pytest
import torch
from domain.scoring.masks import normalize_scores


class TestNormalization:

    @pytest.fixture
    def scores(self):
        return {
            "fc1": torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            "fc2": torch.tensor([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]),
        }

    def test_with_batch_div(self, scores):
        result = normalize_scores({k: v.clone() for k, v in scores.items()},
                                  num_batches=10, needs_batch_div=True)
        assert abs(result["fc1"].max().item() - 1.0) < 1e-5
        assert result["fc1"][0, 0].item() == 0.0

    def test_without_batch_div(self, scores):
        result = normalize_scores({k: v.clone() for k, v in scores.items()},
                                  num_batches=10, needs_batch_div=False)
        assert abs(result["fc1"].max().item() - 1.0) < 1e-5

    def test_all_zero(self):
        scores = {"fc": torch.zeros(3, 4)}
        result = normalize_scores(scores, num_batches=5, needs_batch_div=True)
        assert torch.all(result["fc"] == 0)

    def test_single_element(self):
        scores = {"fc": torch.tensor([[3.14]])}
        result = normalize_scores(scores, num_batches=1, needs_batch_div=False)
        assert abs(result["fc"].item() - 1.0) < 1e-5

    def test_in_place(self, scores):
        scores_copy = {k: v.clone() for k, v in scores.items()}
        result = normalize_scores(scores_copy, num_batches=1, needs_batch_div=False)
        assert result is scores_copy
