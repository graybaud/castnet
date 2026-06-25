"""
test_masking.py — Tests for apply_percentile_mask.
Migrated from legacy.
"""

import torch
from domain.scoring.masks import apply_percentile_mask


class TestPercentileMask:

    def test_keep_50(self):
        score = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        mask = apply_percentile_mask(score, keep_fraction=0.50)
        assert mask.sum().item() == 2

    def test_keep_100(self):
        score = torch.tensor([[0.0, 1.0, 2.0]])
        mask = apply_percentile_mask(score, keep_fraction=1.0)
        assert mask.sum().item() == 2

    def test_keep_0(self):
        score = torch.tensor([[0.0, 1.0, 2.0]])
        mask = apply_percentile_mask(score, keep_fraction=0.0)
        # keep_fraction=0 means threshold = quantile(1.0) = max, so max element is kept
        assert mask.sum().item() == 1
        assert mask[0, 2].item() == 1.0  # max value

    def test_all_zero(self):
        mask = apply_percentile_mask(torch.zeros(3, 4), keep_fraction=0.5)
        assert torch.all(mask == 0)

    def test_single_nonzero(self):
        score = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
        mask = apply_percentile_mask(score, keep_fraction=0.5)
        assert mask[0, 2].item() == 1.0

    def test_output_dtype(self):
        mask = apply_percentile_mask(torch.randn(10, 10).abs(), keep_fraction=0.3)
        assert mask.dtype == torch.float32
