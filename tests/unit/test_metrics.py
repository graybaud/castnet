"""Unit tests for perplexity metrics."""

import math
import torch
import pytest
from domain.metrics.perplexity import (
    compute_perplexity_from_loss,
    compute_perplexity_from_logits,
    compute_perplexity_from_total,
)


class TestPerplexityFromLoss:

    def test_loss_zero(self):
        assert compute_perplexity_from_loss(0.0) == 1.0

    def test_loss_one(self):
        assert math.isclose(compute_perplexity_from_loss(1.0), math.e, rel_tol=1e-6)

    def test_loss_negative(self):
        result = compute_perplexity_from_loss(-1.0)
        assert math.isfinite(result)

    def test_loss_very_high(self):
        assert compute_perplexity_from_loss(15.0) == float("inf")

    def test_loss_just_below_threshold(self):
        assert math.isfinite(compute_perplexity_from_loss(9.99))


class TestPerplexityFromTotal:

    def test_normal_case(self):
        result = compute_perplexity_from_total(total_loss=2.0, total_tokens=2)
        assert math.isclose(result, math.e, rel_tol=1e-6)

    def test_zero_tokens(self):
        assert compute_perplexity_from_total(2.0, 0) == float("inf")

    def test_zero_loss(self):
        assert compute_perplexity_from_total(0.0, 100) == 1.0

    def test_negative_tokens(self):
        result = compute_perplexity_from_total(2.0, -5)
        assert result == float("inf") or math.isfinite(result)


class TestPerplexityFromLogits:

    def test_perfect_prediction(self):
        vocab_size = 10
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, 5] = 100.0  # tres confiant sur le bon token
        targets = torch.tensor([[5]])
        result = compute_perplexity_from_logits(logits, targets, ignore_index=-100)
        assert math.isfinite(result)
        assert result < 2.0  # presque parfait

    def test_batched(self):
        vocab_size = 5
        logits = torch.randn(2, 3, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 3))
        result = compute_perplexity_from_logits(logits, targets)
        assert math.isfinite(result)
        assert result > 0
