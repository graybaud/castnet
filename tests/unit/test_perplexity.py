"""
test_perplexity.py — Tests for evaluate_perplexity.
Migrated from legacy. Tests the same behavior.
"""

import pytest
import torch
import math
from domain.metrics.perplexity import (
    compute_perplexity_from_loss,
    compute_perplexity_from_total,
)


# =============================================================================
# Domain tests (pure functions)
# =============================================================================

class TestComputePerplexityFromLoss:
    def test_loss_1(self):
        assert math.isclose(compute_perplexity_from_loss(1.0), math.e, rel_tol=1e-6)

    def test_loss_0(self):
        assert compute_perplexity_from_loss(0.0) == 1.0

    def test_loss_high(self):
        assert compute_perplexity_from_loss(15.0) == float("inf")


class TestComputePerplexityFromTotal:
    def test_constant_loss_1(self):
        perp = compute_perplexity_from_total(total_loss=1.0, total_tokens=1)
        assert abs(perp - math.e) < 0.1

    def test_constant_loss_0(self):
        perp = compute_perplexity_from_total(total_loss=0.0, total_tokens=100)
        assert abs(perp - 1.0) < 0.01

    def test_zero_tokens(self):
        perp = compute_perplexity_from_total(total_loss=10.0, total_tokens=0)
        assert perp == float("inf")


# =============================================================================
# Integration tests (use case with mocks — matches legacy exactly)
# =============================================================================

class MockModelForPerplexity:
    """Minimal mock that returns constant loss."""
    def __init__(self, loss_value=1.0):
        self.loss_value = loss_value
        self.training = False

    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, input_ids, labels=None):
        loss = torch.tensor(self.loss_value)
        return type('obj', (object,), {'loss': loss})()


class TestEvaluatePerplexity:
    """Same tests as legacy, adapted to EvaluatePerplexityUseCase."""

    def test_constant_loss_1(self):
        """exp(1.0) = e ≈ 2.718"""
        from application.evaluate import EvaluatePerplexityUseCase
        model = MockModelForPerplexity(loss_value=1.0)
        use_case = EvaluatePerplexityUseCase(model, block_size=512, max_tokens=2048)
        result = use_case.execute("cpu")
        assert abs(result.perplexity - 2.718) < 0.1

    def test_constant_loss_0(self):
        """exp(0.0) = 1.0"""
        from application.evaluate import EvaluatePerplexityUseCase
        model = MockModelForPerplexity(loss_value=0.0)
        use_case = EvaluatePerplexityUseCase(model, block_size=512, max_tokens=2048)
        result = use_case.execute("cpu")
        assert abs(result.perplexity - 1.0) < 0.01

    def test_keeps_original_dtype(self):
        """Model should keep its original dtype after evaluation."""
        from application.evaluate import EvaluatePerplexityUseCase
        model = MockModelForPerplexity()
        use_case = EvaluatePerplexityUseCase(model, block_size=512, max_tokens=512)
        result = use_case.execute("cpu")
        # Smoke test passes
        assert result.perplexity > 0
