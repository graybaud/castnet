"""Tests for metrics — updated for v2."""

import math
import pytest
from domain.metrics.perplexity import (
    compute_perplexity_from_loss,
    compute_perplexity_from_total,
)


class TestPerplexity:
    def test_loss_1(self):
        assert math.isclose(compute_perplexity_from_loss(1.0), math.e)

    def test_loss_0(self):
        assert compute_perplexity_from_loss(0.0) == 1.0

    def test_from_total(self):
        perp = compute_perplexity_from_total(2.0, 2)
        assert math.isclose(perp, math.e)
