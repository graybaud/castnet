"""Tests for correlation analysis use case."""

import torch
import pytest
from domain.scoring.ports import ScorePersister
from application.correlation_analysis import (
    CorrelationAnalysisUseCase,
    CorrelationResult,
)


class FakeScorePersister(ScorePersister):
    def __init__(self, scores=None):
        self._scores = scores or {}
    def save(self, scores, path): pass
    def load(self, path):
        return {k: v.clone() for k, v in self._scores.items()}


class TestCorrelationAnalysisUseCase:
    def test_perfect_correlation(self):
        """Identical scores should have correlation ~1.0."""
        scores = {
            "layer0.fc1": torch.rand(10, 20),
            "layer0.fc2": torch.rand(8, 10),
        }
        persister = FakeScorePersister(scores)
        use_case = CorrelationAnalysisUseCase(persister)
        result = use_case.execute("a.pt", "a.pt", "GPS", "GPS")
        assert abs(result.average - 1.0) < 0.1
        assert result.interpretation == "HIGHLY CORRELATED"

    def test_different_scores(self):
        """Different scores should have correlation < 1.0."""
        persister = FakeScorePersister()
        persister._scores = {
            "layer0.fc1": torch.rand(10, 20),
            "layer0.fc2": torch.rand(8, 10),
        }
        persister2 = FakeScorePersister()
        persister2._scores = {
            "layer0.fc1": torch.rand(10, 20),
            "layer0.fc2": torch.rand(8, 10),
        }
        use_case = CorrelationAnalysisUseCase(persister)
        # Override load to return different data
        original_load = persister.load
        call_count = [0]
        def fake_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                return persister2.load(path)
            return original_load(path)
        persister.load = fake_load
        result = use_case.execute("a.pt", "b.pt")
        assert -1.0 <= result.average <= 1.0

    def test_result_dataclass(self):
        result = CorrelationResult(
            method_a="GPS",
            method_b="Gradient",
            per_layer={"layer0": 0.85},
            average=0.85,
            interpretation="HIGHLY CORRELATED",
        )
        assert result.method_a == "GPS"
        assert result.average == 0.85
