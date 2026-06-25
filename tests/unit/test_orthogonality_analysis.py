"""Tests for orthogonality analysis use case."""

import torch
import pytest
from domain.scoring.ports import ScorePersister
from application.orthogonality_analysis import (
    OrthogonalityAnalysisUseCase,
    OrthogonalityResult,
)


class FakeScorePersister(ScorePersister):
    def __init__(self, scores=None):
        self._scores = scores or {}
    def save(self, scores, path): pass
    def load(self, path):
        return {k: v.clone() for k, v in self._scores.items()}


class TestOrthogonalityAnalysisUseCase:
    def test_identical_masks(self):
        scores = {"layer0": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        persister = FakeScorePersister(scores)
        use_case = OrthogonalityAnalysisUseCase(persister)
        result = use_case.execute({"A": "a.pt", "B": "a.pt"}, keep_fraction=0.5)
        assert result.average_overlap == 1.0

    def test_disjoint_masks(self):
        scores_a = {"layer0": torch.tensor([[1.0, 0.0], [0.0, 0.0]])}
        scores_b = {"layer0": torch.tensor([[0.0, 0.0], [0.0, 1.0]])}
        persister = FakeScorePersister()
        persister._scores = scores_a
        use_case = OrthogonalityAnalysisUseCase(persister)

        # Override load to return different data
        original_load = persister.load
        call_count = [0]
        def fake_load(path):
            call_count[0] += 1
            if call_count[0] == 2:
                return scores_b
            return original_load(path)
        persister.load = fake_load
        result = use_case.execute({"A": "a.pt", "B": "b.pt"}, keep_fraction=0.5)
        assert result.average_overlap == 0.0

    def test_result_dataclass(self):
        result = OrthogonalityResult(
            methods=["A", "B"],
            keep_fraction=0.3,
            overlap_matrix=[[1.0, 0.1], [0.1, 1.0]],
            average_overlap=0.1,
            orthogonality_score=90.0,
            most_orthogonal_pairs=[("A", "B", 0.1)],
        )
        assert result.orthogonality_score == 90.0
        assert len(result.most_orthogonal_pairs) == 1
