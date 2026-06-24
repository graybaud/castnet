"""Unit tests for application use cases — fully mocked."""

import torch
import pytest
from unittest.mock import MagicMock, patch
from domain.scoring.ports import ScorePersister, MaskPersister
from application.generate_masks import GenerateMasksUseCase, MaskResult
from application.extract_scores import ExtractScoresUseCase, ExtractResult


class FakeScorePersister(ScorePersister):
    def __init__(self, scores=None):
        self.scores = scores or {}
        self.saved = None
    def save(self, scores, path):
        self.saved = (scores, path)
    def load(self, path):
        return self.scores


class FakeMaskPersister(MaskPersister):
    def __init__(self, masks=None):
        self.masks = masks or {}
        self.saved = None
    def save(self, masks, path):
        self.saved = (masks, path)
    def load(self, path):
        return self.masks


class TestGenerateMasksUseCase:

    def test_generates_masks(self):
        scores = {
            "layer0": torch.tensor([[0.1, 0.5], [0.9, 0.3]]),
            "layer1": torch.tensor([[0.2, 0.8], [0.4, 0.6]]),
        }
        score_persister = FakeScorePersister(scores)
        mask_persister = FakeMaskPersister()

        uc = GenerateMasksUseCase(score_persister, mask_persister)
        result = uc.execute("dummy.pt", "masks.safetensors", keep_fraction=0.5)

        assert isinstance(result, MaskResult)
        assert result.keep_fraction == 0.5
        assert result.sparsity["total_active"] > 0
        assert mask_persister.saved is not None

    def test_keep_fraction_very_small(self):
        scores = {"layer0": torch.rand(10, 10)}
        score_persister = FakeScorePersister(scores)
        mask_persister = FakeMaskPersister()

        uc = GenerateMasksUseCase(score_persister, mask_persister)
        result = uc.execute("dummy.pt", "masks.safetensors", keep_fraction=0.01)

        # Au moins 1 element garde (arrondi), max 100
        assert 1 <= result.sparsity["total_active"] <= 100

    def test_keep_fraction_one(self):
        scores = {"layer0": torch.rand(10, 10)}
        score_persister = FakeScorePersister(scores)
        mask_persister = FakeMaskPersister()

        uc = GenerateMasksUseCase(score_persister, mask_persister)
        result = uc.execute("dummy.pt", "masks.safetensors", keep_fraction=1.0)

        total = scores["layer0"].numel()
        assert result.sparsity["total_active"] == total


class TestExtractScoresUseCase:

    def test_extract_result_dataclass(self):
        result = ExtractResult(
            scores={"L0": torch.rand(5, 10)},
            num_layers=1,
            num_batches=10,
        )
        assert result.num_layers == 1
        assert result.num_batches == 10


class TestPipelineResult:

    def test_pipeline_result_creation(self):
        from application.pipeline import PipelineResult
        result = PipelineResult(
            extract=ExtractResult({}, 0, 0),
            mask=MaskResult({}, {}, 0.3),
            finetune=None,
            evaluate=None,
        )
        assert result.extract is not None
        assert result.mask is not None
        assert result.finetune is None
