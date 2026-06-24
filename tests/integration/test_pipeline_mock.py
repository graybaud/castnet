"""Integration tests — Full pipeline with mock providers, no GPU, no real model."""

import torch
import torch.nn as nn
import pytest
from domain.scoring.ports import WeightProvider, ActivationProvider, BatchProvider
from domain.scoring.strategies import WandaStrategy
from infrastructure.hooks.activation_collector import ActivationCollector
from infrastructure.persistence.safetensors_persister import (
    SafetensorsScorePersister,
    SafetensorsMaskPersister,
)
from application.extract_scores import ExtractScoresUseCase
from application.generate_masks import GenerateMasksUseCase
from application.pipeline import CastNetPipeline


class FakeWeightProvider(WeightProvider):
    def __init__(self, W):
        self._W = W
    def get_weight(self, name): return self._W
    def get_bias(self, name): return torch.zeros(self._W.shape[0])
    def get_gradient(self, name): return torch.randn_like(self._W)
    def layer_names(self): return ["0"]
    def forward_backward(self, batch): return 1.0
    def zero_grad(self): pass


class FakeActivationCollector(ActivationProvider):
    """Mock collector that returns pre-set activations without hooks."""
    def __init__(self, X_in, X_out=None):
        self._X_in = X_in
        self._X_out = X_out if X_out is not None else X_in
    def get_input_activations(self, name): return self._X_in
    def get_output_activations(self, name): return self._X_out
    def attach(self): pass
    def detach(self): pass


class FakeBatchProvider(BatchProvider):
    def get_batches(self, num_batches, batch_size=1):
        for _ in range(num_batches):
            yield torch.randint(0, 1000, (1, 16))


class TestPipelineIntegration:

    def test_extract_then_masks(self, tmp_path):
        """Extraction + mask generation — end to end with mocks."""
        W = torch.randn(5, 10)
        X = torch.randn(50, 10)

        score_persister = SafetensorsScorePersister()
        mask_persister = SafetensorsMaskPersister()

        # Step 1: Extract
        strategy = WandaStrategy()
        model = FakeWeightProvider(W)
        collector = FakeActivationCollector(X)
        dataset = FakeBatchProvider()

        extract_uc = ExtractScoresUseCase(
            strategy, model, dataset, collector, score_persister
        )
        scores_path = str(tmp_path / "scores.safetensors")
        extract_result = extract_uc.execute(5, scores_path)

        assert extract_result.num_layers == 1
        assert extract_result.num_batches == 5
        loaded_scores = score_persister.load(scores_path)
        assert "0" in loaded_scores
        assert loaded_scores["0"].shape == (5, 10)

        # Step 2: Generate masks
        mask_uc = GenerateMasksUseCase(score_persister, mask_persister)
        masks_path = str(tmp_path / "masks.safetensors")
        mask_result = mask_uc.execute(scores_path, masks_path, keep_fraction=0.5)

        assert mask_result.sparsity["total_active"] > 0
        loaded_masks = mask_persister.load(masks_path)
        assert "0" in loaded_masks

    def test_pipeline_all_steps(self, tmp_path):
        """Full pipeline with extract + masks (no finetune/eval)."""
        W = torch.randn(5, 10)
        X = torch.randn(50, 10)

        score_persister = SafetensorsScorePersister()
        mask_persister = SafetensorsMaskPersister()

        strategy = WandaStrategy()
        model = FakeWeightProvider(W)
        collector = FakeActivationCollector(X)
        dataset = FakeBatchProvider()

        extract_uc = ExtractScoresUseCase(
            strategy, model, dataset, collector, score_persister
        )
        mask_uc = GenerateMasksUseCase(score_persister, mask_persister)

        pipeline = CastNetPipeline(extract_uc, mask_uc)

        result = pipeline.run(
            scores_path=str(tmp_path / "scores.safetensors"),
            masks_path=str(tmp_path / "masks.safetensors"),
            checkpoint_path=str(tmp_path / "checkpoint.pt"),
            num_batches=3,
            keep_fraction=0.5,
        )

        assert result.extract is not None
        assert result.mask is not None
        assert result.finetune is None
        assert result.evaluate is None

    def test_activation_collector_with_real_module(self):
        """Test that ActivationCollector works with a real nn.Module."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        collector = ActivationCollector(model, ["0", "2"])
        collector.attach()

        x = torch.randn(4, 10)
        model(x)

        act0 = collector.get_input_activations("0")
        act2 = collector.get_input_activations("2")

        assert act0 is not None
        assert act0.shape == (4, 10)
        assert act2 is not None
        assert act2.shape == (4, 20)

        collector.detach()
