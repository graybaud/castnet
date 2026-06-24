"""Unit tests for infrastructure adapters.

Uses mocking to avoid GPU and real model downloads.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch
from infrastructure.persistence.safetensors_persister import (
    SafetensorsScorePersister,
    SafetensorsMaskPersister,
)


class TestSafetensorsPersisters:

    def test_score_persister_save_load(self, tmp_path):
        persister = SafetensorsScorePersister()
        scores = {
            "layer0": torch.randn(5, 10),
            "layer1": torch.randn(10, 20),
        }
        path = str(tmp_path / "scores.safetensors")

        persister.save(scores, path)
        loaded = persister.load(path)

        assert set(loaded.keys()) == set(scores.keys())
        for name in scores:
            assert torch.equal(loaded[name], scores[name])

    def test_mask_persister_save_load(self, tmp_path):
        persister = SafetensorsMaskPersister()
        masks = {
            "layer0": torch.randint(0, 2, (5, 10)).float(),
            "layer1": torch.randint(0, 2, (10, 20)).float(),
        }
        path = str(tmp_path / "masks.safetensors")

        persister.save(masks, path)
        loaded = persister.load(path)

        assert set(loaded.keys()) == set(masks.keys())
        for name in masks:
            assert torch.equal(loaded[name], masks[name])

    def test_empty_dict(self, tmp_path):
        persister = SafetensorsScorePersister()
        path = str(tmp_path / "empty.safetensors")

        persister.save({}, path)
        loaded = persister.load(path)

        assert loaded == {}

    def test_score_persister_implements_interface(self):
        from domain.scoring.ports import ScorePersister
        assert issubclass(SafetensorsScorePersister, ScorePersister)

    def test_mask_persister_implements_interface(self):
        from domain.scoring.ports import MaskPersister
        assert issubclass(SafetensorsMaskPersister, MaskPersister)


class TestActivationCollector:

    def test_attach_detach(self):
        """Test que le collector s'attache et se detache sans erreur."""
        import torch.nn as nn
        from infrastructure.hooks.activation_collector import ActivationCollector

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        layer_names = ["0", "2"]

        collector = ActivationCollector(model, layer_names)
        collector.attach()
        collector.detach()

    def test_collects_activations(self):
        """Test que les activations sont bien collectees."""
        import torch.nn as nn
        from infrastructure.hooks.activation_collector import ActivationCollector

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        layer_names = ["0", "2"]

        collector = ActivationCollector(model, layer_names)
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

    def test_no_activation_before_forward(self):
        import torch.nn as nn
        from infrastructure.hooks.activation_collector import ActivationCollector

        model = nn.Linear(10, 5)
        collector = ActivationCollector(model, ["0"])
        collector.attach()

        assert collector.get_input_activations("0") is None

        collector.detach()
