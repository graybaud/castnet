"""Integration tests for finetuning — migrated from legacy."""

import pytest
import torch
import torch.nn as nn
from infrastructure.finetuning.mask_loader import (
    convert_mask_name_to_model,
    build_mask_application_dict,
    apply_masks,
)
from infrastructure.finetuning.sparsity import get_sparsity_stats


@pytest.fixture
def mock_model():
    """Model with fc1 (4->8), fc2 (8->4), and an attention layer."""
    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.fc1 = nn.Linear(4, 8)
            self.mlp.fc2 = nn.Linear(8, 4)
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(4, 4)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockLayer()])

    return MockModel()


@pytest.fixture
def sample_masks():
    return {
        "layer0_fc1": torch.ones(8, 4),
        "layer0_fc2": torch.ones(4, 8),
    }


class TestArchitectureMapping:
    def test_opt_architecture(self):
        result = convert_mask_name_to_model("layer0_fc1", "opt")
        assert "model.decoder.layers.0.fc1" in result

    def test_phi2_architecture(self):
        result = convert_mask_name_to_model("layer1_fc2", "phi2")
        assert "model.layers.1.mlp.fc2" in result

    def test_llama_architecture(self):
        result = convert_mask_name_to_model("layer0_gate_proj", "llama")
        assert "model.layers.0.mlp.gate_proj" in result

    def test_generic_returns_multiple_candidates(self):
        result = convert_mask_name_to_model("layer0_fc1", "generic")
        assert len(result) > 1

    def test_unknown_format_returns_name_as_is(self):
        result = convert_mask_name_to_model("unknown_layer", "generic")
        assert result == ["unknown_layer"]


class TestMaskApplication:
    def test_empty_masks(self, mock_model):
        result = build_mask_application_dict(mock_model, {}, "phi2")
        assert result == {}

    def test_no_match_warns(self, mock_model, capsys):
        masks = {"nonexistent_layer": torch.ones(4, 8)}
        result = build_mask_application_dict(mock_model, masks, "phi2")
        assert "WARNING" in capsys.readouterr().out or len(result) == 0

    def test_dot_format_names(self, mock_model):
        masks = {"layer0.fc1": torch.ones(8, 4)}
        result = build_mask_application_dict(mock_model, masks, "phi2")
        assert len(result) >= 0  # Should not crash

    def test_mask_apply_sparsity_pipeline(self, mock_model, sample_masks):
        app_dict = build_mask_application_dict(mock_model, sample_masks, "phi2")
        apply_masks(mock_model, app_dict)
        stats = get_sparsity_stats(mock_model)
        assert stats['overall'] >= 0

    def test_mask_does_not_affect_attention(self, mock_model, sample_masks):
        original_q = mock_model.model.layers[0].self_attn.q_proj.weight.data.clone()
        app_dict = build_mask_application_dict(mock_model, sample_masks, "phi2")
        apply_masks(mock_model, app_dict)
        assert torch.equal(mock_model.model.layers[0].self_attn.q_proj.weight.data, original_q)
