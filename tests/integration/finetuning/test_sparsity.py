"""Tests for sparsity measurement — migrated from legacy."""

import torch
import torch.nn as nn
import pytest
from infrastructure.finetuning.sparsity import get_ffn_layers_generic, get_sparsity_stats


@pytest.fixture
def mock_model():
    """Simple model with fc1 and fc2."""
    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(4, 8))
    model.add_module('fc2', nn.Linear(8, 4))
    return model


class TestFFNDetection:

    def test_detects_fc1_fc2(self, mock_model):
        layers = get_ffn_layers_generic(mock_model)
        names = [n for n, _ in layers]
        assert any('fc1' in n for n in names)
        assert any('fc2' in n for n in names)

    def test_excludes_attention(self, mock_model):
        mock_model.add_module('q_proj', nn.Linear(4, 4))
        layers = get_ffn_layers_generic(mock_model)
        names = [n for n, _ in layers]
        assert not any('q_proj' in n for n in names)

    def test_all_are_linear(self, mock_model):
        layers = get_ffn_layers_generic(mock_model)
        for _, module in layers:
            assert isinstance(module, nn.Linear)


class TestSparsityStats:

    def test_fully_dense(self, mock_model):
        stats = get_sparsity_stats(mock_model)
        assert stats['overall'] == 0.0
        assert stats['total_active'] == 4 * 8 + 8 * 4

    def test_fully_sparse(self, mock_model):
        with torch.no_grad():
            mock_model.fc1.weight.zero_()
            mock_model.fc2.weight.zero_()
        stats = get_sparsity_stats(mock_model)
        assert stats['overall'] == 100.0
        assert stats['total_active'] == 0

    def test_half_sparse(self, mock_model):
        with torch.no_grad():
            mock_model.fc1.weight.zero_()
        stats = get_sparsity_stats(mock_model)
        assert stats['overall'] == 50.0

    def test_layer_stats_present(self, mock_model):
        stats = get_sparsity_stats(mock_model)
        assert 'layers' in stats
        assert len(stats['layers']) == 2
