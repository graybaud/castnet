"""
test_sparsity.py — Tests for get_sparsity_stats.
"""

import torch
from src.finetuning import get_sparsity_stats


class TestSparsityStats:

    def test_fully_dense(self, mock_model):
        stats = get_sparsity_stats(mock_model)
        assert stats['overall'] == 0.0
        assert stats['total_active'] == 4*8 + 8*4  # 64

    def test_fully_sparse(self, mock_model):
        with torch.no_grad():
            mock_model.model.layers[0].mlp.fc1.weight.zero_()
            mock_model.model.layers[0].mlp.fc2.weight.zero_()
        stats = get_sparsity_stats(mock_model)
        assert stats['overall'] == 100.0
        assert stats['total_active'] == 0

    def test_half_sparse(self, mock_model):
        with torch.no_grad():
            mock_model.model.layers[0].mlp.fc1.weight.zero_()
        stats = get_sparsity_stats(mock_model)
        assert stats['overall'] == 50.0

    def test_layer_stats_present(self, mock_model):
        stats = get_sparsity_stats(mock_model)
        assert 'layers' in stats
        assert len(stats['layers']) == 2
