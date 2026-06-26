"""Integration tests for evaluation — migrated from legacy evaluation/test_integration.py."""

import torch
from domain.constants import is_ffn_layer
from domain.scoring.masks import measure_sparsity_from_weights
from domain.metrics.benchmark import apply_magnitude_pruning_all_layers


class MockFFNModel:
    """Minimal model with fc1/fc2 + attn (mirrors legacy conftest.MockFFNModel)."""
    def __init__(self):
        self.fc1_weight = torch.nn.Parameter(torch.randn(8, 4))
        self.fc2_weight = torch.nn.Parameter(torch.randn(4, 8))
        self.attn_weight = torch.nn.Parameter(torch.randn(4, 4))

    def named_modules(self):
        return iter([])  # Not needed for weight-based functions

    def get_weights_dict(self):
        return {
            "fc1": self.fc1_weight.data,
            "fc2": self.fc2_weight.data,
            "attn": self.attn_weight.data,
        }


class TestEvaluationIntegration:

    def test_prune_measure_restore(self):
        """Full cycle: save → prune → measure → restore → verify."""
        model = MockFFNModel()
        weights = model.get_weights_dict()
        ffn_weights = {k: v for k, v in weights.items() if is_ffn_layer(k)}

        # Original edges
        _, original_edges, _, _ = measure_sparsity_from_weights(ffn_weights)

        # Backup
        backup = {k: v.clone() for k, v in ffn_weights.items()}

        # Prune
        pruned, sparsities = apply_magnitude_pruning_all_layers(ffn_weights, threshold=0.05)
        _, pruned_edges, _, _ = measure_sparsity_from_weights(pruned)

        assert pruned_edges <= original_edges
        assert sparsities["overall"] > 0.0

        # Restore
        for k in ffn_weights:
            ffn_weights[k].data.copy_(backup[k])
        _, restored_edges, _, _ = measure_sparsity_from_weights(ffn_weights)
        assert restored_edges == original_edges

    def test_ffn_model_structure(self):
        """Verify FFN patterns detect the right layers."""
        assert is_ffn_layer("fc1")
        assert is_ffn_layer("fc2")
        assert not is_ffn_layer("attn")

        model = MockFFNModel()
        weights = model.get_weights_dict()
        ffn_names = [k for k in weights if is_ffn_layer(k)]
        assert "fc1" in ffn_names
        assert "fc2" in ffn_names
        assert "attn" not in ffn_names
