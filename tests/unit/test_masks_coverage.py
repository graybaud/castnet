"""Complete coverage for masks.py — the missing 5 lines."""

import torch
from domain.scoring.masks import apply_masks_to_weights
from domain.scoring.ports import WeightProvider


class MiniWeightProvider(WeightProvider):
    def __init__(self, W):
        self._W = W
    def get_weight(self, name): return self._W
    def get_bias(self, name): return None
    def get_gradient(self, name): return None
    def layer_names(self): return ["test"]
    def forward_backward(self, batch): return 0.0
    def zero_grad(self): pass


class TestApplyMasksToWeights:

    def test_mask_applied_correctly(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        provider = MiniWeightProvider(W.clone())

        apply_masks_to_weights({"test": mask}, provider)

        expected = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        assert torch.equal(provider.get_weight("test"), expected)

    def test_all_ones_mask_does_nothing(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.ones(2, 2)
        provider = MiniWeightProvider(W.clone())

        apply_masks_to_weights({"test": mask}, provider)

        assert torch.equal(provider.get_weight("test"), W)

    def test_all_zeros_mask_zeroes_everything(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.zeros(2, 2)
        provider = MiniWeightProvider(W.clone())

        apply_masks_to_weights({"test": mask}, provider)

        assert provider.get_weight("test").sum() == 0.0

    def test_multiple_layers(self):
        class MultiProvider(WeightProvider):
            def __init__(self):
                self._weights = {
                    "L0": torch.ones(2, 3),
                    "L1": torch.ones(3, 2) * 2,
                }
            def get_weight(self, name): return self._weights[name]
            def get_bias(self, name): return None
            def get_gradient(self, name): return None
            def layer_names(self): return list(self._weights.keys())
            def forward_backward(self, batch): return 0.0
            def zero_grad(self): pass

        provider = MultiProvider()
        masks = {
            "L0": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            "L1": torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]]),
        }

        apply_masks_to_weights(masks, provider)

        assert provider.get_weight("L0")[0, 0] == 1.0
        assert provider.get_weight("L0")[0, 1] == 0.0
        assert provider.get_weight("L0")[1, 1] == 1.0
        assert provider.get_weight("L1")[0, 0] == 2.0
        assert provider.get_weight("L1")[1, 0] == 0.0
