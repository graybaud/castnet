"""
test_ffn_layers.py — Tests for get_ffn_layers.
"""

import torch
import torch.nn as nn
from tests.extraction.conftest import MockPhi2Model, MockOPTModel, MockLLaMAModel
from src.extraction.config import get_ffn_layers


class TestFFNLayers:

    def test_phi2_count(self):
        layers = get_ffn_layers(MockPhi2Model(num_layers=3), "phi2")
        assert len(layers) == 6

    def test_phi2_names(self):
        layers = get_ffn_layers(MockPhi2Model(num_layers=2), "phi2")
        assert layers[0][0] == "layer0.fc1"
        assert layers[1][0] == "layer1.fc1"
        assert layers[3][0] == "layer1.fc2"

    def test_phi2_shapes(self):
        model = MockPhi2Model(d_model=128, d_ff=384)
        layers = get_ffn_layers(model, "phi2")
        assert layers[0][1].weight.shape == torch.Size([384, 128])
        assert layers[1][1].weight.shape == torch.Size([384, 128])

    def test_opt_count(self):
        layers = get_ffn_layers(MockOPTModel(num_layers=2), "opt")
        assert len(layers) == 4

    def test_llama_count_and_names(self):
        model = MockLLaMAModel(num_layers=2)
        layers = get_ffn_layers(model, "llama")
        assert len(layers) == 6
        names = {n for n, _ in layers}
        assert "layer0.gate_proj" in names
        assert "layer0.up_proj" in names
        assert "layer0.down_proj" in names

    def test_auto_detect(self):
        layers = get_ffn_layers(MockPhi2Model())
        assert len(layers) == 4
