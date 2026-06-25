"""
test_ffn_detection.py — Tests for detect_architecture, get_ffn_layers_generic.
"""

import torch
from src.finetuning import (
    detect_architecture,
    get_ffn_layers_generic,
)


class TestDetectArchitecture:

    def test_phi2(self):
        assert detect_architecture("microsoft/phi-2") == "phi2"

    def test_opt(self):
        assert detect_architecture("facebook/opt-125m") == "opt"

    def test_llama(self):
        assert detect_architecture("meta-llama/Llama-2-7b") == "llama"

    def test_generic(self):
        assert detect_architecture("bert-base-uncased") == "generic"


class TestFFNDetection:

    def test_detects_fc1_fc2(self, mock_model):
        layers = get_ffn_layers_generic(mock_model)
        names = [n for n, _ in layers]
        assert any('fc1' in n for n in names)
        assert any('fc2' in n for n in names)

    def test_excludes_attention(self, mock_model):
        layers = get_ffn_layers_generic(mock_model)
        names = [n for n, _ in layers]
        assert not any('q_proj' in n for n in names)

    def test_all_are_linear(self, mock_model):
        layers = get_ffn_layers_generic(mock_model)
        for _, module in layers:
            assert isinstance(module, torch.nn.Linear)
