"""Tests for architecture detection and layer config."""

import pytest
from infrastructure.models.config import (
    detect_architecture,
    is_deepseek_v2,
    get_ffn_pairs,
    get_attention_layers,
)


class TestDetectArchitecture:

    def test_phi2(self):
        assert detect_architecture("microsoft/phi-2") == "phi2"

    def test_opt(self):
        assert detect_architecture("facebook/opt-125m") == "opt"

    def test_llama(self):
        assert detect_architecture("meta-llama/Llama-2-7b") == "llama"

    def test_deepseek_v2(self):
        assert detect_architecture("deepseek-ai/deepseek-v2") == "deepseek_v2"

    def test_deepseek_moe(self):
        assert detect_architecture("deepseek-ai/deepseek-moe-16b-base") == "deepseek_moe"

    def test_generic(self):
        assert detect_architecture("bert-base-uncased") == "generic"

    def test_case_insensitive(self):
        assert detect_architecture("PHI-2") == "phi2"


class TestIsDeepSeekV2:

    def test_v2(self):
        assert is_deepseek_v2("deepseek_v2") is True

    def test_v3(self):
        assert is_deepseek_v2("deepseek_v3") is True

    def test_moe(self):
        assert is_deepseek_v2("deepseek_moe") is False


class TestGetFFNPairs:

    def test_phi2_structure(self):
        # Mock model with phi2 structure
        import torch.nn as nn
        
        class MockLayer:
            def __init__(self):
                self.mlp = type('obj', (), {
                    'fc1': nn.Linear(4, 8),
                    'fc2': nn.Linear(8, 4),
                })
        
        class MockModel:
            def __init__(self):
                self.model = type('obj', (), {
                    'layers': [MockLayer() for _ in range(2)],
                })
        
        model = MockModel()
        pairs = get_ffn_pairs(model, "phi2")
        assert len(pairs) == 2
        assert pairs[0][0] == "layer0.fc1"
        assert pairs[0][2] == "layer0.fc2"


class TestGetAttentionLayers:

    def test_phi2(self):
        import torch.nn as nn
        
        class MockSelfAttn:
            def __init__(self):
                self.q_proj = nn.Linear(4, 4)
                self.k_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 4)
                self.out_proj = nn.Linear(4, 4)
        
        class MockLayer:
            def __init__(self):
                self.self_attn = MockSelfAttn()
        
        class MockModel:
            def __init__(self):
                self.model = type('obj', (), {
                    'layers': [MockLayer()],
                })
        
        model = MockModel()
        layers = get_attention_layers(model, "phi2")
        assert len(layers) == 4
        assert layers[0][0] == "layer0.attn.q_proj"
