"""
test_architecture.py — Tests for detect_architecture.
"""

import pytest
from tests.extraction.conftest import MockPhi2Model

# Import from src.extraction
from src.extraction.config import detect_architecture


class TestArchitectureDetection:

    def test_phi2_string(self):
        assert detect_architecture("microsoft/phi-2") == "phi2"

    def test_opt_string(self):
        assert detect_architecture("facebook/opt-125m") == "opt"

    def test_llama_string(self):
        assert detect_architecture("meta-llama/Llama-2-7b") == "llama"

    def test_deepseek_moe_string(self):
        assert detect_architecture("deepseek-ai/deepseek-moe-16b-base") == "deepseek_moe"

    def test_generic_fallback(self):
        assert detect_architecture("bert-base-uncased") == "generic"

    def test_model_object(self):
        assert detect_architecture(MockPhi2Model()) == "phi2"

    def test_case_insensitive(self):
        assert detect_architecture("PHI-2") == "phi2"
        assert detect_architecture("DeepSeek-MoE") == "deepseek_moe"
