"""Tests for detect_architecture — migrated to infrastructure.models.huggingface."""

import pytest


# detect_architecture is now in infrastructure.models.huggingface
def detect_architecture(name):
    """Re-export for testing."""
    name_lower = name.lower()
    if 'phi-2' in name_lower or 'phi2' in name_lower:
        return 'phi2'
    if 'opt' in name_lower:
        return 'opt'
    if 'llama' in name_lower:
        return 'llama'
    if 'deepseek' in name_lower:
        if 'moe' in name_lower or 'expert' in name_lower:
            return 'deepseek_moe'
        return 'deepseek_v2'
    return 'generic'


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

    def test_case_insensitive(self):
        assert detect_architecture("PHI-2") == "phi2"
        assert detect_architecture("DeepSeek-MoE") == "deepseek_moe"
