"""
test_ffn_patterns.py — Tests for _check_ffn and FFN_PATTERNS.
"""

from infrastructure.models.huggingface import HuggingFaceWeightProvider
FFN_PATTERNS = HuggingFaceWeightProvider.FFN_PATTERNS

def _check_ffn(name):
    return any(p in name.lower() for p in FFN_PATTERNS)


class TestIsFFNLayer:

    def test_standard_patterns(self):
        assert _check_ffn("model.layers.0.mlp.fc1")
        assert _check_ffn("model.layers.0.mlp.fc2")
        assert _check_ffn("model.layers.0.mlp.c_fc")
        assert _check_ffn("model.layers.0.mlp.c_proj")

    def test_llama_patterns(self):
        assert _check_ffn("model.layers.0.mlp.gate_proj")
        assert _check_ffn("model.layers.0.mlp.up_proj")
        assert _check_ffn("model.layers.0.mlp.down_proj")

    def test_gpt_neox_patterns(self):
        assert _check_ffn("gpt_neox.layers.0.mlp.dense_h_to_4h")
        assert _check_ffn("gpt_neox.layers.0.mlp.dense_4h_to_h")

    def test_non_ffn_patterns(self):
        assert not _check_ffn("model.layers.0.self_attn.q_proj")
        assert not _check_ffn("model.layers.0.self_attn.k_proj")
        assert not _check_ffn("model.layers.0.self_attn.v_proj")
        assert not _check_ffn("model.layers.0.self_attn.o_proj")
        assert not _check_ffn("lm_head")
        assert not _check_ffn("model.embed_tokens")

    def test_case_insensitive(self):
        assert _check_ffn("Model.Layers.0.MLP.FC1")
        assert _check_ffn("MODEL.LAYERS.0.MLP.GATE_PROJ")

    def test_ffn_patterns_completeness(self):
        """Ensure all expected patterns are present."""
        expected = {'fc1', 'fc2', 'c_fc', 'c_proj', 'gate_proj',
                     'up_proj', 'down_proj', 'dense_h_to_4h', 'dense_4h_to_h',
                     'mlp.fc1', 'mlp.fc2', 'mlp.c_fc', 'mlp.c_proj'}
        assert set(FFN_PATTERNS) == expected
