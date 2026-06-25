"""
test_ffn_pairs.py — Tests for get_ffn_pairs.
"""

from collections import Counter
from tests.extraction.conftest import MockPhi2Model, MockOPTModel, MockLLaMAModel
from src.extraction.config import get_ffn_pairs


class TestFFNPairs:

    def test_phi2_structure(self):
        pairs = get_ffn_pairs(MockPhi2Model(num_layers=2), "phi2")
        assert len(pairs) == 2
        assert pairs[0] == ("layer0.fc1", pairs[0][1], "layer0.fc2", pairs[0][3])

    def test_llama_structure(self):
        pairs = get_ffn_pairs(MockLLaMAModel(num_layers=1), "llama")
        assert len(pairs) == 2
        fc2_names = [p[2] for p in pairs]
        assert fc2_names[0] == "layer0.down_proj"
        assert fc2_names[1] == "layer0.down_proj"

    def test_llama_down_proj_count(self):
        model = MockLLaMAModel(num_layers=3)
        pairs = get_ffn_pairs(model, "llama")
        counts = Counter(p[2] for p in pairs)
        for name, count in counts.items():
            assert count == 2, f"{name} appears {count} times, expected 2"

    def test_opt_count(self):
        pairs = get_ffn_pairs(MockOPTModel(num_layers=2), "opt")
        assert len(pairs) == 2
