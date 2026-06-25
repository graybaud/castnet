"""
test_infer_method.py — Tests for infer_method_from_filename.
"""

from src.extraction.utils.io import infer_method_from_filename


class TestInferMethodFromFilename:

    def test_gradient(self):
        assert infer_method_from_filename("phi2_gradient_scores_b100.pt") == "gradient"

    def test_wanda(self):
        assert infer_method_from_filename("opt_wanda_scores.pt") == "wanda"

    def test_wanda_chain(self):
        assert infer_method_from_filename("model_wanda_chain_scores.pt") == "wanda_chain"

    def test_chain(self):
        assert infer_method_from_filename("model_chain_scores.pt") == "chain"

    def test_unknown(self):
        assert infer_method_from_filename("random.pt") == "unknown"

    def test_case_insensitive(self):
        assert infer_method_from_filename("MODEL_WANDA_SCORES.pt") == "wanda"
