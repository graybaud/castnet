"""Integration tests — All strategies on tiny model."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAllStrategiesIntegration:

    @pytest.mark.parametrize("method", ["magnitude", "gradient", "wanda", "gps"])
    def test_extract_with_method(self, method):
        """Each built-in strategy should work end-to-end on tiny model."""
        from infrastructure.models.huggingface import HuggingFaceWeightProvider
        from infrastructure.data.providers import WikiTextProvider
        from infrastructure.hooks.activation_collector import ActivationCollector
        from infrastructure.persistence.safetensors_persister import SafetensorsScorePersister
        from domain.scoring.strategies import get_strategy
        from application.extract_scores import ExtractScoresUseCase

        model = HuggingFaceWeightProvider("sshleifer/tiny-gpt2", "cpu")
        dataset = WikiTextProvider("sshleifer/tiny-gpt2", max_len=32)

        real_names = list(dict(model._model.named_modules()).keys())
        ffn_names = [
            n for n in real_names
            if any(p in n.lower() for p in HuggingFaceWeightProvider.FFN_PATTERNS)
        ]
        collector = ActivationCollector(model._model, ffn_names)
        persister = SafetensorsScorePersister()

        strategy = get_strategy(method)
        uc = ExtractScoresUseCase(strategy, model, dataset, collector, persister)

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name

        try:
            result = uc.execute(2, path)
            assert result.num_layers > 0
            assert os.path.exists(path)

            scores = persister.load(path)
            for name, score in scores.items():
                assert score.max() <= 1.0 + 1e-6
                assert score.min() >= 0.0
        finally:
            if os.path.exists(path):
                os.unlink(path)
