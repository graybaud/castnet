"""
Tests unitaires du domaine metier.

AUCUN import de transformers, datasets, ou torch.nn.
Uniquement torch.Tensor et les strategies du domaine.
"""

import torch
import pytest
from domain.scoring.ports import WeightProvider, ActivationProvider
from domain.scoring.strategies import (
    MagnitudeStrategy,
    GradientStrategy,
    WandaStrategy,
    GPSStrategy,
    get_strategy,
    STRATEGY_REGISTRY,
)
from domain.scoring.masks import apply_percentile_mask, count_sparsity


# ---------------------------------------------------------------------------
# Faux fournisseurs (mocks)
# ---------------------------------------------------------------------------

class FakeWeightProvider(WeightProvider):
    def __init__(self, W, bias=None, grad=None):
        self._W = W
        self._bias = bias
        self._grad = grad

    def get_weight(self, name):
        return self._W

    def get_bias(self, name):
        return self._bias

    def get_gradient(self, name):
        return self._grad

    def layer_names(self):
        return ["test_layer"]

    def forward_backward(self, batch):
        return 1.0

    def zero_grad(self):
        pass


class FakeActivationProvider(ActivationProvider):
    def __init__(self, X_in, X_out=None):
        self._X_in = X_in
        self._X_out = X_out if X_out is not None else X_in

    def get_input_activations(self, name):
        return self._X_in

    def get_output_activations(self, name):
        return self._X_out

    def attach(self):
        pass

    def detach(self):
        pass


# ---------------------------------------------------------------------------
# Tests Magnitude
# ---------------------------------------------------------------------------

class TestMagnitudeStrategy:

    def test_basic(self):
        W = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        strategy = MagnitudeStrategy()
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(None)

        scores = strategy.calculate(weights, activations, "test")

        expected = torch.tensor([[0.25, 0.50], [0.75, 1.00]])
        assert torch.allclose(scores, expected)

    def test_all_zeros(self):
        W = torch.zeros(3, 5)
        strategy = MagnitudeStrategy()
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(None)

        scores = strategy.calculate(weights, activations, "test")

        assert scores.shape == (3, 5)
        assert not torch.isnan(scores).any()


# ---------------------------------------------------------------------------
# Tests Gradient
# ---------------------------------------------------------------------------

class TestGradientStrategy:

    def test_basic(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        grad = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        strategy = GradientStrategy()
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(None)

        scores = strategy.calculate(weights, activations, "test")

        assert scores.shape == (2, 2)
        assert scores.max() == 1.0
        # Verifier que le max est bien en bas a droite (3.0 * 0.5 = 1.5, normalise a 1.0)
        assert scores[1, 1] == 1.0
        # Verifier que le min est bien en haut a gauche (1.0 * 0.5 = 0.5, normalise a 0.25)
        assert torch.allclose(scores[0, 0], torch.tensor(0.25), atol=0.01)

    def test_no_gradient_raises(self):
        W = torch.eye(3)
        strategy = GradientStrategy()
        weights = FakeWeightProvider(W, grad=None)
        activations = FakeActivationProvider(None)

        with pytest.raises(ValueError, match="Pas de gradient"):
            strategy.calculate(weights, activations, "test")


# ---------------------------------------------------------------------------
# Tests Wanda
# ---------------------------------------------------------------------------

class TestWandaStrategy:

    def test_basic(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        X = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        strategy = WandaStrategy()
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(X)

        scores = strategy.calculate(weights, activations, "test")

        assert scores.shape == (2, 2)
        assert scores.max() == 1.0
        assert not torch.isnan(scores).any()

    def test_normalized(self):
        W = torch.randn(5, 10)
        X = torch.randn(100, 10)
        strategy = WandaStrategy()
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(X)

        scores = strategy.calculate(weights, activations, "test")

        assert 0.0 <= scores.min() <= 1.0
        assert scores.max() == 1.0


# ---------------------------------------------------------------------------
# Tests GPS
# ---------------------------------------------------------------------------

class TestGPSStrategy:

    def test_basic(self):
        W = torch.randn(4, 6)
        X_in = torch.randn(50, 6)
        X_out = torch.randn(50, 4)
        strategy = GPSStrategy()
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(X_in, X_out)

        scores = strategy.calculate(weights, activations, "test")

        assert scores.shape == (4, 6)
        assert (scores >= 0).all()
        assert not torch.isnan(scores).any()
        # Apres normalisation, le max doit etre 1.0
        assert torch.allclose(scores.max(), torch.tensor(1.0))


# ---------------------------------------------------------------------------
# Tests Registre
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_all_strategies_registered(self):
        assert "magnitude" in STRATEGY_REGISTRY
        assert "gradient" in STRATEGY_REGISTRY
        assert "wanda" in STRATEGY_REGISTRY
        assert "gps" in STRATEGY_REGISTRY

    def test_get_strategy_returns_correct_type(self):
        assert isinstance(get_strategy("magnitude"), MagnitudeStrategy)
        assert isinstance(get_strategy("wanda"), WandaStrategy)

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(ValueError, match="Strategie inconnue"):
            get_strategy("methode_imaginaire")


# ---------------------------------------------------------------------------
# Tests Masques
# ---------------------------------------------------------------------------

class TestMasks:

    def test_percentile_mask_keep_half(self):
        scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = apply_percentile_mask(scores, keep_fraction=0.5)

        assert mask.sum() == 2
        assert mask[1, 0] == 1.0
        assert mask[1, 1] == 1.0

    def test_percentile_mask_all_zeros(self):
        scores = torch.zeros(3, 3)
        mask = apply_percentile_mask(scores, keep_fraction=0.3)

        assert mask.sum() == 0

    def test_count_sparsity(self):
        masks = {
            "layer1": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            "layer2": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        }
        stats = count_sparsity(masks)

        assert stats["overall_sparsity_pct"] == 37.5
        assert stats["total_active"] == 5
        assert stats["total_connections"] == 8
        assert stats["per_layer"]["layer1"] == 75.0
        assert stats["per_layer"]["layer2"] == 0.0


# ---------------------------------------------------------------------------
# Test parametrique : toutes les strategies
# ---------------------------------------------------------------------------

class TestAllStrategiesIntegration:

    @pytest.mark.parametrize("strategy_name", ["magnitude", "gradient", "wanda", "gps"])
    def test_strategy_runs_without_crash(self, strategy_name):
        W = torch.randn(5, 10)
        X_in = torch.randn(20, 10)
        X_out = torch.randn(20, 5)
        grad = torch.randn(5, 10)

        strategy = get_strategy(strategy_name)
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(X_in, X_out)

        scores = strategy.calculate(weights, activations, "test")

        assert scores.shape == (5, 10)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
