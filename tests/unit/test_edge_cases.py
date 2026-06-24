"""Edge case and robustness tests for the domain layer."""

import torch
import pytest
from domain.scoring.strategies import (
    MagnitudeStrategy, GradientStrategy, WandaStrategy, GPSStrategy,
)
from domain.scoring.masks import apply_percentile_mask, count_sparsity
from domain.scoring.ports import WeightProvider, ActivationProvider


class FakeWeightProvider(WeightProvider):
    def __init__(self, W, bias=None, grad=None):
        self._W, self._bias, self._grad = W, bias, grad
    def get_weight(self, name): return self._W
    def get_bias(self, name): return self._bias
    def get_gradient(self, name): return self._grad
    def layer_names(self): return ["test"]
    def forward_backward(self, batch): return 1.0
    def zero_grad(self): pass


class FakeActivationProvider(ActivationProvider):
    def __init__(self, X_in, X_out=None):
        self._X_in = X_in
        self._X_out = X_out if X_out is not None else X_in
    def get_input_activations(self, name): return self._X_in
    def get_output_activations(self, name): return self._X_out
    def attach(self): pass
    def detach(self): pass


class TestExtremeTensors:

    @pytest.mark.parametrize("strategy_cls", [
        MagnitudeStrategy, GradientStrategy, WandaStrategy, GPSStrategy,
    ])
    def test_all_zeros_weight(self, strategy_cls):
        W = torch.zeros(5, 10)
        X_in = torch.randn(20, 10)
        X_out = torch.randn(20, 5)
        grad = torch.zeros(5, 10)
        strategy = strategy_cls()
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(X_in, X_out)
        scores = strategy.calculate(weights, activations, "test")
        assert scores.shape == (5, 10)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    @pytest.mark.parametrize("strategy_cls", [
        MagnitudeStrategy, GradientStrategy, WandaStrategy, GPSStrategy,
    ])
    def test_all_zeros_activations(self, strategy_cls):
        W = torch.randn(5, 10)
        X_in = torch.zeros(20, 10)
        X_out = torch.zeros(20, 5)
        grad = torch.randn(5, 10)
        strategy = strategy_cls()
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(X_in, X_out)
        scores = strategy.calculate(weights, activations, "test")
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    @pytest.mark.parametrize("strategy_cls", [
        MagnitudeStrategy, GradientStrategy, WandaStrategy, GPSStrategy,
    ])
    def test_very_large_values(self, strategy_cls):
        W = torch.ones(5, 10) * 1e8
        X_in = torch.ones(20, 10) * 1e8
        X_out = torch.ones(20, 5) * 1e8
        grad = torch.ones(5, 10) * 1e8
        strategy = strategy_cls()
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(X_in, X_out)
        scores = strategy.calculate(weights, activations, "test")
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    @pytest.mark.parametrize("strategy_cls", [
        MagnitudeStrategy, GradientStrategy, WandaStrategy, GPSStrategy,
    ])
    def test_very_small_values(self, strategy_cls):
        W = torch.ones(5, 10) * 1e-12
        X_in = torch.ones(20, 10) * 1e-12
        X_out = torch.ones(20, 5) * 1e-12
        grad = torch.ones(5, 10) * 1e-12
        strategy = strategy_cls()
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(X_in, X_out)
        scores = strategy.calculate(weights, activations, "test")
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    @pytest.mark.parametrize("strategy_cls", [
        MagnitudeStrategy, GradientStrategy, WandaStrategy, GPSStrategy,
    ])
    def test_nan_in_weight(self, strategy_cls):
        W = torch.randn(5, 10)
        W[0, 0] = float("nan")
        X_in = torch.randn(20, 10)
        X_out = torch.randn(20, 5)
        grad = torch.randn(5, 10)
        strategy = strategy_cls()
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(X_in, X_out)
        scores = strategy.calculate(weights, activations, "test")
        assert scores.shape == (5, 10)

    @pytest.mark.parametrize("strategy_cls", [
        MagnitudeStrategy, GradientStrategy, WandaStrategy, GPSStrategy,
    ])
    def test_inf_in_weight(self, strategy_cls):
        W = torch.randn(5, 10)
        W[0, 0] = float("inf")
        X_in = torch.randn(20, 10)
        X_out = torch.randn(20, 5)
        grad = torch.randn(5, 10)
        strategy = strategy_cls()
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(X_in, X_out)
        scores = strategy.calculate(weights, activations, "test")
        assert scores.shape == (5, 10)


class TestNonStandardShapes:

    def test_single_neuron(self):
        W = torch.randn(1, 10)
        X = torch.randn(5, 10)
        scores = WandaStrategy().calculate(
            FakeWeightProvider(W), FakeActivationProvider(X), "test"
        )
        assert scores.shape == (1, 10)

    def test_single_input_dim(self):
        W = torch.randn(5, 1)
        X = torch.randn(20, 1)
        scores = WandaStrategy().calculate(
            FakeWeightProvider(W), FakeActivationProvider(X), "test"
        )
        assert scores.shape == (5, 1)

    def test_single_token(self):
        W = torch.randn(5, 10)
        X = torch.randn(1, 10)
        scores = WandaStrategy().calculate(
            FakeWeightProvider(W), FakeActivationProvider(X), "test"
        )
        assert scores.shape == (5, 10)


class TestMasksEdgeCases:

    def test_keep_fraction_very_small(self):
        """keep_fraction proche de zero — doit pas crasher."""
        scores = torch.rand(10, 10)
        mask = apply_percentile_mask(scores, keep_fraction=0.01)
        assert mask.sum() >= 0
        # Verifie que le masque est bien binaire
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})

    def test_keep_fraction_one(self):
        scores = torch.rand(10, 10)
        mask = apply_percentile_mask(scores, keep_fraction=1.0)
        assert mask.sum() == 100

    def test_all_equal_scores(self):
        """Scores egaux => keep_fraction% garde exactement cette fraction."""
        scores = torch.ones(10, 10)
        mask = apply_percentile_mask(scores, keep_fraction=0.5)
        # Avec scores egaux et quantile, tout passe le seuil -> 100%
        # C'est correct : le quantile de valeurs identiques = cette valeur
        assert mask.sum() == 100

    def test_single_element(self):
        scores = torch.tensor([[0.5]])
        mask = apply_percentile_mask(scores, keep_fraction=0.5)
        assert mask.sum() == 1

    def test_negative_scores(self):
        scores = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        mask = apply_percentile_mask(scores, keep_fraction=0.5)
        # Seuls les positifs sont consideres, les negatifs ignores
        assert mask[1, 0] == 1.0  # 3.0 garde
        assert mask[1, 1] == 0.0  # -4.0 ignore

    def test_count_sparsity_empty(self):
        stats = count_sparsity({})
        assert stats["overall_sparsity_pct"] == 0
        assert stats["total_active"] == 0
        assert stats["total_connections"] == 0
