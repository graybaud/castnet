"""Unit tests for APL bridge and APL strategies."""

import torch
import pytest
from domain.scoring.ports import WeightProvider, ActivationProvider
from domain.scoring.apl_strategies import APLFormulaStrategy
from infrastructure.scoring.apl_bridge import APLScoringBridge


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


class TestAPLScoringBridge:

    def test_wanda_formula(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        X = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        bridge = APLScoringBridge("|W| x mean(|act|)")
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(X)
        result = bridge.calculate(weights, activations, "test")
        assert result.shape == (2, 2)
        assert not torch.isnan(result).any()

    def test_magnitude_formula(self):
        W = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        bridge = APLScoringBridge("|W|")
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(None)
        result = bridge.calculate(weights, activations, "test")
        assert torch.allclose(result, W.abs())

    def test_gradient_formula(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        grad = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        bridge = APLScoringBridge("|W| x |grad|")
        weights = FakeWeightProvider(W, grad=grad)
        activations = FakeActivationProvider(None)
        result = bridge.calculate(weights, activations, "test")
        assert result.shape == (2, 2)
        assert not torch.isnan(result).any()

    def test_complex_formula(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        bridge = APLScoringBridge("max(|W|) / mean(|W|)")
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(None)
        result = bridge.calculate(weights, activations, "test")
        assert not torch.isnan(result).any()

    def test_formula_with_act_out(self):
        X_in = torch.randn(20, 10)
        X_out = torch.randn(20, 5)
        W = torch.randn(5, 10)
        bridge = APLScoringBridge("norm(act_out) / norm(act)")
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(X_in, X_out)
        result = bridge.calculate(weights, activations, "test")
        assert not torch.isnan(result).any()

    def test_no_gradient_in_formula(self):
        W = torch.randn(5, 10)
        bridge = APLScoringBridge("|W| x mean(|act|)")
        weights = FakeWeightProvider(W, grad=None)
        activations = FakeActivationProvider(torch.randn(20, 10))
        result = bridge.calculate(weights, activations, "test")
        assert result.shape == (5, 10)


class TestAPLFormulaStrategy:

    def test_wanda_via_apl(self):
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        X = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        strategy = APLFormulaStrategy("|W| x mean(|act|)")
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(X)
        scores = strategy.calculate(weights, activations, "test")
        assert scores.shape == (2, 2)
        assert scores.max() == 1.0
        assert scores.min() >= 0.0

    def test_normalizes_to_01(self):
        W = torch.randn(5, 10)
        X = torch.randn(20, 10)
        strategy = APLFormulaStrategy("|W| x mean(|act|)")
        scores = strategy.calculate(
            FakeWeightProvider(W), FakeActivationProvider(X), "test"
        )
        assert scores.min() >= 0.0
        assert scores.max() == 1.0

    def test_matches_builtin_magnitude(self):
        from domain.scoring.strategies import MagnitudeStrategy
        W = torch.randn(5, 10)
        apl_scores = APLFormulaStrategy("|W|").calculate(
            FakeWeightProvider(W), FakeActivationProvider(torch.zeros(1,10)), "test"
        )
        manual_scores = MagnitudeStrategy().calculate(
            FakeWeightProvider(W), FakeActivationProvider(torch.zeros(1,10)), "test"
        )
        assert torch.allclose(apl_scores, manual_scores, atol=0.01)

    def test_unknown_variable_raises(self):
        W = torch.randn(5, 10)
        bridge = APLScoringBridge("|W| x imaginary_variable")
        weights = FakeWeightProvider(W)
        activations = FakeActivationProvider(torch.randn(20, 10))
        with pytest.raises(Exception):
            bridge.calculate(weights, activations, "test")
